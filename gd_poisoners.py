# poisoning/gd_poisoners.py
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV


def mse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean((y_pred - y_true) ** 2))


class RidgeGDPoisoner:
    """
    Fast, paper-consistent GD poisoning for Ridge regression using analytic gradients
    (in the style of OptP / GD poisoners from the original experiment code).

    Key paper settings supported:
      - eta (step size), beta (line search decay), eps (stopping threshold)
      - objective on validation MSE
      - optimize y as well as x
      - lambda selected by CV and kept fixed during attack
      - clip X, y to [0, 1]

    Notes:
      - No np.matrix usage (compatible with modern sklearn/numpy).
      - No logging/visualization. Designed to run fast and print summary only.
    """

    def __init__(
        self,
        Xtr, ytr,
        Xval, yval,
        Xte, yte,
        eta: float,
        beta: float = 0.75,
        eps: float = 1e-5,
        max_iters: int = 50,
        optimize_y: bool = True,
        random_state: int = 123,
    ):
        self.Xtr = np.asarray(Xtr, dtype=float)
        self.ytr = np.asarray(ytr, dtype=float).reshape(-1)

        self.Xval = np.asarray(Xval, dtype=float)
        self.yval = np.asarray(yval, dtype=float).reshape(-1)

        self.Xte = np.asarray(Xte, dtype=float)
        self.yte = np.asarray(yte, dtype=float).reshape(-1)

        self.eta = float(eta)
        self.beta = float(beta)
        self.eps = float(eps)
        self.max_iters = int(max_iters)
        self.optimize_y = bool(optimize_y)

        self.rng = np.random.RandomState(random_state)

        # paper: lambda selected via cross-validation
        alphas = np.logspace(-4, 3, 20)
        cv = RidgeCV(alphas=alphas, fit_intercept=True)
        cv.fit(self.Xtr, self.ytr)
        self.lam = float(cv.alpha_)

        self.n = self.Xtr.shape[0]   # clean train size
        self.d = self.Xtr.shape[1]   # number of features

        # Precompute (paper-style) Sigma and mu from CLEAN training data
        # Sigma = (X^T X)/n + lam I
        XtX = (self.Xtr.T @ self.Xtr) / self.n
        self.Sigma = XtX + self.lam * np.eye(self.d)
        self.mu = np.mean(self.Xtr, axis=0).reshape(-1, 1)

        # Equation system matrix (d+1)x(d+1): [[Sigma, mu^T],[mu,1]]
        top = np.hstack([self.Sigma, self.mu])
        bot = np.hstack([self.mu.T, np.array([[1.0]])])
        self.eq7lhs = np.vstack([top, bot])

    def fit_ridge(self, X, y) -> Ridge:
        model = Ridge(alpha=self.lam, fit_intercept=True, max_iter=10000)
        model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1))
        return model

    # --- Core analytic gradient pieces (from original GD poisoner math) ---

    def _compute_m(self, w, b, xc, yc):
        """
        m = x_c w^T + (w^T x_c + b - y_c) I
        Shapes:
          w: (d,)
          xc: (d,)
        Returns m: (d,d)
        """
        w = w.reshape(-1)
        xc = xc.reshape(-1)
        err = float(w @ xc + b - yc)
        m = np.outer(xc, w) + err * np.eye(self.d)
        return m

    def _compute_wb_derivatives(self, w, m, xc):
        """
        Solve for [w_xc, b_xc, w_yc, b_yc] from the linear system:
          eq7lhs * wbxc = eq7rhs
        where eq7rhs = -(1/n) * [[m, -x_c^T],
                                 [w^T, -1]]

        Returns:
          wxc: (d,d)
          bxc: (d,)
          wyc: (d,)
          byc: scalar
        """
        xc = xc.reshape(-1)

        # Build eq7rhs as (d+1)x(d+1)
        # Left block: m (dxd)
        # Right col: -x_c^T (dx1)
        # Bottom row: [w^T, -1]
        rhs_top = np.hstack([m, -xc.reshape(-1, 1)])
        rhs_bot = np.hstack([w.reshape(1, -1), np.array([[-1.0]])])
        eq7rhs = -(1.0 / self.n) * np.vstack([rhs_top, rhs_bot])

        wbxc = np.linalg.lstsq(self.eq7lhs, eq7rhs, rcond=None)[0]  # (d+1)x(d+1)

        wxc = wbxc[:-1, :-1]      # (d,d)
        bxc = wbxc[-1, :-1]       # (d,)
        wyc = wbxc[:-1, -1]       # (d,)
        byc = float(wbxc[-1, -1]) # scalar
        return wxc, bxc, wyc, byc

    def _attack_grad_val(self, model, wxc, bxc, wyc, byc):
        """
        Gradient of validation MSE wrt poison point (x_c, y_c).
        Mirrors original comp_attack_vld.

        attackx shape (d,)
        attacky scalar
        """
        res = (model.predict(self.Xval) - self.yval)  # (n_val,)

        gradx = self.Xval @ wxc + bxc  # (n_val, d)
        grady = self.Xval @ wyc.reshape(-1, 1) + byc  # (n_val,1)

        nval = self.Xval.shape[0]
        attackx = (res.reshape(1, -1) @ gradx).reshape(-1) / nval
        attacky = float((res.reshape(1, -1) @ grady).reshape(-1)[0] / nval)

        return attackx, attacky

    def _objective_val(self, model) -> float:
        return mse(self.yval, model.predict(self.Xval))

    def _linesearch(self, Xp, yp, i, attackx, attacky, clip_X, clip_y, max_ls=100):
        """
        Line search for a single poison point i.
        """
        eta = self.eta
        best_x = Xp[i].copy()
        best_y = float(yp[i])

        # Current objective
        Xfull = np.vstack([self.Xtr, Xp])
        yfull = np.concatenate([self.ytr, yp])
        model = self.fit_ridge(Xfull, yfull)
        best_obj = self._objective_val(model)

        for _ in range(max_ls):
            cand_x = np.clip(best_x + eta * attackx, clip_X[0], clip_X[1])
            cand_y = best_y
            if self.optimize_y:
                cand_y = float(np.clip(best_y + eta * attacky, clip_y[0], clip_y[1]))

            Xp_c = Xp.copy()
            yp_c = yp.copy()
            Xp_c[i] = cand_x
            yp_c[i] = cand_y

            Xfull_c = np.vstack([self.Xtr, Xp_c])
            yfull_c = np.concatenate([self.ytr, yp_c])
            model_c = self.fit_ridge(Xfull_c, yfull_c)
            obj_c = self._objective_val(model_c)

            if obj_c >= best_obj:
                return cand_x, cand_y, obj_c  # accept

            eta *= self.beta

        return best_x, best_y, best_obj

    def poison_optp(
        self,
        num_poison: int,
        clip_X=(0.0, 1.0),
        clip_y=(0.0, 1.0),
        max_linesearch_iters: int = 100,
    ):
        """
        Runs GD poisoning, returns:
          Xp_best, yp_best, poisoned_model_best
        """
        num_poison = int(num_poison)

        # Init poisoning points by sampling training points (common + paper-consistent)
        idx = self.rng.choice(self.n, size=num_poison, replace=True)
        Xp = self.Xtr[idx].copy()
        yp = self.ytr[idx].copy()

        # Evaluate starting point
        Xfull = np.vstack([self.Xtr, Xp])
        yfull = np.concatenate([self.ytr, yp])
        model = self.fit_ridge(Xfull, yfull)
        prev_obj = self._objective_val(model)

        best_obj = prev_obj
        best_Xp = Xp.copy()
        best_yp = yp.copy()

        # Main loop
        for it in range(1, self.max_iters + 1):
            # Fit current model on (train + poison)
            Xfull = np.vstack([self.Xtr, Xp])
            yfull = np.concatenate([self.ytr, yp])
            model = self.fit_ridge(Xfull, yfull)

            w = np.asarray(model.coef_, dtype=float).reshape(-1)
            b = float(model.intercept_)

            # Update each poison point
            new_Xp = Xp.copy()
            new_yp = yp.copy()

            for i in range(num_poison):
                xc = Xp[i]
                yc = float(yp[i])

                m = self._compute_m(w, b, xc, yc)
                wxc, bxc, wyc, byc = self._compute_wb_derivatives(w, m, xc)
                attackx, attacky = self._attack_grad_val(model, wxc, bxc, wyc, byc)

                # Normalize including y direction (paper-style)
                if self.optimize_y:
                    vec = np.concatenate([attackx.reshape(-1), np.array([attacky])])
                else:
                    vec = attackx.reshape(-1)
                nrm = np.linalg.norm(vec)
                if nrm > 0:
                    if self.optimize_y:
                        attackx = vec[:-1] / nrm
                        attacky = float(vec[-1] / nrm)
                    else:
                        attackx = vec / nrm

                # Line search
                x_new, y_new, _ = self._linesearch(
                    Xp, yp, i, attackx, attacky, clip_X, clip_y,
                    max_ls=max_linesearch_iters
                )
                new_Xp[i] = x_new
                new_yp[i] = y_new

            # Evaluate objective after the full update
            Xfull_new = np.vstack([self.Xtr, new_Xp])
            yfull_new = np.concatenate([self.ytr, new_yp])
            model_new = self.fit_ridge(Xfull_new, yfull_new)
            obj_new = self._objective_val(model_new)

            # Keep best
            if obj_new > best_obj:
                best_obj = obj_new
                best_Xp = new_Xp.copy()
                best_yp = new_yp.copy()

            # Stopping condition (paper eps) after at least ~15 iters helps stability
            if it >= 15 and abs(obj_new - prev_obj) <= self.eps:
                Xp, yp = new_Xp, new_yp
                break

            Xp, yp = new_Xp, new_yp
            prev_obj = obj_new

        # Final poisoned model from best poison points
        Xfull_best = np.vstack([self.Xtr, best_Xp])
        yfull_best = np.concatenate([self.ytr, best_yp])
        model_best = self.fit_ridge(Xfull_best, yfull_best)

        return best_Xp, best_yp, model_best
