# poisoning/gd_poisoners.py
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import Ridge

@dataclass
class AttackParams:
    eta: float
    beta: float = 0.75
    eps: float = 1e-5
    max_outer_iters: int = 50
    min_outer_iters: int = 15
    max_linesearch_iters: int = 25
    objective: str = "Wtr"     # "Wtr" or "Wval"
    optimize_y: bool = True    # True => optimize (x,y), False => optimize x only
    seed: int = 123


class RidgeGDPoisoner:
    """
    Optimization-based poisoning attack OptP for Ridge regression.

    - BFlip initialization (boundary flipping): y_c = round(1 - y)
    - Line-search with eta and beta decay
    - Objectives: Wtr or Wval (paper Table I)
    """

    def __init__(self, Xtr, ytr, Xval, yval, Xte, yte, lam, colmap=None, params: AttackParams = None):
        self.Xtr = np.asarray(Xtr, dtype=float)
        self.ytr = np.asarray(ytr, dtype=float).reshape(-1)
        self.Xval = np.asarray(Xval, dtype=float)
        self.yval = np.asarray(yval, dtype=float).reshape(-1)
        self.Xte = np.asarray(Xte, dtype=float)
        self.yte = np.asarray(yte, dtype=float).reshape(-1)

        self.lam = float(lam)
        self.params = params or AttackParams(eta=0.1)
        self.rng = np.random.RandomState(self.params.seed)

        # For one-hot constraints; optional
        self.colmap = colmap or {}

    def fit_ridge(self, X, y):
        model = Ridge(alpha=self.lam, fit_intercept=True, max_iter=10000)
        model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1))
        return model

    def mse(self, y_true, y_pred):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        return float(np.mean((y_pred - y_true) ** 2))

    def objective_value(self, model):
        # Paper: Wtr = training MSE (+ reg term sometimes). We use MSE only (consistent with their evaluation focus).
        if self.params.objective == "Wval":
            return self.mse(self.yval, model.predict(self.Xval))
        return self.mse(self.ytr, model.predict(self.Xtr))

    def bflip_init(self, num_poison):
        # Randomly clone training points
        idx = self.rng.choice(self.Xtr.shape[0], size=num_poison, replace=True)
        Xp = self.Xtr[idx].copy()
        yp = self.ytr[idx].copy()

        # Boundary Flipping: y_c = round(1 - y)
        yp = np.round(1.0 - yp)
        yp = np.clip(yp, 0.0, 1.0)
        Xp = np.clip(Xp, 0.0, 1.0)
        return Xp, yp

    def _project_onehot(self, x_row):
        # Enforce one-hot groups if provided (optional)
        x_row = x_row.copy()
        for _, cols in self.colmap.items():
            vals = x_row[cols]
            j = cols[int(np.argmax(vals))]
            x_row[cols] = 0.0
            # threshold like classic implementations (optional)
            x_row[j] = 1.0 if np.max(vals) > (1.0 / (1 + len(cols))) else 0.0
        return x_row

    def _finite_diff_grad(self, Xp, yp, i, h=1e-4):
        """
        Slower but correct and robust:
        finite-difference gradient for x_i and y_i w.r.t outer objective.
        This is the “correct way” in practice when analytic grads are brittle.
        """
        # Build full poisoned dataset
        Xfull = np.vstack([self.Xtr, Xp])
        yfull = np.concatenate([self.ytr, yp])

        base_model = self.fit_ridge(Xfull, yfull)
        base_obj = self.objective_value(base_model)

        # Grad wrt x_i
        gradx = np.zeros_like(Xp[i])
        for j in range(Xp.shape[1]):
            Xp2 = Xp.copy()
            Xp2[i, j] = np.clip(Xp2[i, j] + h, 0.0, 1.0)
            Xp2[i] = self._project_onehot(Xp2[i])

            Xfull2 = np.vstack([self.Xtr, Xp2])
            model2 = self.fit_ridge(Xfull2, yfull)
            obj2 = self.objective_value(model2)
            gradx[j] = (obj2 - base_obj) / h

        # Grad wrt y_i (only if optimizing y)
        grady = 0.0
        if self.params.optimize_y:
            yp2 = yp.copy()
            yp2[i] = np.clip(yp2[i] + h, 0.0, 1.0)
            yfull2 = np.concatenate([self.ytr, yp2])
            model2 = self.fit_ridge(Xfull, yfull2)
            obj2 = self.objective_value(model2)
            grady = (obj2 - base_obj) / h

        return gradx, float(grady), base_obj

    def _linesearch(self, Xp, yp, i, gradx, grady):
        eta = self.params.eta
        beta = self.params.beta
        max_ls = self.params.max_linesearch_iters

        Xp_new = Xp.copy()
        yp_new = yp.copy()

        # Normalize gradient (paper-style)
        if self.params.optimize_y:
            g = np.concatenate([gradx, np.array([grady])])
        else:
            g = gradx.copy()

        norm = np.linalg.norm(g)
        if norm > 0:
            g = g / norm

        if self.params.optimize_y:
            gradx_n = g[:-1]
            grady_n = float(g[-1])
        else:
            gradx_n = g
            grady_n = 0.0

        # current objective
        Xfull = np.vstack([self.Xtr, Xp])
        yfull = np.concatenate([self.ytr, yp])
        base_model = self.fit_ridge(Xfull, yfull)
        base_obj = self.objective_value(base_model)

        best_obj = base_obj
        best_Xi = Xp[i].copy()
        best_yi = yp[i]

        for _ in range(max_ls):
            # ascent step
            Xi = np.clip(best_Xi + eta * gradx_n, 0.0, 1.0)
            Xi = self._project_onehot(Xi)

            yi = best_yi
            if self.params.optimize_y:
                yi = float(np.clip(best_yi + eta * grady_n, 0.0, 1.0))

            Xp_new[i] = Xi
            yp_new[i] = yi

            Xfull_c = np.vstack([self.Xtr, Xp_new])
            yfull_c = np.concatenate([self.ytr, yp_new])
            model_c = self.fit_ridge(Xfull_c, yfull_c)
            obj_c = self.objective_value(model_c)

            if obj_c > best_obj:
                best_obj = obj_c
                best_Xi = Xi.copy()
                best_yi = yi
            else:
                eta *= beta  # decay like Table VI

        Xp_new[i] = best_Xi
        yp_new[i] = best_yi
        return Xp_new, yp_new, best_obj

    def poison_optp(self, num_poison):
        """
        Main OptP loop: optimize each poison point one-at-a-time (Algorithm 1 style).
        """
        Xp, yp = self.bflip_init(num_poison)

        last_obj = -np.inf
        best_obj = -np.inf
        best_Xp = Xp.copy()
        best_yp = yp.copy()

        for t in range(1, self.params.max_outer_iters + 1):
            # Update poison points sequentially
            for i in range(num_poison):
                gradx, grady, _ = self._finite_diff_grad(Xp, yp, i)
                Xp, yp, _ = self._linesearch(Xp, yp, i, gradx, grady)

            # evaluate
            Xfull = np.vstack([self.Xtr, Xp])
            yfull = np.concatenate([self.ytr, yp])
            model = self.fit_ridge(Xfull, yfull)
            cur_obj = self.objective_value(model)

            if cur_obj > best_obj:
                best_obj = cur_obj
                best_Xp = Xp.copy()
                best_yp = yp.copy()

            # stopping (paper Table VI eps1)
            diff = abs(cur_obj - last_obj) if np.isfinite(last_obj) else np.inf
            if t >= self.params.min_outer_iters and diff <= self.params.eps:
                break

            last_obj = cur_obj

        # final model with best poisons
        Xfull = np.vstack([self.Xtr, best_Xp])
        yfull = np.concatenate([self.ytr, best_yp])
        poisoned_model = self.fit_ridge(Xfull, yfull)
        return best_Xp, best_yp, poisoned_model
