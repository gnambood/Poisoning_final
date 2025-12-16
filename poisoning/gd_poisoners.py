from dataclasses import dataclass
import numpy as np

@dataclass
class AttackParams:
    eta: float
    beta: float = 0.75
    eps1: float = 1e-5
    max_outer_iters: int = 50
    min_outer_iters: int = 15
    max_linesearch_iters: int = 25
    objective: str = "Wtr"
    optimize_y: bool = True
    clip_X: bool = True
    clip_y: bool = True
    seed: int = 123
    per_point_linesearch: bool = False

@dataclass
class SimpleRidgeModel:
    coef_: np.ndarray
    intercept_: float

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

def fit_ridge_closed_form(X, y, lam):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    n, d = X.shape
    X_aug = np.hstack([X, np.ones((n, 1))])
    R = np.zeros((d + 1, d + 1))
    R[:d, :d] = lam * np.eye(d)
    w = np.linalg.solve(X_aug.T @ X_aug + R, X_aug.T @ y)
    return SimpleRidgeModel(w[:d], float(w[d]))

class RidgeGDPoisoner:
    def __init__(self, Xtr, ytr, Xval, yval, Xte, yte, lam, params):
        self.Xtr = np.asarray(Xtr, float)
        self.ytr = np.asarray(ytr, float).reshape(-1)
        self.Xval = np.asarray(Xval, float)
        self.yval = np.asarray(yval, float).reshape(-1)
        self.Xte = np.asarray(Xte, float)
        self.yte = np.asarray(yte, float).reshape(-1)
        self.lam = float(lam)
        self.p = params
        self.rng = np.random.RandomState(params.seed)

    def _clip(self, Xp, yp):
        if self.p.clip_X:
            Xp = np.clip(Xp, 0.0, 1.0)
        if self.p.optimize_y and self.p.clip_y:
            yp = np.clip(yp, 0.0, 1.0)
        return Xp, yp

    def bflip_init(self, k):
        idx = np.argsort(np.abs(self.ytr - 0.5))[:k]
        Xp = self.Xtr[idx].copy()
        yp = np.where(self.ytr[idx] >= 0.5, 0.0, 1.0)
        return Xp, yp

    def fit_ridge(self, X, y):
        return fit_ridge_closed_form(X, y, self.lam)

    def outer_objective(self, model):
        if self.p.objective.lower() == "wval":
            return -np.mean((self.yval - model.predict(self.Xval)) ** 2)
        return -np.mean((self.ytr - model.predict(self.Xtr)) ** 2)

    def poison_optp(self, k, init="bflip"):
        Xp, yp = self.bflip_init(k)
        best_obj = -np.inf

        for _ in range(self.p.max_outer_iters):
            model = self.fit_ridge(np.vstack([self.Xtr, Xp]), np.hstack([self.ytr, yp]))
            obj = self.outer_objective(model)
            if obj > best_obj:
                best_obj = obj
            else:
                break
        return Xp, yp, model
