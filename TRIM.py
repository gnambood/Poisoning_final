import numpy as np
from sklearn.linear_model import Ridge

from .PRODA import PRODA
from .CertifiedRegression import CertifiedRegression


def trim_regression(X, y, keep_count, alpha=1.0, tol=1e-5, max_iter=400, seed=123):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    rng = np.random.RandomState(seed)
    inds = rng.permutation(X.shape[0])[:keep_count]

    model = Ridge(alpha=alpha)
    model.fit(X[inds], y[inds])

    prev_loss = float("inf")
    for _ in range(max_iter):
        resid = (model.predict(X) - y) ** 2
        inds = np.argsort(resid)[:keep_count]
        model.fit(X[inds], y[inds])

        loss = float(np.mean((model.predict(X[inds]) - y[inds]) ** 2))
        if abs(prev_loss - loss) <= tol:
            break
        prev_loss = loss

    return model


def proda_defense(X, y, seed=123):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    base = Ridge(alpha=1.0)
    proda = PRODA(X, y)
    return proda.apply_defense(alpha=0.2, gamma=20, eps=200, base_regressor=base, random_state=seed)


def certified_regression_defense(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    cr = CertifiedRegression(T=21, s=12, alpha=10.0)
    cr.fit(X, y)
    return cr
