import numpy as np
from sklearn.linear_model import Ridge

def trim_regression(X, y, keep_count, alpha, seed=123, max_iters=400, tol=1e-5):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    n = X.shape[0]
    if keep_count <= 0 or keep_count > n:
        raise ValueError("invalid keep_count")

    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(n, keep_count, replace=False))
    last_err = np.inf

    for _ in range(max_iters):
        model = Ridge(alpha=alpha, fit_intercept=True, max_iter=10000)
        model.fit(X[idx], y[idx])
        sq = (model.predict(X) - y) ** 2
        new_idx = np.sort(np.argsort(sq)[:keep_count])
        err = float(np.sum(np.sort(sq)[:keep_count]))
        if np.array_equal(new_idx, idx) and (last_err - err) <= tol:
            return model
        idx, last_err = new_idx, err

    model.fit(X[idx], y[idx])
    return model
