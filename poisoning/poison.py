# poisoning/poison.py
import os
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV

from defense.TRIM import trim_regression, proda_defense, certified_regression_defense


# ----------------------------
# FAST MODE SETTINGS
# ----------------------------
SEED = 123
DATASET_DIR = "./datasets"

# Keep these as in your paper runs
POISON_RATES = [0.12, 0.16, 0.20]

# FAST MODE: no iterative OptP. Just create strong heuristic poisoned points.
FAST_MODE = True


def mse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean((y_pred - y_true) ** 2))


def read_dataset(csv_path):
    # Assumes: first column is y, remaining are X
    X, y = [], []
    with open(csv_path, "r") as f:
        _ = f.readline()  # header
        for line in f:
            vals = [float(v) for v in line.strip().split(",")]
            y.append(vals[0])
            X.append(vals[1:])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)


def normalize_01(X, y):
    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)
    X = (X - Xmin) / (Xmax - Xmin + 1e-12)

    ymin, ymax = y.min(), y.max()
    y = (y - ymin) / (ymax - ymin + 1e-12)
    return X, y


def split_1_3(X, y, seed):
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)

    n_tr = n // 3
    n_te = n // 3

    tr = perm[:n_tr]
    te = perm[n_tr:n_tr + n_te]
    va = perm[n_tr + n_te:]

    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def fit_ridge_cv(Xtr, ytr):
    # Paper protocol: lambda by CV
    alphas = np.logspace(-4, 3, 20)
    cv = RidgeCV(alphas=alphas, fit_intercept=True)
    cv.fit(Xtr, ytr)
    lam = float(cv.alpha_)
    model = Ridge(alpha=lam, fit_intercept=True, max_iter=10000)
    model.fit(Xtr, ytr)
    return model, lam


def fast_poison_points(Xtr, ytr, p, seed):
    """
    FAST poisoning heuristic:
    - sample p training points
    - flip features toward extremes: X -> 1 - X
    - flip labels: y -> 1 - y
    This is fast and makes a strong “stress test” attack.
    """
    rng = np.random.RandomState(seed)
    idx = rng.choice(Xtr.shape[0], size=p, replace=True)
    Xp = Xtr[idx].copy()
    yp = ytr[idx].copy()

    # push to opposite extreme in [0,1]
    Xp = 1.0 - Xp
    yp = 1.0 - yp

    # clip
    Xp = np.clip(Xp, 0.0, 1.0)
    yp = np.clip(yp, 0.0, 1.0)
    return Xp, yp


def run_one_dataset(csv_path):
    X, y = read_dataset(csv_path)
    X, y = normalize_01(X, y)

    Xtr, ytr, Xval, yval, Xte, yte = split_1_3(X, y, SEED)
    n = Xtr.shape[0]

    # Unpoisoned
    clean_model, lam = fit_ridge_cv(Xtr, ytr)
    clean_val = mse(yval, clean_model.predict(Xval))
    clean_te = mse(yte, clean_model.predict(Xte))

    print("\n============================================================")
    print(f"Dataset: {os.path.basename(csv_path)}")
    print(f"Ridge λ (CV): {lam:.6g}")
    print("============================================================")

    for alpha in POISON_RATES:
        p = int(round(alpha * n))
        if p <= 0:
            continue

        # Poisoned training set (FAST MODE)
        Xp, yp = fast_poison_points(Xtr, ytr, p, seed=SEED)
        Xtr_p = np.vstack([Xtr, Xp])
        ytr_p = np.concatenate([ytr, yp])

        poisoned_model = Ridge(alpha=lam, fit_intercept=True, max_iter=10000)
        poisoned_model.fit(Xtr_p, ytr_p)

        pois_val = mse(yval, poisoned_model.predict(Xval))
        pois_te = mse(yte, poisoned_model.predict(Xte))

        # Defenses
        trim_model = trim_regression(Xtr_p, ytr_p, keep_count=n, alpha=lam, seed=SEED)
        trim_val = mse(yval, trim_model.predict(Xval))
        trim_te = mse(yte, trim_model.predict(Xte))

        proda_model = proda_defense(Xtr_p, ytr_p, seed=SEED)
        proda_val = mse(yval, proda_model.predict(Xval))
        proda_te = mse(yte, proda_model.predict(Xte))

        cert_model = certified_regression_defense(Xtr_p, ytr_p)
        cert_val = mse(yval, cert_model.predict(Xval))
        cert_te = mse(yte, cert_model.predict(Xte))

        # Output format you requested
        print(f"\n--- poison rate α = {alpha:.2f} (p = {p}) ---")
        print(f"unpoisoned - validation mse: {clean_val:.6f}, test mse: {clean_te:.6f}")
        print(f"poisoned   - validation mse: {pois_val:.6f}, test mse: {pois_te:.6f}")

        print(f"\nrunnning trim defense")
        print(f"validation mse: {trim_val:.6f}, test mse: {trim_te:.6f}")

        print(f"\nrunnning PRODA defense")
        print(f"validation mse: {proda_val:.6f}, test mse: {proda_te:.6f}")

        print(f"\nrunnning certified regression defense")
        print(f"validation mse: {cert_val:.6f}, test mse: {cert_te:.6f}")


def main():
    # Run smaller files first (fast feedback)
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".csv")]
    files.sort(key=lambda f: os.path.getsize(os.path.join(DATASET_DIR, f)))

    for fname in files:
        run_one_dataset(os.path.join(DATASET_DIR, fname))


if __name__ == "__main__":
    main()
