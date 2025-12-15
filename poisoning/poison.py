# poisoning/poison.py
import os, json
import numpy as np
from sklearn.linear_model import RidgeCV, Ridge

from poisoning.gd_poisoners import RidgeGDPoisoner, AttackParams
from defense.TRIM import trim_regression
from defense.PRODA import PRODA
from defense.CertifiedRegression import CertifiedRegression

DATASET_DIR = "./datasets"
SEED = 123

# Paper Table VI
ETA_GRID = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0]
BETA = 0.75
EPS1 = 1e-5

# Poison rates you are using
POISON_RATES = [0.12, 0.16, 0.20]

# Paper: use 1400 points per dataset for experiments
SUBSAMPLE_N = 1400

OUT_JSON = "optp_hyperparams.json"


def mse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean((y_pred - y_true) ** 2))


def read_dataset(csv_path):
    X, y = [], []
    with open(csv_path, "r") as f:
        _ = f.readline()  # header
        for line in f:
            vals = [float(v) for v in line.strip().split(",")]
            y.append(vals[0])
            X.append(vals[1:])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)


def normalize_01(X, y):
    Xmin, Xmax = X.min(axis=0), X.max(axis=0)
    X = (X - Xmin) / (Xmax - Xmin + 1e-12)
    ymin, ymax = y.min(), y.max()
    y = (y - ymin) / (ymax - ymin + 1e-12)
    return X, y


def subsample(X, y, n, seed):
    rng = np.random.RandomState(seed)
    if X.shape[0] <= n:
        return X, y
    idx = rng.choice(X.shape[0], size=n, replace=False)
    return X[idx], y[idx]


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
    alphas = np.logspace(-4, 3, 20)
    cv = RidgeCV(alphas=alphas, fit_intercept=True)
    cv.fit(Xtr, ytr)
    lam = float(cv.alpha_)
    model = Ridge(alpha=lam, fit_intercept=True, max_iter=10000)
    model.fit(Xtr, ytr)
    return model, lam


def proda_defense(X, y, seed):
    base = Ridge(alpha=1.0)
    p = PRODA(X, y)
    return p.apply_defense(alpha=0.2, gamma=20, eps=200, base_regressor=base, random_state=seed)


def certified_regression_defense(X, y):
    cr = CertifiedRegression(T=21, s=12, alpha=10.0)
    cr.fit(X, y)
    return cr


def paper_config_for_dataset(name):
    """
    Paper Table I (Ridge):
    - Health: BFlip (x,y), Wtr
    - Loan:   BFlip x-only, Wval
    - House:  BFlip (x,y), Wtr

    We map:
    - pharm-preproc.csv -> Health-like
    """
    n = name.lower()
    if "loan" in n:
        return {"objective": "Wval", "optimize_y": False}
    if "pharm" in n:
        return {"objective": "Wtr", "optimize_y": True}
    if "house" in n:
        return {"objective": "Wtr", "optimize_y": True}
    # default: Wtr, optimize both
    return {"objective": "Wtr", "optimize_y": True}


def run_one_dataset(csv_path, hyperlog):
    fname = os.path.basename(csv_path)

    X, y = read_dataset(csv_path)
    X, y = normalize_01(X, y)
    X, y = subsample(X, y, SUBSAMPLE_N, SEED)

    Xtr, ytr, Xval, yval, Xte, yte = split_1_3(X, y, SEED)

    clean_model, lam = fit_ridge_cv(Xtr, ytr)
    clean_val = mse(yval, clean_model.predict(Xval))
    clean_te = mse(yte, clean_model.predict(Xte))

    cfg = paper_config_for_dataset(fname)

    print("\n============================================================")
    print(f"Dataset: {fname}")
    print(f"Ridge λ (CV): {lam:.6g}")
    print(f"OptP config (paper Table I): objective={cfg['objective']} optimize_y={cfg['optimize_y']}")
    print("============================================================")

    for alpha in POISON_RATES:
        p = int(round(alpha * Xtr.shape[0]))
        if p <= 0:
            continue

        best = None

        # Sweep eta (paper Table VI)
        for eta in ETA_GRID:
            params = AttackParams(
                eta=eta,
                beta=BETA,
                eps=EPS1,
                objective=cfg["objective"],
                optimize_y=cfg["optimize_y"],
                seed=SEED,
                max_outer_iters=50,
                min_outer_iters=15,
                max_linesearch_iters=25
            )

            poisoner = RidgeGDPoisoner(Xtr, ytr, Xval, yval, Xte, yte, lam=lam, params=params)
            Xp, yp, poisoned_model = poisoner.poison_optp(num_poison=p)

            pois_val = mse(yval, poisoned_model.predict(Xval))
            pois_te = mse(yte, poisoned_model.predict(Xte))

            # choose by outer objective (maximize Wtr or Wval)
            score = pois_val if cfg["objective"] == "Wval" else mse(ytr, poisoned_model.predict(Xtr))
            if best is None or score > best["score"]:
                best = {
                    "eta": eta,
                    "alpha": alpha,
                    "p": p,
                    "lam": lam,
                    "objective": cfg["objective"],
                    "optimize_y": cfg["optimize_y"],
                    "val_mse": pois_val,
                    "test_mse": pois_te,
                    "Xp": Xp,
                    "yp": yp,
                    "poisoned_model": poisoned_model,
                    "score": score
                }

        # Build poisoned training set using best poisons
        Xp, yp = best["Xp"], best["yp"]
        Xtr_p = np.vstack([Xtr, Xp])
        ytr_p = np.concatenate([ytr, yp])

        # Defenses
        trim_model = trim_regression(Xtr_p, ytr_p, keep_count=Xtr.shape[0], alpha=lam, seed=SEED)
        trim_val = mse(yval, trim_model.predict(Xval))
        trim_te = mse(yte, trim_model.predict(Xte))

        proda_model = proda_defense(Xtr_p, ytr_p, seed=SEED)
        proda_val = mse(yval, proda_model.predict(Xval))
        proda_te = mse(yte, proda_model.predict(Xte))

        cert_model = certified_regression_defense(Xtr_p, ytr_p)
        cert_val = mse(yval, cert_model.predict(Xval))
        cert_te = mse(yte, cert_model.predict(Xte))

        # Print in your requested format
        print(f"\n--- poison rate α = {alpha:.2f} (p = {p}) ---")
        print(f"unpoisoned - validation mse: {clean_val:.6f}, test mse: {clean_te:.6f}")
        print(f"poisoned   - validation mse: {best['val_mse']:.6f}, test mse: {best['test_mse']:.6f}")
        print(f"(best eta = {best['eta']})")

        print("\nrunnning trim defense")
        print(f"validation mse: {trim_val:.6f}, test mse: {trim_te:.6f}")

        print("\nrunnning PRODA defense")
        print(f"validation mse: {proda_val:.6f}, test mse: {proda_te:.6f}")

        print("\nrunnning certified regression defense")
        print(f"validation mse: {cert_val:.6f}, test mse: {cert_te:.6f}")

        # Record hyperparameters
        hyperlog.append({
            "dataset": fname,
            "alpha": alpha,
            "p": p,
            "ridge_lambda_cv": lam,
            "objective": cfg["objective"],
            "optimize_y": cfg["optimize_y"],
            "beta": BETA,
            "eps1": EPS1,
            "eta_best": best["eta"],
            "poisoned_val_mse": best["val_mse"],
            "poisoned_test_mse": best["test_mse"],
            "trim_val_mse": trim_val,
            "trim_test_mse": trim_te,
            "proda_val_mse": proda_val,
            "proda_test_mse": proda_te,
            "cert_val_mse": cert_val,
            "cert_test_mse": cert_te
        })


def main():
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".csv")]
    # run smaller first
    files.sort(key=lambda f: os.path.getsize(os.path.join(DATASET_DIR, f)))

    hyperlog = []

    for f in files:
        run_one_dataset(os.path.join(DATASET_DIR, f), hyperlog)

    with open(OUT_JSON, "w") as fp:
        json.dump(hyperlog, fp, indent=2)

    print(f"\nSaved hyperparameter log to: {OUT_JSON}")


if __name__ == "__main__":
    main()
