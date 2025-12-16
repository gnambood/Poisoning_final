import sys
sys.dont_write_bytecode = True

import os
import numpy as np

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error

from .gd_poisoners import AttackParams, RidgeGDPoisoner
from defense.TRIM import trim_regression
from defense.PRODA import PRODA
from defense.CertifiedRegression import CertifiedRegression

SEED = 123
POISON_RATES = [0.00, 0.04, 0.08, 0.12, 0.16, 0.20]
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")


def read_dataset(path):
    X, y = [], []
    with open(path, "r") as f:
        next(f)
        for line in f:
            row = list(map(float, line.strip().split(",")))
            y.append(row[0])
            X.append(row[1:])
    return np.asarray(X), np.asarray(y)


def normalize_01(X, y):
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    return X, y


def split_data(X, y, n_train, n_val, n_test):
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(X))
    tr = perm[:n_train]
    va = perm[n_train:n_train + n_val]
    te = perm[n_train + n_val:n_train + n_val + n_test]
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def fit_ridge_cv(X, y):
    grid = np.logspace(-4, 3, 20)
    cv = RidgeCV(alphas=grid).fit(X, y)
    model = Ridge(alpha=float(cv.alpha_), max_iter=10000)
    model.fit(X, y)
    return model, float(cv.alpha_)


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def print_optp_table(dataset, clean_test, rows, objective):
    print("=" * 78)
    print(f"OptP sweep | dataset={dataset} | objective={objective}")
    print("=" * 78)
    print(" alpha     p   clean_test    optp_test   xClean   eta*")
    print("-" * 78)
    for r in rows:
        eta_str = f"{r['eta']:.3f}" if r['eta'] is not None else "-"
        print(f" {r['alpha']:>4.2f} {r['p']:>5d}   {clean_test:>10.6f}   {r['optp_test']:>10.6f}   {r['xclean']:>6.2f}   {eta_str:>5}")
    print("-" * 78)


def print_defense_table(dataset, alpha_star, clean_val, clean_test, res, objective, optimize_y):
    print("=" * 78)
    print(f"Defenses on max-poison | dataset={dataset} | alpha*={alpha_star:.2f}")
    print("=" * 78)
    print("Method               Val MSE    Test MSE   Test xClean")
    print("-" * 78)

    for name, (v, t) in res.items():
        xc = t / clean_test
        print(f"{name:<20} {v:>10.6f} {t:>10.6f} {xc:>12.2f}")

    print("-" * 78)
    print(f"OptP knobs: objective={objective}, optimize_y={optimize_y}")


MAKE_PLOTS = True


def main():
    datasets = [
        ("house-processed.csv", "Wtr", True),
        ("loan-processed.csv", "Wval", False),
        ("pharm-preproc.csv", "Wtr", True),
    ]

    n_train, n_val, n_test = 300, 300, 300

    for fname, objective, optimize_y in datasets:
        X, y = read_dataset(os.path.join(DATASET_DIR, fname))
        X, y = normalize_01(X, y)

        Xtr, ytr, Xva, yva, Xte, yte = split_data(X, y, n_train, n_val, n_test)

        clean_model, lam = fit_ridge_cv(Xtr, ytr)
        clean_val = mse(yva, clean_model.predict(Xva))
        clean_test = mse(yte, clean_model.predict(Xte))

        params = AttackParams(
            eta=0.1,
            objective=objective,
            optimize_y=optimize_y,
            seed=SEED
        )

        poisoner = RidgeGDPoisoner(Xtr, ytr, Xva, yva, Xte, yte, lam, params)

        sweep_rows = []
        best_alpha = None
        best_p = None
        best_poison = None
        best_score = -np.inf

        for alpha in POISON_RATES:
            p = int(round(alpha * len(Xtr)))
            if p == 0:
                optp_test = clean_test
                sweep_rows.append({
                    "alpha": alpha,
                    "p": 0,
                    "optp_test": optp_test,
                    "xclean": 1.0,
                    "eta": None
                })
                continue

            Xp, yp, poisoned_model = poisoner.poison_optp(p)
            optp_test = mse(yte, poisoned_model.predict(Xte))
            score = mse(ytr, poisoned_model.predict(Xtr))

            sweep_rows.append({
                "alpha": alpha,
                "p": p,
                "optp_test": optp_test,
                "xclean": optp_test / clean_test,
                "eta": params.eta
            })

            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_p = p
                best_poison = (Xp, yp)

        print_optp_table(fname, clean_test, sweep_rows, objective)

        Xp, yp = best_poison
        Xtr_p = np.vstack([Xtr, Xp])
        ytr_p = np.concatenate([ytr, yp])

        results = {}

        results["Clean"] = (clean_val, clean_test)

        optp_model = poisoner.fit_ridge(Xtr_p, ytr_p)
        results["OptP"] = (
            mse(yva, optp_model.predict(Xva)),
            mse(yte, optp_model.predict(Xte))
        )

        trim_model = trim_regression(Xtr_p, ytr_p, len(Xtr), lam, seed=SEED)
        results["TRIM"] = (
            mse(yva, trim_model.predict(Xva)),
            mse(yte, trim_model.predict(Xte))
        )

        proda_model = PRODA(Xtr_p, ytr_p).apply_defense(
            alpha=best_alpha,
            gamma=20,
            eps=200,
            base_regressor=Ridge(alpha=lam),
            random_state=SEED
        )
        results["PRODA"] = (
            mse(yva, proda_model.predict(Xva)),
            mse(yte, proda_model.predict(Xte))
        )

        cert = CertifiedRegression(alpha=lam)
        cert.fit(Xtr_p, ytr_p)
        results["Certified"] = (
            mse(yva, cert.predict(Xva)),
            mse(yte, cert.predict(Xte))
        )

        print_defense_table(fname, best_alpha, clean_val, clean_test, results, objective, optimize_y)

        if MAKE_PLOTS:
            import matplotlib.pyplot as plt

            alphas = [r['alpha'] for r in sweep_rows if r['p'] > 0]
            optp_curve = [r['optp_test'] for r in sweep_rows if r['p'] > 0]

            trim_curve = []
            proda_curve = []
            cert_curve = []

            for a in alphas:
                p = int(round(a * len(Xtr)))
                Xp, yp, _ = poisoner.poison_optp(p)
                Xtr_p = np.vstack([Xtr, Xp])
                ytr_p = np.concatenate([ytr, yp])

                trim_m = trim_regression(Xtr_p, ytr_p, len(Xtr), lam, seed=SEED)
                trim_curve.append(mse(yte, trim_m.predict(Xte)))

                proda_m = PRODA(Xtr_p, ytr_p).apply_defense(
                    alpha=a,
                    gamma=20,
                    eps=200,
                    base_regressor=Ridge(alpha=lam),
                    random_state=SEED
                )
                proda_curve.append(mse(yte, proda_m.predict(Xte)))

                cert_m = CertifiedRegression(alpha=lam)
                cert_m.fit(Xtr_p, ytr_p)
                cert_curve.append(mse(yte, cert_m.predict(Xte)))

            plt.figure(figsize=(6, 4))
            plt.plot(alphas, optp_curve, marker='o', label='OptP')
            plt.plot(alphas, trim_curve, marker='s', label='TRIM')
            plt.plot(alphas, proda_curve, marker='^', label='PRODA')
            plt.plot(alphas, cert_curve, marker='D', label='Certified')
            plt.axhline(clean_test, linestyle='--', color='black', label='Clean')
            plt.xlabel('Poisoning fraction α')
            plt.ylabel('Test MSE')
            plt.title(fname)
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
