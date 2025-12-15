import pandas as pd
import numpy as np
from math import log, ceil
from sklearn.linear_model import Ridge
from sklearn.base import clone

class PRODA:
    def __init__(self, X_train_poisoned, y_train_poisoned):
        self.X = X_train_poisoned
        self.y = y_train_poisoned

    def apply_defense(self, alpha, gamma, eps, base_regressor=None, random_state=None):
        """
        Implementation of Algorithm 2 (Proda algorithm) from the paper.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
        y : np.ndarray, shape (N,)
        alpha : float
            Attack fraction α (0 < alpha < 1).
        gamma : int
            Subset size γ (1 <= gamma <= N).
        eps : float
            ε in the algorithm. β is computed as β = ε / (1 - α^γ).
            In practice, you often choose eps so that β is a reasonable integer
            (e.g., eps=100 → β≈100 if α^γ ≈ 0).
        base_regressor : sklearn estimator or None
            If None, use Ridge(alpha=1.0).
        random_state : int or None

        Returns
        -------
        best_model : sklearn estimator
            The model S^(j*) trained on the best Q^(j*).
        best_Q_idx : np.ndarray
            Indices of the optimizing set Q^(j*).
        M_values : list of float
            The list of M(i) values (MSEs) for each iteration i.
        """
        rng = np.random.default_rng(random_state)
        N, d = self.X.shape

        # Number of attacked points p = alpha * n, and n = N - p
        p = int(alpha * N)
        n = N - p  # expected number of clean points

        if n <= 0:
            raise ValueError("alpha too large: no clean points left (n <= 0).")

        if gamma > N:
            raise ValueError("gamma (subset size) cannot exceed N.")

        # Base regressor
        if base_regressor is None:
            base_regressor = Ridge(alpha=1.0)

        # Line 1: β = ε / (1 - α^γ)
        denom = 1.0 - (alpha ** gamma)
        if denom <= 0:
            raise ValueError("Denominator 1 - alpha**gamma <= 0; choose smaller gamma or alpha.")
        beta_float = eps / denom
        beta = int(np.ceil(beta_float))
        beta = max(beta, 1)  # at least 1 iteration

        # print(f"PRODA Algorithm 2:")
        # print(f"  N = {N}, d = {d}")
        # print(f"  alpha = {alpha}, gamma = {gamma}, eps = {eps}")
        # print(f"  p (attack points) = {p}, n (clean points) = {n}")
        # print(f"  beta ≈ {beta_float:.3f}, using β = {beta} iterations")

        best_M = np.inf
        best_Q_idx = None
        best_model = None
        M_values = []

        # Lines 3–11: for i ≤ β do
        for i in range(beta):
            # Line 4: J^(i) ← random subset of size γ
            J_i = rng.choice(N, size=gamma, replace=False)

            # Line 5: L^(i) ← argmin_θ L(J^(i), θ)
            model_subset = clone(base_regressor)
            model_subset.fit(self.X[J_i], self.y[J_i])

            # Line 6: list^(i) ← distance of N points to L^(i)
            y_pred_all = model_subset.predict(self.X)
            residuals = (self.y - y_pred_all) ** 2

            # Line 7: Q^(i) ← sorted(list^(i))[ : n ]
            sorted_idx = np.argsort(residuals)
            Q_i = sorted_idx[:n]

            # Line 8: S^(i) ← argmin_θ L(Q^(i), θ)
            model_clean = clone(base_regressor)
            model_clean.fit(self.X[Q_i], self.y[Q_i])

            # Line 9: M(i) ← MSE between S^(i) and Q^(i)
            y_pred_Q = model_clean.predict(self.X[Q_i])
            M_i = np.mean((self.y[Q_i] - y_pred_Q) ** 2)
            M_values.append(M_i)

            # Track best
            if M_i < best_M:
                best_M = M_i
                best_Q_idx = Q_i
                best_model = model_clean

        # Line 12: choose j with min M(i) (we already tracked best_M, best_Q_idx, best_model)
        return best_model