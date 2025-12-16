import numpy as np
from sklearn.linear_model import Ridge
from sklearn.base import clone

class PRODA:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def apply_defense(self, alpha, gamma, eps, base_regressor=None, random_state=None):
        rng = np.random.default_rng(random_state)
        N = self.X.shape[0]
        n = N - int(alpha * N)
        if n <= 0:
            raise ValueError("alpha too large")
        if base_regressor is None:
            base_regressor = Ridge(alpha=1.0)

        beta = max(int(np.ceil(eps / (1.0 - alpha ** gamma))), 1)
        best_M = np.inf
        best_model = None

        for _ in range(beta):
            J = rng.choice(N, gamma, replace=False)
            m0 = clone(base_regressor).fit(self.X[J], self.y[J])
            sq = (self.y - m0.predict(self.X)) ** 2
            Q = np.argsort(sq)[:n]
            m = clone(base_regressor).fit(self.X[Q], self.y[Q])
            M = np.mean((self.y[Q] - m.predict(self.X[Q])) ** 2)
            if M < best_M:
                best_M, best_model = M, m

        return best_model
