import numpy as np
import hashlib
from sklearn.linear_model import Ridge

class CertifiedRegression:
    def __init__(self, T=21, s=12, alpha=10.0):
        self.T = T
        self.s = s
        self.alpha = alpha
        self.models = []

    def _hash(self, x, t):
        return int(hashlib.sha256(x.tobytes() + str(t).encode()).hexdigest(), 16)

    def _partitions(self, X):
        parts = [[] for _ in range(self.T)]
        for i, x in enumerate(X):
            hs = sorted((self._hash(x, t), t) for t in range(self.T))[:self.s]
            for _, t in hs:
                parts[t].append(i)
        return [np.asarray(p, int) for p in parts]

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.models = []
        for idx in self._partitions(X):
            if len(idx) < 5:
                raise ValueError("partition too small")
            m = Ridge(alpha=self.alpha).fit(X[idx], y[idx])
            self.models.append(m)
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = np.vstack([m.predict(X) for m in self.models])
        return np.median(preds, axis=0)
