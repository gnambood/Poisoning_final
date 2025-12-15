import numpy as np
from sklearn.linear_model import Ridge
import hashlib


class CertifiedRegression:
    """
    Certified Regression Defense
    (Hammoudeh & Lowd, SaTML 2023)

    Empirical implementation adapted for small-data regimes:
      - Overlapping deterministic partitions
      - Feature-based universal hashing
      - Strong regularization
      - Median aggregation
    """

    def __init__(self, T=21, s=12, alpha=10.0):
        assert s <= T, "s must be <= T"
        self.T = T
        self.s = s
        self.alpha = alpha
        self.models = []
        self.partitions = None

    # ------------------------------------------------------------
    # Feature-based deterministic hash
    # ------------------------------------------------------------
    def _hash(self, x_row, t):
        """
        Hash a feature vector + model index to integer.
        Deterministic and attack-independent.
        """
        key = x_row.tobytes() + str(t).encode()
        return int(hashlib.sha256(key).hexdigest(), 16)

    # ------------------------------------------------------------
    # Build overlapping partitions
    # ------------------------------------------------------------
    def _build_partitions(self, X):
        n_samples = X.shape[0]
        partitions = [[] for _ in range(self.T)]

        for i in range(n_samples):
            hvals = []
            for t in range(self.T):
                hv = self._hash(X[i], t)
                hvals.append((hv, t))

            # assign sample i to s submodels
            hvals.sort(key=lambda x: x[0])
            chosen_models = [t for (_, t) in hvals[:self.s]]

            for t in chosen_models:
                partitions[t].append(i)

        # convert to numpy arrays
        partitions = [np.array(p, dtype=int) for p in partitions]
        return partitions

    # ------------------------------------------------------------
    # Train ensemble
    # ------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.partitions = self._build_partitions(X)
        self.models = []

        for t in range(self.T):
            idx = self.partitions[t]

            # Safety check: skip tiny partitions
            if len(idx) < 5:
                raise ValueError(
                    f"Submodel {t} has too few samples ({len(idx)}). "
                    "Increase s or reduce T."
                )

            model = Ridge(alpha=self.alpha)
            model.fit(X[idx], y[idx])
            self.models.append(model)

        return self

    # ------------------------------------------------------------
    # Predict via median aggregation
    # ------------------------------------------------------------
    def predict(self, X):
        X = np.asarray(X)
        preds = np.zeros((X.shape[0], self.T))

        for t, model in enumerate(self.models):
            preds[:, t] = model.predict(X)

        return np.median(preds, axis=1)