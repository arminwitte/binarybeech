import numpy as np


def print_bars(d, max_width=70):
    max_val = max(d.values())
    usable_width = max_width - 19
    for key, val in d.items():
        l = int(round(usable_width * val / max_val))
        print(f"{key:10}|{'#'*l}{' '*(usable_width-l)}{val:4.2}")

def gini_impurity(x):
    unique, counts = np.unique(x, return_counts=True)
    N = x.size
    p = counts / N
    return 1.0 - np.sum(p**2)

def shannon_entropy(x):
    unique, counts = np.unique(x, return_counts=True)
    N = y.size
    p = counts / N
    return -np.sum(p * np.log2(p))

def misclassification_cost(self, x):
    unique, counts = np.unique(y, return_counts=True)
    N = y.size
    p = np.max(counts) / N
    return 1.0 - p

def logistic_loss(y, p):
    p_ = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.sum(-y * np.log(p) - (1 - y) * np.log(1 - p))

def mean_squared_error(y, y_hat):
        e = y - y_hat
        return 1 / e.size * (e.T @ e)

def majority_class(x):
    unique, counts = np.unique(x, return_counts=True)
    ind_max = np.argmax(counts)
    return unique[ind_max]

def odds(x):
    unique, counts = np.unique(x, return_counts=True)
    d = {0: 0, 1: 0}
    for i, u in enumerate(unique):
        d[u] = counts[i]
    if d[0] == 0:
        return np.Inf
    odds = d[1] / d[0]
    return odds

def log_odds(x):
    odds = odds(x)
    odds = np.clip(odds, 1e-12, 1e12)
    logodds = np.log(odds)
    return logodds

def _probability(self, df):
        odds = self._odds(df)
        if odds == np.Inf:
            return 1.0
        p = odds / (1 + odds)
        # print(f"odds: {odds:.2f} probability: {p:.4f}")
        return p

    def _classification_metrics(self, y_hat, df=None):
        confmat = self._confusion_matrix(y_hat, df)
        P = self._precision(confmat)
        # print(f"precision: {P}")
        R = self._recall(confmat)
        # print(f"recall: {R}")
        F = np.mean(self._F1(P, R))
        # print(f"F-score: {F}")
        A = self._accuracy(confmat)
        return {"precision": P, "recall": R, "F-score": F, "accuracy": A}

    @staticmethod
    def _precision(m):
        return np.diag(m) / np.sum(m, axis=1)

    @staticmethod
    def _recall(m):
        return np.diag(m) / np.sum(m, axis=0)

    @staticmethod
    def _F1(P, R):
        # F = np.zeros_like(P)
        # for i in range(len(
        return 2 * P * R / (P + R)

    @staticmethod
    def _accuracy(m):
        return np.sum(np.diag(m)) / np.sum(np.sum(m))
