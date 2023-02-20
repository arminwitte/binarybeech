import numpy as np
import treelib


def print_bars(d, max_width=70):
    max_val = max(d.values())
    usable_width = max_width - 19
    for key, val in d.items():
        l = int(round(usable_width * val / max_val))
        print(f"{key:10}|{'#'*l}{' '*(usable_width-l)}{val:4.2}")

def print_tree(tree)

    def _show(self, node, tree_view, parent=None, prefix=""):
        name = str(hash(node))
        if node.is_leaf:
            text = f"{prefix}{node.value}"
        else:
            if isinstance(node.threshold, (int, float, np.number)):
                text = f"{prefix}{node.attribute}<{node.threshold:.2f}"
            else:
                text = f"{prefix}{node.attribute} in {node.threshold}"
        tree_view.create_node(text, name, parent=parent)

        if not node.is_leaf:
            for i, b in enumerate(node.branches):
                p = "True: " if i == 0 else "False:"
                _show(b, tree_view, parent=name, prefix=p)
    tree_view = treelib.Tree()
    _show(tree.root, tree_view)
    tree_view.show()
    
def k_fold_split(df, k=1, frac=None, random=False, shuffle=True, replace=True):
    if shuffle:
        df = df.sample(frac=1., replace=False)
        
    if frac is None:
        frac = 1. - 1./(k + 1.)
        
    N = len(df.index)
    n = int(np.ceil(N / k))
    sets = []
    for i in reversed(range(k)):
        if random:
            test = df.sample(frac=1.-frac, replace=replace)
        else:
            test = df.iloc[i * n : min(N, (i + 1) * n), :]
        training = df.loc[df.index.difference(test.index), :]
        sets.append((training, test))
    return sets


def gini_impurity(x):
    unique, counts = np.unique(x, return_counts=True)
    N = x.size
    p = counts / N
    return 1.0 - np.sum(p**2)

def shannon_entropy(x):
    unique, counts = np.unique(x, return_counts=True)
    N = x.size
    p = counts / N
    return -np.sum(p * np.log2(p))

def misclassification_cost(x):
    unique, counts = np.unique(x, return_counts=True)
    N = x.size
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
    o = odds(x)
    o = np.clip(o, 1e-12, 1e12)
    logodds = np.log(o)
    return logodds

def probability(x):
    if x == np.Inf:
        return 1.0
    return x / (1 + x)
    
def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

def precision(m):
    return np.diag(m) / np.sum(m, axis=1)

def recall(m):
    return np.diag(m) / np.sum(m, axis=0)

def F1(P, R):
    return 2 * P * R / (P + R)

def accuracy(m):
    return np.sum(np.diag(m)) / np.sum(np.sum(m))
