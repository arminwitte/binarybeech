#!/usr/bin/env python
# coding: utf-8
import numpy as np
import treelib

from binarybeech.binarybeech import CART
from binarybeech.extra import k_fold_split


def print_bars(d, max_width=70):
    max_val = max(d.values())
    usable_width = max_width - 19
    for key, val in d.items():
        l = int(round(usable_width * val / max_val))
        print(f"{key:10}|{'#'*l}{' '*(usable_width-l)}{val:4.2}")


def print_tree(tree):
    def _show(node, tree_view, parent=None, prefix=""):
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
        df = df.sample(frac=1.0, replace=False)

    if frac is None:
        frac = 1.0 - 1.0 / (k + 1.0)

    N = len(df.index)
    n = int(np.ceil(N / k))
    sets = []
    for i in reversed(range(k)):
        if random:
            test = df.sample(frac=1.0 - frac, replace=replace)
        else:
            test = df.iloc[i * n : min(N, (i + 1) * n), :]
        training = df.loc[df.index.difference(test.index), :]
        sets.append((training, test))
    return sets


def model_missings(df, y_name, X_names=None, cart_settings={}):
    if X_names is None:
        X_names = [n for n in df.columns]
        X_names.remove(y_name)
    df_ = df.copy()
    has_missings = df.isnull().any()
    for x_name in X_names:

        if not has_missings[x_name]:
            continue

        m_X_names = [n for n in df.columns]
        m_X_names.remove(x_name)
        m_X_names.remove(y_name)
        kwargs = dict(
            max_depth=3,
            min_leaf_samples=5,
            min_split_samples=4,
        )
        kwargs = {**kwargs, **cart_settings}
        mod = CART(df[~df[x_name].isnull()], x_name, X_names=m_X_names, **cart_settings)
        mod.create_tree()
        df_.loc[df[x_name].isnull(), x_name] = mod.predict(df[df[x_name].isnull()])

    return df_


# def plot_areas(df):
#     x, y = np.meshgrid(np.linspace(1,7,101),np.linspace(0,2.5,101))
#     col = []
#     for i in range(len(x.ravel())):
#         d = df_iris.iloc[120].copy()
#         d["petal_length"] = x.ravel()[i]
#         d["petal_width"] = y.ravel()[i]
#         col.append(c_iris.tree.traverse(d).value)
#     unique = [u for u in np.unique(col)]
#     for i, c in enumerate(col):
#         col[i] = unique.index(c)
#     z = np.array(col).reshape(x.shape)
