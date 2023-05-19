#!/usr/bin/env python
# coding: utf-8
import numpy as np
import treelib

from binarybeech.binarybeech import CART


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
