#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import treelib


def plot_areas(
    mod, x_name, y_name, df=None, rng=None, slice_definition=None, N=101, ax=None
):
    if rng is None:
        x_min, x_max = df[x_name].min(), df[x_name].max()
        y_min, y_max = df[y_name].min(), df[y_name].max()
        rng = ((x_min, x_max), (y_min, y_max))

    x, y = np.meshgrid(
        np.linspace(rng[0][0], rng[0][1], N), np.linspace(rng[1][0], rng[1][1], N)
    )
    col = []

    if slice_definition is None:
        slice_definition = pd.DataFrame(df.iloc[0])

    for i in range(len(x.ravel())):
        d = slice_definition.copy()
        d[x_name] = x.ravel()[i]
        d[y_name] = y.ravel()[i]
        col.append(mod.predict(d)[0])
    unique = [u for u in np.unique(col)]
    for i, c in enumerate(col):
        col[i] = unique.index(c)
    z = np.array(col).reshape(x.shape)

    if ax is None:
        plt.pcolormesh(x, y, z, shading="auto")
        if df is not None:
            plt.scatter(df[x_name], df[y_name])
    else:
        ax.pcolormesh(x, y, z, shading="auto")
        if df is not None:
            ax.scatter(df[x_name], df[y_name])


def plot_pruning_quality(beta=None, qual_mean=None, qual_sd=None):
    plt.errorbar(beta, qual_mean, yerr=qual_sd)


def print_bars(d, max_width=70):
    max_val = max(d.values())
    usable_width = max_width - 19
    for key, val in d.items():
        L = int(round(usable_width * val / max_val))
        print(f"{key:10}|{'#'*L}{' '*(usable_width-L)}{val:4.2}")


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

def print_rules(tree):
    leafs = tree.leafs()
    d = {}
    for i, L in enumerate(leafs):
        rules = []
        node = L
        while node is not None:
            rules.append((node.attribute,node.threshold))
            node = node.parent
        d[i] = rules
    return d
        
