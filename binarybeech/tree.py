#!/usr/bin/env python
# coding: utf-8

import copy
import itertools
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.optimize as opt

import treelib
from binarybeech.metrics import metrics_factory
import binarybeech.utils as utils


class Node:
    def __init__(self, branches=None, attribute=None, threshold=None, value=None):
        if branches is None and value is None:
            raise ValueError(
                "You have to specify either the branches emerging from this node or a value for this leaf."
            )

        self.branches = branches
        self.threshold = threshold
        self.attribute = attribute
        self.is_leaf = True if self.branches is None else False
        self.value = value
        self.pinfo = {}

    def get_child(self, df):
        if isinstance(self.threshold, (int, float, np.number)):
            return (
                self.branches[0]
                if df[self.attribute] < self.threshold
                else self.branches[1]
            )
        else:
            return (
                self.branches[0]
                if df[self.attribute] in self.threshold
                else self.branches[1]
            )


class Tree:
    def __init__(self, root):
        self.root = root

    def traverse(self, x):
        item = self.root
        while not item.is_leaf:
            item = item.get_child(x)
        return item

    def leaf_count(self):
        return self._leaf_count(self.root)

    def _leaf_count(self, node):
        if node.is_leaf:
            return 1
        else:
            return np.sum([self._leaf_count(b) for b in node.branches])

    def nodes(self):
        return self._nodes(self.root)

    def _nodes(self, node):
        if node.is_leaf:
            return [node]

        nl = [node]
        for b in node.branches:
            nl += self._nodes(b)
        return nl

    def classes(self):
        nodes = self.nodes()
        c = []
        for n in nodes:
            c.append(n.value)
        return np.unique(c).tolist()

    def show(self):
           tree_view = treelib.Tree()
           self._show(self.root, tree_view)
           tree_view.show()

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
                self._show(b, tree_view, parent=name, prefix=p)