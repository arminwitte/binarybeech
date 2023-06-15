#!/usr/bin/env python
# coding: utf-8

import numpy as np
import json

class Node:
    def __init__(
        self,
        branches=None,
        attribute=None,
        threshold=None,
        value=None,
        decision_fun=None,
        parent=None,
    ):
        if branches is None and value is None:
            raise ValueError(
                "You have to specify either the branches emerging from this node or a value for this leaf."
            )

        self.branches = branches
        self.threshold = threshold
        self.attribute = attribute
        self.decision_fun = decision_fun
        self.is_leaf = True if self.branches is None else False
        self.value = value
        self.pinfo = {}
        self.parent = parent

    def get_child(self, df):
        return (
            self.branches[0]
            if self.decision_fun(df[self.attribute], self.threshold)
            else self.branches[1]
        )
        
    def to_dict(self):
        d = {}
        d["branches"] = None
        d["threshold"] = self.threshold
        d["attribute"] = self.attribute
        d["decision_fun"] = None
        d["is_leaf"] = self.is_leaf
        d["value"] = self.value
        d["pinfo"] = self.pinfo
        return d
        
    def to_json(self,filename=None):
        d = self.to_dict()
        if filename is None:
            return json.dumps(d)
        else:
            with open(filename,"w") as f:
                json.dump(d,f)


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
        
    def leafs(self):
        return [n for n in self.nodes() if n.is_leaf]

    def classes(self):
        nodes = self.nodes()
        c = []
        for n in nodes:
            c.append(n.value)
        return np.unique(c).tolist()

    def to_dict(self):
        return self._to_dict(self.root)
    
    def _to_dict(self, node):
        if node.is_leaf:
            return node.to_dict()
        else:
            d = node.to_dict()
            d["branches"] = []
            for b in node.branches:
                d_ = self._to_dict(b)
                d["branches"].append(d_)
            return d
                
        
    def to_json(self,filename=None):
        d = self.to_dict()
        if filename is None:
            return json.dumps(d)
        else:
            with open(filename,"w") as f:
                json.dump(d,f)
    