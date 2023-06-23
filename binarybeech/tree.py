#!/usr/bin/env python
# coding: utf-8

import numpy as np
import json
from binarybeech.attributehandler import attribute_handler_factory

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
        d["decision_fun"] = self.decision_fun
        d["is_leaf"] = self.is_leaf
        d["value"] = self.value
        d["pinfo"] = self.pinfo
        return d
        
    def to_json(self,filename=None):
        d = self.to_dict()
        d["decision_fun"] = d["decision_fun"].__qualname__
        if filename is None:
            return json.dumps(d)
        else:
            with open(filename,"w") as f:
                json.dump(d,f)
                
    @classmethod
    def from_dict(cls,d):
        n = cls(branches=d.get("branches"),
                attribute=d.get("attribute"),
                threshold=d.get("threshold"),
                value=d.get("value"),
                decision_fun=d.get("decision_fun"),
                parent=d.get("parent"),
            )
        n.pinfo = d.get("pinfo",{})
        return n
        


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
        d = node.to_dict()
        if not node.is_leaf:
            d["branches"] = []
            for b in node.branches:
                d_ = self._to_dict(b)
                d["branches"].append(d_)
        return d
                
        
    def to_json(self,filename=None):
        d = self.to_dict()
        print(d)
        self._replace_fun(d)
        if filename is None:
            return json.dumps(d)
        else:
            with open(filename,"w") as f:
                json.dump(d,f)
                
    def _replace_fun(self, d):
        if "decision_fun" in d and d["decision_fun"] is not None:
            d["decision_fun"] = d["decision_fun"].__qualname__.split(".")[-2]
        if "branches" in d and d["branches"] is not None:
            for b in d["branches"]:
                self._replace_fun(b)
                
    @classmethod
    def from_dict(cls,d):
        root = cls._from_dict(d)
        return cls(root)
    
    @staticmethod
    def _from_dict(d):
        # if the dict does not describe a leaf, process the branches first.
        if not d["is_leaf"]:
            branches = []
            for b in d["branches"]:
                branches.append(Tree._from_dict(b))
            d["branches"] = branches
        return Node.from_dict(d)
        
    @classmethod
    def from_json(cls,filename=None,string=None):
        if not filename and not string:
            raise ValueError("Either filename or a string has to be passed as argument to from_json.")
        
        if filename is not None:
            with open(filename, "r") as f:
                d = json.load(f)
        else:
            d = json.loads(string)
    
        cls._replace_str_with_fun(d)
        
        tree = cls.from_dict(d)
        return tree
        
    @staticmethod
    def _replace_str_with_fun(d):
        if not d["is_leaf"]:
            s = d["decision_fun"]
            d["decision_fun"] = attribute_handler_factory[s].decide
            for b in d["branches"]:
                Tree._replace_str_with_fun(b)
        
        
        
        
    