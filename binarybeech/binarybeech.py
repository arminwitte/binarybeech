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

    def predict(self, x):
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


class Reporter:
    def __init__(self, labels):
        self.labels = labels
        self.buffer = {}
    
    def set(self, **kwargs):
        self.buffer.update(kwargs)
        
    def print(self):
        s = ""
        for l in self.labels:
            v = self.buffer.get(l)
            if v is None:
                s += " - \t"
            elif isinstance(v,float):
                s += f"{v:4.2f}\t"
            elif isinstance(v,int):
                s += f"{v:6}\t"
            elif isinstance(v,str):
                s += f"{v[:9]}\t"
            else:
                s += f"{v:10}\t"
        print(s)
        self.buffer = {}


class Splitter(ABC):
    def __init__(self, y_name, attribute,  metrics_type):
        self.y_name = y_name
        self.attribute = attribute
        self.metrics = metrics_factory.create_metrics(metrics_type, self.y_name)
        
        self.loss = None
        self.split_df = []
        self.threshold = None
    
    @abstractmethod
    def split(self, df):
        pass


class NominalSplitter(Splitter):
    def __init__(self, y_name, attribute, metrics_type):
        super().__init__(y_name, metrics_type)
        
    def split(self, df):
        self.loss = np.Inf
        self.split_df = []
        self.threshold = None
        
        success = False
        
        unique = np.unique(df[self.attribute])
        
        if len(unique) < 1:
            return success
        
        comb = []
        name = self.attribute
        
        if len(unique) > 5:
            comb = [(u,) for u in unique]
        else:
            for i in range(1, len(unique)):
                comb += list(itertools.combinations(unique, i))
        
        loss = np.Inf
        
        for c in comb:
            threshold = c
            split_df = [
                df[df[name].isin(threshold)],
                        df[~df[name].isin(threshold)],
                    ]
            N = len(df.index)
            n = [len(df_.index) for df_ in split_df]
            loss = n[0] / N * self.metrics.loss(split_df[0]) + n[1] / N * self.metrics.loss(
                        split_df[1]
                    )
            if loss < self.loss:
                success = True
                self.loss = loss
                self.threshold = threshold
                self.split_df = split_df
                
        return success

class DichotomousSplitter(Splitter):
    def __init__(self, y_name, attribute, metrics_type):
        super().__init__(y_name, metrics_type)
        
    def split(self, df):
        self.loss = np.Inf
        self.split_df = []
        self.threshold = None
        
        success = False
        
        N = len(df.index)
        
        if np.sum(df[self.attribute]) >= N or np.sum(df[self.attribute]) <= 0.:
            return success
        
        self.threshold = 0.5
        self.split_df = [
        df[df[self.attribute] == 1],
        df[df[self.attribute] == 0]]
        N = len(df.index)
        n = [len(df_.index) for df_ in split_df]
        self.loss = n[0] / N * self.metrics.loss(split_df[0]) + n[1] / N * self.metrics.loss(split_df[1])
        
        return success
        
        
class IntervalSplitter(Splitter):
    def __init__(self, y_name, attribute, metrics_type):
        super().__init__(y_name, metrics_type)
        
    def split(self, df):
        self.loss = np.Inf
        self.split_df = []
        self.threshold = None
        
        success = False
        
        if -df[self.attribute].min() + df[self.attribute].max() < np.finfo(float).tiny:
            return success
            
        mame = self.attribute
        
        res = opt.minimize_scalar(
            self._opt_fun(df),
            bounds=(df[self.attribute].min(), df[self.attribute].max()),
            method="bounded",
        )
        self.threshold = res.x
        self.split_df = [df[df[self.attribute] < self.threshold], df[df[self.attribute] >= self.threshold]]
        self.loss = res.fun
        return res.success
                
    def _opt_fun(self, df):
        split_name = self.attribute
        N = len(df.index)
        def fun(x):
            split_df = [df[df[split_name] < x], df[df[split_name] >= x]]
            n = [len(df_.index) for df_ in split_df]
            return n[0] / N * self.metrics.loss(split_df[0]) + n[1] / N * self.metrics.loss(
                    split_df[1]
                )
    
        return fun
        
class NullSplitter(Splitter):
    def __init__(self, y_name, attribute, metrics_type):
        super().__init__(y_name, metrics_type)
            
    def split(self, df):
        self.loss = np.Inf
        self.split_df = []
        self.threshold = None
            
        success = False
            
        return success
    
class Model(ABC):
    def __init__(self,df, y_name, X_names=None, handle_missings='simple', variable_levels=None):
        self.y_name = y_name
        
        if X_names is None:
            X_names = list(df.columns)
            X_names.remove(self.y_name)
            self.X_names = X_names
        
        self.variable_levels = self._variable_levels(variable_levels)
        self.df = self._handle_missings(df, handle_missings)
            
    def _handle_missings(self, df_in, mode):
        df_out = df_in.dropna(subset=[self.y_name])
        
        if mode is None:
            return df_out
        # use nan as category
        # use mean if numerical
        for name, level in self.variable_levels.items():
            if level == "unknown":
                continue
            elif level == "constant":
                val = np.unique(df_out[~np.isnan(df_out[name])])[0]
                df_out.loc[:,name] = df_out[name].fillna(val)
            elif level == 'interval':
                df_out.loc[:,name] = df_out[name].fillna(np.nanmedian(df_out[name].values))
            elif level == 'dichotomous':
                unique, counts = np.unique(df_out[~np.isnan(df_out[name])], return_counts=True)
                ind_max = np.argmax(counts)
                val = unique[ind_max]
                df_out.loc[:,name] = df_out[name].fillna(val)
            elif level == 'nominal':
                df_out.loc[:,name] = df_out[name].fillna("missing")
            else:
                raise ValueError("Unknown variable level")
        return df_out
        
    def _variable_levels(self, variable_levels=None):
        if variable_levels is not None:
            #TODO: include check whether variable_levels is comprehensive/complete
            return variable_levels
            
        d = {}
        vars = [self.y_name] + self.X_names
        for name in vars:
            df = self.df[name].dropna()
            unique = np.unique(df)
            if len(unique) == 0:
                d[name] = "unknown"
            elif len(unique) == 1:
                d[name] = "constant"
            elif len(unique) == 2 and 0 in unique and 1 in unique:
                d[name] = "dichotomous"
            else: 
                if np.issubdtype(df.values.dtype, np.number):
                    d[name] = "interval"
                else:
                    d[name] = "nominal"
            print(f"{name} is {d[name]}")
        return d
        
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def predict(self, df):
        pass
    
    @abstractmethod
    def validate(self, df=None):
        pass

class CART(Model):
    
    available_splitters = {"unknown":None,
        "constant":Null,
        "dichotomous":DichotomousSplitter,
        "nominal":NominalSplitter,
        "interval":IntervalSplitter
    }
    
    def __init__(
        self,
        df,
        y_name,
        X_names=None,
        min_leaf_samples=1,
        min_split_samples=1,
        max_depth=10,
        metrics_type="regression",
        handle_missings="simple",
        variable_levels=None,
    ):
        super().__init__(df, y_name, X_names=X_names, handle_missings=handle_missings, variable_levels=variable_levels)
        self.tree = None
        #self.splittyness = 1.0
        self.leaf_loss_threshold = 1e-12
        self.metrics_type = self._metrics_type(metrics_type)
        self.metrics = metrics_factory.create_metrics(self.metrics_type, self.y_name)

        self.classes = np.unique(df[self.y_name]).tolist()
        self.variable_levels = self._variable_levels()
        self.splitters = self._init_splitters()

        # pre-pruning
        self.min_leaf_samples = min_leaf_samples
        self.min_split_samples = min_split_samples
        self.max_depth = max_depth

        self.depth = 0

        self.logger = logging.getLogger(__name__)
        
    def _metrics_type(self, override=None):
        if override is not None:
            return override
            
        y_level = self.variable_levels[self.y_name]
        
        d = {"dichotomous":"logistic",
            "nominal":"classification",
            "interval":"regression"}
            
        return d[y_level]
                
    def _init_splitters(self):
        d = {}
        for key, val in self.variable_levels:
            splttr = self.available_splitters[val]
            d[key] = splttr(self.y_name,key,metrics_type=self.metrics_type)
        return d

    def predict_all(self, df):
        y_hat = np.empty((len(df.index),))
        for i, x in enumerate(df.iloc):
            y_hat[i] = self.tree._predict(x).value
        return y_hat

    def train(self, k=5, plot=True, slack=1.0):
        """
        train desicion tree by k-fold cross-validation
        """
        # shuffle dataframe
        df = self.df.sample(frac=1.0)

        # train tree with full dataset
        self.create_tree()
        pres = self.prune()
        beta = self._beta(pres["alpha"])
        qual_cv = np.zeros((len(beta), k))
        # split df for k-fold cross-validation
        sets = utils.k_fold_split(df, k)
        for i, data in enumerate(sets)):
            c = CART(
                data[0],
                self.y_name,
                X_names=self.X_names,
                min_leaf_samples=self.min_leaf_samples,
                min_split_samples=self.min_split_samples,
                max_depth=self.max_depth,
                metrics_type=self.metrics_type,
            )
            c.create_tree()
            pres = c.prune(test_set=data[1])
            qual = self._qualities(beta, pres)
            qual_cv[:, i] = np.array(qual)
        qual_mean = np.mean(qual_cv, axis=1)
        qual_sd = np.std(qual_cv, axis=1)
        qual_sd_mean = np.mean(qual_sd)
        import matplotlib.pyplot as plt

        plt.errorbar(beta, qual_mean, yerr=qual_sd)

        qual_max = np.nanmax(qual_mean)
        ind_max = np.argmax(qual_mean)
        qual_max_sd = qual_sd[ind_max]
        qual_upper = qual_mean + qual_sd * slack
        ind_best = ind_max
        for i in range(ind_max, len(qual_upper)):
            if qual_mean[i] > qual_max - qual_max_sd * slack:
                ind_best = i
        beta_best = beta[ind_best]
        self.logger.info(f"beta_best: {beta_best}")
        self.create_tree()
        self.prune(alpha_max=beta_best)

    def _beta(self, alpha):
        beta = []
        for i in range(len(alpha) - 1):
            if alpha[i] <= 0:
                continue
            b = np.sqrt(alpha[i] * alpha[i + 1])
            beta.append(b)
        return beta

    def _quality_at(self, b, data):
        for i, a in enumerate(data["alpha"]):
            if a > b:
                return data["A_cv"][i - 1]
        return 0.0

    def _qualities(self, beta, data):
        return [self._quality_at(b, data) for b in beta]

    def _handle_missings(self, df_in):
        df_out = df_in.dropna(subset=[self.y_name])
        # use nan as category
        # use mean if numerical
        for name in self.X_names:
            if np.issubdtype(df_out[name].values.dtype, np.number):
                df_out.loc[:,name] = df_out[name].fillna(np.nanmedian(df_out[name].values))
            else:
                df_out.loc[:,name] = df_out[name].fillna("missing")
        return df_out

    def create_tree(self, leaf_loss_threshold=1e-12):
        self.leaf_loss_threshold = leaf_loss_threshold
        root = self._node_or_leaf(self.df)
        self.tree = Tree(root)
        n_leafs = self.tree.leaf_count()
        self.logger.info(f"A tree with {n_leafs} leafs was created")
        return self.tree

    def _node_or_leaf(self, df):
        loss_parent = self._loss(df)
        # p = self._probability(df)
        if (
            loss_parent < self.leaf_loss_threshold
            # p < 0.025
            # or p > 0.975
            or len(df.index) < self.min_leaf_samples
            or self.depth >= self.max_depth
        ):
            return self._leaf(df)

        loss_best, split_df, split_threshold, split_name = self._loss_best(df)
        if split_df is None:
            return self._leaf(df)
        self.logger.debug(
            f"Computed split:\nloss: {loss_best:.2f} (parent: {loss_parent:.2f})\nattribute: {split_name}\nthreshold: {split_threshold}\ncount: {[len(df_.index) for df_ in split_df]}"
        )
        if loss_best < loss_parent:
            # print(f"=> Node({split_name}, {split_threshold})")
            branches = []
            self.depth += 1
            for i in range(2):
                branches.append(self._node_or_leaf(split_df[i]))
            self.depth -= 1
            unique, counts = np.unique(df[self.y_name], return_counts=True)
            value = self._node_value(df)
            item = Node(
                branches=branches,
                attribute=split_name,
                threshold=split_threshold,
                value=value,
            )
            item.pinfo["N"] = len(df.index)
            item.pinfo["r"] = self.metrics.loss_prune(df)
            item.pinfo["R"] = item.pinfo["N"] / len(self.df.index) * item.pinfo["r"]
        else:
            item = self._leaf(df)

        return item

    def _leaf(self, df):
        value = self._node_value(df)
        leaf = Node(value=value)

        leaf.pinfo["N"] = len(df.index)
        leaf.pinfo["r"] = self.metrics.loss_prune(df)
        leaf.pinfo["R"] = leaf.pinfo["N"] / len(self.df.index) * leaf.pinfo["r"]
        return leaf

    def _loss_best(self, df):
        loss = np.Inf
        split_df = None
        split_threshold = None
        split_name = None
        for name in self.X_names:
            loss_ = np.Inf
            splitter = self.splitters[name]
            success = splitter.split(df)
            if not success:
                continue
            loss_ = splitter.loss
            split_df_ = splitter.split_df
            split_threshold_ = splitter.threshold
            # print(loss_)
            if (
                loss_ < loss
                and np.min([len(df_.index) for df_ in split_df_])
                >= self.min_split_samples
            ):
                loss = loss_
                split_threshold = split_threshold_
                split_df = split_df_
                split_name = name

        return loss, split_df, split_threshold, split_name

    #def _loss(self, df):
    #    return self.metrics.loss(df)

    def _node_value(self, df):
        return self.metrics.node_value(df)

    def validate(self, df=None):
        if df is None:
            df = self.df
        y_hat = []
        for x in df.iloc:
            y_hat.append(self.tree._predict(x).value)
        y_hat = np.array(y_hat)
        return self.metrics.validate(y_hat, df)

    def prune(self, alpha_max=None, test_set=None, metrics_only=False):
        if metrics_only:
            tree = copy.deepcopy(self.tree)
        else:
            tree = self.tree

        d = {}
        d["alpha"] = []
        d["R"] = []
        d["n_leafs"] = []
        if test_set is not None:
            d["A_cv"] = []
            d["R_cv"] = []
            d["P_cv"] = []
            d["F_cv"] = []
        n_iter = 0
        g_min = 0
        alpha = 0
        # print("n_leafs\tR\talpha")
        n_leafs, R = self._g2(tree.root)
        # print(f"{n_leafs}\t{R:.4f}\t{g_min:.2e}")
        while tree.leaf_count() > 1 and n_iter < 100:
            n_iter += 1

            alpha = g_min
            if alpha_max is not None and alpha > alpha_max:
                break
            # compute g
            nodes = tree.nodes()
            g = []
            pnodes = []
            for n in nodes:
                if not n.is_leaf:
                    g.append(self._g(n))
                    pnodes.append(n)

            g_min = max(0, np.min(g))
            for i, n in enumerate(pnodes):
                if g[i] <= g_min:
                    n.is_leaf = True
            N, R = self._g2(tree.root)
            # print(f"{N}\t{R:.4f}\t{alpha:.2e}")
            if test_set is not None:
                metrics = self.validate(df=test_set)
                d["A_cv"].append(metrics["accuracy"])
                d["R_cv"].append(metrics["recall"])
                d["P_cv"].append(metrics["precision"])
                d["F_cv"].append(metrics["F-score"])
            d["alpha"].append(alpha)
            d["n_leafs"].append(N)
            d["R"].append(R)
        return d

    def _g(self, node):
        n_leafs, R_desc = self._g2(node)
        R = node.pinfo["R"]
        # print(n_leafs, R, R_desc)
        return (R - R_desc) / (n_leafs - 1)

    def _g2(self, node):
        n_leafs = 0
        R_desc = 0
        if node.is_leaf:
            return 1, node.pinfo["R"]

        for b in node.branches:
            nl, R = self._g2(b)
            n_leafs += nl
            R_desc += R
        return n_leafs, R_desc

class GradientBoostedTree(Model):
    def __init__(
        self,
        df,
        y_name,
        X_names=None,
        sample_frac=1,
        n_attributes=None,
        learning_rate=0.1,
        cart_settings={},
        init_metrics_type="logistic",
        gamma=None,
        handle_missings="simple",
        variable_levels=None,
        ):
        super().__init__(df, y_name, X_names=X_names, handle_missings=handle_missings, variable_levels=variable_levels)
        self.df = self.df.copy()
        self.N = len(self.df.index)

        self.init_tree = None
        self.trees = []
        self.gamma = []
        self.learning_rate = learning_rate
        self.cart_settings = cart_settings
        self.init_metrics_type = init_metrics_type
        self.metrics = metrics_factory.create_metrics(
            self.init_metrics_type, self.y_name
        )
        self.sample_frac = sample_frac
        self.n_attributes = n_attributes
        self.gamma_setting = gamma

        self.logger = logging.getLogger(__name__)

    def _initial_tree(self):
        c = CART(
            self.df,
            self.y_name,
            X_names=self.X_names,
            max_depth=0,
            metrics_type=self.init_metrics_type,
        )
        c.create_tree()
        c.prune()
        self.init_tree = c.tree
        return c

    #@staticmethod
    #def logistic(x):
    #    return 1.0 / (1.0 + np.exp(-x))

    def _predict_log_odds(self, x):
        p = self.init_tree._predict(x).value
        p = np.log(p / (1.0 - p))
        for i, t in enumerate(self.trees):
            p += self.learning_rate * self.gamma[i] * t._predict(x).value
        return p

    def _predict(self, x):
        p = self._predict_log_odds(x)
        return utils.logistic(p)

    def predict_all_log_odds(self, df):
        y_hat = np.empty((len(df.index),))
        for i, x in enumerate(df.iloc):
            y_hat[i] = self.predict_log_odds(x)
        return y_hat

    def predict(self, df):
        p = self.predict_all_log_odds(df)
        return utils.logistic(p)

    def _pseudo_residuals(self):
        # res = np.empty_like(self.df[self.y_name].values).astype(np.float64)
        # for i, x in enumerate(self.df.iloc):
        # res[i] = x[self.y_name] - self._predict(x)
        res = self.df[self.y_name] - self.predict_all(self.df)
        return res

    def create_trees(self, M):
        self._initial_tree()
        res = self._pseudo_residuals()
        df = self.df
        df["pseudo_residuals"] = res
        self.trees = []
        self.gamma = []
        for i in range(M):
            res = self._pseudo_residuals()
            self.logger.info(f"Norm of pseudo-residuals: {np.linalg.norm(res)}")
            df["pseudo_residuals"] = res
            if self.n_attributes is None:
                X_names = self.X_names
            else:
                rng = np.random.default_rng()
                X_names = rng.choice(self.X_names, self.n_attributes, replace=False)
            kwargs = dict(
                max_depth=3,
                min_leaf_samples=5,
                min_split_samples=4,
                metrics_type="regression",
            )
            kwargs = {**kwargs, **self.cart_settings}
            c = CART(
                df.sample(frac=self.sample_frac, replace=True),
                "pseudo_residuals",
                X_names=X_names,
                **kwargs,
            )
            c.create_tree()
            if self.gamma_setting is None:
                gamma = self._gamma(c.tree)
            else:
                gamma = self.gamma_setting
            self.trees.append(c.tree)
            self.gamma.append(gamma)

    def _gamma(self, tree):
        res = opt.minimize_scalar(self._opt_fun(tree), bounds=[0.0, 10.0])
        print(f"{res.x:.2f}\t {res.fun/self.N:.4f}")
        return res.x

    def _opt_fun(self, tree):
        y_hat = self.predict_all_log_odds(self.df)
        delta = np.empty_like(y_hat)
        for i, x in enumerate(self.df.iloc):
            delta[i] = tree._predict(x).value
        y = self.df[self.y_name].values
        def fun(gamma):
            y_ = y_hat + gamma * delta  # * self.learning_rate
            p = self.logistic(y_)
            return utils.logistic_loss(y, p)

        return fun

    def validate(self, df=None):
        if df is None:
            df = self.df
        y_hat = self.predict_all(df)
        # from binarybeech.metrics import LogisticMetrics
        # m = LogisticMetrics(self.y_name)
        return self.metrics.validate(y_hat, df)


class RandomForest(Model):
    def __init__(
        self,
        df,
        y_name,
        X_names=None,
        verbose=False,
        sample_frac=1,
        n_attributes=None,
        cart_settings={},
        metrics_type="regression",
        handle_missings="simple",
        variable_levels=None,
        ):
        super().__init__(df, y_name, X_names=X_names, handle_missings=handle_missings, variable_levels=variable_levels)
        self.df = self.df.copy()
        self.N = len(self.df.index)

        self.trees = []
        self.oob_indices = []
        self.cart_settings = cart_settings
        self.metrics_type = metrics_type
        self.metrics = metrics_factory.create_metrics(self.metrics_type, self.y_name)
        self.sample_frac = sample_frac
        self.n_attributes = n_attributes

        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

    def create_trees(self, M):
        self.trees = []
        for i in range(M):
            df = self.df.sample(frac=self.sample_frac, replace=True)
            if self.n_attributes is None:
                X_names = self.X_names
            else:
                rng = np.random.default_rng()
                X_names = rng.choice(self.X_names, self.n_attributes, replace=False)
            kwargs = dict(
                max_depth=3,
                min_leaf_samples=5,
                min_split_samples=4,
                metrics_type=self.metrics_type,
            )
            kwargs = {**kwargs, **self.cart_settings}
            c = CART(df, self.y_name, X_names=X_names, **kwargs)
            c.create_tree()
            self.trees.append(c.tree)
            self.oob_indices.append(self.df.index.difference(df.index))
            if self.verbose:
                print(f"{i:4d}: Tree with {c.tree.leaf_count()} leaves created.")

    def _predict(self, x):
        y = []
        for t in self.trees:
            y.append(t._predict(x).value)
        unique, counts = np.unique(y, return_counts=True)
        ind_max = np.argmax(counts)
        return unique[ind_max]

    def predict(self, df):
        y_hat = []
        for x in df.iloc:
            y_hat.append(self._predict(x))

        return y_hat

    def validate_oob(self):
        df = self._oob_df()
        df = self._oob_predict(df)
        for index, row in df.iterrows():
            if not row["votes"]:
                continue
            unique, counts = np.unique(row["votes"], return_counts=True)
            idx_max = np.argmax(counts)
            df.loc[index, "majority_vote"] = unique[idx_max]
        df = df.astype({"majority_vote": "int"})
        df = df.dropna(subset=["majority_vote"])
        return self.metrics.validate(df["majority_vote"].values, df)

    def _oob_predict(self, df):
        for i, t in enumerate(self.trees):
            idx = self.oob_indices[i]
            for j in idx:
                x = self.df.loc[j, :]
                y = t._predict(x).value
                df.loc[j]["votes"].append(y)
        return df

    def _oob_df(self):
        df = pd.DataFrame(index=self.df.index, dtype="object")
        df[self.y_name] = self.df[self.y_name].values
        df["votes"] = np.empty((len(df), 0)).tolist()
        df["majority_vote"] = np.NaN
        return df

    def validate(self, df=None):
        if df is None:
            df = self.df
        y_hat = self.predict_all(df)
        return self.metrics.validate(y_hat, df)

    def variable_importance(self):
        d = {}
        for x in self.X_names:
            d[x] = 0.0
        for t in self.trees:
            nodes = t.nodes()
            for n in nodes:
                if n.is_leaf:
                    continue
                name = n.attribute
                R_parent = n.pinfo["R"]
                R_children = np.sum([b.pinfo["R"] for b in n.branches])
                R_delta = R_parent - R_children
                d[name] += R_delta
        max_val = max(d.values())
        for key in d.keys():
            d[key] /= max_val
        d = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
        return d
