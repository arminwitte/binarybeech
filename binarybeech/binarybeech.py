#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import copy
import treelib
import itertools
import scipy.optimize as opt
import logging

from binarybeech.metrics import metrics_factory

class Node:
    def __init__(self,branches=None,attribute=None,threshold=None,value=None):
        if branches is None and value is None:
            raise ValueError("You have to specify either the branches emerging from this node or a value for this leaf.")
        
        self.branches = branches
        self.threshold = threshold
        self.attribute = attribute
        self.is_leaf = True if self.branches is None else False
        self.value = value
        self.pinfo = {}
        
    def get_child(self,df):
        if isinstance(self.threshold,(int,float,np.number)):
            return self.branches[0] if df[self.attribute] < self.threshold else self.branches[1]
        else:
            return self.branches[0] if df[self.attribute] in self.threshold else self.branches[1]
        
class Tree:
    def __init__(self,root):
        self.root = root
        
    def predict(self,x):
        item = self.root
        while not item.is_leaf:
            item = item.get_child(x)
        return item
    
    def leaf_count(self):
        return self._leaf_count(self.root)
    
    def _leaf_count(self,node):
        if node.is_leaf:
            return 1
        else:
            return np.sum([self._leaf_count(b) for b in node.branches])
    
    def nodes(self):
        return self._nodes(self.root)
    
    def _nodes(self,node):
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
        self._show(self.root,tree_view)
        tree_view.show()
        
    def _show(self,node,tree_view,parent=None,prefix=""):
        name = str(hash(node))
        if node.is_leaf:
            text = f"{prefix}{node.value}"
        else:
            if isinstance(node.threshold,(int,float,np.number)):
                text = f"{prefix}{node.attribute}<{node.threshold:.2f}"
            else:
                text = f"{prefix}{node.attribute} in {node.threshold}"
        tree_view.create_node(text,name,parent=parent)
        
        if not node.is_leaf:
            for i, b in enumerate(node.branches):
                p = "True: " if i == 0 else "False:"
                self._show(b,tree_view,parent=name,prefix=p)
    
class CART:
    def __init__(self,df,y_name,X_names=None,min_leaf_samples=1,min_split_samples=1,max_depth=32767,metrics_type="regression"):
        self.y_name = y_name
        if X_names is None:
            X_names = list(df.columns)
            X_names.remove(self.y_name)
        self.X_names = X_names
        self.df = self._handle_missings(df)
        self.tree = None
        self.splittyness = 1.
        self.leaf_loss_threshold = 1e-12
        
        self.classes = np.unique(df[self.y_name]).tolist()
        
        self.min_leaf_samples = min_leaf_samples
        self.min_split_samples = min_split_samples
        self.max_depth = max_depth
        
        self.depth = 0
        
        self.metrics_type = metrics_type
        self.metrics = metrics_factory.create_metrics(metrics_type, self.y_name)

        self.logger = logging.getLogger(__name__)
        
    def train(self,k=5, plot=True, slack=1.):
        """
        train desicion tree by k-fold cross-validation
        """
        #shuffle dataframe
        df = self.df.sample(frac=1.)
        
        # train tree with full dataset
        self.create_tree()
        pres = self.prune()
        beta = self._beta(pres["alpha"])
        qual_cv = np.zeros((len(beta),k))
        #split df for k-fold cross-validation
        training_sets, test_sets = self._k_fold_split(df,k)
        for i in range(len(training_sets)):
            c = CART(training_sets[i],
                     self.y_name,
                     X_names = self.X_names, 
                     min_leaf_samples=self.min_leaf_samples,
                     min_split_samples=self.min_split_samples,
                     max_depth=self.max_depth,
                     metrics_type=self.metrics_type)
            c.create_tree()          
            pres = c.prune(test_set=test_sets[i])
            qual = self._qualities(beta,pres)
            qual_cv[:,i] = np.array(qual)
        qual_mean = np.mean(qual_cv, axis=1)
        qual_sd = np.std(qual_cv, axis = 1)
        qual_sd_mean = np.mean(qual_sd)
        import matplotlib.pyplot as plt
        plt.errorbar(beta,qual_mean,yerr=qual_sd)
        
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
    
    def _beta(self,alpha):
        beta = []
        for i in range(len(alpha)-1):
            if alpha[i] <= 0:
                continue
            b = np.sqrt(alpha[i]*alpha[i+1])
            beta.append(b)
        return beta
            
    def _quality_at(self,b,data):
        for i, a in enumerate(data["alpha"]):
            if a > b:
                return data["A_cv"][i-1]
        return 0.
    
    def _qualities(self,beta,data):
        return [self._quality_at(b,data) for b in beta]
    
    @staticmethod
    def _k_fold_split(df,k):
        N = len(df.index)
        n = int(np.ceil(N/k))
        training_sets = []
        test_sets = []
        for i in range(k):
            test = df.iloc[i*n:min(N,(i+1)*n),:]
            training = df.loc[df.index.difference(test.index),:]
            test_sets.append(test)
            training_sets.append(training)
        return training_sets, test_sets
    
    def _handle_missings(self,df_in):
        df_out = df_in.dropna(subset=[self.y_name])
        # use nan as category
        # use mean if numerical
        for name in self.X_names:
            if np.issubdtype(df_out[name].values.dtype, np.number):
                df_out[name] = df_out[name].fillna(np.nanmean(df_out[name].values))
            else:
                df_out[name] = df_out[name].fillna("missing")
        return df_out
        
    def create_tree(self, leaf_loss_threshold=1e-12):
        self.leaf_loss_threshold = leaf_loss_threshold
        root = self._node_or_leaf(self.df)
        self.tree = Tree(root)
        n_leafs = self.tree.leaf_count()
        self.logger.info(f"A tree with {n_leafs} leafs was created")
        return self.tree
           
    def _opt_fun(self,df,split_name):
        def fun(x):
            split_df = [df[df[split_name]<x],
                        df[df[split_name]>=x]]
            N = len(df.index)
            n = [len(df_.index) for df_ in split_df]
            return n[0]/N * self._loss(split_df[0]) + n[1]/N * self._loss(split_df[1])
        return fun
        
    def _node_or_leaf(self,df):
        loss_parent = self._loss(df)
        #p = self._probability(df)
        if (loss_parent < self.leaf_loss_threshold
            #p < 0.025
            #or p > 0.975
            or len(df.index) < self.min_leaf_samples
            or self.depth > self.max_depth):
            return self._leaf(df)
        
        loss_best, split_df, split_threshold, split_name = self._loss_best(df)
        if split_df is None:
            return self._leaf(df)
        self.logger.debug(f"Computed split:\nloss: {loss_best:.2f} (parent: {loss_parent:.2f})\nattribute: {split_name}\nthreshold: {split_threshold}\ncount: {[len(df_.index) for df_ in split_df]}")
        if loss_best < loss_parent:
            #print(f"=> Node({split_name}, {split_threshold})")
            branches = []
            self.depth += 1
            for i in range(2):
                branches.append(self._node_or_leaf(split_df[i]))  
            self.depth -= 1
            unique, counts = np.unique(df[self.y_name], return_counts=True)
            value = self._node_value(df)
            item = Node(branches=branches,attribute=split_name,threshold=split_threshold,value=value)
            item.pinfo["N"] = len(df.index)
            item.pinfo["r"] = self.metrics.loss_prune(df)
            item.pinfo["R"] = item.pinfo["N"]/len(self.df.index) * item.pinfo["r"]
        else:
            item = self._leaf(df)
            
        return item
    
    def _leaf(self,df):
        #unique, counts = np.unique(df[self.y_name].values,return_counts=True)
        #print([(unique[i], counts[i]) for i in range(len(counts))])
        #sort_ind = np.argsort(-counts)
        value = self._node_value(df)#unique[sort_ind[0]]
        leaf = Node(value=value)
        
        leaf.pinfo["N"] = len(df.index)
        leaf.pinfo["r"] = self.metrics.loss_prune(df)
        leaf.pinfo["R"] = leaf.pinfo["N"]/len(self.df.index) * leaf.pinfo["r"]
        #print(f"=> Leaf({value}, N={len(df.index)})")
        return leaf
    
    def _loss_best(self,df):
        loss = np.Inf
        split_df = None
        split_threshold = None
        split_name = None
        for name in self.X_names:
            loss_ = np.Inf
            if np.issubdtype(df[name].values.dtype, np.number):
                loss_, split_df_, split_threshold_ = self._split_by_number(df,name)
            else:
                loss_, split_df_, split_threshold_ = self._split_by_class(df,name)
            #print(loss_)
            if (loss_ < loss
                and np.min([len(df_.index) for df_ in split_df_]) >= self.min_split_samples):
                loss = loss_
                split_threshold = split_threshold_
                split_df = split_df_
                split_name = name

        return loss, split_df, split_threshold, split_name
    
    def _split_by_number(self,df,name):
        if -df[name].min()+df[name].max() < np.finfo(float).tiny:
            return np.Inf, None, None
        res = opt.minimize_scalar(self._opt_fun(df,name),bounds=(df[name].min(),df[name].max()),method="bounded")
        split_threshold = res.x
        split_df = [df[df[name]<split_threshold],
                    df[df[name]>=split_threshold]]
        loss = res.fun
        return loss, split_df, split_threshold
    
    def _split_by_class(self,df,name):
        unique = np.unique(df[name])
        comb = []
        if len(unique) > 5:
            comb = [(u,) for u in unique]
        else:
            for i in range(1,len(unique)):
                comb += list(itertools.combinations(unique,i))
            
        if len(comb) < 1:
            return np.Inf, None, None
        
        loss_ = np.Inf
        loss = np.Inf
        for c in comb:
            split_threshold_ = c
            split_df_ =[df[df[name].isin(split_threshold_)],
                        df[~df[name].isin(split_threshold_)]]
            N = len(df.index)
            n = [len(df_.index) for df_ in split_df_]
            loss_ = n[0]/N * self._loss(split_df_[0]) + n[1]/N * self._loss(split_df_[1])
            if loss_ < loss:
                loss = loss_
                split_threshold = split_threshold_
                split_df = split_df_
        return loss, split_df, split_threshold
    
    def _loss(self,df):
        return self.metrics.loss(df)

    def _node_value(self,df):
        return self.metrics.node_value(df)
    
    def validate(self,df=None):
        if df is None:
            df = self.df
        y_hat = []
        for x in df.iloc:
            y_hat.append(self.tree.predict(x).value)
        y_hat = np.array(y_hat)
        return self.metrics.validate(y_hat, df)
    
    def prune(self,alpha_max=None, test_set=None):
        #if not alpha_max:
        #    tree = copy.deepcopy(self.tree)
        #else:
        tree = self.tree
                
        d={}
        d["alpha"]=[]
        d["R"]=[]
        d["n_leafs"]=[]
        if test_set is not None:
            d["A_cv"] = []
            d["R_cv"] = []
            d["P_cv"] = []
            d["F_cv"] = []
        n_iter = 0
        g_min = 0
        alpha = 0
        #print("n_leafs\tR\talpha")
        n_leafs, R = self._g2(tree.root)
        #print(f"{n_leafs}\t{R:.4f}\t{g_min:.2e}")
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
                    
            g_min = max(0,np.min(g))
            for i, n in enumerate(pnodes):
                if g[i] <= g_min:
                    n.is_leaf = True
            N, R = self._g2(tree.root)
            #print(f"{N}\t{R:.4f}\t{alpha:.2e}")
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
            
    
    def _g(self,node):
        n_leafs, R_desc = self._g2(node)
        R = node.pinfo["R"]
        #print(n_leafs, R, R_desc)
        return (R - R_desc)/(n_leafs - 1)
                              
    def _g2(self,node):
        n_leafs = 0
        R_desc = 0
        if node.is_leaf:
            return 1, node.pinfo["R"]
        
        for b in node.branches:
            nl, R = self._g2(b)
            n_leafs += nl
            R_desc += R
        return n_leafs, R_desc


class GradientBoostedTree:
    def __init__(self,df,y_name,X_names=None,sample_frac=1, n_attributes=None, learning_rate=0.1,cart_settings={}, init_metrics_type="logistic"):
        self.df = df.copy()
        self.y_name = y_name
        if X_names is None:
            self.X_names = [s for s in df.columns if s not in [y_name]]
        else:
            self.X_names = X_names
        
        self.init_tree = None
        self.trees = []
        self.gamma = []
        self.learning_rate = learning_rate
        self.cart_settings = cart_settings
        self.init_metrics_type = init_metrics_type
        self.metrics = metrics_factory.create_metrics(self.init_metrics_type, self.y_name)
        self.sample_frac = sample_frac
        self.n_attributes = n_attributes

        self.logger = logging.getLogger(__name__)
    
    def _initial_tree(self):
        c = CART(self.df,self.y_name,X_names=self.X_names, max_depth=0, metrics_type=self.init_metrics_type)
        c.create_tree()
        c.prune()
        self.init_tree = c.tree
        return c
            
    def predict(self,x):
        p = self.init_tree.predict(x).value
        for i, t in enumerate(self.trees):
            p += self.learning_rate * self.gamma[i] * t.predict(x).value
        return p

    def predict_all(self,df):
        y_hat = np.empty((len(df.index),))
        for i, x in enumerate(df.iloc):
            y_hat[i] = self.predict(x)
        return y_hat
        
    def _pseudo_residuals(self):
        res = np.empty_like(self.df[self.y_name].values).astype(np.float64)
        for i, x in enumerate(self.df.iloc):
            res[i] = x[self.y_name] - self.predict(x)
        return res
    
    def create_trees(self,M):
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
                X_names = rng.choice(self.X_names,self.n_attributes,replace=False)
            kwargs = dict(max_depth=3,min_leaf_samples=5,min_split_samples=4,metrics_type="regression")
            kwargs = {**kwargs, **self.cart_settings}
            c = CART(df.sample(frac=self.sample_frac),"pseudo_residuals",X_names=X_names,**kwargs)
            c.create_tree()
            gamma = self._gamma(c.tree)
            self.trees.append(c.tree)
            self.gamma.append(gamma)

    def _gamma(self, tree):
        res = opt.minimize_scalar(self._opt_fun(tree), bracket=[0.,2.])
        print(res.x, res.fun)
        return res.x

    def _opt_fun(self, tree):
        y_hat = self.predict_all(self.df)
        delta = np.empty_like(y_hat)
        for i, x in enumerate(self.df.iloc):
            delta[i] = tree.predict(x).value
        def fun(gamma):
            y_hat_new = y_hat + gamma * delta
            return self._logistic_loss(y_hat_new)
        return fun
    
    def _logistic_loss(self,y_hat_new):
        y = self.df[self.y_name].values
        p = y_hat_new
        p = np.clip(p,1e-12,1.-1e-12)
        l = np.sum(-y*np.log(p)-(1-y)*np.log(1-p))
        return l 

    @staticmethod
    def _dichotomize(y_hat):
        y_hat = np.clip(y_hat,0.,1.)
        return np.round(y_hat).astype(int)

    def validate(self, df=None):
        if df is None:
            df = self.df
        y_hat = self.predict_all(df)
        #from binarybeech.metrics import LogisticMetrics
        #m = LogisticMetrics(self.y_name)
        #m = metrics_factory.create_metrics(self.init_metrics_type,self.y_name)
        return self.metrics.validate(y_hat, df)
            
