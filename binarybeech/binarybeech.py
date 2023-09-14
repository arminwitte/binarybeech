#!/usr/bin/env python
# coding: utf-8

import copy
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from binarybeech.datamanager import DataManager
from binarybeech.minimizer import minimize
from binarybeech.reporter import reporter
from binarybeech.trainingdata import TrainingData
from binarybeech.tree import Node, Tree

# import scipy.optimize as opt


class Model(ABC):
    def __init__(
        self,
        training_data,
        df,
        y_name,
        X_names,
        attribute_handlers,
        method,
        handle_missings,
        algorithm_kwargs,
    ):
        if isinstance(training_data, TrainingData):
            self.training_data = training_data
        elif isinstance(df, pd.DataFrame):
            self.training_data = TrainingData(
                df, y_name=y_name, X_names=X_names, handle_missings=handle_missings
            )
        else:
            raise TypeError(
                "Wrong data type. Either pass training_data as a TrainingData object or df as a pandas DataFrame."
            )

        self.y_name = self.training_data.y_name
        self.X_names = self.training_data.X_names

        self.dmgr = DataManager(
            self.training_data, method, attribute_handlers, algorithm_kwargs
        )

        self.algorithm_kwargs = algorithm_kwargs

        # self.training_data.df = self._handle_missings(df, handle_missings)

    # def _handle_missings(self, df, mode):
    #     df = df.dropna(subset=[self.y_name])

    #     if mode is None:
    #         return df
    #     elif mode == "simple":
    #         # use nan as category
    #         # use mean if numerical
    #         for name, dh in self.dmgr.items():
    #             df = dh.handle_missings(df)
    #     elif mode == "model":
    #         raise ValueError("Not implemented")

    #     return df

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, df):
        pass

    def validate(self, df=None):
        if df is None:
            df = self.training_data.df
        y_hat = self.predict(df)
        y = df[self.y_name]
        return self.dmgr.metrics.validate(y, y_hat)

    def goodness_of_fit(self, df=None):
        if df is None:
            df = self.training_data.df
        y_hat = self.predict(df)
        y = df[self.y_name]
        return self.dmgr.metrics.goodness_of_fit(y, y_hat)


class CART(Model):
    def __init__(
        self,
        training_data=None,
        df=None,
        y_name=None,
        X_names=None,
        min_leaf_samples=1,
        min_split_samples=1,
        max_depth=10,
        min_split_loss = 0.,
        method="regression",
        handle_missings="simple",
        attribute_handlers=None,
        seed=None,
        algorithm_kwargs={},
    ):
        super().__init__(
            training_data,
            df,
            y_name,
            X_names,
            attribute_handlers,
            method,
            handle_missings,
            algorithm_kwargs,
        )
        self.tree = None
        self.leaf_loss_threshold = 1e-12

        # pre-pruning
        self.min_leaf_samples = min_leaf_samples
        self.min_split_samples = min_split_samples
        self.max_depth = max_depth
        self.min_split_loss = min_split_loss

        self.depth = 0
        self.seed = seed

        self.logger = logging.getLogger(__name__)

    def _predict1(self, x):
        return self.tree.traverse(x).value

    def _predict_raw(self, df):
        y_hat = [self._predict1(x) for x in df.iloc]
        return np.array(y_hat)

    def predict(self, df):
        y_hat = self._predict_raw(df)
        return self.dmgr.metrics.output_transform(y_hat)

    def train(self, k=5, plot=True, slack=1.0):
        """
        train decision tree by k-fold cross-validation
        """
        # shuffle dataframe
        # df = self.training_data.df.sample(frac=1.0)

        # train tree with full dataset
        self.create_tree()
        pres = self.prune()
        beta = self._beta(pres["alpha"])
        qual_cv = np.zeros((len(beta), k))
        # split df for k-fold cross-validation
        self.training_data.split(k=k, seed=self.seed)
        sets = self.training_data.data_sets
        for i, data in enumerate(sets):
            c = CART(
                df=data[0],
                y_name=self.y_name,
                X_names=self.X_names,
                min_leaf_samples=self.min_leaf_samples,
                min_split_samples=self.min_split_samples,
                max_depth=self.max_depth,
                method=self.dmgr.method,
                attribute_handlers=self.dmgr,
            )
            c.create_tree()
            pres = c.prune(test_set=data[1])
            qual = self._qualities(beta, pres)
            qual_cv[:, i] = np.array(qual)
        qual_mean = np.mean(qual_cv, axis=1)
        qual_sd = np.std(qual_cv, axis=1)
        # qual_sd_mean = np.mean(qual_sd)
        # import matplotlib.pyplot as plt

        # plt.errorbar(beta, qual_mean, yerr=qual_sd)

        self.pruning_quality = {
            "beta": beta,
            "qual_mean": qual_mean,
            "qual_sd": qual_sd,
        }

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
                return data["goodness_of_fit"][i - 1]
        return 0.0

    def _qualities(self, beta, data):
        return [self._quality_at(b, data) for b in beta]

    def create_tree(self, leaf_loss_threshold=1e-12):
        self.leaf_loss_threshold = leaf_loss_threshold
        root = self._node_or_leaf(self.training_data.df)
        self.tree = Tree(root)
        n_leafs = self.tree.leaf_count()
        reporter.message(f"A tree with {n_leafs} leafs was created", 1)
        return self.tree

    def _node_or_leaf(self, df):
        y = df[self.y_name]

        loss_args = {}
        if "__weights__" in df:
            loss_args["weights"] = df["__weights__"].values

        y_hat = self.dmgr.metrics.node_value(y, **loss_args)
        loss_parent = self.dmgr.metrics.loss(y, y_hat, **loss_args)
        # p = self._probability(df)
        if (
            loss_parent < self.leaf_loss_threshold
            # p < 0.025
            # or p > 0.975
            or len(df.index) < self.min_leaf_samples
            or self.depth >= self.max_depth
        ):
            return self._leaf(y, y_hat)

        loss_best, split_df, split_threshold, split_name = self._loss_best(df)
        if not split_df:
            return self._leaf(y, y_hat)
        # print(
        #    f"Computed split:
        # \nloss: {loss_best:.2f} (parent: {loss_parent:.2f})
        # \nattribute: {split_name}
        # \nthreshold: {split_threshold}
        # \ncount: {[len(df_.index) for df_ in split_df]}"
        # )
        # print(f"gain: {loss_parent - loss_best}")
        if loss_best < loss_parent - self.min_split_loss:
            # print(f"=> Node({split_name}, {split_threshold})")
            branches = []
            self.depth += 1
            for i in range(2):
                branches.append(self._node_or_leaf(split_df[i]))
            self.depth -= 1
            # unique, counts = np.unique(df[self.y_name], return_counts=True)
            value = y_hat
            item = Node(
                branches=branches,
                attribute=split_name,
                threshold=split_threshold,
                value=value,
                decision_fun=self.dmgr[split_name].decide,
            )
            item.pinfo["N"] = len(df.index)
            loss_args ={}
            item.pinfo["r"] = self.dmgr.metrics.loss_prune(y, y_hat, **loss_args)
            item.pinfo["R"] = (
                item.pinfo["N"] / len(self.training_data.df.index) * item.pinfo["r"]
            )
            for b in item.branches:
                b.parent = item
        else:
            item = self._leaf(y, y_hat)

        return item

    def _leaf(self, y, y_hat):
        leaf = Node(value=y_hat)

        leaf.pinfo["N"] = y.size
        loss_args = {}
        leaf.pinfo["r"] = self.dmgr.metrics.loss_prune(y, y_hat, **loss_args)
        leaf.pinfo["R"] = (
            leaf.pinfo["N"] / len(self.training_data.df.index) * leaf.pinfo["r"]
        )
        return leaf

    def _loss_best(self, df):
        loss = np.Inf
        split_df = None
        split_threshold = None
        split_name = None
        for name in self.X_names:
            loss_ = np.Inf
            dh = self.dmgr[name]
            success = dh.split(df)
            if not success:
                continue
            loss_ = dh.loss
            split_df_ = dh.split_df
            split_threshold_ = dh.threshold
            # print(name[:7],"\t:",loss_,"(",len(split_df_[0].index),",",len(split_df_[1].index),")")
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
            d["goodness_of_fit"] = []
        #    d["A_cv"] = []
        #    d["R_cv"] = []
        #    d["P_cv"] = []
        #    d["F_cv"] = []
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
                # metrics = self.validate(df=test_set)
                # for key, val in metrics.items():
                #    d["".join((key,"_cv"))].append(val)
                d["goodness_of_fit"].append(self.goodness_of_fit(df=test_set))
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
        training_data=None,
        df=None,
        y_name=None,
        X_names=None,
        sample_frac=1,
        n_attributes=None,
        learning_rate=0.1,
        cart_settings={},
        init_method="logistic",
        gamma=None,
        handle_missings="simple",
        attribute_handlers=None,
        seed=None,
        algorithm_kwargs={},
    ):
        super().__init__(
            training_data,
            df,
            y_name,
            X_names,
            attribute_handlers,
            init_method,
            handle_missings,
            algorithm_kwargs,
        )
        self.df = self.training_data.df.copy()
        self.N = len(self.df.index)

        self.init_tree = None
        self.trees = []
        self.gamma = []
        self.learning_rate = learning_rate
        self.cart_settings = cart_settings
        self.init_method = init_method
        self.sample_frac = sample_frac
        self.n_attributes = n_attributes
        self.gamma_setting = gamma
        self.seed = seed

        self.logger = logging.getLogger(__name__)
        reporter.reset(["iter", "res_norm", "gamma", "sse"])

    def _initial_tree(self):
        c = CART(
            df=self.df,
            y_name=self.y_name,
            X_names=self.X_names,
            max_depth=0,
            method=self.init_method,
            attribute_handlers=self.dmgr,
            seed=None,
        )
        c.create_tree()
        self.init_tree = c.tree
        return c

    def _predict1(self, x, m=None):
        p = self.init_tree.traverse(x).value
        p = self.dmgr.metrics.inverse_transform(p)
        M = len(self.trees)
        if not m:
            m = M
        for i in range(m):
            t = self.trees[i]
            p += self.learning_rate * self.gamma[i] * t.traverse(x).value
        return p

    def _predict_raw(self, df, m=None):
        y_hat = [self._predict1(x, m) for x in df.iloc]
        return np.array(y_hat)

    def predict(self, df, m=None):
        y_hat = self._predict_raw(df, m)
        return self.dmgr.metrics.output_transform(y_hat)

    def _pseudo_residuals(self, df, m=None):
        res = df[self.y_name] - self.predict(df, m=m)
        return res

    def train(self, M):
        self._initial_tree()
        df = self.df
        self.trees = []
        self.gamma = []

        for i in range(M):
            res = self._pseudo_residuals(df)
            # print(f"Norm of pseudo-residuals: {np.linalg.norm(res)}")
            reporter["iter"] = i
            reporter["res_norm"] = np.linalg.norm(res)
            df["pseudo_residuals"] = res

            c = self._append_regression_tree(df)

            if self.gamma_setting is None:
                gamma = self._gamma(c.tree)
            else:
                gamma = self.gamma_setting
            self.trees.append(c.tree)
            self.gamma.append(gamma)
            reporter.print(level=2)

    def _append_regression_tree(self, df):
        if self.n_attributes is None:
            X_names = self.X_names
        else:
            rng = np.random.default_rng(seed=self.seed)
            if self.seed is not None:
                self.seed += 1
            X_names = rng.choice(self.X_names, self.n_attributes, replace=False)

        kwargs = dict(
            max_depth=3,
            min_leaf_samples=5,
            min_split_samples=4,
            method="regression",
        )
        kwargs = {**kwargs, **self.cart_settings}

        c = CART(
            df=df.sample(frac=self.sample_frac, replace=True, random_state=self.seed),
            y_name="pseudo_residuals",
            X_names=X_names,
            **kwargs,
        )
        c.create_tree()

        if self.seed is not None:
            self.seed += 1

        return c

    def _gamma(self, tree):
        # minimizer = BrentsScalarMinimizer()
        # x, y = minimizer.minimize(self._opt_fun(tree), 0.0, 10.0)
        method = self.algorithm_kwargs.get("minimizer_method", "brent")
        x, y = minimize(
            self._opt_fun(tree), 0.0, 10.0, method=method, options=self.algorithm_kwargs
        )
        reporter["gamma"] = x
        reporter["sse"] = y / self.N
        return x

    def _opt_fun(self, tree):
        y_hat = self._predict_raw(self.df)
        delta = np.empty_like(y_hat)
        for i, x in enumerate(self.df.iloc):
            delta[i] = tree.traverse(x).value
        y = self.df[self.y_name].values
        
        loss_args = {}
        if "__weights__" in self.df:
            loss_args["weights"] = self.df["__weights__"].values

        def fun(gamma):
            y_ = y_hat + gamma * delta
            p = self.dmgr.metrics.output_transform(y_)
            return self.dmgr.metrics.loss(y, p, **loss_args)

        return fun

    def validate(self, df=None):
        if df is None:
            df = self.df
        y_hat = self.predict(df)
        y = df[self.y_name]
        return self.dmgr.metrics.validate(y, y_hat)

    def update(self, df, update_method="elastic"):
        if update_method == "gamma":
            self._update_gamma(df)
        elif update_method == "elastic":
            self._update_elastic(df)
        else:
            raise ValueError(f"unknown update method {update_method}")

    def _update_elastic(self, df):
        # Wang, K., Liu, A., Lu, J., Zhang, G., Xiong, L. (2020). An Elastic Gradient
        # Boosting Decision Tree for Concept Drift Learning. In: Gallagher, M.,
        # Moustafa, N., Lakshika, E. (eds) AI 2020: Advances in Artificial
        # Intelligence. AI 2020. Lecture Notes in Computer Science(), vol 12576.
        # Springer, Cham. https://doi.org/10.1007/978-3-030-64984-5_33
        M = len(self.trees)
        res_norm = np.Inf
        m = M
        for i in range(M):
            res_norm_old = res_norm
            res_norm = np.linalg.norm(self._pseudo_residuals(df, i))

            print(res_norm_old, "->", res_norm)

            if res_norm_old < res_norm:
                reporter.message("update required")
                m = i

        if m == M:
            reporter.message("no update necessary")
        self.trees = self.trees[:m]

        for i in range(m, M):
            res = self._pseudo_residuals(df)
            reporter["iter"] = i
            reporter["res_norm"] = np.linalg.norm(res)
            df["pseudo_residuals"] = res
            c = self._append_regression_tree(df)

            if self.gamma_setting is None:
                gamma = self._gamma(c.tree)
            else:
                gamma = self.gamma_setting
            self.trees.append(c.tree)
            self.gamma.append(gamma)
            reporter.print()

    def _update_gamma(self, df):
        if self.gamma_setting is not None:
            print("fixed gamma specified. No update required")
            return

        M = len(self.trees)

        bag_of_trees = self.trees.copy()
        self.trees = []
        self.gamma = []

        for i in range(M):
            res = self._pseudo_residuals(df)
            reporter["iter"] = i
            reporter["res_norm"] = np.linalg.norm(res)
            df["pseudo_residuals"] = res
            tree = bag_of_trees[i]
            gamma = self._gamma(tree)
            self.trees.append(tree)
            self.gamma.append(gamma)
            reporter.print()


class AdaBoostTree(Model):
    def __init__(
        self,
        training_data=None,
        df=None,
        y_name=None,
        X_names=None,
        sample_frac=1,
        n_attributes=None,
        cart_settings={},
        method="classification",
        handle_missings="simple",
        attribute_handlers=None,
        seed=None,
        algorithm_kwargs={},
    ):
        super().__init__(
            training_data,
            df,
            y_name,
            X_names,
            attribute_handlers,
            method,
            handle_missings,
            algorithm_kwargs,
        )
        self.df = self.training_data.df.copy()
        self.N = len(self.df.index)
        self.method = method

        self.trees = []
        self.alpha = []  # gamma
        self.cart_settings = cart_settings
        self.sample_frac = sample_frac
        self.n_attributes = n_attributes
        self.seed = seed

        self.logger = logging.getLogger(__name__)
        reporter.reset(["iter", "err", "alpha", "w_ratio"])

    def _predict1(self, x, m=None):
        M = len(self.trees)
        if not m:
            m = M
        d = {}
        for i in range(m):
            alpha = self.alpha[i]
            if alpha <= 0:
                continue
            t = self.trees[i]
            # print(t)
            label = t.traverse(x).value
            # print(">>>>>>>> label:", label)
            if label in d:
                d[label] += alpha
            else:
                d[label] = alpha
        labels = [k for k in d.keys()]
        scores = [s for s in d.values()]
        # print(labels)
        # print(scores)
        ind_max = np.argmax(scores)
        return labels[ind_max]

    def _predict_raw(self, df, m=None):
        y_hat = [self._predict1(x, m) for x in df.iloc]
        return np.array(y_hat)

    def predict(self, df, m=None):
        y_hat = self._predict_raw(df, m)
        return self.dmgr.metrics.output_transform(y_hat)

    def train(self, M):
        df = self.df.copy()
        self.trees = []
        self.alpha = []

        # Initialize the observation weights
        N = len(df.index)
        # w = np.ones((N,)) * 1/N
        if "__weights__" not in df:
            df["__weights__"] = 1 / N
        K = len(np.unique(df[self.y_name]))

        for i in range(M):
            reporter["iter"] = i

            # Fit a classifier
            c = self._decision_stump(df)

            mis = self._I(df, c)
            err = self._err(df, mis)
            reporter["err"] = err

            alpha = self._alpha(err, K)
            alpha = max(0, alpha)
            reporter["alpha"] = alpha

            w = df["__weights__"] * np.exp(alpha * mis)
            reporter["w_ratio"] = np.max(w) / np.min(w)
            df["__weights__"] = w

            self.trees.append(c.tree)
            self.alpha.append(alpha)
            reporter.print()

    def _decision_stump(self, df):
        if self.n_attributes is None:
            X_names = self.X_names
        else:
            rng = np.random.default_rng(seed=self.seed)
            if self.seed is not None:
                self.seed += 1
            X_names = rng.choice(self.X_names, self.n_attributes, replace=False)

        kwargs = dict(
            max_depth=1,
            min_leaf_samples=5,
            min_split_samples=4,
            method=self.method,
        )
        kwargs = {**kwargs, **self.cart_settings}

        c = CART(
            df=df.sample(frac=self.sample_frac, replace=True, random_state=self.seed),
            y_name=self.y_name,
            X_names=X_names,
            **kwargs,
        )
        c.create_tree()

        if self.seed is not None:
            self.seed += 1

        return c

    def _I(self, df, c):
        y_hat = np.array(c.predict(df)).ravel()
        mis = np.empty_like(y_hat)
        for i, x in enumerate(df.iloc):
            mis[i] = 1 if x[self.y_name] != y_hat[i] else 0
        return mis.astype(int)

    def _err(self, df, mis):
        err = np.sum(mis * df["__weights__"])
        # err = 0
        # for i, x in enumerate(df.iloc):
        #     err += x["__weights__"] if df[self.y_name] != y_hat[i] else 0.
        return err / np.sum(df["__weights__"])

    def _alpha(self, err, K):
        return np.log((1.0 - err) / err) + np.log(K - 1)

    def validate(self, df=None):
        if df is None:
            df = self.df
        y_hat = self.predict(df)
        y = df[self.y_name]
        return self.dmgr.metrics.validate(y, y_hat)


class RandomForest(Model):
    def __init__(
        self,
        training_data=None,
        df=None,
        y_name=None,
        X_names=None,
        verbose=False,
        sample_frac=1,
        n_attributes=None,
        cart_settings={},
        method="regression",
        handle_missings="simple",
        attribute_handlers=None,
        seed=None,
        algorithm_kwargs={},
    ):
        super().__init__(
            training_data,
            df,
            y_name,
            X_names,
            attribute_handlers,
            method,
            handle_missings,
            algorithm_kwargs,
        )
        self.df = self.training_data.df.copy()
        self.N = len(self.df.index)

        self.trees = []
        self.oob_indices = []
        self.cart_settings = cart_settings
        self.sample_frac = sample_frac
        self.n_attributes = n_attributes
        self.seed = seed

        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        reporter.reset(["no", "n_leafs"])

    def train(self, M):
        self.trees = []
        seed = self.seed
        for i in range(M):
            df = self.df.sample(frac=self.sample_frac, replace=True, random_state=seed)
            if seed is not None:
                seed += 1

            if self.n_attributes is None:
                X_names = self.X_names
            else:
                rng = np.random.default_rng(seed=seed)
                if seed is not None:
                    seed += 1
                X_names = rng.choice(self.X_names, self.n_attributes, replace=False)
            kwargs = dict(
                max_depth=3,
                min_leaf_samples=5,
                min_split_samples=4,
                method=self.dmgr.method,
                attribute_handlers=self.dmgr,
            )
            kwargs = {**kwargs, **self.cart_settings}
            c = CART(df=df, y_name=self.y_name, X_names=X_names, **kwargs)
            c.create_tree()
            self.trees.append(c.tree)
            self.oob_indices.append(self.df.index.difference(df.index))
            if self.verbose:
                reporter["no"] = i
                reporter["n_leafs"] = c.tree.leaf_count()
                reporter.print()

    def _predict1(self, x):
        y = []
        for t in self.trees:
            y.append(t.traverse(x).value)
        unique, counts = np.unique(y, return_counts=True)
        ind_max = np.argmax(counts)
        return unique[ind_max]

    def _predict_raw(self, df):
        y_hat = [self._predict1(x) for x in df.iloc]
        return np.array(y_hat)

    def predict(self, df):
        y_hat = self._predict_raw(df)
        return self.dmgr.metrics.output_transform(y_hat)

    def validate_oob(self):
        df = self._oob_df()
        df = self._oob_predict(df)
        for index, row in df.iterrows():
            if not row["votes"]:
                continue
            unique, counts = np.unique(row["votes"], return_counts=True)
            idx_max = np.argmax(counts)
            df.loc[index, "majority_vote"] = unique[idx_max]
        df = df.dropna(subset=["majority_vote"])
        # df = df.astype({"majority_vote": "int"})
        y = df[self.y_name]
        return self.dmgr.metrics.validate(y, df["majority_vote"].values)

    def _oob_predict(self, df):
        for i, t in enumerate(self.trees):
            idx = self.oob_indices[i]
            for j in idx:
                x = self.df.loc[j, :]
                y = t.traverse(x).value
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
        y_hat = self.predict(df)
        y = df[self.y_name]
        return self.dmgr.metrics.validate(y, y_hat)

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
