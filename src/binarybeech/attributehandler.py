#!/usr/bin/env python
# coding: utf-8

import itertools
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, List, Dict


import numpy as np
import pandas as pd

import binarybeech.math as math
from binarybeech.minimizer import minimize

# import pandas as pd
# import scipy.optimize as opt


class AttributeHandlerBase(ABC):
    def __init__(
        self, y_name: str, attribute: str, metrics: Any, algorithm_kwargs: dict
    ) -> None:
        self.y_name = y_name
        self.attribute = attribute
        self.metrics = metrics
        self.algorithm_kwargs = algorithm_kwargs

        self.loss: Optional[float] = None
        self.split_df: List[pd.DataFrame] = []
        self.threshold: Any = None

    @abstractmethod
    def split(self, df: "pd.DataFrame") -> bool:
        pass

    @staticmethod
    @abstractmethod
    def decide(x: Any, threshold: Any) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def check(x: Any) -> bool:
        pass


# =========================


class NominalAttributeHandler(AttributeHandlerBase):
    def __init__(self, y_name, attribute, metrics, algorithm_kwargs):
        super().__init__(y_name, attribute, metrics, algorithm_kwargs)

    def split(self, df):
        self.loss = np.inf
        self.split_df = []
        self.threshold = None

        success = False

        unique = np.unique(df[self.attribute])

        if len(unique) < 2:
            return success

        comb = []
        name = self.attribute

        if len(unique) > 5:
            comb = [(u,) for u in unique]
        else:
            for i in range(1, len(unique)):
                comb += list(itertools.combinations(unique, i))

        loss = np.inf

        for c in comb:
            threshold = c
            split_df = [
                df[df[name].isin(threshold)],
                df[~df[name].isin(threshold)],
            ]
            N = len(df.index)
            n = [len(df_.index) for df_ in split_df]

            loss_args = {
                key: self.algorithm_kwargs[key] for key in ["lambda_l1", "lambda_l2"]
            }
            loss_args = [loss_args.copy(), loss_args.copy()]
            if "__weights__" in df:
                for i, df_ in enumerate(split_df):
                    loss_args[i]["weights"] = df_["__weights__"].values

            val = [
                self.metrics.node_value(df_[self.y_name], **loss_args[i])
                for i, df_ in enumerate(split_df)
            ]
            loss = n[0] / N * self.metrics.loss(
                split_df[0][self.y_name], val[0], **loss_args[0]
            ) + n[1] / N * self.metrics.loss(
                split_df[1][self.y_name], val[1], **loss_args[1]
            )
            if loss < self.loss:
                success = True
                self.loss = loss
                self.threshold = threshold
                self.split_df = split_df

        return success

    @staticmethod
    def decide(x, threshold):
        return True if x in threshold else False

    @staticmethod
    def check(x):
        return math.check_nominal(
            x, max_unique_fraction=0.2, exclude_dichotomous=True, high=5
        )


class HighCardinalityNominalAttributeHandler(AttributeHandlerBase):
    def __init__(self, y_name, attribute, metrics, algorithm_kwargs):
        super().__init__(y_name, attribute, metrics, algorithm_kwargs)

    def split(self, df):
        self.loss = np.inf
        self.split_df = []
        self.threshold = None

        success = False

        unique = np.unique(df[self.attribute])

        if len(unique) < 2:
            return success

        bins = self.metrics.bins(df, self.y_name, self.attribute)
        b = bins[0] if len(bins[0]) > 0 else bins[1]
        # print(b)
        # m = ScalarSimulatedAnnealing()
        # m._new = ScalarSimulatedAnnealing._choice
        # m.max_iter = 100
        # x, y = m.minimize(self._opt_fun(df), unique.tolist(), b)
        # print("x:", x)
        x, y = minimize(
            self._opt_fun(df),
            unique.tolist(),
            b,
            method="simulated_annealing",
            options=self.algorithm_kwargs,
        )
        success = True
        self.loss = y
        self.threshold = x
        self.split_df = [
            df[df[self.attribute].isin(x)],
            df[~df[self.attribute].isin(x)],
        ]

        return success

    def _opt_fun(self, df):
        split_name = self.attribute
        N = len(df.index)

        def fun(x):
            split_df = [
                df[df[split_name].isin(x)],
                df[~df[split_name].isin(x)],
            ]
            n = [len(df_.index) for df_ in split_df]
            if min(n) == 0:
                return np.inf

            loss_args = {
                key: self.algorithm_kwargs[key] for key in ["lambda_l1", "lambda_l2"]
            }
            loss_args = [loss_args.copy(), loss_args.copy()]
            if "__weights__" in df:
                for i, df_ in enumerate(split_df):
                    loss_args[i]["weights"] = df_["__weights__"].values

            val = [
                self.metrics.node_value(df_[self.y_name], **loss_args[i])
                for i, df_ in enumerate(split_df)
            ]
            return n[0] / N * self.metrics.loss(
                split_df[0][self.y_name], val[0], **loss_args[0]
            ) + n[1] / N * self.metrics.loss(
                split_df[1][self.y_name], val[1], **loss_args[1]
            )

        return fun

    @staticmethod
    def decide(x, threshold):
        return True if x in threshold else False

    @staticmethod
    def check(x):
        return math.check_nominal(
            x, max_unique_fraction=0.2, exclude_dichotomous=True, low=6
        )


class DichotomousAttributeHandler(AttributeHandlerBase):
    def __init__(self, y_name, attribute, metrics, algorithm_kwargs):
        super().__init__(y_name, attribute, metrics, algorithm_kwargs)

    def split(self, df):
        self.loss = np.inf
        self.split_df = []
        self.threshold = None

        success = False

        N = len(df.index)

        unique = np.unique(df[self.attribute])

        if len(unique) < 2:
            return success

        success = True
        self.threshold = (unique[0],)
        self.split_df = [
            df[df[self.attribute].isin(self.threshold)],
            df[~df[self.attribute].isin(self.threshold)],
        ]
        N = len(df.index)
        n = [len(df_.index) for df_ in self.split_df]

        loss_args = {
            key: self.algorithm_kwargs[key] for key in ["lambda_l1", "lambda_l2"]
        }
        loss_args = [loss_args.copy(), loss_args.copy()]
        if "__weights__" in df:
            for i, df_ in enumerate(self.split_df):
                loss_args[i]["weights"] = df_["__weights__"].values

        val = [
            self.metrics.node_value(df_[self.y_name], **loss_args[i])
            for i, df_ in enumerate(self.split_df)
        ]
        self.loss = n[0] / N * self.metrics.loss(
            self.split_df[0][self.y_name], val[0], **loss_args[0]
        ) + n[1] / N * self.metrics.loss(
            self.split_df[1][self.y_name], val[1], **loss_args[1]
        )

        return success

    @staticmethod
    def decide(x, threshold):
        return True if x in threshold else False

    @staticmethod
    def check(x):
        return math.check_dichotomous(x)


class BinnedAttributeHandler(AttributeHandlerBase):
    def __init__(self, y_name, attribute, metrics, algorithm_kwargs):
        super().__init__(y_name, attribute, metrics, algorithm_kwargs)

    def split(self, df):
        self.loss = np.inf
        self.split_df = []
        self.threshold = None

        success = False

        # Try optimized histogram-based split from metrics when available.
        # Filter NaNs first (can appear during preprocessing)
        clean_df = df.dropna(subset=[self.attribute])
        if len(clean_df) < 2:
            return success

        loss_args = {}
        if "__weights__" in df:
            loss_args["weights"] = clean_df["__weights__"].values

        try:
            best_loss, best_threshold = self.metrics.binned_loss(
                clean_df, self.y_name, self.attribute, **loss_args
            )
        except AttributeError:
            # Fallback to classic scan if metric doesn't implement binned_loss
            bin_edges = pd.Series(clean_df[self.attribute]).dropna().unique()
            if len(bin_edges) < 2:
                return success

            bin_edges = np.sort(bin_edges)
            loss = np.inf
            N = len(df.index)

            for i in range(1, len(bin_edges)):
                threshold = (bin_edges[i - 1] + bin_edges[i]) / 2.0
                split_df = [df[df[self.attribute] < threshold], df[df[self.attribute] >= threshold]]
                n = [len(df_.index) for df_ in split_df]

                if min(n) == 0:
                    continue

                loss_args_local = {key: self.algorithm_kwargs[key] for key in ["lambda_l1", "lambda_l2"] if key in self.algorithm_kwargs}
                loss_args_local = [loss_args_local.copy(), loss_args_local.copy()]
                if "__weights__" in df:
                    for j, df_ in enumerate(split_df):
                        loss_args_local[j]["weights"] = df_["__weights__"].values

                val = [
                    self.metrics.node_value(df_[self.y_name], **loss_args_local[j])
                    for j, df_ in enumerate(split_df)
                ]

                current_loss = (
                    n[0] / N * self.metrics.loss(split_df[0][self.y_name], val[0], **loss_args_local[0])
                    + n[1] / N * self.metrics.loss(split_df[1][self.y_name], val[1], **loss_args_local[1])
                )

                if current_loss < loss:
                    success = True
                    loss = current_loss
                    self.loss = loss
                    self.threshold = threshold
                    self.split_df = split_df

            return success

        # If we get a valid threshold from binned_loss, set split and return
        if best_threshold is not None and best_loss < np.Inf:
            success = True
            self.loss = best_loss
            self.threshold = best_threshold
            self.split_df = [
                df[df[self.attribute] < self.threshold],
                df[df[self.attribute] >= self.threshold],
            ]

        return success

    @staticmethod
    def decide(x, threshold):
        return True if x < threshold else False

    @staticmethod
    def check(x):
        try:
            if not pd.api.types.is_numeric_dtype(x):
                return False
        except Exception:
            return False

        # count uniques, handle Series vs array
        try:
            unique_count = x.nunique() if isinstance(x, pd.Series) else len(np.unique(x[~np.isnan(x)]))
        except Exception:
            unique_count = len(np.unique(x))

        is_integer_type = pd.api.types.is_integer_dtype(x)

        return is_integer_type or unique_count <= 256


class IntervalAttributeHandler(AttributeHandlerBase):
    def __init__(self, y_name, attribute, metrics, algorithm_kwargs):
        super().__init__(y_name, attribute, metrics, algorithm_kwargs)

    def split(self, df):
        self.loss = np.inf
        self.split_df = []
        self.threshold = None

        success = False

        if -df[self.attribute].min() + df[self.attribute].max() < np.finfo(float).tiny:
            return success

        # name = self.attribute

        # res = opt.minimize_scalar(
        #     self._opt_fun(df),
        #     bounds=(df[self.attribute].min(), df[self.attribute].max()),
        #    method="bounded",
        # )
        # self.threshold = res.x

        # mini = BrentsScalarMinimizer(rtol=0.5 / len(df.index))
        # x, y = mini.minimize(
        #     self._opt_fun(df), df[self.attribute].min(), df[self.attribute].max()
        # )
        method = self.algorithm_kwargs.get("minimizer_method", "brent")
        options = self.algorithm_kwargs
        options["minimizer_rtol"] = self.algorithm_kwargs.get(
            "minimizer_rtol", 0.5 / len(df.index)
        )
        x, y = minimize(
            self._opt_fun(df),
            df[self.attribute].min(),
            df[self.attribute].max(),
            method=method,
            options=options,
        )
        self.threshold = x

        self.split_df = [
            df[df[self.attribute] < self.threshold],
            df[df[self.attribute] >= self.threshold],
        ]
        # self.loss = res.fun
        # return res.success
        self.loss = y
        return True

    def _opt_fun(self, df):
        split_name = self.attribute
        N = len(df.index)

        def fun(x):
            split_df = [df[df[split_name] < x], df[df[split_name] >= x]]
            n = [len(df_.index) for df_ in split_df]

            loss_args = {
                key: self.algorithm_kwargs[key] for key in ["lambda_l1", "lambda_l2"]
            }
            loss_args = [loss_args.copy(), loss_args.copy()]
            if "__weights__" in df:
                for i, df_ in enumerate(split_df):
                    loss_args[i]["weights"] = df_["__weights__"].values

            val = [
                self.metrics.node_value(df_[self.y_name], **loss_args[i])
                for i, df_ in enumerate(split_df)
            ]
            return n[0] / N * self.metrics.loss(
                split_df[0][self.y_name], val[0], **loss_args[0]
            ) + n[1] / N * self.metrics.loss(
                split_df[1][self.y_name], val[1], **loss_args[1]
            )

        return fun

    @staticmethod
    def decide(x, threshold):
        return True if x < threshold else False

    @staticmethod
    def check(x):
        return math.check_interval(x)


class NullAttributeHandler(AttributeHandlerBase):
    def __init__(self, y_name, attribute, metrics, algorithm_kwargs):
        super().__init__(y_name, attribute, metrics, algorithm_kwargs)

    def split(self, df):
        self.loss = np.inf
        self.split_df = []
        self.threshold = None

        success = False

        return success

    @staticmethod
    def decide(x, threshold):
        return None

    @staticmethod
    def check(x):
        return True


# =========================


class ClusteringIntervalAttributeHandler(AttributeHandlerBase):
    def __init__(self, y_name, attribute, metrics, algorithm_kwargs):
        super().__init__(y_name, attribute, metrics, algorithm_kwargs)

    def split(self, df):
        self.loss = np.inf
        self.split_df = []
        self.threshold = None

        success = False

        if -df[self.attribute].min() + df[self.attribute].max() < np.finfo(float).tiny:
            return success

        name = self.attribute

        valleys = math.valley(df[name])
        if not valleys:
            return success

        loss = np.inf
        for v in valleys:
            threshold_candidate = v
            split_df_candidate = [
                df[df[self.attribute] < threshold_candidate],
                df[df[self.attribute] >= threshold_candidate],
            ]
            H = math.shannon_entropy_histogram(df[name], normalized=False)
            H_ = [
                math.shannon_entropy_histogram(df_[name], normalized=False)
                for df_ in split_df_candidate
            ]
            loss_candidate = (-np.sum(H_) + H) / np.abs(H)
            if loss_candidate < loss:
                loss = loss_candidate
                split_df = split_df_candidate
                threshold = threshold_candidate

        # loss = math.shannon_entropy_histogram(df[name], normalized=True)

        print(f"{self.attribute} loss: {loss}")

        tol = self.algorithm_kwargs.get(
            "unsupervised_minimum_relative_entropy_improvement"
        )

        if tol is not None and loss > tol:
            return success

        success = True

        self.threshold = threshold
        self.split_df = split_df
        self.loss = loss
        return success

    @staticmethod
    def decide(x, threshold):
        return True if x < threshold else False

    @staticmethod
    def check(x):
        return math.check_interval(x)


class ClusteringNominalAttributeHandler(AttributeHandlerBase):
    def __init__(self, y_name, attribute, metrics, algorithm_kwargs):
        super().__init__(y_name, attribute, metrics, algorithm_kwargs)

    def split(self, df):
        self.loss = np.inf
        self.split_df = []
        self.threshold = None

        success = False

        unique = np.unique(df[self.attribute])

        if len(unique) < 2:
            return success

        comb = []
        name = self.attribute

        if len(unique) > 5:
            comb = [(u,) for u in unique]
        else:
            for i in range(1, len(unique)):
                comb += list(itertools.combinations(unique, i))

        loss = np.inf

        for c in comb:
            threshold = c
            split_df = [
                df[df[name].isin(threshold)],
                df[~df[name].isin(threshold)],
            ]
            loss = math.shannon_entropy(df[c])
            if loss < self.loss:
                success = True
                self.loss = loss
                self.threshold = threshold
                self.split_df = split_df

        return success

    @staticmethod
    def decide(x, threshold):
        return True if x in threshold else False

    @staticmethod
    def check(x):
        return math.check_nominal(x, max_unique_fraction=0.2, exclude_dichotomous=False)


# =========================


class AttributeHandlerFactory:
    def __init__(self):
        self.attribute_handlers = {"default": []}

    def register_method_group(self, method_group):
        self.attribute_handlers[method_group] = []

    def register_handler(self, attribute_handler_class, method_group="default"):
        self.attribute_handlers[method_group].append(attribute_handler_class)

    def __getitem__(self, name):
        ahc = None
        for val in self.attribute_handlers.values():
            for a in val:
                if name == a.__name__:
                    ahc = a
        return ahc

    def get_attribute_handler_class(self, arr, method_group="default"):
        """
        This function returns the first match in the attribute_handlers list.
        Therefore, you have to be mindful of the order of the list.
        """
        for attribute_handler_class in self.attribute_handlers[method_group]:
            if attribute_handler_class.check(arr):
                return attribute_handler_class

        raise ValueError("no data handler class for this type of data")

    def create_attribute_handlers(
        self, training_data, metrics, method_group, algorithm_kwargs
    ):
        df = training_data.df
        y_name = training_data.y_name
        X_names = training_data.X_names

        if method_group not in self.attribute_handlers.keys():
            # raise ValueError(f"{method} is not a registered method_group")
            # print(
            #     f"WARNING: '{method_group}' is not a registered method group. Chosing 'default'."
            # )
            method_group = "default"

        ahc = self.get_attribute_handler_class(df[y_name], method_group=method_group)

        d = {y_name: ahc(y_name, y_name, metrics, algorithm_kwargs)}

        for name in X_names:
            ahc = self.get_attribute_handler_class(df[name], method_group=method_group)
            d[name] = ahc(y_name, name, metrics, algorithm_kwargs)

        return d


attribute_handler_factory = AttributeHandlerFactory()
attribute_handler_factory.register_handler(NominalAttributeHandler)
attribute_handler_factory.register_handler(HighCardinalityNominalAttributeHandler)
attribute_handler_factory.register_handler(DichotomousAttributeHandler)
attribute_handler_factory.register_handler(IntervalAttributeHandler)
attribute_handler_factory.register_handler(BinnedAttributeHandler)
attribute_handler_factory.register_handler(NullAttributeHandler)
attribute_handler_factory.register_method_group("clustering")
attribute_handler_factory.register_handler(
    ClusteringIntervalAttributeHandler, method_group="clustering"
)
attribute_handler_factory.register_handler(
    ClusteringNominalAttributeHandler, method_group="clustering"
)
attribute_handler_factory.register_handler(
    NullAttributeHandler, method_group="clustering"
)
