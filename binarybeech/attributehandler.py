#!/usr/bin/env python
# coding: utf-8

import itertools
from abc import ABC, abstractmethod

import numpy as np
#import pandas as pd
#import scipy.optimize as opt

import binarybeech.math as math
from binarybeech.brentsscalarminimizer import BrentsScalarMinimizer


class AttributeHandlerBase(ABC):
    def __init__(self, y_name, attribute, metrics, algorithm_kwargs):
        self.y_name = y_name
        self.attribute = attribute
        self.metrics = metrics
        self.algorithm_kwargs = algorithm_kwargs

        self.loss = None
        self.split_df = []
        self.threshold = None

    @abstractmethod
    def split(self, df):
        pass

    @staticmethod
    @abstractmethod
    def decide(x, threshold):
        pass

    @staticmethod
    @abstractmethod
    def check(x):
        pass


# =========================


class NominalAttributeHandler(AttributeHandlerBase):
    def __init__(self, y_name, attribute, metrics, algorithm_kwargs):
        super().__init__(y_name, attribute, metrics, algorithm_kwargs)

    def split(self, df):
        self.loss = np.Inf
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

        loss = np.Inf

        for c in comb:
            threshold = c
            split_df = [
                df[df[name].isin(threshold)],
                df[~df[name].isin(threshold)],
            ]
            N = len(df.index)
            n = [len(df_.index) for df_ in split_df]
            val = [self.metrics.node_value(df_[self.y_name]) for df_ in split_df]
            loss = n[0] / N * self.metrics.loss(split_df[0][self.y_name], val[0]) + n[
                1
            ] / N * self.metrics.loss(split_df[1][self.y_name], val[1])
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
        return math.check_nominal(x, max_unique_fraction=0.2, exclude_dichotomous=True)


class DichotomousAttributeHandler(AttributeHandlerBase):
    def __init__(self, y_name, attribute, metrics, algorithm_kwargs):
        super().__init__(y_name, attribute, metrics, algorithm_kwargs)

    def split(self, df):
        self.loss = np.Inf
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
        val = [self.metrics.node_value(df_[self.y_name]) for df_ in self.split_df]
        self.loss = n[0] / N * self.metrics.loss(
            self.split_df[0][self.y_name], val[0]
        ) + n[1] / N * self.metrics.loss(self.split_df[1][self.y_name], val[1])

        return success

    @staticmethod
    def decide(x, threshold):
        return True if x in threshold else False

    @staticmethod
    def check(x):
        return math.check_dichotomous(x)


class IntervalAttributeHandler(AttributeHandlerBase):
    def __init__(self, y_name, attribute, metrics, algorithm_kwargs):
        super().__init__(y_name, attribute, metrics, algorithm_kwargs)

    def split(self, df):
        self.loss = np.Inf
        self.split_df = []
        self.threshold = None

        success = False

        if -df[self.attribute].min() + df[self.attribute].max() < np.finfo(float).tiny:
            return success

        #name = self.attribute

        # res = opt.minimize_scalar(
        #     self._opt_fun(df),
        #     bounds=(df[self.attribute].min(), df[self.attribute].max()),
        #    method="bounded",
        # )
        # self.threshold = res.x
        
        mini =BrentsScalarMinimizer(rtol=0.5/len(df.index))
        x, y = mini.minimize(self._opt_fun(df),df[self.attribute].min(), df[self.attribute].max())
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
            val = [self.metrics.node_value(df_[self.y_name]) for df_ in split_df]
            return n[0] / N * self.metrics.loss(split_df[0][self.y_name], val[0]) + n[
                1
            ] / N * self.metrics.loss(split_df[1][self.y_name], val[1])

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
        self.loss = np.Inf
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


class UnsupervisedIntervalAttributeHandler(AttributeHandlerBase):
    def __init__(self, y_name, attribute, metrics, algorithm_kwargs):
        super().__init__(y_name, attribute, metrics, algorithm_kwargs)

    def split(self, df):
        self.loss = np.Inf
        self.split_df = []
        self.threshold = None

        success = False

        if -df[self.attribute].min() + df[self.attribute].max() < np.finfo(float).tiny:
            return success

        name = self.attribute

        valleys = math.valley(df[name])
        if not valleys:
            return success

        loss = np.Inf
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


class UnsupervisedNominalAttributeHandler(AttributeHandlerBase):
    def __init__(self, y_name, attribute, metrics, algorithm_kwargs):
        super().__init__(y_name, attribute, metrics, algorithm_kwargs)

    def split(self, df):
        self.loss = np.Inf
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

        loss = np.Inf

        for c in comb:
            threshold = c
            split_df = [
                df[df[name].isin(threshold)],
                df[~df[name].isin(threshold)],
            ]
            # N = len(df.index)
            # n = [len(df_.index) for df_ in split_df]
            # val = [self.metrics.node_value(None) for df_ in split_df]
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
        self.attribute_handlers = {"default": {}}

    def register_group(self, group_name):
        self.attribute_handlers[group_name] = {}

    def register_handler(self, data_level, attribute_handler_class, group_name="default"):
        self.attribute_handlers[group_name][data_level] = attribute_handler_class

    def get_attribute_handler_class(self, arr, group_name="default"):
        for attribute_handler_class in self.attribute_handlers[group_name].values():
            if attribute_handler_class.check(arr):
                return attribute_handler_class

        raise ValueError("no data handler class for this type of data")

    def create_attribute_handlers(self, training_data, metrics, algorithm_kwargs):
        df = training_data.df
        y_name = training_data.y_name
        X_names = training_data.X_names
        ahc = self.get_attribute_handler_class(
            df[y_name], group_name=metrics.attribute_handler_group()
        )

        d = {y_name: ahc(y_name, y_name, metrics, algorithm_kwargs)}

        for name in X_names:
            ahc = self.get_attribute_handler_class(
                df[name], group_name=metrics.attribute_handler_group()
            )
            d[name] = ahc(y_name, name, metrics, algorithm_kwargs)

        return d


attribute_handler_factory = AttributeHandlerFactory()
attribute_handler_factory.register_handler("nominal", NominalAttributeHandler)
attribute_handler_factory.register_handler("dichotomous", DichotomousAttributeHandler)
attribute_handler_factory.register_handler("interval", IntervalAttributeHandler)
attribute_handler_factory.register_handler("null", NullAttributeHandler)
attribute_handler_factory.register_group("unsupervised")
attribute_handler_factory.register_handler(
    "interval", UnsupervisedIntervalAttributeHandler, group_name="unsupervised"
)
attribute_handler_factory.register_handler(
    "nominal", UnsupervisedNominalAttributeHandler, group_name="unsupervised"
)
attribute_handler_factory.register_handler(
    "null", NullAttributeHandler, group_name="unsupervised"
)
