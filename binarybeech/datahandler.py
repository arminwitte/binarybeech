#!/usr/bin/env python
# coding: utf-8

import itertools
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.optimize as opt


class DataHandlerBase(ABC):
    def __init__(self, y_name, attribute, metrics):
        self.y_name = y_name
        self.attribute = attribute
        self.metrics = metrics

        self.loss = None
        self.split_df = []
        self.threshold = None

    @abstractmethod
    def split(self, df):
        pass

    @abstractmethod
    def handle_missings(self, df):
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


class NominalDataHandler(DataHandlerBase):
    def __init__(self, y_name, attribute, metrics):
        super().__init__(y_name, attribute, metrics)

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

    def handle_missings(self, df):
        name = self.attribute
        df.loc[:, name] = df[name].fillna("missing")
        return df

    @staticmethod
    def decide(x, threshold):
        return True if x in threshold else False

    @staticmethod
    def check(x):
        x = x[~pd.isna(x)]
        unique = np.unique(x)
        l = len(unique)

        if l / len(x) > 0.2:
            return False

        dtype = x.values.dtype

        if not np.issubdtype(dtype, np.number) and l > 2:
            return True

        return False


class DichotomousDataHandler(DataHandlerBase):
    def __init__(self, y_name, attribute, metrics):
        super().__init__(y_name, attribute, metrics)

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

    def handle_missings(self, df):
        name = self.attribute
        unique, counts = np.unique(df[name].dropna(), return_counts=True)
        ind_max = np.argmax(counts)
        val = unique[ind_max]
        df.loc[:, name] = df[name].fillna(val)
        return df

    @staticmethod
    def decide(x, threshold):
        return True if x in threshold else False

    @staticmethod
    def check(x):
        x = x[~pd.isna(x)]
        unique = np.unique(x)
        l = len(unique)

        if l / len(x) > 0.2:
            return False

        dtype = x.values.dtype

        if l == 2:
            return True

        return False


class IntervalDataHandler(DataHandlerBase):
    def __init__(self, y_name, attribute, metrics):
        super().__init__(y_name, attribute, metrics)

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
        self.split_df = [
            df[df[self.attribute] < self.threshold],
            df[df[self.attribute] >= self.threshold],
        ]
        self.loss = res.fun
        return res.success

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

    def handle_missings(self, df):
        name = self.attribute
        df.loc[:, name] = df[name].fillna(np.nanmedian(df[name].values))
        return df

    @staticmethod
    def decide(x, threshold):
        return True if x < threshold else False

    @staticmethod
    def check(x):
        x = x[~pd.isna(x)]
        unique = np.unique(x)
        l = len(unique)
        dtype = x.values.dtype

        if np.issubdtype(dtype, np.number) and l > 2:
            return True

        return False


class NullDataHandler(DataHandlerBase):
    def __init__(self, y_name, attribute, metrics):
        super().__init__(y_name, attribute, metrics)

    def split(self, df):
        self.loss = np.Inf
        self.split_df = []
        self.threshold = None

        success = False

        return success

    def handle_missings(self, df):
        return df

    @staticmethod
    def decide(x, threshold):
        return None

    @staticmethod
    def check(x):
        return True


# =========================

class IntervalUnsupervisedDataHandler(DataHandlerBase):
    def __init__(self, y_name, attribute, metrics):
        super().__init__(y_name, attribute, metrics)

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
        self.split_df = [
            df[df[self.attribute] < self.threshold],
            df[df[self.attribute] >= self.threshold],
        ]
        self.loss = res.fun
        return res.success

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

    def handle_missings(self, df):
        name = self.attribute
        df.loc[:, name] = df[name].fillna(np.nanmedian(df[name].values))
        return df

    @staticmethod
    def decide(x, threshold):
        return True if x < threshold else False

    @staticmethod
    def check(x):
        
        if y_name:
            return False
            
        x = x[~pd.isna(x)]
        unique = np.unique(x)
        l = len(unique)
        dtype = x.values.dtype

        if np.issubdtype(dtype, np.number) and l > 2:
            return True

        return False



# =========================


class DataHandlerFactory:
    def __init__(self):
        self.data_handlers = {}

    def register(self, data_level, data_handler_class):
        self.data_handlers[data_level] = data_handler_class

    def get_data_handler_class(self, df):
        for data_handler_class in self.data_handlers.values():
            if data_handler_class.check(df):
                return data_handler_class

        raise ValueError("no data handler class for this type of data")

    def create_data_handlers(self, df, y_name, X_names, metrics):
        dhc = self.get_data_handler_class(df[y_name])

        d = {y_name: dhc(y_name, y_name, metrics)}

        for name in X_names:
            dhc = self.get_data_handler_class(df[name])
            d[name] = dhc(y_name, name, metrics)

        return d


data_handler_factory = DataHandlerFactory()
data_handler_factory.register("nominal", NominalDataHandler)
data_handler_factory.register("dichotomous", DichotomousDataHandler)
data_handler_factory.register("interval", IntervalDataHandler)
data_handler_factory.register("null", NullDataHandler)
