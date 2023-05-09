#!/usr/bin/env python
# coding: utf-8

import itertools
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.optimize as opt

import binarybeech.math as math


class MissingsHandlerBase(ABC):
    def __init__(self, df, attribute):
        self.df = df
        self.attribute = attribute

    @abstractmethod
    def handle_missings(self, df=None):
        pass

    @staticmethod
    @abstractmethod
    def check(x):
        pass


# =========================


class DropMissingsHandler(MissingsHandlerBase):
    def __init__(self, df, attribute):
        super().__init__(df, attribute)

    def handle_missings(self, df=None):
        if df is None:
            df = self.df
        df = df.dropna(subset=[self.attribute])
        return df

    @staticmethod
    def check(arr):
        return True


class NaiveFillNominalMissingsHandler(MissingsHandlerBase):
    def __init__(self, df, attribute):
        super().__init__(df, attribute)

    def handle_missings(self, df=None):
        if df is None:
            df = self.df
        name = self.attribute
        df.loc[:, name] = df[name].fillna("missing")
        return df

    @staticmethod
    def check(x):
        return math.check_nominal(x, max_unique_fraction=0.2, exclude_dichotomous=True)


class HighestProbabilityNominalMissingsHandler(MissingsHandlerBase):
    def __init__(self, df, attribute):
        super().__init__(df, attribute)

    def handle_missings(self, df=None):
        if df is None:
            df = self.df
        name = self.attribute
        unique, counts = np.unique(df[name].dropna(), return_counts=True)
        ind_max = np.argmax(counts)
        val = unique[ind_max]
        df.loc[:, name] = df[name].fillna(val)
        return df

    @staticmethod
    def check(x):
        return math.check_nominal(x, max_unique_fraction=0.2, exclude_dichotomous=True)


class HighestProbabilityDichotomousMissingsHandler(MissingsHandlerBase):
    def __init__(self, df, attribute):
        super().__init__(df, attribute)

    def handle_missings(self, df=None):
        if df is None:
            df = self.df
        name = self.attribute
        unique, counts = np.unique(df[name].dropna(), return_counts=True)
        ind_max = np.argmax(counts)
        val = unique[ind_max]
        df.loc[:, name] = df[name].fillna(val)
        return df

    @staticmethod
    def check(x):
        return math.check_dichotomous(x)


class MedianIntervalMissingsHandler(MissingsHandlerBase):
    def __init__(self, df, attribute):
        super().__init__(df, attribute)

    def handle_missings(self, df=None):
        name = self.attribute
        df.loc[:, name] = df[name].fillna(np.nanmedian(df[name].values))
        return df

    @staticmethod
    def check(x):
        return math.check_interval(x)


class NullMissingsHandler(MissingsHandlerBase):
    def __init__(self, df, attribute):
        super().__init__(df, attribute)

    def handle_missings(self, df=None):
        return df

    @staticmethod
    def check(x):
        return np.issubdtype(x.dtype, np.number) and np.sum(0)


# =========================


class MissingsHandlerFactory:
    def __init__(self):
        self.missings_handlers = {"default": {}}

    def register_group(self, group_name):
        self.missings_handlers[group_name] = {}

    def register_handler(
        self, data_level, missings_handler_class, group_names=["default"]
    ):
        for name in group_names:
            self.missings_handlers[name][data_level] = missings_handler_class

    def get_missings_handler_class(self, arr, group_name="default"):
        for missings_handler_class in self.missings_handlers[group_name].values():
            if missings_handler_class.check(arr):
                return missings_handler_class

        raise ValueError("no missings handler class for this type of data")

    def create_missings_handlers(self, df, y_name, X_names, handle_missings):
        mhc = self.get_missings_handler_class(df[y_name], group_name=handle_missings)

        d = {y_name: mhc(df, y_name)}

        for name in X_names:
            ahc = self.get_missings_handler_class(df[name], group_name=handle_missings)
            d[name] = ahc(df, name)

        return d


missings_handler_factory = MissingsHandlerFactory()
missings_handler_factory.register_handler("drop", DropMissingsHandler)
missings_handler_factory.register_group("simple")
missings_handler_factory.register_handler(
    "highestProbabilityDichotomous", HighestProbabilityDichotomousMissingsHandler, group_names=["simple"]
)
missings_handler_factory.register_handler(
    "naiveFillNominal", NaiveFillNominalMissingsHandler, group_names=["simple"]
)
missings_handler_factory.register_handler(
    "medianInterval", MedianIntervalMissingsHandler, group_names=["simple"]
)
missings_handler_factory.register_handler(
    "null", NullMissingsHandler, group_names=["simple"]
)
