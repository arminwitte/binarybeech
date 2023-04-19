#!/usr/bin/env python
# coding: utf-8

import itertools
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.optimize as opt

import binarybeech.math as math


class MissingsHandlerBase(ABC):
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



# =========================

class MissingsHandlerFactory:
    def __init__(self):
        self.missings_handlers = {"default": {}}

    def register_group(self, group_name):
        self.missings_handlers[group_name] = {}

    def register_handler(self, data_level, missings_handler_class, group_names=["default"]):
        for name in group_names:
        self.missings_handlers[name][data_level] = missings_handler_class

    def get_missings_handler_class(self, arr, group_name="default"):
        for missings_handler_class in self.missings_handlers[group_name].values():
            if missings_handler_class.check(arr):
                return missings_handler_class

        raise ValueError("no missings handler class for this type of data")

missings_handler_factory = MissingsHandlerFactory()
missings_handler_factory.register("nominal", NominalAttributeHandler)
missings_handler_factory.register("dichotomous", DichotomousAttributeHandler)
missings_handler_factory.register("interval", IntervalAttributeHandler)
missings_handler_factory.register("null", NullAttributeHandler)
missings_handler_factory.register_group("unsupervised")
missings_handler_factory.register(
    "interval", UnsupervisedIntervalAttributeHandler, group_name="unsupervised"
)
missings_handler_factory.register(
    "nominal", UnsupervisedNominalAttributeHandler, group_name="unsupervised"
)
missings_handler_factory.register("null", NullAttributeHandler, group_name="unsupervised")
