#!/usr/bin/env python
# coding: utf-8

import itertools
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.optimize as opt

import binarybeech.math as math


class MissingsHandlerBase(ABC):
    def __init__(self, df):
        self.df = df

    @abstractmethod
    def handle_missings(self, df=None):
        pass

    @staticmethod
    @abstractmethod
    def check(x):
        pass

# =========================

class DropMissingsHandler(MissingsHandlerBase):
    
    def __init__(self,df):
        super().__init__(df)
        
    def handle_missings(self,df=None):
        if df is None:
            df = self.df
        df = df.dropna()
        return df
        
    @staticmethod
    def check(arr):
        pass

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


missings_handler_factory = MissingsHandlerFactory()
missings_handler_factory.register_handler("drop", DropMissingsHandler)
