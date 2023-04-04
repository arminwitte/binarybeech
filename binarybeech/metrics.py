#!/usr/bin/env python
# coding: utf-8
# Initially Created with ChatGPT
import uuid
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# import binarybeech.utils as utils
import binarybeech.math as math


class Metrics(ABC):
    def __init__(self):
        pass

    def _classification_metrics(self, y_hat, df=None):
        confmat = self._confusion_matrix(y_hat, df)
        P = self._precision(confmat)
        # print(f"precision: {P}")
        R = self._recall(confmat)
        # print(f"recall: {R}")
        F = np.mean(self._F1(P, R))
        # print(f"F-score: {F}")
        A = self._accuracy(confmat)
        return {"precision": P, "recall": R, "F-score": F, "accuracy": A}

    @staticmethod
    def _precision(m):
        return np.diag(m) / np.sum(m, axis=1)

    @staticmethod
    def _recall(m):
        return np.diag(m) / np.sum(m, axis=0)

    @staticmethod
    def _F1(P, R):
        # F = np.zeros_like(P)
        # for i in range(len(
        return 2 * P * R / (P + R)

    @staticmethod
    def _accuracy(m):
        return np.sum(np.diag(m)) / np.sum(np.sum(m))

    @staticmethod
    def output_transform(arr):
        return arr

    @staticmethod
    def inverse_transform(arr):
        return arr

    @staticmethod
    def data_handler_group():
        return "default"

    @abstractmethod
    def loss(self, y, y_hat):
        pass

    @abstractmethod
    def loss_prune(self, y, y_hat):
        pass

    @abstractmethod
    def node_value(self, y):
        pass

    @abstractmethod
    def validate(self, y_hat, data):
        pass

    @abstractmethod
    def goodness_of_fit(self, y_hat, data):
        pass

    @staticmethod
    @abstractmethod
    def check_data_type(arr):
        pass


class RegressionMetrics(Metrics):
    def __init__(self):
        pass

    def loss(self, y, y_hat):
        # Implementation of the loss calculation for regression
        return math.mean_squared_error(y, y_hat)

    def loss_prune(self, y, y_hat):
        # Implementation of the loss pruning calculation for regression
        return self.loss(y, y_hat)

    def node_value(self, y):
        # Implementation of the node value calculation for regression
        return np.nanmean(y)

    def validate(self, y, y_hat):
        return self._regression_metrics(y, y_hat)

    def _regression_metrics(self, y, y_hat):
        R2 = math.r_squared(y, y_hat)
        return {"R_squared": R2}

    def goodness_of_fit(self, y, y_hat):
        R2 = math.r_squared(y, y_hat)
        return R2

    @staticmethod
    def check_data_type(arr):
        x = arr[~pd.isna(arr)]
        unique = np.unique(x)
        l = len(unique)
        r = l / x.size
        dtype = x.values.dtype

        if np.issubdtype(dtype, np.number) and l > 2 and r > 0.01:
            return True

        return False


class LogisticMetrics(Metrics):
    def __init__(self):
        pass

    def loss(self, y, y_hat):
        # Implementation of the loss calculation for logistic
        return math.logistic_loss(y, y_hat)

    def loss_prune(self, y, y_hat):
        # Implementation of the loss pruning calculation for logistic
        return math.misclassification_cost(y)

    def node_value(self, y):
        # Implementation of the node value calculation for logistic
        return math.max_probability(y)

    def validate(self, y, y_hat):
        return self._classification_metrics(y, y_hat)

    def _confusion_matrix(self, y, y_hat):
        m = np.zeros((2, 2), dtype=int)
        y_hat = np.round(np.clip(y_hat, 0.0, 1.0)).astype(int)
        for i, y_ in enumerate(y):
            y_ = int(y_)
            y_hat_i = y_hat[i]
            m[y_, y_hat_i] += 1
        return m

    def goodness_of_fit(self, y, y_hat):
        confmat = self._confusion_matrix(y, y_hat)
        A = self._accuracy(confmat)
        return A

    @staticmethod
    def output_transform(arr):
        return math.logistic(arr)

    @staticmethod
    def inverse_transform(arr):
        return math.logit(arr)

    @staticmethod
    def check_data_type(arr):
        x = arr[~pd.isna(arr)]
        unique = np.unique(x)
        l = len(unique)
        r = l / x.size
        dtype = x.values.dtype

        if (
            np.issubdtype(dtype, np.number)
            and l == 2
            and np.min(x) == 0
            and np.max(x) == 1
        ):
            return True

        return False


class ClassificationMetrics(Metrics):
    def __init__(self):
        pass

    def loss(self, y, y_hat):
        # Implementation of the loss calculation for classification
        return math.gini_impurity(y)

    def loss_prune(self, y, y_hat):
        # Implementation of the loss pruning calculation for classification
        return math.misclassification_cost(y)

    def node_value(self, y):
        # Implementation of the node value calculation for classification
        return math.majority_class(y)

    def validate(self, y, y_hat):
        return self._classification_metrics(y, y_hat)

    def _confusion_matrix(self, y, y_hat):
        unique = np.unique(y)
        classes = unique.tolist()  # self.tree.classes()
        n_classes = len(classes)
        confmat = np.zeros((n_classes, n_classes))

        for i, y_ in enumerate(y):
            val_pred = y_hat[i]
            val_true = y_
            i_pred = classes.index(val_pred)
            i_true = classes.index(val_true)
            confmat[i_true, i_pred] += 1
        return confmat

    def goodness_of_fit(self, y, y_hat):
        confmat = self._confusion_matrix(y, y_hat)
        A = self._accuracy(confmat)
        return A

    @staticmethod
    def check_data_type(arr):
        x = arr[~pd.isna(arr)]
        unique = np.unique(x)
        l = len(unique)
        # r = l/x.size
        dtype = x.values.dtype

        if not np.issubdtype(dtype, np.number) and l > 1:
            return True

        return False


class ClassificationMetricsEntropy(ClassificationMetrics):
    def __init__(self):
        pass

    def loss(self, y, y_hat):
        # Implementation of the loss calculation for classification
        return math.shannon_entropy(y)


# =============================


class UnsupervisedMetrics(Metrics):
    def __init__(self):
        pass

    def loss(self, y, y_hat):
        return np.inf

    def loss_prune(self, y, y_hat):
        return self.loss(y, y_hat)

    def node_value(self, y):
        return f"cluster {str(uuid.uuid4())}"

    def validate(self, y, y_hat):
        return {}

    def goodness_of_fit(self, y, y_hat):
        return 0.0

    @staticmethod
    def data_handler_group():
        return "unsupervised"

    @staticmethod
    def check_data_type(arr):
        if not arr:
            return True

        return False


# =============================


class MetricsFactory:
    def __init__(self):
        self.metrics = {}

    def register(self, metrics_type, metrics_class):
        self.metrics[metrics_type] = metrics_class

    def create_metrics(self, metrics_type):
        if metrics_type in self.metrics:
            return self.metrics[metrics_type]()
        else:
            raise ValueError("Invalid metrics type")

    def from_data(self, y):
        for name, cls in self.metrics.items():
            if cls.check_data_type(y):
                return cls(), name


metrics_factory = MetricsFactory()
metrics_factory.register("regression", RegressionMetrics)
metrics_factory.register("classification:gini", ClassificationMetrics)
metrics_factory.register("classification:entropy", ClassificationMetricsEntropy)
metrics_factory.register("logistic", LogisticMetrics)
metrics_factory.register("classification", ClassificationMetrics)
metrics_factory.register("clustering", UnsupervisedMetrics)
