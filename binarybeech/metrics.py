# Initially Created with ChatGPT
from abc import ABC, abstractmethod

import numpy as np

import binarybeech.utils as utils


class Metrics(ABC):
    def __init__(self, y_name):
        self.y_name = y_name

    def _y(self, df):
        return df[self.y_name].values

    def _gini_impurity(self, df):
        y = self._y(df)
        return utils.gini_impurity(y)

    def _shannon_entropy(self, df):
        y = self._y(df)
        return utils.shannon_entropy(y)

    def _misclassification_cost(self, df):
        y = self._y(df)
        return utils.misclassification_cost(y)

    def _logistic_loss(self, df):
        y = self._y(df)
        p_ = self.node_value(df)
        p = np.ones_like(y) * p_
        return utils.logistic_loss(y, p)

    def _mean_squared_error(self, df):
        y = self._y(df)
        y_hat = self.node_value(df)
        return utils.mean_squared_error(y, y_hat)

    def _mean(self, df):
        y = self._y(df)
        return np.nanmean(y)

    def _majority_class(self, df):
        y = self._y(df)
        return utils.majority_class(y)

    def _odds(self, df):
        y = self._y(df)
        return utils.odds(y)

    def _log_odds(self, df):
        odds = self._odds(df)
        odds = np.clip(odds, 1e-12, 1e12)
        logodds = np.log(odds)
        # print(f"logodds: {logodds}")
        return logodds

    def _probability(self, df):
        odds = self._odds(df)
        if odds == np.Inf:
            return 1.0
        p = odds / (1 + odds)
        # print(f"odds: {odds:.2f} probability: {p:.4f}")
        return p

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

    @abstractmethod
    def loss(self, data):
        pass

    @abstractmethod
    def loss_prune(self, data):
        pass

    @abstractmethod
    def node_value(self, data):
        pass

    @abstractmethod
    def validate(self, y_hat, data):
        pass


class RegressionMetrics(Metrics):
    def __init__(self, y_name):
        super().__init__(y_name)

    def loss(self, data):
        # Implementation of the loss calculation for regression
        return self._mean_squared_error(data)

    def loss_prune(self, data):
        # Implementation of the loss pruning calculation for regression
        return self.loss(data)

    def node_value(self, data):
        # Implementation of the node value calculation for regression
        return self._mean(data)
        pass

    def validate(self, y_hat, data):
        return self._regression_metrics(y_hat, data)

    def _regression_metrics(self, y_hat, df):
        R2 = self._r_squared(y_hat, df)
        return {"R_squared": R2}

    def _r_squared(self, y_hat, df):
        y = self._y(df)
        e = y - y_hat
        sse = e.T @ e
        sst = np.sum((y - np.nanmean(y)) ** 2)
        return 1 - sse / sst


class LogisticMetrics(Metrics):
    def __init__(self, y_name):
        super().__init__(y_name)

    def loss(self, data):
        # Implementation of the loss calculation for logistic
        return self._logistic_loss(data)

    def loss_prune(self, data):
        # Implementation of the loss pruning calculation for logistic
        return self._misclassification_cost(data)

    def node_value(self, data):
        # Implementation of the node value calculation for logistic
        return self._probability(data)

    def validate(self, y_hat, data):
        return self._classification_metrics(y_hat, data)

    def _confusion_matrix(self, y_hat, df):
        m = np.zeros((2, 2), dtype=int)
        y_hat = np.round(np.clip(y_hat, 0.0, 1.0)).astype(int)
        for i, x in enumerate(df.iloc):
            y = int(x[self.y_name])
            y_hat_i = y_hat[i]
            m[y, y_hat_i] += 1
        return m


class ClassificationMetrics(Metrics):
    def __init__(self, y_name):
        super().__init__(y_name)

    def loss(self, data):
        # Implementation of the loss calculation for classification
        return self._gini_impurity(data)

    def loss_prune(self, data):
        # Implementation of the loss pruning calculation for classification
        return self._misclassification_cost(data)

    def node_value(self, data):
        # Implementation of the node value calculation for classification
        return self._majority_class(data)

    def validate(self, y_hat, data):
        return self._classification_metrics(y_hat, data)

    def _confusion_matrix(self, y_hat, df):
        y = self._y(df)
        unique = np.unique(y)
        classes = unique.tolist()  # self.tree.classes()
        n_classes = len(classes)
        confmat = np.zeros((n_classes, n_classes))

        for i in range(len(df.index)):
            val_pred = y_hat[i]
            val_true = df[self.y_name].iloc[i]
            i_pred = classes.index(val_pred)
            i_true = classes.index(val_true)
            confmat[i_true, i_pred] += 1
        return confmat


class MetricFactory:
    def __init__(self):
        self.metrics = {}

    def register(self, metrics_type, metrics_class):
        self.metrics[metrics_type] = metrics_class

    def create_metrics(self, metrics_type, y_name):
        if metrics_type in self.metrics:
            return self.metrics[metrics_type](y_name)
        else:
            raise ValueError("Invalid metrics type")


metrics_factory = MetricFactory()
metrics_factory.register("regression", RegressionMetrics)
metrics_factory.register("logistic", LogisticMetrics)
metrics_factory.register("classification", ClassificationMetrics)
