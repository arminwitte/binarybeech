# Initially Created with ChatGPT
from abc import ABC, abstractmethod
import numpy as np

class Metric(ABC):
    def __init__(self,y_name):
        self.y_name = y_name

    def _y(self,df):
        return df[self.y_name].values

    def _gini_impurity(self, df):
        y = self._y(df)
        unique, counts = np.unique(y, return_counts=True)
        N = y.size
        p = counts/N
        #print(unique)
        #print(p)
        return 1. - np.sum(p**2)
    
    def _shannon_entropy(self,df):
        y = self._y(df)
        unique, counts = np.unique(y, return_counts=True)
        N = y.size
        p = counts/N
        return -np.sum(p * np.log2(p))

    def _misclassification_cost(self,df):
        y = self._y(df)
        unique, counts = np.unique(y, return_counts=True)
        N = y.size
        p = np.max(counts)/N
        return 1. - p
    
    def _logistic_loss(self,df):
        y = self._y(df)
        p_ = self.node_value(df) #np.nanmax(counts)/np.sum(counts)
        p_ = np.clip(p_,1e-12,1.-1e-12)
        p = np.ones_like(y) * p_
        l = np.sum(-y*np.log(p)-(1-y)*np.log(1-p))
        return l
    
    def _mean_squared_error(self,df):
        y = self._y(df)
        y_hat = self.node_value(df)
        e = y - y_hat
        return 1/e.size * (e.T @ e)
    
    def _mean(self,df):
        y = self._y(df)
        return np.nanmean(y)
    
    def _majority_class(self,df):
        y = self._y(df)
        unique, counts = np.unique(y,return_counts=True)
        ind_max = np.argmax(counts)
        return unique[ind_max]
    
    def _odds(self,df):
        y = self._y(df)
        unique, counts = np.unique(y, return_counts=True)
        d={0:0,1:0}
        for i, u in enumerate(unique):
            d[u] = counts[i]
        if d[0] == 0:
            return np.Inf
        odds = d[1]/d[0]
        #print(f"odds: {odds}")
        return odds
    
    def _log_odds(self,df):
        odds = self._odds(df)
        odds = np.clip(odds,1e-12,1e12)
        logodds = np.log(odds)
        #print(f"logodds: {logodds}")
        return logodds
    
    def _probability(self,df):
        odds = self._odds(df)
        if odds == np.Inf:
            return 1.
        p = odds/(1+odds)
        #print(f"odds: {odds:.2f} probability: {p:.4f}")
        return p

    @abstractmethod
    def loss(self, data):
        pass

    @abstractmethod
    def loss_prune(self, data):
        pass

    @abstractmethod
    def node_value(self, data):
        pass

class RegressionMetrics(Metric):
    def __init__(self,y_name):
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

class LogisticMetrics(Metric):
    def __init__(self,y_name):
        super().__init__(y_name)

    def loss(self, data):
        # Implementation of the loss calculation for logistic
        pass

    def loss_prune(self, data):
        # Implementation of the loss pruning calculation for logistic
        pass

    def node_value(self, data):
        # Implementation of the node value calculation for logistic
        pass

class ClassificationMetrics(Metric):
    def __init__(self,y_name):
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

class MetricFactory:
    def __init__(self):
        self.metrics = {}

    def register(self, metric_type, metric_class):
        self.metrics[metric_type] = metric_class

    def create_metric(self, metric_type):
        if metric_type in self.metrics:
            return self.metrics[metric_type]()
        else:
            raise ValueError("Invalid metric type")

metrics_factory = MetricFactory()
metrics_factory.register("regression", RegressionMetrics)
metrics_factory.register("logistic", LogisticMetrics)
metrics_factory.register("classification", ClassificationMetrics)
