# Initially Created with ChatGPT
from abc import ABC, abstractmethod
import numpy as np

class Metrics(ABC):
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

    def _classification_metrics(self, y_hat, df=None):
        confmat = self._confusion_matrix(y_hat, df)
        P = self._precision(confmat)
        #print(f"precision: {P}")
        R = self._recall(confmat)
        #print(f"recall: {R}")
        F = np.mean(self._F1(P,R))
        #print(f"F-score: {F}")
        A = self._accuracy(confmat)
        return {"precision":P,
                "recall":R,
                "F-score":F,
                "accuracy":A}

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
        pass

    def validate(self, y_hat, data):
        return self._regression_metrics(y_hat, data)

    def _regression_metrics(self, y_hat, df):
        R2 = self._r_squared(y_hat, df)
        return {"R_squared":R2}
    
    def _r_squared(self, y_hat, df):
        y = self._y()
        e = y - y_hat
        sse = e.T @ e
        sst = np.sum((y - np.nanmean(y))**2)
        return 1 - sse/sst

class LogisticMetrics(Metrics):
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

    def validate(self, y_hat, data):
        return self._classification_metrics(y_hat, data)

    def _confusion_matrix(self, y_hat, df):
        m = np.zeros((2,2),dtype=int)
        y_hat = np.round(y_hat).astype(int)
        for i, x in enumerate(df.iloc):
            y = int(x[self.y_name])
            y_hat_i = y_hat[i]
            m[y,y_hat_i] += 1
        return m
            

class ClassificationMetrics(Metrics):
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

    def validate(self, y_hat, data):
        return self._classification_metrics(y_hat, data)

    def _confusion_matrix(self, y_hat, df):
        unique = np.unique(self.df[self.y_name].values)
        classes = unique.tolist()#self.tree.classes()
        n_classes = len(classes)
        confmat = np.zeros((n_classes,n_classes))
        
        for i in range(len(df.index)):
            val_pred = y_hat[i]
            val_true = df[self.y_name].iloc[i]
            i_pred = classes.index(val_pred)
            i_true = classes.index(val_true)
            confmat[i_true,i_pred] += 1
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
