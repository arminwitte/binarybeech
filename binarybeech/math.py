#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import scipy.signal


def gini_impurity(x):
    unique, counts = np.unique(x, return_counts=True)
    N = x.size
    p = counts / N
    return 1.0 - np.sum(p**2)


def shannon_entropy(x):
    unique, counts = np.unique(x, return_counts=True)
    N = x.size
    p = counts / N
    return -np.sum(p * np.log2(p))


def misclassification_cost(x):
    unique, counts = np.unique(x, return_counts=True)
    N = x.size
    p = np.max(counts) / N
    return 1.0 - p


def logistic_loss(y, p):
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def mean_squared_error(y, y_hat):
    e = y - y_hat
    return 1 / e.size * (e.T @ e)


def r_squared(y, y_hat):
    e = y - y_hat
    sse = e.T @ e
    sst = np.sum((y - np.nanmean(y)) ** 2)
    return 1 - sse / sst


def majority_class(x):
    unique, counts = np.unique(x, return_counts=True)
    ind_max = np.argmax(counts)
    return unique[ind_max]


def odds(x):
    unique, counts = np.unique(x, return_counts=True)
    d = {0: 0, 1: 0}
    for i, u in enumerate(unique):
        d[u] = counts[i]
    if d[0] == 0:
        return np.Inf
    odds = d[1] / d[0]
    return odds


def log_odds(x):
    o = odds(x)
    o = np.clip(o, 1e-12, 1e12)
    logodds = np.log(o)
    return logodds


def probability(x):
    # if x == np.Inf:
    #    return 1.0
    return x / (1 + x)


def max_probability(x):
    unique, counts = np.unique(x, return_counts=True)
    return np.max(counts) / x.size


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(x):
    return np.log(x / (1.0 - x))


def precision(m):
    return np.diag(m) / np.sum(m, axis=1)


def recall(m):
    return np.diag(m) / np.sum(m, axis=0)


def F1(P, R):
    return 2 * P * R / (P + R)


def accuracy(m):
    return np.sum(np.diag(m)) / np.sum(np.sum(m))


def distance_matrix(X):
    n = X.shape[0]
    D = np.empty((n, n))
    for i in range(n):
        D[i, :] = np.linalg.norm(X - X[i, :], axis=1)
    return D


def proximity_matrix(D):
    d_max = np.max(np.max(D))
    return 1.0 - D / d_max


def ambiguity(X):
    D = distance_matrix(X)
    mu = proximity_matrix(D)
    return -np.sum(mu * (1 - mu))


def valley(x):
    hist, bin_edges = np.histogram(x, bins="auto")
    valley_ind, _ = scipy.signal.find_peaks(-hist)
    v = [(bin_edges[i] + bin_edges[i + 1]) * 0.5 for i in valley_ind]
    return v


def shannon_entropy_histogram(x, normalized=False):
    hist, bin_edges = np.histogram(x, bins="auto")
    hist = np.maximum(hist, 1e-12)
    s = -np.sum(hist * np.log2(hist))
    
    if normalized:
        n_bins = bin_edges.size - 1
        n_samples = x.size
        s_ref = n_samples * np.log2(n_samples/n_bins)
        s /= s_ref

    return s


# =====================================


def check_nominal(x, max_unique_fraction=0.2, exclude_dichotomous=True):
    x = x[~pd.isna(x)]
    unique = np.unique(x)
    l = len(unique)

    if exclude_dichotomous and l <= 2:
        return False

    if l / len(x) > max_unique_fraction:
        return False

    dtype = x.values.dtype
    if not np.issubdtype(dtype, np.number):
        return True

    return False


def check_dichotomous(x):
    x = x[~pd.isna(x)]
    unique = np.unique(x)
    l = len(unique)

    if l == 2:
        return True

    return False


def check_interval(x):
    x = x[~pd.isna(x)]
    unique = np.unique(x)
    l = len(unique)

    if l <= 2:
        return False

    dtype = x.values.dtype

    if np.issubdtype(dtype, np.number):
        return True

    return False
