#!/usr/bin/env python
# coding: utf-8
import numpy as np


def k_fold_split(df, k=1, frac=None, random=False, shuffle=True, replace=True):
    if shuffle:
        df = df.sample(frac=1.0, replace=False)

    if frac is None:
        frac = 1.0 - 1.0 / (k + 1.0)

    N = len(df.index)
    n = int(np.ceil(N / k))
    sets = []
    for i in reversed(range(k)):
        if random:
            test = df.sample(frac=1.0 - frac, replace=replace)
        else:
            test = df.iloc[i * n : min(N, (i + 1) * n), :]
        training = df.loc[df.index.difference(test.index), :]
        sets.append((training, test))
    return sets
