#!/usr/bin/env python
# coding: utf-8
import numpy as np

def k_fold_split(
    df, k=1, frac=None, random=False, shuffle=True, replace=True, seed=None
):
    if shuffle:
        df = df.sample(frac=1.0, replace=False, random_state=seed)

    if frac is None:
        frac = 1.0 - 1.0 / (k + 1.0)

    N = len(df.index)
    n = int(np.ceil(N / k))
    sets = []
    for i in reversed(range(k)):
        if random:
            if seed is not None:
                seed += 1
            test = df.sample(frac=1.0 - frac, replace=replace, random_state=seed)
        else:
            test = df.iloc[i * n : min(N, (i + 1) * n), :]
        training = df.loc[df.index.difference(test.index), :]
        sets.append((training, test))
    return sets
