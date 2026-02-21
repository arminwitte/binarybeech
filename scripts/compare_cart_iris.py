#!/usr/bin/env python3
"""Compare CART on a dataset with scalar vs binned attribute handlers.

This script runs CART (from the package) on a dataset twice:
- once with the original (scalar/interval) numeric columns
- once with numeric columns discretized to integer bin codes so the
    `BinnedAttributeHandler` is selected.

By default it now targets the forest covertype dataset stored in
`data/covtype_categorical.parquet` (no one-hot encoding).
It reports accuracy and fit/predict timings (mean over repeats).
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from binarybeech.binarybeech import CART


def stratified_train_test_split(df: pd.DataFrame, y_col: str, test_size: float = 0.3, seed: int = 42):
    rng = np.random.RandomState(seed)
    groups = df.groupby(y_col)
    train_parts = []
    test_parts = []
    for _, g in groups:
        g_shuffled = g.sample(frac=1.0, random_state=rng)
        n_test = int(len(g_shuffled) * test_size)
        test_parts.append(g_shuffled.iloc[:n_test])
        train_parts.append(g_shuffled.iloc[n_test:])
    train = pd.concat(train_parts).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    return train, test


def bin_df_to_int(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    df_b = df.copy()
    for col in df_b.columns:
        if pd.api.types.is_numeric_dtype(df_b[col]) and col != "Cover_Type":
            # equal-width bins, label with integer codes
            df_b[col] = pd.cut(df_b[col], bins=n_bins, labels=False, duplicates="drop").astype("Int64").astype(int)
    return df_b


def accuracy(y_true: pd.Series, y_pred: List) -> float:
    return (y_true.values == np.array(y_pred)).mean()


def run_cart(
    train_df: pd.DataFrame, test_df: pd.DataFrame, method: str = "classification", max_depth: int | None = None
) -> Tuple[float, float, float]:
    # measure training (tree creation) and prediction
    kwargs = {}
    if max_depth is not None:
        kwargs["max_depth"] = max_depth
    c = CART(df=train_df, y_name="Cover_Type", method=method, **kwargs)
    t0 = time.perf_counter()
    c.create_tree()
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = c.predict(test_df)
    t_pred = time.perf_counter() - t0

    acc = accuracy(test_df["Cover_Type"], y_pred)
    return acc, t_fit, t_pred


def main(repeats: int = 5, n_bins: int = 5, sample_size: int | None = None, max_depth: int | None = None):
    df = pd.read_parquet("data/covtype_categorical.parquet")
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    results = {"scalar": [], "binned": []}

    for i in range(repeats):
        train, test = stratified_train_test_split(df, "Cover_Type", test_size=0.3, seed=42 + i)

        # scalar (original floats -> IntervalAttributeHandler)
        r = run_cart(train, test, method="classification", max_depth=max_depth)
        results["scalar"].append(r)

        # binned (discretize numeric columns to integer codes -> BinnedAttributeHandler)
        train_b = bin_df_to_int(train, n_bins=n_bins)
        test_b = bin_df_to_int(test, n_bins=n_bins)
        r = run_cart(train_b, test_b, method="classification", max_depth=max_depth)
        results["binned"].append(r)

    def summarize(arr: List[Tuple[float, float, float]]):
        a = np.array(arr)
        return a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()

    s_acc, s_fit, s_pred = summarize(results["scalar"])
    b_acc, b_fit, b_pred = summarize(results["binned"])

    print("CART on Covtype comparison (mean over {} runs):".format(repeats))
    print("Scalar CART:    accuracy={:.4f}, fit={:.6f}s, predict={:.6f}s".format(s_acc, s_fit, s_pred))
    print("Binned CART:    accuracy={:.4f}, fit={:.6f}s, predict={:.6f}s".format(b_acc, b_fit, b_pred))


if __name__ == "__main__":
    # default: sample to keep runtime reasonable for quick tests
    # set max_depth to limit tree growth and speed up runs
    main(repeats=1, sample_size=20000, max_depth=6)
