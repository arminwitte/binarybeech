#!/usr/bin/env python3
"""Compare GradientBoostedTree on a dataset with scalar vs binned attribute handlers.

This script mirrors `compare_cart_iris.py` but uses `GradientBoostedTree` from
`binarybeech` instead of `CART`.
"""
from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
import pandas as pd

from binarybeech.binarybeech import GradientBoostedTree, CART


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
            df_b[col] = pd.cut(df_b[col], bins=n_bins, labels=False, duplicates="drop").astype("Int64").astype(int)
    return df_b


def accuracy(y_true: pd.Series, y_pred: List) -> float:
    return (y_true.values == np.array(y_pred)).mean()


def run_gbt(train_df: pd.DataFrame, test_df: pd.DataFrame, n_trees: int = 20, learning_rate: float = 0.1, cart_max_depth: int | None = None) -> Tuple[float, float, float]:
    kwargs = {}
    cart_settings = {}
    if cart_max_depth is not None:
        cart_settings["max_depth"] = cart_max_depth
    kwargs["cart_settings"] = cart_settings

    # request classification init to encourage classification-oriented trees
    gbt = GradientBoostedTree(df=train_df, y_name="Cover_Type", learning_rate=learning_rate, seed=42, init_method="classification", **kwargs)
    t0 = time.perf_counter()
    gbt.train(n_trees)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = gbt.predict(test_df)
    t_pred = time.perf_counter() - t0

    # If predictions are numeric scores, map to nearest class label from training set
    try:
        y_pred_arr = np.asarray(y_pred, dtype=float)
        classes = np.unique(train_df["Cover_Type"])
        # map each numeric prediction to nearest class value
        y_pred_mapped = np.array([classes[np.argmin(np.abs(classes - v))] for v in y_pred_arr])
        y_pred = y_pred_mapped
    except Exception:
        # if mapping fails, leave predictions as-is
        pass

    acc = accuracy(test_df["Cover_Type"], y_pred)
    return acc, t_fit, t_pred


def run_cart(train_df: pd.DataFrame, test_df: pd.DataFrame, max_depth: int | None = None) -> Tuple[float, float, float]:
    kwargs = {}
    if max_depth is not None:
        kwargs["max_depth"] = max_depth

    c = CART(df=train_df, y_name="Cover_Type", method="classification", **kwargs)
    t0 = time.perf_counter()
    c.create_tree()
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = c.predict(test_df)
    t_pred = time.perf_counter() - t0

    acc = accuracy(test_df["Cover_Type"], y_pred)
    return acc, t_fit, t_pred


def main(repeats: int = 3, n_bins: int = 5, sample_size: int | None = None, n_trees: int = 20, learning_rate: float = 0.1, cart_max_depth: int | None = 6):
    df = pd.read_parquet("data/covtype_categorical.parquet")

    for cat_col in ("Wilderness_Area", "Soil_Type"):
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype("category")

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    results = {"cart": [], "binned_gbt": []}

    for i in range(repeats):
        train, test = stratified_train_test_split(df, "Cover_Type", test_size=0.3, seed=42 + i)

        r = run_cart(train, test, max_depth=cart_max_depth)
        results["cart"].append(r)

        train_b = bin_df_to_int(train, n_bins=n_bins)
        test_b = bin_df_to_int(test, n_bins=n_bins)
        r = run_gbt(train_b, test_b, n_trees=n_trees, learning_rate=learning_rate, cart_max_depth=cart_max_depth)
        results["binned_gbt"].append(r)

    def summarize(arr: List[Tuple[float, float, float]]):
        a = np.array(arr)
        return a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()

    c_acc, c_fit, c_pred = summarize(results["cart"])
    b_acc, b_fit, b_pred = summarize(results["binned_gbt"])

    print("CART vs Binned GBT comparison (mean over {} runs):".format(repeats))
    print("CART (scalar): accuracy={:.4f}, fit={:.6f}s, predict={:.6f}s".format(c_acc, c_fit, c_pred))
    print("Binned GBT:     accuracy={:.4f}, fit={:.6f}s, predict={:.6f}s".format(b_acc, b_fit, b_pred))


if __name__ == "__main__":
    main(repeats=1, sample_size=20000, n_trees=20, learning_rate=0.1, cart_max_depth=6)
