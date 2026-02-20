#!/usr/bin/env python
# coding: utf-8
# import numpy as np

from binarybeech.binarybeech import CART


def model_missings(df, y_name, X_names=None, cart_settings={}):
    if X_names is None:
        X_names = [n for n in df.columns]
        X_names.remove(y_name)
    df_ = df.copy()
    has_missings = df.isnull().any()
    # Fill missings for predictor columns (X_names)
    for x_name in X_names:
        if not df[x_name].isnull().any():
            continue

        m_X_names = [n for n in df.columns if n not in (x_name, y_name)]
        kwargs = dict(
            max_depth=3,
            min_leaf_samples=5,
            min_split_samples=4,
        )
        kwargs = {**kwargs, **cart_settings}
        mod = CART(df=df[~df[x_name].isnull()], y_name=x_name, X_names=m_X_names, **kwargs)
        mod.create_tree()
        df_.loc[df[x_name].isnull(), x_name] = mod.predict(df[df[x_name].isnull()])

    # Fill missings for the target variable (y_name) using provided X_names as predictors
    if df[y_name].isnull().any():
        m_X_names = list(X_names)
        kwargs = dict(
            max_depth=3,
            min_leaf_samples=5,
            min_split_samples=4,
        )
        kwargs = {**kwargs, **cart_settings}
        mod = CART(df=df[~df[y_name].isnull()], y_name=y_name, X_names=m_X_names, **kwargs)
        mod.create_tree()
        df_.loc[df[y_name].isnull(), y_name] = mod.predict(df[df[y_name].isnull()])

    return df_
