#!/usr/bin/env python
# coding: utf-8
import numpy as np

from binarybeech.binarybeech import CART


def model_missings(df, y_name, X_names=None, cart_settings={}):
    if X_names is None:
        X_names = [n for n in df.columns]
        X_names.remove(y_name)
    df_ = df.copy()
    has_missings = df.isnull().any()
    for x_name in X_names:

        if not has_missings[x_name]:
            continue

        m_X_names = [n for n in df.columns]
        m_X_names.remove(x_name)
        m_X_names.remove(y_name)
        kwargs = dict(
            max_depth=3,
            min_leaf_samples=5,
            min_split_samples=4,
        )
        kwargs = {**kwargs, **cart_settings}
        mod = CART(df[~df[x_name].isnull()], x_name, X_names=m_X_names, **cart_settings)
        mod.create_tree()
        df_.loc[df[x_name].isnull(), x_name] = mod.predict(df[df[x_name].isnull()])

    return df_
