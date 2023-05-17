#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_areas(mod, x_name, y_name, df=None, rng=None, slice_definition=None, N=101, ax=None):
    if rng is None:
        x_min, x_max = df[x_name].min(), df[x_name].max()
        y_min, y_max = df[y_name].min(), df[y_name].max()
        rng = ((x_min, x_max), (y_min, y_max))
        
    x, y = np.meshgrid(
        np.linspace(rng[0][0],rng[0][1],N),
        np.linspace(rng[1][0],rng[1][1],N))
    col = []
    
    if slice_definition is None:
        slice_definition = pd.DataFrame(df.iloc[0])
    
    for i in range(len(x.ravel())):
        d = slice_definition.copy()
        d[x_name] = x.ravel()[i]
        d[y_name] = y.ravel()[i]
        col.append(mod.tree.traverse(d).value)
    unique = [u for u in np.unique(col)]
    for i, c in enumerate(col):
        col[i] = unique.index(c)
    z = np.array(col).reshape(x.shape)
    
    if ax is None:
        plt.pcolormesh(x,y,z)
        if df is not None:
            plt.scatter(df[x_name],df[y_name])
    else:
        ax.pcolormesh(x,y,z)
        if df is not None:
            ax.scatter(df[x_name],df[y_name])

def plot_pruning_quality(beta=None, qual_mean=None, qual_sd=None)
    plt.errorbar(beta, qual_mean, yerr=qual_sd)
