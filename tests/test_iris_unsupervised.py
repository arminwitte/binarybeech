import numpy as np
import pandas as pd

from binarybeech.binarybeech import CART, GradientBoostedTree, RandomForest


def test_iris_cart_create():
    df_iris_orig = pd.read_csv("data/iris.csv")
    df_iris = df_iris_orig.drop(columns=["species"])
    c = CART(df=df_iris, metrics_type="clustering", max_depth=2)
    c.create_tree()
    p = c.predict(df_iris)
    assert isinstance(p, np.ndarray)
    assert isinstance(p[0], str)
    assert p[0] == p[1]
    assert p[0] != p[-1]


def test_iris_cart_tolerance():
    df_iris_orig = pd.read_csv("data/iris.csv")
    df_iris = df_iris_orig.drop(columns=["species"])
    c = CART(
        df=df_iris,
        metrics_type="clustering",
        max_depth=10,
        min_leaf_samples=10,
        algorithm_kwargs={"unsupervised_minimum_relative_entropy_improvement": -0.15},
    )
    c.create_tree()
    p = c.predict(df_iris)
    assert c.tree.leaf_count() == 16
