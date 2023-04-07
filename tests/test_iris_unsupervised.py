import numpy as np
import pandas as pd

from binarybeech.binarybeech import CART, GradientBoostedTree, RandomForest


def test_iris_cart_create():
    df_iris_orig = pd.read_csv("data/iris.csv")
    df_iris = df_iris_orig.drop(columns=["species"])
    c = CART(df_iris, None, metrics_type="clustering", max_depth=2)
    c.create_tree()
    p = c.predict(df_iris)
    assert isinstance(p, np.ndarray)
    assert isinstance(p[0], str)
    assert p[0] == p[1]
    assert p[0] != p[-1]


