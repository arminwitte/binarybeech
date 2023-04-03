import numpy as np
import pandas as pd

from binarybeech.binarybeech import CART, GradientBoostedTree, RandomForest


def test_iris_cart_create():
    df_iris = pd.read_csv("data/iris.csv")
    c = CART(df_iris, None, metrics_type="clustering")
    c.create_tree()
    p = c.predict(df_iris)
    assert isinstance(p, np.ndarray)


