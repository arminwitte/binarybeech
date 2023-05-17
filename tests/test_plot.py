import numpy as np
import pandas as pd

from binarybeech.binarybeech import CART


def test_iris_cart_train():
    df_iris = pd.read_csv("data/iris.csv")
    c = CART(df=df_iris, y_name="species", metrics_type="classification", seed=42)
    c.train()
    assert isinstance(c.pruning_quality, dict)
