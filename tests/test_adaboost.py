import numpy as np
import pandas as pd

from binarybeech.binarybeech import AdaBoostTree
from binarybeech.tree import Tree


def test_iris_cart_create():
    df_iris = pd.read_csv("data/iris.csv")
    c = AdaBoostTree(df=df_iris, y_name="species", method="classification")
    # c.create_tree()
    # p = c.predict(df_iris)
    # val = c.validate()
    # acc = val["accuracy"]
    # np.testing.assert_array_equal(p[:10], ["setosa"] * 10)
    # assert acc <= 1.0 and acc > 0.95

