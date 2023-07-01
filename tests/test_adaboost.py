import numpy as np
import pandas as pd

from binarybeech.binarybeech import AdaBoostTree
from binarybeech.tree import Tree


def test_adaboost_iris():
    df_iris = pd.read_csv("data/iris.csv")
    c = AdaBoostTree(df=df_iris, y_name="species", method="classification")
    c.train(100)
    p = c.predict(df_iris)
    val = c.validate()
    acc = val["accuracy"]
    np.testing.assert_array_equal(p[:10], ["setosa"] * 10)
    assert acc <= 1.0 and acc > 0.98


def test_adaboost_titanic():
    df_titanic = pd.read_csv("data/titanic.csv")
    c = AdaBoostTree(df=df_titanic, y_name="Survived", method="classification")
    c.train(100)
    p = c.predict(df_titanic)
    val = c.validate()
    acc = val["accuracy"]
    np.testing.assert_allclose(p[:10], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    assert acc < 1.0 and acc > 0.82
