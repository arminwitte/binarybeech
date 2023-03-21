import numpy as np
import pandas as pd

from binarybeech.binarybeech import CART, GradientBoostedTree, RandomForest


def test_iris_cart_create():
    df_iris = pd.read_csv("data/iris.csv")
    c = CART(df_iris, "Survived", metrics_type="classification")
    c.create_tree()
    p = c.predict(df_iris)
    val = c.validate()
    acc = val["accuracy"]
    np.testing.assert_allclose(p[:10], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    assert acc < 1.0 and acc > 0.95


def test_iris_cart_train():
    df_iris = pd.read_csv("data/iris.csv")
    c = CART(df_iris, "Survived", metrics_type="classification")
    c.train()
    p = c.predict(df_iris)
    val = c.validate()
    acc = val["accuracy"]
    np.testing.assert_allclose(p[:10], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    assert acc < 1.0 and acc > 0.95


def test_iris_randomforest():
    df_iris = pd.read_csv("data/iris.csv")
    rf = RandomForest(df_iris, "Survived", metrics_type="classification")
    rf.train(20)
    p = rf.predict(df_iris)
    val = rf.validate_oob()
    acc = val["accuracy"]
    np.testing.assert_allclose(p[:10], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    assert acc < 1.0 and acc > 0.8
