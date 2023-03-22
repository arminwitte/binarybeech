import numpy as np
import pandas as pd

from binarybeech.binarybeech import CART, GradientBoostedTree, RandomForest


def test_housing_cart_create():
    df_housing = pd.read_csv("data/Housing.csv")
    c = CART(df_housing, "price", metrics_type="regression")
    c.create_tree()
    p = c.predict(df_housing)
    val = c.validate()
    acc = val["R2"]
    np.testing.assert_allclose(p[:10], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    assert acc < 1.0 and acc > 0.78


def test_housing_cart_train():
    df_housing = pd.read_csv("data/Housing.csv")
    c = CART(df_housing, "price", metrics_type="classification")
    c.train()
    p = c.predict(df_housing)
    val = c.validate()
    acc = val["R2"]
    np.testing.assert_allclose(p[:10], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    assert acc < 1.0 and acc > 0.78


def test_housing_gradientboostedtree():
    df_housing = pd.read_csv("data/Housing.csv")
    gbt = GradientBoostedTree(
        df_housing, "price", learning_rate=0.5, init_metrics_type="regression"
    )
    gbt.train(20)
    p = gbt.predict(df_housing)
    val = gbt.validate()
    acc = val["R2"]
    np.testing.assert_allclose(
        np.round(p[:10]).astype(int), [0, 1, 1, 1, 0, 0, 0, 0, 1, 1]
    )
    assert acc < 1.0 and acc > 0.8


def test_housing_randomforest():
    df_housing = pd.read_csv("data/Housing.csv")
    rf = RandomForest(df_housing, "price", metrics_type="regression")
    rf.train(20)
    p = rf.predict(df_housing)
    val = rf.validate_oob()
    acc = val["R2"]
    np.testing.assert_allclose(p[:10], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    assert acc < 1.0 and acc > 0.8
