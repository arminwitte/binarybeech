import numpy as np
import pandas as pd

from binarybeech.binarybeech import CART, GradientBoostedTree, RandomForest


def test_titanic_cart_create():
    df_titanic = pd.read_csv("data/titanic.csv")
    c = CART(df=df_titanic, y_name="Survived", method="classification")
    c.create_tree()
    p = c.predict(df_titanic)
    val = c.validate()
    acc = val["accuracy"]
    np.testing.assert_allclose(p[:10], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    assert acc < 1.0 and acc > 0.78
    assert c.tree.leaf_count() == 109


def test_titanic_cart_create_min_split_loss():
    df_titanic = pd.read_csv("data/titanic.csv")
    c = CART(
        df=df_titanic, y_name="Survived", method="classification", min_split_loss=0.1
    )
    c.create_tree()
    p = c.predict(df_titanic)
    val = c.validate()
    acc = val["accuracy"]
    np.testing.assert_allclose(p[:10], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    assert acc < 1.0 and acc > 0.78
    assert c.tree.leaf_count() == 2


def test_titanic_cart_train():
    df_titanic = pd.read_csv("data/titanic.csv")
    c = CART(df=df_titanic, y_name="Survived", method="classification", seed=42)
    c.train()
    p = c.predict(df_titanic)
    val = c.validate()
    acc = val["accuracy"]
    np.testing.assert_allclose(p[:10], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    assert acc < 1.0 and acc > 0.78


def test_titanic_gradientboostedtree():
    df_titanic = pd.read_csv("data/titanic.csv")
    gbt = GradientBoostedTree(
        df=df_titanic,
        y_name="Survived",
        learning_rate=0.5,
        init_method="logistic",
        seed=42,
    )
    gbt.train(20)
    p = gbt.predict(df_titanic)
    val = gbt.validate()
    acc = val["accuracy"]
    np.testing.assert_allclose(
        np.round(p[:10]).astype(int), [0, 1, 1, 1, 0, 0, 0, 0, 1, 1]
    )
    assert acc < 1.0 and acc > 0.8


def test_titanic_randomforest():
    df_titanic = pd.read_csv("data/titanic.csv")
    rf = RandomForest(
        df=df_titanic, y_name="Survived", method="classification", seed=42
    )
    rf.train(20)
    p = rf.predict(df_titanic)
    val = rf.validate_oob()
    acc = val["accuracy"]
    np.testing.assert_allclose(p[:10], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    assert acc < 1.0 and acc > 0.75
