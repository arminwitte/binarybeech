import numpy as np
import pandas as pd

from binarybeech.binarybeech import CART, GradientBoostedTree, RandomForest


def test_prostate_cart_create():
    df_prostate = pd.read_csv("data/prostate.data", sep="\t")
    train = df_prostate["train"].isin(["T"])
    df_prostate.drop(columns=["Unnamed: 0", "train"])

    c = CART(
        df=df_prostate[train], y_name="lpsa", method="regression:regularized", seed=42
    )
    c.create_tree()
    p = c.predict(df_prostate[~train])
    val = c.validate(df_prostate[~train])
    acc = val["R_squared"]
    np.testing.assert_allclose(
        p[:10],
        [
            0.765468,
            1.266948,
            1.266948,
            1.348073,
            1.695616,
            1.800058,
            1.800058,
            1.800058,
            2.008214,
            2.008214,
        ],
        rtol=1e-5,
    )
    assert acc <= 1.0 and acc > 0.98
    assert c.tree.leaf_count() == 63


def test_prostate_cart_l1():
    df_prostate = pd.read_csv("data/prostate.data", sep="\t")
    train = df_prostate["train"].isin(["T"])
    df_prostate.drop(columns=["Unnamed: 0", "train"])
    c = CART(
        df=df_prostate,
        y_name="lpsa",
        method="regression:regularized",
        seed=42,
        lambda_l1=0.1,
        lambda_l2=0.0,
    )
    c.create_tree()
    p = c.predict(df_prostate[~train])
    val = c.validate(df_prostate[~train])
    acc = val["R_squared"]
    np.testing.assert_allclose(
        p[:10],
        [
            0.761784,
            0.997319,
            0.997319,
            1.405271,
            1.623057,
            1.745681,
            1.745681,
            1.745681,
            2.014268,
            2.014268,
        ],
        rtol=1e-5,
    )
    assert acc <= 1.0 and acc > 0.99
    assert c.tree.leaf_count() == 33


def test_prostate_cart_l2():
    df_prostate = pd.read_csv("data/prostate.data", sep="\t")
    train = df_prostate["train"].isin(["T"])
    df_prostate.drop(columns=["Unnamed: 0", "train"])
    c = CART(
        df=df_prostate,
        y_name="lpsa",
        method="regression:regularized",
        seed=42,
        lambda_l1=0.0,
        lambda_l2=0.1,
    )
    c.create_tree()
    p = c.predict(df_prostate[~train])
    val = c.validate(df_prostate[~train])
    acc = val["R_squared"]
    np.testing.assert_allclose(
        p[:10],
        [
            0.769468,
            0.997447,
            0.997447,
            1.444482,
            1.60786,
            1.73106,
            1.73106,
            1.73106,
            1.99438,
            1.99438,
        ],
        rtol=1e-5,
    )
    assert acc <= 1.0 and acc > 0.98
    assert c.tree.leaf_count() == 24


def test_prostate_gradientboostedtree():
    df_prostate = pd.read_csv("data/prostate.data", sep="\t")
    train = df_prostate["train"].isin(["T"])
    df_prostate.drop(columns=["Unnamed: 0", "train"])
    gbt = GradientBoostedTree(
        df=df_prostate[train],
        y_name="lpsa",
        learning_rate=0.5,
        # lambda_l1=1.,
        # lambda_l2=1.,
        init_method="regression:regularized",
        seed=42,
        cart_settings={
            "method": "regression:regularized",
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "min_split_loss": 0.01,
        },
    )
    gbt.train(20)
    p = gbt.predict(df_prostate[~train])
    val = gbt.validate(df_prostate[~train])
    acc = val["R_squared"]
    np.testing.assert_allclose(
        p[:10],
        [1.095208, 0.951133, 0.869174, 1.409681, 1.754471, 1.754471,
               1.528437, 1.610396,1.905683, 1.905683],
        rtol=1e-5,
    )
    assert acc <= 1.0 and acc > 0.93
