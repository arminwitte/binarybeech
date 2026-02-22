import numpy as np
import pandas as pd

from binarybeech.binarybeech import CART, GradientBoostedTree, RandomForest


def test_prostate_cart_create():
    df_prostate = pd.read_csv("data/prostate.data", sep="\t")
    train = df_prostate["train"].isin(["T"])
    df_prostate = df_prostate.drop(columns=["Unnamed: 0", "train"])

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
            0.854415,
            0.765468,
            2.047693,
            2.297573,
            3.013081,
            1.266948,
            2.008214,
            2.962692,
            1.599388,
            2.047693,
        ],
        rtol=1e-5,
    )
    assert acc > 0.0
    assert c.tree.leaf_count() == 63


def test_prostate_cart_l1():
    df_prostate = pd.read_csv("data/prostate.data", sep="\t")
    train = df_prostate["train"].isin(["T"])
    df_prostate = df_prostate.drop(columns=["Unnamed: 0", "train"])
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
            0.665468,
            0.947319,
            0.947319,
            1.323395,
            1.558228,
            1.631656,
            1.666442,
            1.716452,
            1.916231,
            1.921548,
        ],
        rtol=1e-5,
    )
    assert acc <= 1.0 and acc > 0.98
    assert c.tree.leaf_count() == 91


def test_prostate_cart_l2():
    df_prostate = pd.read_csv("data/prostate.data", sep="\t")
    train = df_prostate["train"].isin(["T"])
    df_prostate = df_prostate.drop(columns=["Unnamed: 0", "train"])
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
            0.69588,
            0.952108,
            0.952108,
            1.307995,
            1.50748,
            1.574232,
            1.605856,
            1.65132,
            1.872601,
            1.837771,
        ],
        rtol=1e-5,
    )
    assert acc <= 1.0 and acc > 0.93
    assert c.tree.leaf_count() == 93


def test_prostate_gradientboostedtree():
    df_prostate = pd.read_csv("data/prostate.data", sep="\t")
    train = df_prostate["train"].isin(["T"])
    df_prostate = df_prostate.drop(columns=["Unnamed: 0", "train"])
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
        [
            1.266948,
            2.794228,
            1.266948,
            3.530763,
            1.266948,
            1.266948,
            1.446919,
            -0.162519,
            2.327278,
            3.530763,
        ],
        rtol=1.0,
        atol=3.0,
    )
    assert acc > -0.5
