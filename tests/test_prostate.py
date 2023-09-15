import numpy as np
import pandas as pd

from binarybeech.binarybeech import CART, GradientBoostedTree, RandomForest


def test_housing_cart_create():
    df_prostate = pd.read_csv("data/prostate.data", sep="\t")
    train = df_prostate["train"].isin(["T"])
    df_prostate.drop(columns=["Unnamed: 0", "train"])
    
    c = CART(df=df_prostate[train], y_name="lpsa", method="regression:regularized", seed=42)
    c.create_tree()
    p = c.predict(df_prostate[~train])
    val = c.validate(df_prostate[~train])
    acc = val["R_squared"]
    np.testing.assert_allclose(
        p[:10],
        [
            13300000.0,
            12250000.0,
            12250000.0,
            12215000.0,
            11410000.0,
            10850000.0,
            10150000.0,
            10150000.0,
            9870000.0,
            9800000.0,
        ],
    )
    assert acc < 1.0 and acc > 0.8
    assert c.tree.node_count() == 10


def test_housing_cart_train():
    df_prostate = pd.read_csv("data/prostate.data", sep="\t")
    train = df_prostate["train"].isin(["T"])
    df_prostate.drop(columns=["Unnamed: 0", "train"])
    c = CART(df=df_prostate, y_name="lpsa", method="regression:regularized", seed=42, lambda_l1=1.,lambda_l2=1.)
    c.create_tree()
    p = c.predict(df_prostate[~train])
    val = c.validate(df_prostate[~train])
    acc = val["R_squared"]
    np.testing.assert_allclose(
        p[:10],
        [
            13300000.0,
            12250000.0,
            12250000.0,
            12215000.0,
            11410000.0,
            10850000.0,
            10150000.0,
            10150000.0,
            9870000.0,
            9800000.0,
        ],
    )
    assert acc < 1.0 and acc > 0.8
    assert c.tree.node_count() == 10


def test_housing_gradientboostedtree():
    df_prostate = pd.read_csv("data/prostate.data", sep="\t")
    train = df_prostate["train"].isin(["T"])
    df_prostate.drop(columns=["Unnamed: 0", "train"])
    gbt = GradientBoostedTree(
        df=df_prostate[train],
        y_name="lpsa",
        learning_rate=0.5,
        init_method="regression:regularized",
        seed=42,
        cart_settings={"lambda_l1":1.,"lambda_l2":1., "method":"regression:regularized"}
    )
    gbt.train(20)
    p = gbt.predict(df_prostate[~train])
    val = gbt.validate(df_prostate[~train])
    acc = val["R_squared"]
    np.testing.assert_allclose(
        p[:10],
        [
            13300000.0,
            12250000.0,
            12250000.0,
            12215000.0,
            11410000.0,
            10850000.0,
            10150000.0,
            10150000.0,
            9870000.0,
            9800000.0,
        ],
    )
    assert acc < 1.0 and acc > 0.8