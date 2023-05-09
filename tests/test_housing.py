import numpy as np
import pandas as pd

from binarybeech.binarybeech import CART, GradientBoostedTree, RandomForest


def test_housing_cart_create():
    df_housing = pd.read_csv("data/Housing.csv")
    c = CART(df=df_housing, y_name="price", metrics_type="regression", seed=42)
    c.create_tree()
    p = c.predict(df_housing)
    val = c.validate()
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
    assert acc < 1.0 and acc > 0.78


def test_housing_cart_train():
    df_housing = pd.read_csv("data/Housing.csv")
    c = CART(df=df_housing, y_name="price", metrics_type="regression", seed=42)
    c.train()
    p = c.predict(df_housing)
    val = c.validate()
    acc = val["R_squared"]
    np.testing.assert_allclose(
        p[:10],
        [
            12757500.0,
            12250000.0,
            12250000.0,
            12757500.0,
            11410000.0,
            10500000.0,
            10500000.0,
            9915500.0,
            9835000.0,
            9450000.0,
        ],
    )
    assert acc < 1.0 and acc > 0.78


def test_housing_gradientboostedtree():
    df_housing = pd.read_csv("data/Housing.csv")
    gbt = GradientBoostedTree(
        df=df_housing,
        y_name="price",
        learning_rate=0.5,
        init_metrics_type="regression",
        seed=42,
    )
    gbt.train(20)
    p = gbt.predict(df_housing)
    val = gbt.validate()
    acc = val["R_squared"]
    np.testing.assert_allclose(
        p[:10],
        [
            10153886.825137,
            10228043.358084,
            9250209.100109,
            9625320.38567,
            8441615.154178,
            10046719.36346,
            11027888.728298,
            9269609.157115,
            8175317.976215,
            7653074.341622,
        ],
        rtol=1e-2,
    )
    assert acc < 1.0 and acc > 0.7


def test_housing_randomforest():
    df_housing = pd.read_csv("data/Housing.csv")
    rf = RandomForest(df=df_housing, y_name="price", metrics_type="regression", seed=42)
    rf.train(20)
    p = rf.predict(df_housing)
    val = rf.validate_oob()
    acc = val["R_squared"]
    np.testing.assert_allclose(
        p[:10],
        [
            6505708.888889,
            11025000.0,
            4975176.470588,
            6505708.888889,
            5074850.609756,
            11025000.0,
            11025000.0,
            4419049.54955,
            5074850.609756,
            5588586.956522,
        ],
        # rtol=0.7,
    )
    assert acc < 1.0 and acc > 0.2
