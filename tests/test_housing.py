import numpy as np
import pandas as pd

from binarybeech.binarybeech import CART, GradientBoostedTree, RandomForest


def test_housing_cart_create():
    df_housing = pd.read_csv("data/Housing.csv")
    c = CART(df=df_housing, y_name="price", method="regression", seed=42)
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
    assert acc < 1.0 and acc > 0.8


def test_housing_cart_train():
    df_housing = pd.read_csv("data/Housing.csv")
    c = CART(df=df_housing, y_name="price", method="regression", seed=42)
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
            10150000.0,
            9870000.0,
            9800000.0,
        ],
    )
    assert acc < 1.0 and acc > 0.8


def test_housing_gradientboostedtree():
    df_housing = pd.read_csv("data/Housing.csv")
    gbt = GradientBoostedTree(
        df=df_housing,
        y_name="price",
        learning_rate=0.5,
        init_method="regression",
        seed=42,
    )
    gbt.train(20)
    p = gbt.predict(df_housing)
    val = gbt.validate()
    acc = val["R_squared"]
    np.testing.assert_allclose(
        p[:10],
        [
            8801218.895703, 10775516.637133, 10085858.184785,  9840551.810912,
            8338360.153133,  8876988.814029, 11119599.903721,  9731685.064304,
            8656771.468612,  8211774.820057
        ],
        # rtol=1e-2,
    )
    assert acc < 1.0 and acc > 0.75


def test_housing_randomforest():
    df_housing = pd.read_csv("data/Housing.csv")
    rf = RandomForest(df=df_housing, y_name="price", method="regression", seed=42)
    rf.train(20)
    p = rf.predict(df_housing)
    val = rf.validate_oob()
    acc = val["R_squared"]
    np.testing.assert_allclose(
        p[:10],
        [
            6505708.888889,
            10850000.0,
            4877212.765957,
            6468358.974359,
            5015773.913043,
            10850000.0,
            10850000.0,
            4471566.666667,
            5015773.913043,
            5388055.555556,
        ],
        # rtol=0.7,
    )
    assert acc < 1.0 and acc > 0.3
