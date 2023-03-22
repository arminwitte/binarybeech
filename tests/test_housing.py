import numpy as np
import pandas as pd

from binarybeech.binarybeech import CART, GradientBoostedTree, RandomForest


def test_housing_cart_create():
    df_housing = pd.read_csv("data/Housing.csv")
    c = CART(df_housing, "price", metrics_type="regression")
    c.create_tree()
    p = c.predict(df_housing)
    val = c.validate()
    acc = val["R_squared"]
    np.testing.assert_allclose(p[:10], [13300000., 12250000., 12250000., 12215000., 11410000., 10850000.,
           10150000., 10150000.,  9870000.,  9800000.])
    assert acc < 1.0 and acc > 0.78


def test_housing_cart_train():
    df_housing = pd.read_csv("data/Housing.csv")
    c = CART(df_housing, "price", metrics_type="regression")
    c.train()
    p = c.predict(df_housing)
    val = c.validate()
    acc = val["R_squared"]
    np.testing.assert_allclose(p[:10], [12757500., 12250000., 12250000., 12757500., 11410000., 10500000.,
           10500000.,  9915500.,  9835000.,  9450000.])
    assert acc < 1.0 and acc > 0.78


def test_housing_gradientboostedtree():
    df_housing = pd.read_csv("data/Housing.csv")
    gbt = GradientBoostedTree(
        df_housing, "price", learning_rate=0.5, init_metrics_type="regression"
    )
    gbt.train(20)
    p = gbt.predict(df_housing)
    val = gbt.validate()
    acc = val["R_squared"]
    np.testing.assert_allclose(
        p[:10], [10710888, 10946083, 7583931, 10526538,  7151695,  9834308, 11651867,  8528326,  7642070,  8127825]
    )
    assert acc < 1.0 and acc > 0.8


def test_housing_randomforest():
    df_housing = pd.read_csv("data/Housing.csv")
    rf = RandomForest(df_housing, "price", metrics_type="regression")
    rf.train(20)
    p = rf.predict(df_housing)
    val = rf.validate_oob()
    acc = val["R_squared"]
    np.testing.assert_allclose(p[:10], [6321721.649485, 11375000.      ,  4827425.287356,  6321721.649485, 4568416.666667, 11375000.      , 11375000.      ,  5116847.826087, 4775413.793103,  6399243.421053],rtol=0.5)
    assert acc < 1.0 and acc > 0.3
