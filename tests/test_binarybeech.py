from binarybeech.binarybeech import *
import pandas as pd


def test_node():
    n = Node(value=1.0)
    assert n.is_leaf
    
def test_cart_create():
    df_titanic = pd.read_csv("data/titanic.csv")
    c = CART(df_titanic,"Survived", metrics_type="classification")
    c.create_tree()
    p = c.predict(df_titanic)
    val = c.validate()
    acc = val["accuracy"]
    np.testing.assert_allclose(p[:10], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    assert acc < 1. and acc > 0.78
    
def test_cart_train():
    df_titanic = pd.read_csv("data/titanic.csv")
    c = CART(df_titanic,"Survived", metrics_type="classification")
    c.train()
    p = c.predict(df_titanic)
    val = c.validate()
    acc = val["accuracy"]
    np.testing.assert_allclose(p[:10], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    assert acc < 1. and acc > 0.78
    
def test_gradientboostedtree():
    df_titanic = pd.read_csv("data/titanic.csv")
    gbt = GradientBoostedTree(df_titanic,"Survived",learning_rate=0.5,init_metrics_type="logistic")
    gbt.train(20)
    p = gbt.predict(df_titanic)
    val = gbt.validate()
    acc = val["accuracy"]
    np.testing.assert_allclose(np.round(p[:10]).astype(int), [0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    assert acc < 1. and acc > 0.8
    
def test_randomforest():
    df_titanic = pd.read_csv("data/titanic.csv")
    rf = RandomForest(df_titanic,"Survived",metrics_type="classification")
    rf.train(20)
    p = rf.predict(df_titanic)
    val = rf.validate_oob()
    acc = val["accuracy"]
    np.testing.assert_allclose(p[:10], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    assert acc < 1. and acc > 0.8
    