from binarybeech.binarybeech import *
import pandas as pd


def test_node():
    n = Node(value=1.0)
    assert n.is_leaf
    
def test_cart():
    df_titanic = pd.read_csv("data/titanic.csv")
    c = CART(df_titanic,"Survived", metrics_type="classification")
    c.create_tree()
    p = c.predict(df_titanic)
    val = c.validate()
    np.testing.assert_allclose(p[:10], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    acc = val["accuracy"]
    assert acc < 1. and acc > 0.8
    
def test_cart2():
    df_titanic = pd.read_csv("data/titanic.csv")
    c = CART(df_titanic,"Survived", metrics_type="classification")
    c.train()
    p = c._predict(df_titanic.iloc[0])
    print(c.validate())
    assert p == 1
    assert c._predict(df_titanic.iloc[1]) == 1
    
def test_gbt():
    df_titanic = pd.read_csv("data/titanic.csv")
    gbt = GradientBoostedTree(df_titanic,"Survived",learning_rate=0.5,init_metrics_type="logistic")
    gbt.train(10)
    p = gbt._predict(df_titanic.iloc[0])
    assert p == 0.9
    assert gbt._predict(df_titanic.iloc[1]) == 1
    
def test_randomforest():
    df_titanic = pd.read_csv("data/titanic.csv")
    rf = RandomForest(df_titanic,"Survived",metrics_type="logistic")
    rf.train(10)
    p = rf._predict(df_titanic.iloc[0])
    assert p == 0.9
    assert rf._predict(df_titanic.iloc[1]) == 1
    
