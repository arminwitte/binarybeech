from binarybeech.binarybeech import *
import pandas as pd


def test_node():
    n = Node(value=1.0)
    assert n.is_leaf
    
def test_cart():
    df_titanic = pd.read_csv("data/titanic.csv")
    c = CART(df_titanic,"Survived", metrics_type="classification")
    c.create_tree()
    p = c._predict(df_titanic.iloc[0])
    assert p == 0.9
    
def test_cart():
    df_titanic = pd.read_csv("data/titanic.csv")
    c = CART(df_titanic,"Survived", metrics_type="classification")
    c.train(10)
    p = c._predict(df_titanic.iloc[0])
    assert p == 0.9
    
    
def test_gbt():
    df_titanic = pd.read_csv("data/titanic.csv")
    gbt = GradientBoostedTree(df_titanic,"Survived",learning_rate=0.5)
    gbt.train(10)
    p = gbt._predict(df_titanic.iloc[0])
    assert p == 0.9
    
