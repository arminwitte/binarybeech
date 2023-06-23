import numpy as np
import pandas as pd

from binarybeech.binarybeech import CART, RandomForest
from binarybeech.tree import Tree


def test_iris_cart_create():
    df_iris = pd.read_csv("data/iris.csv")
    c = CART(df=df_iris, y_name="species", method="classification")
    c.create_tree()
    p = c.predict(df_iris)
    val = c.validate()
    acc = val["accuracy"]
    np.testing.assert_array_equal(p[:10], ["setosa"] * 10)
    assert acc <= 1.0 and acc > 0.95


def test_iris_cart_train():
    df_iris = pd.read_csv("data/iris.csv")
    c = CART(df=df_iris, y_name="species", method="classification", seed=42)
    c.train()
    p = c.predict(df_iris)
    val = c.validate()
    acc = val["accuracy"]
    np.testing.assert_array_equal(p[:10], ["setosa"] * 10)
    assert acc <= 1.0 and acc > 0.95


def test_iris_randomforest():
    df_iris = pd.read_csv("data/iris.csv")
    rf = RandomForest(df=df_iris, y_name="species", method="classification", seed=42)
    rf.train(20)
    p = rf.predict(df_iris)
    val = rf.validate_oob()
    acc = val["accuracy"]
    np.testing.assert_array_equal(p[:10], ["setosa"] * 10)
    assert acc <= 1.0 and acc > 0.9
    

def test_iris_from_dict():
    df_iris = pd.read_csv("data/iris.csv")
    c = CART(df=df_iris, y_name="species", method="classification")
    c.create_tree()
    
    tree_dict = c.tree.to_dict()
    assert isinstance(tree_dict, dict)
    
    # print(tree_dict)
    
    tree = Tree.from_dict(tree_dict)
    assert isinstance(tree, Tree)
    assert len(tree.nodes()) == 21
    assert tree.leaf_count() == 4
        
    
    c.tree = tree
    p = c.predict(df_iris)
    val = c.validate()
    acc = val["accuracy"]
    np.testing.assert_array_equal(p[:10], ["setosa"] * 10)
    assert acc <= 1.0 and acc > 0.95
    
def test_iris_from_json():
    df_iris = pd.read_csv("data/iris.csv")
    c = CART(df=df_iris, y_name="species", method="classification")
    c.train()
    
    tree_json = c.tree.to_json()
    assert isinstance(tree_json, str)
    
    # print(tree_json)
    
    tree = Tree.from_json(string=tree_json)
    assert isinstance(tree, Tree)
    assert len(tree.nodes()) == 5
    assert tree.leaf_count() == 4
    
    c.tree = tree
    p = c.predict(df_iris)
    val = c.validate()
    acc = val["accuracy"]
    np.testing.assert_array_equal(p[:10], ["setosa"] * 10)
    assert acc <= 1.0 and acc > 0.95