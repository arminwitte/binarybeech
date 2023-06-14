import numpy as np
import pandas as pd

from binarybeech.binarybeech import CART
from binarybeech.visualize import plot_areas, plot_pruning_quality, extract_rules, print_rules


def test_plot_areas():
    df_iris = pd.read_csv("data/iris.csv")
    c = CART(df=df_iris, y_name="species", method="classification", seed=42)
    c.train()
    plot_areas(c, "petal_width", "petal_length", df=df_iris)


def test_plot_pruning_quality():
    df_iris = pd.read_csv("data/iris.csv")
    c = CART(df=df_iris, y_name="species", method="classification", seed=42)
    c.train()
    assert isinstance(c.pruning_quality, dict)
    
def test_print_rules():
    df_iris = pd.read_csv("data/iris.csv")
    c = CART(df=df_iris, y_name="species", method="classification", seed=42)
    c.train()
    rules = extract_rules(c.tree)
    print(rules)
    
    print_rules(rules)
    
    assert 1 == 2
    assert isinstance(rules,dict)
