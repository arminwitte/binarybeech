import numpy as np
import pandas as pd

from binarybeech.binarybeech import CART


def test_iris_binned_cart():
    # load iris and bin one numeric column
    df = pd.read_csv("data/iris.csv")

    # create binned versions of numeric columns
    for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        df[f"{col}_binned"] = pd.cut(df[col], bins=6, labels=False)

    # use the binned columns as feature set; keep species as target
    X_names = [f"{col}_binned" for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    c = CART(df=df, y_name="species", X_names=X_names, method="classification", seed=42)
    c.train()

    preds = c.predict(df)
    val = c.validate()
    acc = val["accuracy"]

    # sanity checks: predictions should be strings (class labels) and accuracy reasonable
    assert isinstance(preds[0], str)
    # With multiple binned features we expect decent accuracy
    assert acc > 0.9
