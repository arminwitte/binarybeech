import numpy as np
import pandas as pd

from binarybeech.binarybeech import CART, GradientBoostedTree, RandomForest


def test_titanic_update_elastic():
    df_titanic = pd.read_csv("data/titanic.csv").sample(
        frac=1.0, replace=False, random_state=42
    )
    N = len(df_titanic.index)
    n = int(round(N / 2))
    df_train = df_titanic.iloc[:n].copy()
    df_update = df_titanic.iloc[n:].copy()
    # df_update.loc[:,"Survived"] = np.abs(df_update["Survived"].values - 1)
    gbt = GradientBoostedTree(
        df=df_train,
        y_name="Survived",
        learning_rate=0.5,
        init_method="logistic",
        seed=42,
    )
    gbt.train(20)

    gbt.update(df_update, update_method="elastic")

    p = gbt.predict(df_train)
    val = gbt.validate()
    acc = val["accuracy"]
    np.testing.assert_allclose(
        np.round(p[:10]).astype(int), [0, 0, 0, 0, 0, 1, 0, 0, 1, 1]
    )
    assert acc < 1.0 and acc > 0.8


def test_titanic_update_gamma():
    df_titanic = pd.read_csv("data/titanic.csv").sample(
        frac=1.0, replace=False, random_state=42
    )
    N = len(df_titanic.index)
    n = int(round(N / 2))
    df_train = df_titanic.iloc[:n].copy()
    df_update = df_titanic.iloc[n:].copy()
    # df_update.loc[:,"Survived"] = np.abs(df_update["Survived"].values - 1)
    gbt = GradientBoostedTree(
        df=df_train,
        y_name="Survived",
        learning_rate=0.5,
        init_method="logistic",
        seed=42,
    )
    gbt.train(20)

    gbt.update(df_update, update_method="gamma")

    p = gbt.predict(df_train)
    val = gbt.validate()
    acc = val["accuracy"]
    np.testing.assert_allclose(
        np.round(p[:10]).astype(int), [0, 0, 0, 0, 0, 1, 0, 0, 1, 1]
    )
    assert acc < 1.0 and acc > 0.8
