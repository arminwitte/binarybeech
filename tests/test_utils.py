import pandas as pd

from binarybeech.utils import model_missings


def test_model_missings():
    df_titanic = pd.read_csv("data/titanic.csv")

    df_new = model_missings(df_titanic, "Survived", X_names=["Age"])

    has_missings = df_new.isnull().any()
    assert not has_missings["Age"]
