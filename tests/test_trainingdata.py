# import numpy as np
import pandas as pd
import pytest

from binarybeech.trainingdata import TrainingData


@pytest.fixture
def df_iris():
    return pd.read_csv("data/iris.csv")


def test_training_data(df_iris):
    data = TrainingData(df=df_iris, y_name="species")
    assert isinstance(data, TrainingData)


def test_training_data_split(df_iris):
    data = TrainingData(df=df_iris, y_name="species")
    data.split(k=5)
    assert len(data.data_sets) == 5
    assert len(data.data_sets[0]) == 2
