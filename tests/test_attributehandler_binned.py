import numpy as np
import pandas as pd

from binarybeech.attributehandler import BinnedAttributeHandler
from binarybeech.metrics import RegressionMetrics


def test_binned_check():
    # create continuous data and bin it into 10 bins
    N = 100
    x = np.linspace(0, 99, N)
    binned = pd.Series(pd.cut(x, bins=10, labels=False))

    assert BinnedAttributeHandler.check(binned)


def test_binned_split():
    np.random.seed(0)
    N = 200
    x_cont = np.linspace(0, 199, N)
    # bin into 8 discrete bins
    x_binned = pd.cut(x_cont, bins=8, labels=False)

    # create a target correlated with the original continuous values
    y = x_cont + np.random.normal(scale=1.0, size=N)

    df = pd.DataFrame({"y": y, "x": x_binned})

    metrics = RegressionMetrics()
    handler = BinnedAttributeHandler("y", "x", metrics, algorithm_kwargs={})

    success = handler.split(df)
    assert success is True
    assert handler.threshold is not None
    # split_df should partition the dataframe
    left, right = handler.split_df
    assert len(left) + len(right) == len(df)
    # decision function should be consistent with threshold
    if not pd.isna(df.loc[0, "x"]):
        # since binned values are integers, comparing with threshold should work
        val = df.loc[0, "x"]
        assert isinstance(handler.decide(val, handler.threshold), bool)
