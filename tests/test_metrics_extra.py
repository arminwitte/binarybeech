import numpy as np
import pandas as pd

from binarybeech.binarybeech import CART
from binarybeech.metrics import (
    RegressionMetricsRegularized,
    LogisticMetrics,
    ClassificationMetrics,
    metrics_factory,
)


def test_regression_regularized_node_value():
    y = np.array([1.0, -2.0, 0.5])
    rm = RegressionMetricsRegularized()
    # without regularization should be sum/n
    val = rm.node_value(y)
    assert isinstance(val, float)

    # test with l1/l2 producing shrinkage
    val2 = rm.node_value(y, lambda_l1=0.5, lambda_l2=1.0)
    assert isinstance(val2, float)


def test_logistic_confusion_and_transforms():
    lm = LogisticMetrics()
    y = np.array([0, 1, 1, 0])
    probs = np.array([0.2, 0.8, 0.6, 0.1])
    cm = lm._confusion_matrix(y, probs)
    # y_true: [0,1,1,0], preds round clipped -> [0,1,1,0]
    assert cm.shape == (2, 2)
    assert cm[0, 0] == 2
    assert cm[1, 1] == 2

    # transforms
    arr = np.array([0.0, 1.0])
    out = lm.output_transform(arr)
    inv = lm.inverse_transform(out)
    # inverse_transform(output_transform(x)) should be close to x after transformations
    np.testing.assert_allclose(inv, arr, rtol=1e-6, atol=1e-6)


def test_classification_bins_and_confusion():
    cm = ClassificationMetrics()
    df = pd.DataFrame({
        "y": [0, 0, 1, 1, 2, 2],
        "a": ["x", "x", "y", "y", "z", "z"],
    })
    bins = cm.bins(df, "y", "a")
    # bins should partition categories into two lists
    assert isinstance(bins, list) and len(bins) == 2
    conf = cm._confusion_matrix(df["y"].values, df["y"].values)
    # perfect prediction -> diagonal equals counts
    assert np.sum(np.diag(conf)) == np.sum(conf)


def test_metrics_factory_from_data():
    y = np.array([1.0, 2.0, 3.0])
    m, name = metrics_factory.from_data(y, {})
    assert name in metrics_factory.metrics
    assert m is not None
