import pandas as pd

from binarybeech.attributehandler import HighCardinalityNominalAttributeHandler
from binarybeech.binarybeech import CART


def test_adult():
    adult_train = pd.read_csv("data/adult_data.csv", header=None)
    adult_test = pd.read_csv("data/adult_test.csv", skiprows=1, header=None)
    c = CART(df=adult_train, y_name=14, max_depth=4, method="classification")
    c.create_tree()
    val = c.validate(adult_test)
    assert val["accuracy"] > 0.82
    assert isinstance(
        c.dmgr.attribute_handlers[1], HighCardinalityNominalAttributeHandler
    )
