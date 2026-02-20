import numpy as np
import pandas as pd

from binarybeech.utils import model_missings


def test_model_missings_simple():
    # Create simple dataset where B is sum of A and C
    df = pd.DataFrame({
        "A": [1.0, 2.0, 3.0, 4.0],
        "B": [2.0, 4.0, np.nan, 8.0],
        "C": [1.0, 2.0, 3.0, 4.0],
    })
    df_filled = model_missings(df, y_name="B", X_names=["A", "C"], cart_settings={})
    # Missing value should be filled (not NaN)
    assert not pd.isna(df_filled.loc[2, "B"]) 
    # The filled value should be numeric
    assert isinstance(df_filled.loc[2, "B"], (int, float, np.floating))
