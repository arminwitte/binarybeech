from binarybeech.brentsscalarminimizer import BrentsScalarMinimizer
import numpy as np

def test_node():
    m = BrentsScalarMinimizer()
    x, y = m.minimize(np.sin,0.,7.)
    assert np.isclose(x, 4.7123889715201255)
    assert np.isclose(y, -1.)