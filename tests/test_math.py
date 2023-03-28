import binarybeech.math as math
import numpy as np

def test_distance_matrix():
    X = np.arange(12).reshape(4,3)
    D = math.distance_matrix(X)
    
    assert D.shape == (4,4)
    np.testing.assert_allclose(D,[[0., 5., 10., 15.]]*4)