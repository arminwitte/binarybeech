import binarybeech.math as math
import numpy as np

def test_distance_matrix():
    X = np.arange(12).reshape(4,3)
    D = math.distance_matrix(X)
    
    assert D.shape == (4,4)
    
    np.testing.assert_allclose(D,np.array([[0., 1, 2, 3],[1, 0, 1, 2],[2,1,0,1],[3,2,1,0]])*np.sqrt(27))