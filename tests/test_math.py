import binarybeech.math as math
import numpy as np

def test_distance_matrix():
    X = np.arange(12).reshape(4,3)
    D = math.distance_matrix(X)
    
    assert D.shape == (4,4)
    
    np.testing.assert_allclose(D,np.array([[0., 1, 2, 3],[1, 0, 1, 2],[2,1,0,1],[3,2,1,0]])*np.sqrt(27))
    
def test_proximity_matrix():
    X = np.arange(12).reshape(4,3)
    D = math.distance_matrix(X)
    mu = math.proximity_matrix(D)
    
    assert mu.shape == (4,4)
    
    np.testing.assert_allclose(mu,np.array([[3, 2, 1, 0],[2, 3, 2, 1],[1,2,3,2],[0,1,2,3]])/3)
    
def test_valley():
    rng = np.random.RandomState(10)  # deterministic random data
    x = np.hstack((rng.normal(size=1000),
                   rng.normal(loc=5, scale=2, size=1000)))
    v = math.valley(x)
    
    np.testing.assert_allclose(v[0],2.5417373)
    
def test_shannon_entropy_histogram():
    rng = np.random.RandomState(10)  # deterministic random data
    x = np.hstack((rng.normal(size=1000),
                   rng.normal(loc=5, scale=2, size=1000)))
    H = math.shannon_entropy_histogram(x)
    
    assert H == 12.
    