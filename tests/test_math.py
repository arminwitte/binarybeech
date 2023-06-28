import numpy as np

import binarybeech.math as math


def test_distance_matrix():
    X = np.arange(12).reshape(4, 3)
    D = math.distance_matrix(X)

    assert D.shape == (4, 4)

    np.testing.assert_allclose(
        D,
        np.array([[0.0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]])
        * np.sqrt(27),
    )


def test_proximity_matrix():
    X = np.arange(12).reshape(4, 3)
    D = math.distance_matrix(X)
    mu = math.proximity_matrix(D)

    assert mu.shape == (4, 4)

    np.testing.assert_allclose(
        mu, np.array([[3, 2, 1, 0], [2, 3, 2, 1], [1, 2, 3, 2], [0, 1, 2, 3]]) / 3
    )


def test_valley():
    rng = np.random.RandomState(10)  # deterministic random data
    x = np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2, size=1000)))
    v = math.valley(x)

    np.testing.assert_allclose(v[0], 3.116351)


def test_shannon_entropy_histogram():
    rng = np.random.RandomState(10)  # deterministic random data
    x = np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2, size=1000)))
    H = math.shannon_entropy_histogram(x)

    assert H == -15642.544876423011
    

def test_unique_weighted():
    x = ["A"]*5 + ["B"]*4 + ["C"]*3 + ["A"]*2 + ["B"]*1
    w = np.linspace(0,1,num=len(x))
    unique, share = math.unique_weighted(x, w)
    np.testing.assert_array_equal(unique,["A","B","C"])
    print(share)
    np.testing.assert_allclose(share,[0.33333333, 0.38095238, 0.28571429])
    
def test_gini_impurity_weighted():
    x = ["A"]*5 + ["B"]*4 + ["C"]*3 + ["A"]*2 + ["B"]*1
    w = np.linspace(0,1,num=len(x))
    loss = math.gini_impurity_weighted(x, w)
    print(loss)
    np.testing.assert_allclose(loss,[0.6621315192743764])
    
def test_shannon_entropy_weighted():
    x = ["A"]*5 + ["B"]*4 + ["C"]*3 + ["A"]*2 + ["B"]*1
    w = np.linspace(0,1,num=len(x))
    loss = math.shannon_entropy_weighted(x, w)
    #print("shannon", loss)
    np.testing.assert_allclose(loss,[1.575114591410657])
    
def test_misclassification_cost_weighted():
    x = ["A"]*5 + ["B"]*4 + ["C"]*3 + ["A"]*2 + ["B"]*1
    w = np.linspace(0,1,num=len(x))
    loss = math.misclassification_cost_weighted(x, w)
    print("miss",loss)
    np.testing.assert_allclose(loss,[0.6190476190476191])
    
def test_mean_squared_error_weighted():
    x = np.linspace(0,1,50)
    y = np.linspace(0.1,1.2,50)
    w = np.linspace(0.5,5,50)
    mse = math.mean_squared_error_weighted(x,y,w)
    print("mse",mse)
    np.testing.assert_allclose(mse,[0.2833752029341517])
    
    
    
