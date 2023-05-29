from binarybeech.minimizer import BrentsScalarMinimizer, ScalarSimulatedAnnealing, minimize
import numpy as np

def test_brentsscalarminimizer():
    m = BrentsScalarMinimizer()
    x, y = m.minimize(np.sin,0.,7.)
    assert np.isclose(x, 4.7123889715201255)
    assert np.isclose(y, -1.)

def test_scalarsimulatedannealing():
    m = ScalarSimulatedAnnealing()
    x, y = m.minimize(np.sin,0.,7.)
    assert np.isclose(x, 4.7123889715201255, rtol=0.01)
    assert np.isclose(y, -1., rtol=0.01)
    
    
def test_minimize():
    x, y = minimize(np.sin,0.,7.)
    assert np.isclose(x, 4.7123889715201255)
    assert np.isclose(y, -1.)