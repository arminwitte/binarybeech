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
    assert np.isclose(x, 4.7123889715201255, rtol=0.0001)
    assert np.isclose(y, -1., rtol=0.0001)

def test_scalarsimulatedannealing_choice():
    m = ScalarSimulatedAnnealing()
    m._new = ScalarSimulatedAnnealing._choice
    def f(tup):
        n = 0
        for s in tup:
            n += len(s)
        return n
    x, y = m.minimize(f,["1","22","333"],None)
    assert np.isclose(y, 1, rtol=0.0001)
    assert 1 == 2
    
    
def test_minimize():
    x, y = minimize(np.sin,0.,7.)
    assert np.isclose(x, 4.7123889715201255)
    assert np.isclose(y, -1.)