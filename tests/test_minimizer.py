from binarybeech.minimizer import (
    BrentsScalarMinimizer,
    ScalarSimulatedAnnealing,
    minimize,
)
import numpy as np


def test_brentsscalarminimizer():
    m = BrentsScalarMinimizer()
    x, y = m.minimize(np.sin, 0.0, 7.0)
    assert np.isclose(x, 4.7123889715201255)
    assert np.isclose(y, -1.0)


def test_scalarsimulatedannealing():
    m = ScalarSimulatedAnnealing()
    x, y = m.minimize(np.sin, 0.0, 7.0)
    assert np.isclose(x, 4.7123889715201255, rtol=1 / m.max_iter)
    assert np.isclose(y, -1.0, rtol=1 / m.max_iter)


def test_scalarsimulatedannealing_choice():
    m = ScalarSimulatedAnnealing()
    # m._new = ScalarSimulatedAnnealing._choice

    def f(tup):
        n = 0
        for s in tup:
            L = len(s)
            if L == 2:
                n -= L
            else:
                n += L
        return n

    m.max_iter = 100
    x, y = m.minimize(f, ["1", "22", "333", "44", "5", "6", "77", "888", "9999"], None)
    assert y < 0
    assert isinstance(x, list)
    # assert np.assert_array_equal(sorted(x), ["22", "44", "77"])


def test_minimize():
    x, y = minimize(np.sin, 0.0, 7.0)
    assert np.isclose(x, 4.7123889715201255)
    assert np.isclose(y, -1.0)
