import numpy as np

from binarybeech.minimizer import (
    BrentsScalarMinimizer,
    NewtonRaphsonStepMinimizer,
    ScalarSimulatedAnnealing,
    minimize,
)


def test_brentsscalarminimizer():
    m = BrentsScalarMinimizer()
    x, y = m.minimize(np.sin, 0.0, 7.0)
    assert np.isclose(x, 4.7123889715201255)
    assert np.isclose(y, -1.0)


def test_scalarsimulatedannealing():
    m = ScalarSimulatedAnnealing()
    x, y = m.minimize(np.sin, 0.0, 7.0)
    assert np.isclose(x, 4.7123889715201255, rtol=2 / m.max_iter)
    assert np.isclose(y, -1.0, rtol=2 / m.max_iter)


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
    assert y < 4
    assert isinstance(x, list)
    # assert np.assert_array_equal(sorted(x), ["22", "44", "77"])


def test_minimize():
    x, y = minimize(np.sin, 0.0, 7.0)
    x1, y1 = minimize(
        np.sin, 0.0, 7.0, method="scipy_bounded", options={"max_iter": 100}
    )
    assert np.isclose(x, 4.7123889715201255)
    assert np.isclose(y, -1.0)
    assert np.isclose(y, y1)


def test_newton_raphson_step():
    # Simple case: 3-class problem, class 0
    # pred = tree predictions, r = residuals, p = probabilities
    pred = np.array([1.0, 0.5, -0.3, 0.8])
    p = np.array([0.3, 0.4, 0.2, 0.5])
    r = np.array([0.7, 0.1, -0.2, 0.5])  # Y_k - P_k

    m = NewtonRaphsonStepMinimizer()
    gamma, _ = m.minimize({"pred": pred, "r": r, "p": p}, 0.1, 10.0)

    # manually compute expected
    num = np.sum(pred * r)
    den = np.sum(pred**2 * p * (1.0 - p)) + 1e-10
    expected = float(np.clip(num / den, 0.1, 10.0))
    assert np.isclose(gamma, expected)


def test_newton_raphson_step_weighted():
    pred = np.array([1.0, 0.5, -0.3, 0.8])
    p = np.array([0.3, 0.4, 0.2, 0.5])
    r = np.array([0.7, 0.1, -0.2, 0.5])
    w = np.array([1.0, 2.0, 1.0, 0.5])

    m = NewtonRaphsonStepMinimizer()
    gamma, _ = m.minimize({"pred": pred, "r": r, "p": p, "weights": w}, 0.1, 10.0)

    num = np.sum(w * pred * r)
    den = np.sum(w * pred**2 * p * (1.0 - p)) + 1e-10
    expected = float(np.clip(num / den, 0.1, 10.0))
    assert np.isclose(gamma, expected)


def test_newton_via_factory():
    # Verify the newton method is accessible via minimize()
    data = {"pred": np.ones(5), "r": np.ones(5) * 0.5, "p": np.ones(5) * 0.3}
    gamma, _ = minimize(data, 0.1, 10.0, method="newton")
    assert 0.1 <= gamma <= 10.0
