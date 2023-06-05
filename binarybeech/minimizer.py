#!/usr/bin/env python
# coding: utf-8
import math
import random
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import scipy.optimize

INVPHISQR = (3 - math.sqrt(5)) * 0.5
EPSSQRT = math.sqrt(np.finfo(float).eps)
TINY = np.finfo(float).tiny


class Minimizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def minimize(self, f: callable, a: float, b: float) -> Tuple[float, float]:
        pass


# ====================================================================================


class BrentsScalarMinimizer(Minimizer):
    def __init__(self, atol=0, rtol=0, max_iter=100):
        self.a = None
        self.b = None
        self.e = None
        self.m = None
        self.u = None
        self.fu = None
        self.v = None
        self.fv = None
        self.w = None
        self.fw = None
        self.x = None
        self.fx = None
        self.max_iter = max_iter
        self.n_iter = None
        self.rtol = max(EPSSQRT, rtol)
        self.atol = atol
        self.tol1 = None
        self.tol2 = None

    def minimize(self, f: callable, a: float, b: float) -> Tuple[float, float]:
        self._initialize(f, a, b)
        while self._iterate():
            self.m = 0.5 * (self.a + self.b)
            self.tol1 = self.rtol * abs(self.x) + self.atol
            self.tol2 = 2 * self.tol1
            if abs(self.x - self.m) <= self.tol2 - 0.5 * (self.b - self.a):
                break

            d = self._spi()
            if not d:
                d = self._gss()

            if abs(d) >= self.tol1:
                u = self.x + d
            elif d > 0:
                u = self.x + self.tol1
            else:
                u = self.x - self.tol1

            # update
            self.e = d
            self.u = u
            self.fu = f(self.u)

            self._update()

        return self.x, self.fx

    def _update(self):
        if self.fu <= self.fx:

            if self.u < self.x:
                self.b = self.x
            else:
                self.a = self.x

            self.v = self.w
            self.fv = self.fw
            self.w = self.x
            self.fw = self.fx
            self.x = self.u
            self.fx = self.fu

        else:

            if self.u < self.x:
                self.a = self.u
            else:
                self.b = self.u

            if self.fu <= self.fw or math.isclose(self.w, self.x):
                self.v = self.w
                self.fv = self.fw
                self.w = self.u
                self.fw = self.fu
            elif (
                self.fu <= self.fv
                or math.isclose(self.v, self.x)
                or math.isclose(self.v, self.w)
            ):
                self.v = self.u
                self.fv = self.fu

    def _initialize(self, f: callable, a: float, b: float):
        self.a = min(a, b)
        self.b = max(a, b)
        self.x = self.a + INVPHISQR * (self.b - self.a)
        self.w = self.x
        self.v = self.x
        self.e = 0
        self.n_iter = 0

        self.fx = f(self.x)
        self.fw = self.fx
        self.fv = self.fx

    def _gss(self):
        if self.x < self.m:
            e = self.b - self.x
        else:
            e = self.a - self.x

        d = INVPHISQR * e
        return d

    def _spi(self):
        xw = self.x - self.w
        xv = self.x - self.v
        r = xw * (self.fx - self.fv)
        q = xv * (self.fx - self.fw)
        p = xv * q - xw * r
        q = 2 * (q - r)

        if abs(p) >= (0.5 * q * self.e):
            return None
        if abs(q) < TINY:
            return None

        d = p / q

        if abs(d) < 0.5 * self.e:
            return None

        u = self.x + d

        if u <= self.a:
            return None
        if u >= self.b:
            return None

        if (u - self.a) < self.tol2:
            d = np.sign(d) * self.tol1
        if (self.b - u) < self.tol2:
            d = np.sign(d) * self.tol1
        return d

    def _iterate(self):
        self.n_iter += 1
        if self.n_iter > self.max_iter:
            return False
        return True


# ====================================================================================


class ScalarSimulatedAnnealing(Minimizer):
    def __init__(self):
        self.init_temp = 1
        self.max_iter = 30
        self._new = None

    def minimize(self, f, a, b):
        if isinstance(a,list):
            self._new = self._choice
            if b is None:
                m = [s for s in a[: int(np.ceil(len(a) / 2))]]
            else:
                m = b
        else:
            self._new = self._rand
            a, b = min(a, b), max(a, b)
            m = (a + b) / 2
        ym = f(m)
        best = m
        ybest = ym
        current = m
        ycurrent = ym
        n_accept = 0
        # print("\tTemperature\tx\t\tnew\t\tcurrent\t\tbest")
        init_iter = max(int(math.ceil(self.max_iter * 0.1)), 5)
        main_iter = self.max_iter - init_iter
        T = 1e12
        dE = []
        for i in range(self.max_iter):
            new = self._new(current, a, b)
            ynew = f(new)
            if i < init_iter:
                dE.append(abs(ycurrent - ynew))
            elif i == init_iter:
                dE.append(abs(ycurrent - ynew))
                self.init_temp = np.mean(dE)  # dE/ln(0.8)
                # print(f"setting initial temperature to {self.init_temp}")
            else:
                T = self.init_temp * (1 - (i - init_iter) / main_iter)

            if ynew < ybest:
                best = new
                ybest = ynew
            if self._accept(ycurrent, ynew, T):
                n_accept += 1
                current = new
                ycurrent = ynew
            # if b is None:
            #     print(f"{T:15.2e}\t{new}\t{ynew:15.2e}\t{ycurrent:15.2e}\t{ybest:15.2e}")
            # else:
            #     print(f"{T:15.2e}\t{new:15.2e}\t{ynew:15.2e}\t{ycurrent:15.2e}\t{ybest:15.2e}")
        # print(f"acceptance rate: {n_accept/self.max_iter}")
        # print("")
        return (best, ybest)

    @staticmethod
    def _rand(current, a, b):
        delta = b - a
        new = a - 1
        while new < a or new > b:
            new = random.normalvariate(mu=current, sigma=delta / 6)
            # new = a + random.random() * delta
        return new

    @staticmethod
    def _choice(current, a, b):
        unique = a
        Lu = len(unique)
        r = random.random()
        L = len(current)
        size_change_probability = 0.2
        new = current.copy()
        if (r < size_change_probability and L > 1) or L == Lu:
            del new[0]
        if r > 1 - size_change_probability and L < (Lu - 1):
            new.append(None)
        # rng = np.random.default_rng()
        # new = rng.choice(unique, L, replace=False)

        pool = [s for s in unique if s not in new]

        flip_probability = 0.4
        for i, x in enumerate(new):
            r = random.random()
            if r < flip_probability or x is None:
                v = new[i]
                u = random.choice(pool)
                new[i] = u
                pool.remove(u)
                pool.append(v)

        return [s for s in new]

    @staticmethod
    def _accept(ycurrent, ynew, T):
        if ynew < ycurrent:
            return True
        # if T < 1e-12:
        #     return False
        p = ScalarSimulatedAnnealing._acceptance_probability(ycurrent, ynew, T)
        # print(f"acceptance probability: {p}")
        return random.random() < p

    @staticmethod
    def _acceptance_probability(ycurrent, ynew, T):
        if T < TINY:
            T = TINY
        return math.exp((ycurrent - ynew) / T)


class ScipyBoundedScalarMinimizer(Minimizer):
    def __init__(self):
        pass
    
    def minimize(f, a, b):
        res = scipy.optimize.minimize_scalar(f,bounds=[a,b], method="bounded")
        return res.x, res.fun







class MinimizerFactory:
    def __init__(self):
        self.minimizer = {}

    def register_minimizer(self, name, minimizer_class):
        self.minimizer[name] = minimizer_class

    def get_minimizer_class(self, name):
        return self.minimizer[name]


minimizer_factory = MinimizerFactory()
minimizer_factory.register_minimizer("brent", BrentsScalarMinimizer)
minimizer_factory.register_minimizer("simulated_annealing", ScalarSimulatedAnnealing)
minimizer_factory.register_minimizer("scipy_bounded", ScipyBoundedScalarMinimizer)


def minimize(f, a, b, method="brent"):
    M = minimizer_factory.get_minimizer_class(method)
    m = M()
    return m.minimize(f, a, b)
