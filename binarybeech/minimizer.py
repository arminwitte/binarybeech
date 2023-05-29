#!/usr/bin/env python
# coding: utf-8
import math
import random
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

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
        self.init_temp = 10
        self.max_iter = 30

    def minimize(self, f, a, b):
        a, b = min(a, b), max(a, b)
        delta = b - a
        m = (a + b) / 2
        ym = f(m)
        best = m
        ybest = ym
        current = m
        ycurrent = ym
        for i in range(self.max_iter):
            T = 1 - (i + 1) / self.max_iter
            # print(T)
            new = random.normalvariate(mu=current, sigma=delta)
            if new < a:
                new = a + (a - new)
            if new > b:
                new = b + (b - new)
            ynew = f(new)
            if ynew < ybest:
                best = new
                ybest = ynew
            if ynew < ycurrent or self._accept(ycurrent, ynew, T):
                current = new
                ycurrent = ynew
        return (best, ybest)

    @staticmethod
    def _accept(ycurrent, ynew, T):
        if T < 1e-12:
            return False
        p = math.exp((ycurrent - ynew) / T)
        return random.random() > p
    
    @staticmethod
    def _acceptance_probability(ycurrent, ynew, T):
        return math.exp((ycurrent - ynew)/ycurrent / T)