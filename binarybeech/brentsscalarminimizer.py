#!/usr/bin/env python
# coding: utf-8
import math
from operator import itemgetter


class BrentsScalarMinimizer:
    def __init__(self,tol=1e7):
        self.a = None
        self.b = None
        self.c = None
        self.d = None
        self.x = None
        self.uvw = None
        self.fun = None
        self.tol = tol
        self.iter = None

    invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
    invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2

    def minimize(self, f, a, b):
        self._initialize(f, a, b)
        while self._iterate():
            #print(self.h)
            #print(self.uvw[0])
            #print(self.a, self.c)
            x = self._para()
            #print(x)
            if self._accept(x):
                #print(self.iter, "para")
                self.uvw[3] = x
            else:
                #print(self.iter, "gss")
                x_ = self._gss()
                if x_:
                    self.uvw[3] = x_
            self.uvw = sorted(self.uvw, key=itemgetter(1))
            #print(self.uvw[0])
        
        return self.uvw[0]
   
    def _initialize(self, f, a, b):
        a, b = min(a, b), max(a, b)
        ya = f(a)
        yb = f(b)
        self.a = (a, ya)
        self.b = (b, yb)
        self.h = b - a
        self.delta = self.h
        c = self.a[0] + self.invphi2 * self.h
        yc = f(c)
        self.c = (c, yc)
        d = self.a[0] + self.invphi * self.h
        yd = f(d)
        self.d = (d, yd)
        self.fun = f
        self.uvw = sorted([self.a,self.b,self.c,self.d],key = itemgetter(1))
        self.x = self.uvw[0]
        self.iter = 0
        
    def _gss(self):
        f = self.fun
        
        if not self.d:
            d = self.a[0] + self.invphi * self.h
            yd = f(d)
            self.d = (d, yd)
            self.uvw[3] = self.d

        if self.c[1] < self.d[1]:  # yc > yd to find the maximum
            self.b = self.d
            self.d = self.c
            self.h *= self.invphi
            c = self.a[0] + self.invphi2 * self.h
            yc = f(c)
            self.c = (c, yc)
            if self.c[1] < self.d[1]:
                return self.c
        else:
            self.a = self.c
            self.c = self.d
            self.h *= self.invphi
            d = self.a[0] + self.invphi * self.h
            yd = f(d)
            self.d = (d, yd)
            if self.d[1] < self.c[1]:
                return self.d
        return None

        #if yc < yd:
        #    return (a, d)
        #else:
        #    return (c, b)
    
    def _para(self):
        f = self.fun
        #u, v, w = sorted(self.uvw[:3], key= lambda x: x[0])
        u, v, w = self.uvw[:3]
        x = v[0] - 0.5 * (
            ((v[0]-u[0])**2*(v[1]-w[1]) - (v[0]-w[0])**2*(v[1]-u[1]))/
            ((v[0]-u[0])   *(v[1]-w[1]) - (v[0]-w[0])   *(v[1]-u[1]))
        )
        yx = f(x)
        return (x, yx)
    
    def _accept(self, x):
        if x[1] > self.x[1]:
            #print("not smaller")
            return False
        if x[0] < self.a[0]:
            #print("left from bounds")
            return False
        if x[0] > self.b[0]:
            #print("right from bounds")
            return False
        delta = abs(x[0] - self.x[0])
        if delta > self.invphi * self.delta:
            #print("supralinear")
            return False
        self.delta = delta
        return True
    
    def _iterate(self):
        if self.h < self.tol:
            #print("small interval")
            return False
        if self.delta < self.tol:
            #print("small step")
            return False
        self.iter += 1
        if self.iter > 100:
            return False
        if abs(self.x[1] - self.uvw[2][1]) < self.tol:
            return False
        self.x = self.uvw[0]
        return True
        
        