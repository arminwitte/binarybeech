#!/usr/bin/env python
# coding: utf-8


class Reporter:
    def __init__(self):
        self.buffer = {}
        self.n = -1
        self.flight_level = 2

    def reset(self, labels):
        self.labels = labels

    def set(self, **kwargs):
        self.buffer.update(kwargs)

    def __setitem__(self, key, value):
        if key not in self.labels:
            print(f"{key} is not a registered label.")
            return
        self.buffer[key] = value

    def print(self, level=2):
        if self.flight_level > level:
            return

        self.n += 1
        if self.n == 0:
            for L in self.labels:
                print(f"{L:>12}", end=" ")
            print("")

        if self.n > 19:
            self.n = -1

        s = ""
        for L in self.labels:
            v = self.buffer.get(L)
            if v is None:
                s += "       -       "
            elif isinstance(v, float):
                s += f"{v:12.4f}"
            elif isinstance(v, int):
                s += f"{v:12d}"
            elif isinstance(v, str):
                s += f"{v:12s}"
            else:
                s += f"{str(v)[:12]:12s}"
            s += " "
        print(s)
        self.buffer = {}

    def message(self, s, level=0):
        if self.flight_level > level:
            return
        print(s)


reporter = Reporter()
