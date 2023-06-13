#!/usr/bin/env python
# coding: utf-8


class Reporter:
    def __init__(self, labels):
        self.labels = labels
        self.buffer = {}
        self.n = -1

    def set(self, **kwargs):
        self.buffer.update(kwargs)

    def __setitem__(self, key, value):
        if key not in self.labels:
            print(f"{key} is not a registered label.")
            return
        self.buffer[key] = value

    def print(self):
        self.n += 1
        if self.n == 0:
            for L in self.labels:
                print(L, end=" ")
            print("")

        if self.n > 19:
            self.n = -1

        s = ""
        for L in self.labels:
            v = self.buffer.get(L)
            if v is None:
                s += " - \t"
            elif isinstance(v, float):
                s += f"{v:4.2f}\t"
            elif isinstance(v, int):
                s += f"{v:6}\t"
            elif isinstance(v, str):
                s += f"{v[:9]}\t"
            else:
                s += f"{v:10}\t"
        print(s)
        self.buffer = {}

    def message(self, s):
        print(s)
