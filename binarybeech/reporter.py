#!/usr/bin/env python
# coding: utf-8


class Reporter:
    def __init__(self, labels):
        self.labels = labels
        self.buffer = {}

    def set(self, **kwargs):
        self.buffer.update(kwargs)

    def print(self):
        s = ""
        for l in self.labels:
            v = self.buffer.get(l)
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
