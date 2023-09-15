#!/usr/bin/env python
# coding: utf-8
from binarybeech.datamanager import DataManager

# def test_datamanager_init():
#     pass


def test_datamanager_info():
    ah, m = DataManager.info()
    assert ah == ["default", "clustering"]
    assert m == [
        "regression",
        "regression:regularized",
        "classification:gini",
        "classification:entropy",
        "logistic",
        "classification",
        "clustering",
    ]
