#!/usr/bin/env python
# coding: utf-8
from binarybeech.attributehandler import attribute_handler_factory
from binarybeech.metrics import metrics_factory


class DataManager:
    def __init__(self, training_data, method, attribute_handlers, algorithm_kwargs):
        self.algorithm_kwargs = algorithm_kwargs

        if method is None:
            method, metrics = metrics_factory.from_data(
                training_data.df[training_data.y_name], self.algorithm_kwargs
            )
        else:
            metrics = metrics_factory.create_metrics(method, self.algorithm_kwargs)
        self.metrics = metrics
        self.method = method

        if attribute_handlers is None:
            attribute_handlers = attribute_handler_factory.create_attribute_handlers(
                training_data,
                self.metrics,
                self.method,
                self.algorithm_kwargs,
            )
        self.attribute_handlers = attribute_handlers
        self.items = self.attribute_handlers.items

    def __getitem__(self, key):
        return self.attribute_handlers[key]

    @staticmethod
    def info():
        ah = [k for k in attribute_handler_factory.attribute_handlers.keys()]
        m = [k for k in metrics_factory.metrics.keys()]
        return ah, m
