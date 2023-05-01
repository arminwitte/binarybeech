#!/usr/bin/env python
# coding: utf-8
from binarybeech.attributehandler import attribute_handler_factory
from binarybeech.metrics import metrics_factory

class DataManager:
    def __init__(self, training_data, method, attribute_handlers):
        self.method = method
        self.attribute_handlers = {}
        
        if method is None:
            metrics_type, metrics = metrics_factory.from_data(training_data.df[training_data.y_name])
        else:
            metrics = metrics_factory.create_metrics(method)
            metrics_type = method
        self.metrics = metrics
        self.metrics_type = metrics_type
        
        if attribute_handlers is None:
            attribute_handlers = attribute_handler_factory.create_attribute_handlers(
                training_data, self.metrics
            )
        self.attribute_handlers = attribute_handlers
        self.items = self.attribute_handlers.items

    
    def __getitem__(self, key):
        return self.attribute_handlers[key]