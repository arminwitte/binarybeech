#!/usr/bin/env python
# coding: utf-8
from binarybeech.attributehandler import attribute_handler_factory
from binarybeech.extra import k_fold_split
from binarybeech.metrics import metrics_factory


class TrainingData:
    def __init__(
        self,
        df,
        y_name=None,
        X_names=None,
        attribute_handlers=None,
        metrics_type=None,
        handle_missings="simple",
    ):
        self.y_name = y_name

        if X_names is None:
            X_names = list(df.columns)
            if y_name is not None:
                X_names.remove(self.y_name)
        self.X_names = X_names

        if metrics_type is None:
            metrics_type, metrics = metrics_factory.from_data(df[self.y_name])
        else:
            metrics = metrics_factory.create_metrics(metrics_type)
        self.metrics_type = metrics_type
        self.metrics = metrics

        if attribute_handlers is None:
            attribute_handlers = attribute_handler_factory.create_attribute_handlers(
                df, y_name, X_names, self.metrics
            )
        self.attribute_handlers = attribute_handlers

        self.df = df

    def handle_missings(self, mode):
        df = self.df
        if self.y_name is not None:
            df = df.dropna(subset=[self.y_name])

        if mode == "simple":
            # use nan as category
            # use mean if numerical
            for name, dh in self.attribute_handlers.items():
                df = dh.handle_missings(df)
        elif mode == "model":
            raise ValueError("Not implemented")

        self.df = df

        return df

    def clean(self):
        # remove nan cols and rows
        pass

    def split(self):
        pass

    def report(self):
        # first loop over y and X
        # second show pandas stats
        pass
