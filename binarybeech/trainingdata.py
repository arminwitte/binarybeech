#!/usr/bin/env python
# coding: utf-8
from binarybeech.extra import k_fold_split
from binarybeech.missingshandler import missings_handler_factory


class TrainingData:
    def __init__(
        self,
        df,
        y_name=None,
        X_names=None,
        handle_missings=None,
        missings_handlers=None,
    ):

        if not y_name:
            y_name = "__internal_placeholder_for_y__"
            df[y_name] = 0

        self.y_name = y_name

        if X_names is None:
            X_names = list(df.columns)
            if y_name is not None:
                X_names.remove(self.y_name)
        self.X_names = X_names

        self.df = df

        if handle_missings is not None:

            if missings_handlers is None:
                missings_handlers = missings_handler_factory.create_missings_handlers(
                    self.df, self.y_name, self.X_names, handle_missings
                )
            self.missings_handlers = missings_handlers
            self.handle_missings(df=df)

        self.data_sets = [
            (self.df, None),
        ]

    def handle_missings(self, df=None):
        if df is None:
            df = self.df

        # --------------------

        for name, missings_handler in self.missings_handlers.items():
            df = missings_handler.handle_missings(df=df)

        # --------------------

        if df is None:
            self.df = df
        return df

    def clean(self):
        # remove nan cols and rows
        self.df.dropna(inplace=True, how="all", axis=0)
        self.df.dropna(inplace=True, how="all", axis=1)

    def split(
        self, k=1, frac=None, random=False, shuffle=True, replace=True, seed=None
    ):
        sets = k_fold_split(
            self.df,
            k=k,
            frac=frac,
            random=random,
            shuffle=shuffle,
            replace=replace,
            seed=seed,
        )
        self.data_sets = sets

    def report(self):
        # first loop over y and X
        # second show pandas stats
        # - missings
        # - Pearson correlation
        pass
