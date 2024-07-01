#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of decorators.

@author: bgrammer
"""
import functools

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


def validate_yearly_time_series_input(func):
    """
    Wrapper that validates if the "dataframe" argument of a function is a numeric dataframe with consecutively labelled
    yearly columns.

    Args:
        func(function): Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        dataframe = kwargs.get("dataframe", args[0])

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a Pandas DataFrame")

        num_cols = dataframe.select_dtypes(include='number').columns
        if not dataframe.columns.equals(num_cols):
            raise ValueError(
                "Passed a dataframe with non-numeric columns to a function that expects only numeric input")
        # check if columns themselves only contain numbers

        if not is_numeric_dtype(dataframe.columns):
            raise ValueError(
                "Passed non-numeric column header to function which requires yearly time intervals in columns"
            )

        years = np.arange(dataframe.columns.min(), dataframe.columns.max()+1)
        if not np.array_equal(dataframe.columns.values, years):
            raise ValueError("Dataframe should be a yearly time-series, but column names could not be related to years")

        return func(*args, **kwargs)
    return wrapper
