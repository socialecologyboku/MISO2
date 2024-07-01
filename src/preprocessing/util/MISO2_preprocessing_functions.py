#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:30:51 2022

Util data preparation functions.

@author: bgrammer

"""
import logging as logger


def check_value_difference(df1, df2, column_name, parameter_name):
    """
    Checks differences in values between a shared column of two dataframes.
    """
    # names of this function need to be generalized
    df1_values = set(df1[column_name])
    df2_values = set(df2[column_name])

    missing_df1 = df1_values.difference(df2_values)
    missing_df2 = df2_values.difference(df1_values)

    if missing_df1:
        logger.warning(f"Parameter {parameter_name} column {column_name} missing in \
                       uncertainty: {missing_df1}")
    if missing_df2:
        logger.warning(f"Parameter {parameter_name} column {column_name} missing in \
                       original: {missing_df2}")


def log_values_by_index(df, indices, mode):
    """
    Logs values from dataframe index by index with level warning

    Primarily meant for exact location of outliers.

    Args:
        df(pd.DataFrame): Dataframe to print values from.
        indices(tuple): Tuple holding a list of row and column indices each.
        mode(str): Explanation of how values were gathered.
    """
    for row_index, col_index in zip(indices[0], indices[1]):
        row_name = df.index[row_index]
        col_name = df.columns[col_index]
        logger.warning(f"Value with ({mode}) found at index: \
                       {row_name} {col_name}: {df.iloc[row_index, col_index]}")
