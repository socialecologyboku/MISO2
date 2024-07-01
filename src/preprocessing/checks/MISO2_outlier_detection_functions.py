#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of outlier detection routines.

Created on Mon Nov  7 14:29:28 2022

@author: bgrammer

"""

import logging as logger
import numpy as np
from util.MISO2_decorators import validate_yearly_time_series_input


@validate_yearly_time_series_input
def next_value_absolute_change(df, change_threshold=0.5):
    """
    Detects row-wise outliers that change more than absolute given value between data points.

    Args:
        df(pd.DataFrame): Pandas dataframe that is checked.
        change_threshold(float): Threshold for maximum allowed absolute changes. Defaults to 0.5.

    Returns:
        indices_pos(list): List with two elements: the list of row indices and column indices where \
            outliers are located.
    """
    logger.info(f"Detecting outliers by comparing neighbours, threshold {change_threshold}")
    abs_change = df.diff(axis=1).abs()
    outliers = abs_change > abs(change_threshold)

    indices = df[outliers]
    indices_pos = np.where(indices[indices.notna()].notna())

    return indices_pos


@validate_yearly_time_series_input
def next_value_relative_change(df, relative_change_threshold):
    """
    Detects row-wise outliers that change more than absolute given value between data points.

    Args:
        df(pd.DataFrame): Pandas dataframe that is checked.
        relative_change_threshold(float): Threshold for maximum allowed absolute changes. Defaults to 0.5.

    Returns:
        indices_pos(list): List with two elements: the list of row indices and column indices where \
            outliers are located.
    """
    logger.info(f"Detecting outliers by comparing neighbours, threshold {relative_change_threshold}")
    abs_change = df.diff(axis=1).abs()
    outliers = abs_change > abs(relative_change_threshold)

    indices = df[outliers]
    indices_pos = np.where(indices[indices.notna()].notna())

    return indices_pos


@validate_yearly_time_series_input
def centered_mean_window(dataframe, windows, change_threshold):
    """
    Detects outliers that deviate from mean over given window size by the given threshold.

    Args:
        dataframe(pd.DataFrame): Pandas dataframe.
        windows(list): List of window sizes.
        change_threshold(list): List of corresponding change thresholds.

    Returns:
        outlier_indices(dict): Dictionary with indices of outliers.
    """

    if len(windows) != len(change_threshold):
        raise ValueError("Number of windows and change thresholds do not match")

    outlier_indices = {}
    for window, threshold in zip(windows, change_threshold):
        logger.info(f"Detecting outliers with rolling window {window} and threshold {threshold}")
        rolling_window = dataframe.rolling(window=window, center=True, axis=1)
        over_mean = rolling_window.mean() + threshold
        under_mean = rolling_window.mean() - threshold
        indices_over = np.where(dataframe[dataframe > over_mean].notna())
        indices_under = np.where(dataframe[dataframe < under_mean].notna())
        outlier_indices[f"abs_outlier_window_{window}_threshold_{threshold}_over"] = indices_over
        outlier_indices[f"abs_outlier_window_{window}_threshold_{threshold}_under"] = indices_under

    return outlier_indices


@validate_yearly_time_series_input
def check_values_against_limits(dataframe, min_value, max_value):
    """
    Check if any value of dataframe violates given minimum and maximum value.

    Args:
        dataframe(pd.DataFrame): Dataframe to be checked.
        max_value(float): Maximum value.
        min_value(float): Minimum value.
    """

    logger.info(f"Checking values against min {min_value} and max {max_value} limits")

    indices_below_min = np.where(dataframe[dataframe < min_value].notna())
    indices_above_max = np.where(dataframe[dataframe > max_value].notna())

    return indices_below_min, indices_above_max
