#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outlier detection routines that can be run independently of input creation.

@author: bgrammer
"""

import logging as logger
from preprocessing.checks.MISO2_outlier_detection_functions import next_value_absolute_change
from preprocessing.input_creation.MISO2_multipliers_input_creator import MISO2MultipliersInputCreator
from preprocessing.input_creation.MISO2_recycling_input_creator import MISO2RecyclingInputCreator
from preprocessing.util.MISO2_preprocessing_functions import log_values_by_index


class MISO2OutlierDetector:
    def __init__(self, miso_file_manager):
        self.miso_file_manager = miso_file_manager

    def outlier_detection_by_parameter(self):
        """
        Returns an outlier detection function per parameter name.

        May contain "None" for unspecified functions. Any function that follows the outlier detection interface
        (pd.DataFrame, parameter_name, bool) may be called.

        Returns:
             functions(dict): Dictionary with parameter as key and an outlier detection function as value.
        """
        rate_recycling = MISO2RecyclingInputCreator(self.miso_file_manager).outlier_detection
        multipliers = MISO2MultipliersInputCreator(self.miso_file_manager).outlier_detection

        functions = {
            "MISO2_Lifetimes": None,
            "MISO2_EoLRateRecycling": rate_recycling,
            "MISO2_EoLRateDowncycling": rate_recycling,
            "MISO2_WasteRate_recov_p5": rate_recycling,
            "MISO2_WasteRate_unrecov_p5": rate_recycling,
            "MISO2_WasteRate_recov_p7": rate_recycling,
            "MISO2_WasteRate_unrecov_p7": rate_recycling,
            "MISO2_WasteRate_recov_p9": rate_recycling,
            "MISO2_WasteRate_unrecov_p9": rate_recycling,
            "MISO2_Production_p3_total": None,
            "MISO2_Production_p3_primary": None,
            "MISO2_Import_p4": None,
            "MISO2_Export_p4": None,
            "MISO2_Import_p6": None,
            "MISO2_Export_p6": None,
            "MISO2_Import_p8": None,
            "MISO2_Export_p8": None,
            "MISO2_Import_p12": None,
            "MISO2_Export_p12": None,
            "MISO2_Lifetimes_deviation": None,
            "MISO2_RoadMultiplier": multipliers,
            "MISO2_BuildingsMultiplier": multipliers,
            "MISO2_EoLAbsoluteRecycling": None,
            "MISO2_EndUseShares": enduse_outlier_detection,
            "MISO2_EoLAbsoluteDowncycling": None,
            "MISO2_TotalAbsoluteRecycling": None,
            "MISO2_TotalRateRecycling": rate_recycling,
        }
        return functions

    def detect_outliers(self, parameters=None):
        """
        Runs specified outlier detection routines for specified parameters or all parameters that file manager can load.

        Args:
            parameters(list): List of parameters. If None (default), all parameters will be checked.

        Returns:
            dfs_by_parameter(dict): Dict with parameter name as key and pd.DataFrame as value.
            outliers(dict): Dict with parameter name as key and outlier indices as value.
        """

        outlier_detection_functions = self.outlier_detection_by_parameter()
        outlier_detection_functions_active = {}

        if parameters is None:
            outlier_detection_functions_active = outlier_detection_functions
        else:
            for parameter in parameters:
                outlier_detection_functions_active[parameter] = outlier_detection_functions[parameter]

        outliers = {}
        dfs_by_parameter = {}

        for parameter, df in outlier_detection_functions_active.items():
            detection_function = outlier_detection_functions[parameter]

            if detection_function:
                df = self.miso_file_manager.load_parameters(parameter)[parameter]
                outlier_indices = detection_function(
                    df=df, parameter=parameter, print_outliers=True)
                outliers[parameter] = outlier_indices
                dfs_by_parameter[parameter] = df
            else:
                logger.warning(f"No outlier detection set for {parameter}")

        return dfs_by_parameter, outliers


def enduse_outlier_detection(df, parameter="Enduses", print_outliers=True):
    """
    Outlier detection wrapper for enduses.

    Args:
        df(pd.DataFrame): Dataframe to be checked.
        parameter(str): Name of parameter
        print_outliers(bool): If outliers should be logged as warning. Defaults to true.

    Returns:
        outlier_indices(dict): Dictionary with method of outlier detection as key and indices as values.
    """
    logger.info(f"Starting outlier detection routine for {parameter}. Print outliers set to {print_outliers}")
    outlier_indices = {}
    change_threshold = 0.3
    # maximum allowed absolute value change between two years

    absolute_value_change_outliers = next_value_absolute_change(df, change_threshold)

    if print_outliers:
        log_values_by_index(df, absolute_value_change_outliers, "absolute change threshold exceeded")

    return outlier_indices
