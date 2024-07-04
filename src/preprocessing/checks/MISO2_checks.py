#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of value checks.


@author: bgrammer

"""

import logging as logger

import numpy as np
from util.MISO2_exceptions import EnduseSummationError, ValueOverlapError, WasteRateError


def check_mutually_exclusive_arrays(values, mode="normal"):
    """
    Given values, check that their value holding (> 0) indices are mutually exclusive.
    In normal mode, this compares the list in successive order. In other modes, a dict with specified keywords
    is expected.

    Args:
        values(list/dict): List of tuples holding array name and np.array
        (normal mode) or dict (downcycling mode)
        mode(str): Mode for comparison. One of "normal" (values may not overlap) or "downcycling"

    Raises:
         ValueOverlapError: If mutually exclusive values are found.
    """
    logger.info("Checking that arrays are mutually exclusive")

    if mode == "downcycling":
        # either rates are set for asphalt, bricks, concrete or absolutes for aggr_downcycle
        # in any given year for any given region
        materials_to_check = ["asphalt", "bricks", "concrete"]
        material_positions = values["material_position"]
        rate_indices = [material_positions[material] for material in materials_to_check]
        absolute_indices = [material_positions["aggr_downcycl"]]

        absolutes = values["absolute"][:, :, absolute_indices, :]
        rates = values["rate"][:, :, rate_indices, :]
        any_rates_set = np.any(rates, axis=2)
        any_absolutes_set = np.any(absolutes, axis=2)

        if np.logical_and(any_absolutes_set, any_rates_set).any():
            raise ValueOverlapError(f"Found overlapping values in exclusive absolute and rate ({materials_to_check}) downcycling")

        #if not np.logical_or(all_rates_set, absolutes_set).all():
        #    raise AttributeError("Missing downcycling rates, either rate or absolute must be set")

    else:
        values_set = np.zeros(values[0][1].shape, dtype=bool)
        for array_tuple in values:
            array = array_tuple[1]
            if np.logical_and(array, values_set).any():
                raise ValueOverlapError(f"Found overlapping values in mutually exclusive arrays: {array_tuple[0]}")
            values_set = np.logical_or(values_set, array)
    return True


def check_enduse_values(enduses, enduse_selectors, non_enduse_selectors, error_tolerance=None):
    """
    Checks validity of MISO2 Enduse parameter.

    Enduses must sum up to exactly 1 for non-aggregate materials and must be zero for aggregates.

    Args:
        enduses(msc.Parameter): ODYM-parameter of enduses
        enduse_selectors(list): Indices of materials for which enduses need to sum up to 1.
        non_enduse_selectors(list): Indices of materials that need to sum up to 0.
        error_tolerance(float): Value by which enduse sums may diverge from 1.
            This defaults to machine epsilon (float) times the number of enduses.

    Raises:
        EnduseSummationError: When validity of enduses is violated.

    """
    logger.info(f"Checking enduse shares properly sum up. Enduse selectors: {enduse_selectors}"
                f"non enduses: {non_enduse_selectors}, error tolerance {error_tolerance}")
    enduse_values = enduses.Values
    m_index = enduses.Indices.find('g')

    enduse_sums_expected_one = np.sum(enduse_values[:, :, enduse_selectors, :, :-1], axis=m_index)
    enduse_sums_expected_zero = np.sum(enduse_values[:, :, non_enduse_selectors, :, :-1], axis=m_index)

    enduse_sums = np.sum(enduse_values[:, :, :, :, :-1], axis=m_index)
    expected_sums = np.zeros_like(enduse_sums)
    expected_sums[:, :, enduse_selectors, :] = 1

    if error_tolerance is None:
        no_enduses = enduse_values.shape[m_index]
        error_tolerance = no_enduses*np.finfo(float).eps
    # we exclude last year values
    exceed_one = np.transpose(np.nonzero(enduse_sums_expected_one > 1+error_tolerance))
    below_one = np.transpose(np.nonzero(enduse_sums_expected_one < 1-error_tolerance))

    if exceed_one.size > 0:
        error_msg = f"Enduse shares sums exceed one by error tolerance for {exceed_one.size} entries"
        logger.error(exceed_one)
        logger.error(error_msg)
        raise EnduseSummationError(error_msg)
    if below_one.size > 0:
        error_msg = f"Enduse shares sums below one by error tolerance for {below_one.size} entries"
        logger.error(enduse_sums_expected_one)
        logger.error(enduse_sums_expected_one[enduse_sums_expected_one != 1])
        logger.error(np.where(enduse_sums_expected_one[enduse_sums_expected_one < 1-error_tolerance]))
        logger.error(error_msg)
        raise EnduseSummationError(error_msg)

    non_zeros = np.transpose(np.nonzero(~np.isclose(enduse_sums_expected_zero, 0)))
    if non_zeros.size > 0:
        error_msg = f"Enduse shares which should be zero are not {non_zeros.size} entries"
        logger.error(error_msg)
        raise EnduseSummationError(error_msg)

    return True


def check_exclusive_values(parameter_dict, exclusive_parameters):
    """
    Checks for two parameters files if their values are mutually exclusive.

    When an array has a value set greater than zero, it must be zero in the other array and vice versa.

    Args:
        parameter_dict(dict): Dict of ODYM parameters.
        exclusive_parameters(list): List of tuples that give the name of the exclusive parameters.
    """

    logger.info("Checking exclusive values")
    for array_names in exclusive_parameters:
        try:
            array1 = parameter_dict[array_names[0]].Values
            array2 = parameter_dict[array_names[1]].Values

            non_exclusive_elements = (np.where(array2[np.where(array1 > 0)] > 0))
            non_exclusive = len(non_exclusive_elements[0])
            if non_exclusive:
                raise ValueOverlapError(f"Found overlapping values in Total and EoL parameter dicts {array_names}")
        except KeyError as e:
            logger.exception(e)
            logger.error(f"Could not find parameter in parameter dict: {array_names} \
                           The data may be missing intentionally")


def check_limits(parameter_dict, uncertainty_settings):
    """
    Checks if all parameter values are within the specified hard limits.

    Args:
        parameter_dict(dict): A dictionary of ODYM Parameters containing src WasteRates.
        uncertainty_settings(dict): A dictionary with parameter names as keys, \
            containing dictionaries of Min,Max and Type with their respective values.

    Raises:
        InputLimitError: When any parameter value is below the given minimum or above the given maximum limit.
    """

    logger.info("Checking if input data is within value limits")
    for k, v in parameter_dict.items():
        if k in uncertainty_settings:
            values = v.Values
            min_limit = uncertainty_settings[k]["Min"]
            max_limit = uncertainty_settings[k]["Max"]

            invalid_condition = np.where((values < min_limit) | (values > max_limit))
            invalid = values[invalid_condition]

            if len(invalid) > 0:
                error_msg = f"Invalid values found in parameter {k}: {invalid}"
                logger.error(error_msg)
                # raise InputLimitError(error_msg)
                # turned off for debugging
        else:
            logger.warning(f"Parameter {k} not found in parameter limits dictionary")


def check_waste_ratios(parameter_dict):
    """
    Checks MISO2 waste rate ratios of recovered and unrecovered and throws a
    WasteRateException when they exceed 100%. This likely happens in the context
    of Monte-Carlo randomization.

    Args:
        parameter_dict(dict): A dictionary of ODYM Parameters containing src WasteRates.
    Raises:
        WasteRateError: When the sum of recov and unrecov values of a process exceed 1.
    """

    p5_ratios = parameter_dict["MISO2_WasteRate_recov_p5"].Values + \
        parameter_dict["MISO2_WasteRate_unrecov_p5"].Values

    p5_ratios = p5_ratios[p5_ratios > 1]

    if np.any(p5_ratios):
        error_msg = f"p5 waste rate exceed 1: {p5_ratios}"
        raise WasteRateError(error_msg)

    p7_ratios = parameter_dict["MISO2_WasteRate_recov_p7"].Values + \
        parameter_dict["MISO2_WasteRate_unrecov_p7"].Values

    p7_ratios = p7_ratios[p7_ratios > 1]

    if np.any(p7_ratios):
        error_msg = f"p7 waste rate exceed 1: {p7_ratios}"
        raise WasteRateError(error_msg)

    p9_ratios = parameter_dict["MISO2_WasteRate_recov_p9"].Values + \
        parameter_dict["MISO2_WasteRate_unrecov_p9"].Values

    p9_ratios = p9_ratios[p9_ratios > 1]

    if np.any(p9_ratios):
        error_msg = f"p9 waste rate exceed 1: {p9_ratios}"
        raise WasteRateError(error_msg)


def check_buffer_year(parameter_dict):
    """
    Check that last year of input data is an all-zero buffer year.

    To correctly calculate mass-balances, we need to have a buffer year at the end of our input time series.
    This is usually achieved by simply setting the model runtime in ODYM's master classification to
        include an additional year.

    Args:
        parameter_dict(dict): Dictionary of ODYM parameters

    Raises:
        ValueError: If no buffer year is provided.
    """

    for param_name, param in parameter_dict.items():
        if not (param.Values[..., -1] == 0).all():
            error_msg = f"No empty buffer year at end of input data found for parameter {param_name} \
                Make sure that the model runtime (in the Master classification file) is set to be one year longer \
                    than the actual input data to create such a buffer year."
            logger.error(error_msg)
            raise ValueError(error_msg)
