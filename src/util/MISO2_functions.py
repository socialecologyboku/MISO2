#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of helper methods used throughout the src code.

Created on Wed Jul  6 09:46:18 2022
@author: bgrammer
"""

import logging as logger
import numpy as np
import numba as nb
import odym.ODYM_Classes as msc


def set_arrays_exclusive_in_order(array_tuple_list):
    """
    Set values overlapping by index to zero in successive order.

    Args:
        array_tuple_list(list): List of tuples with (name, np.ndarray) format
    """

    logger.info(f"Setting arrays to non-overlapping, starting with {array_tuple_list[0][0]}")
    combined_mask = np.zeros(array_tuple_list[0][1].shape, dtype=bool)
    for i, array_tuple in enumerate(array_tuple_list[0:-1]):
        array = array_tuple[1]
        new_mask = array > 0
        combined_mask = (combined_mask == 1) | (new_mask == 1)
        array_tuple_list[i+1][1][combined_mask] = 0


def copy_and_subset_parameter_dict(parameter_dict, nr_start, nr_stop):
    """
    Creates a copy of a Parameter dictionary subset by the region index.

    Args:
        parameter_dict(dict): A dictionary of ODYM Parameter objects.
        nr_start(int): Start region index.
        nr_stop(int): Stop region index. As usual with indexing, the subset will NOT include this row.

    Returns:
        parameter_dict_copy(dict): A copy of the original parameter dict, subset by the region index.
    """

    parameter_dict_copy = {}
    for k, v in parameter_dict.items():

        values_copy = np.copy(v.Values[nr_start:nr_stop, ...])

        uncert_copy_list = None
        if v.Uncert is not None:
            uncert_copy = np.array(v.Uncert).reshape(v.Values.shape)[nr_start:nr_stop, ...]
            uncert_copy_list = uncert_copy.flatten().tolist()
            # to list implicit copy call

        new_param = msc.Parameter(Name=v.Name, ID=v.ID, UUID=v.UUID,
                                  P_Res=v.P_Res, MetaData=v.MetaData, Indices=v.Indices,
                                  Values=values_copy,
                                  Uncert=uncert_copy_list)
        parameter_dict_copy[k] = new_param
    return parameter_dict_copy


@nb.jit(nopython=True)
def nb_any(array):
    """
    Short circuit version of np.any.
    Returns True if an array contains a null value. Note that np.nan counts as a value.

    Args:
        array (np.ndarray): Array to be checked
    Returns:
        bool:
    """
    for x in array.flat:
        if x:
            return True
    return False


def compare_parameter_dict(parameter_dict, other_parameter_dict):
    """
    Compare two parameter dicts values for equality.

    Args:
        parameter_dict(dict): First dict of ODYM parameters
        other_parameter_dict(): Second dict of ODYM parameters

    Returns:
        dict: Bool comparison by parameter name
    """

    comparisons = {}

    for parameter_name, parameter_item in parameter_dict.items():
        logger.info(f"Comparing {parameter_name}")
        allclose = np.allclose(parameter_item.Values, other_parameter_dict[parameter_name].Values)
        logger.info(f"All close: {allclose}")
        comparisons[parameter_name] = allclose

    return comparisons


def swap_dict_by_mat_to_dict_by_param(values_dict):
    """
    Swap a nested dictionary which is ordered by material {parameter: dataframe} into format parameter: [dataframes]

    Args:
        values_dict(dict)
    """
    param_values_dict = {}

    for df_dict in values_dict.values():
        for parameter, df in df_dict.items():
            if parameter not in param_values_dict:
                param_values_dict[parameter] = [df]
            else:
                param_values_dict[parameter].append(df)
    return param_values_dict


def equalize_enduses(enduse_array, enduse_selectors):
    """
    Equalizes values in the enduse dimension of an array to 1.

    Args:
        enduse_array(np.ndarray): Numpy array of enduses
        enduse_selectors(list): List of material indices where enduses must sum up to 1

    Returns:
         equalized_enduses(np.ndarray): Copy of the equalized array.
    """
    logger.info(f"Rescaling enduse array for enduses: {enduse_selectors}")
    # remgt
    assert len(enduse_array.shape) == 5
    equalized_enduses = enduse_array.copy()

    def adjust_array_sum(array, target_sum):
        current_sum = np.sum(array)
        scaling_factor = target_sum / current_sum
        equalized_array = array * scaling_factor
        return equalized_array

    for r in range(enduse_array.shape[0]):
        for e in range(enduse_array.shape[1]):
            for m in enduse_selectors:
                for t in range(enduse_array.shape[4]):
                    if np.isclose(np.sum(enduse_array[r, e, m, :, t]), 0):
                        continue
                    if not np.isclose(np.sum(enduse_array[r, e, m, :, t]), 1.):
                        equalized_enduses[r, e, m, :, t] = adjust_array_sum(enduse_array[r, e, m, :, t], 1.)

    return equalized_enduses
