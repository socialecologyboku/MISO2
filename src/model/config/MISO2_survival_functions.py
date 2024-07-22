#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:13:21 2022

@author: jstreeck, bgrammer
"""

import glob
import pickle
import os
import logging as logger
import random
import numpy as np
import odym.dynamic_stock_model as dsm


class MISO2SurvivalFunctions:
    """
    Contains the survival function array and helper methods for constructing and saving them to disk.

    The parameters for constructing the SF_Array are taken from a MISO2Config object.
    As an object of this class is dependent on the correct parametrization, the UUID of the config
    is saved and both are cross-checked by the model to prevent running the model on non-matching
    MISO2Configs and MISO2SurvivalFunctions.

    Args:
        miso_config(MISOConfig): The MISOConfig from which the survival arrays are constructed.
        parametrization(str): Lifetime distribution as expected by ODYM's dynamic stock model. \
            One of "MISO2LogNormal", "LogNormal", "Normal" (default), "Weibull", "Fixed", "FoldedNormal".
        cache(dict): Dictionary to look up cached survival functions. This may be reused between survival functions to
        speed up computation.
        randomization(bool): Randomise survival functions
        bias(str): Randomise survival functions into biased direction, e.g. always "higher" or "lower" than mean.

    Attributes:
        config_path(str): Default save path for pickling. Set to value found in MISO2Config.
        miso_config_id(UUID): Unique ID of the MISO2Config used to create the array.
        SF_Array(np.array):
        nr_start(int): Region start index for which SF were created.
        nr_stop(int): Region stop index for which SF were created.
        parametrization(str): Lifetime distribution

    """
    __slots__ = "SF_Array", "miso_config_id", "config_path", "nr_start", "nr_stop", "parametrization"

    def __init__(self, miso_config=None, parametrization="LogNormal", cache=None, randomization=False, bias=None):

        if miso_config:
            self._construct_valid(miso_config, parametrization, cache, randomization, bias)
        elif miso_config is None:
            self._construct_dummy()
        else:
            raise AttributeError("Illegal argument combination for constructor.")

    def _construct_valid(self, miso_config, parametrization, cache=None, randomization=False, bias=None):
        valid_parametrizations = ["MISO2LogNormal", "LogNormal", "Normal", "Weibull", "Fixed", "FoldedNormal"]
        if parametrization not in valid_parametrizations:
            raise AttributeError(f"Passed non-valid paramtrization: {parametrization}")

        self.nr_start = miso_config.nr_start
        self.nr_stop = miso_config.nr_stop
        self.parametrization = parametrization
        # parametrization needs to be defined before creation of lifetime functions
        self.SF_Array = self.create_lifetime_functions(
            miso_config=miso_config, cache=cache, randomization=randomization, bias=bias)
        self.miso_config_id = miso_config.unique_id
        self.config_path = miso_config.config_path

    def _construct_dummy(self):
        self.nr_start = None
        self.nr_stop = None
        self.SF_Array = None
        self.miso_config_id = None
        self.config_path = None
        self.parametrization = None

    def save_to_pickle(self, filename=None, folder=None):
        """
        Save the Survival functions to a pickle with the highest protocol.

        Args:
            filename(str): Defaults to <UUID>_SF.pickle
            folder(str): Defaults to MISO2Configs config path.
        """

        logger.info("Saving MISO survival function to pickle file: ")

        if folder is None:
            folder = self.config_path
            logger.info("No folder path specified, saving to default folder: " + folder)

        if filename is None:
            filename = str(self.miso_config_id) + "_SF.pickle"
            logger.info("No file name specified, saving to default filename (UUID): " + filename)
        with open(os.path.join(folder, filename), 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def create_lifetime_functions(self, miso_config, cache=None, randomization=False, bias=None):
        """
        Create Survival Functions with the help of ODYMs Dynamic Stock Model.

        Note that the SF Array will have the full size (over all regions) of the MISO2Config,
        but survival functions will only be calculated for the regions specified with
        nr_start and nr_stop.

        Parametrization is dependent on the format of the MISO2_Lifetimes and MISO2_Lifetimes_deviation
        input parameters.

        The function has an unbounded cached to speed up computation. Please note that when you supply a lot of
        different input arguments, the cache will grow without bound and add significant memory requirements to
        the call.

        Args:
            miso_config(MISO2Config): Config object holding the relevant parameters.
            cache(dict): Already existing cache of survival functions. If None (default), cache is created.
            randomization(bool): Randomize survival functions. Defaults to false.
            bias(str): Return "higher" or "lower" biased survival functions.
        Returns:
            SF_Array(np.array): A (r,e,m,g,t,t)-shaped array of floats. Over the last two dimensions, this a lower
            triangular matrix with lifetime functions, ones in the diagonal, and zeros in the upper half.

        """

        def calculate_lifetimes():
            time = np.arange(0, miso_config.Nc, 1)
            for r in range(miso_config.nr_start, miso_config.nr_stop):
                for g in range(0, miso_config.Ng):
                    for e in range(0, miso_config.Ne):
                        for m in range(0, miso_config.Nm):
                            for t in range(0, miso_config.Nt):
                                mean = miso_config.parameter_dict['MISO2_Lifetimes'].Values[r, e, m, g, t]
                                stddev = miso_config.parameter_dict['MISO2_Lifetimes_deviation'].Values[r, e, m, g, t]
                                input_args = (tuple([mean]), tuple([stddev]), t)
                                if input_args not in cache:
                                    lt = {'Type': self.parametrization,
                                          'Mean': [mean],
                                          'StdDev': [stddev]}
                                    sf = dsm.DynamicStockModel(
                                        t=time, lt=lt).compute_sf()[:, t]
                                    SF_Array[r, e, m, g, :, t] = sf
                                    cache[input_args] = sf
                                else:
                                    SF_Array[r, e, m, g, :, t] = cache[input_args]
                                np.fill_diagonal(SF_Array[r, e, m, g, :, :], 1)

        def calculate_lifetimes_random(bias=None):

            if bias == "higher":
                logger.info("Constructing SF with higher bias")
                options = ["high"]
            elif bias == "lower":
                logger.info("Constructing SF with lower bias")
                options = ["low"]
            elif bias == "random":
                logger.info("Constructing SF without bias")
                options = ["high", "normal", "low"]
            else:
                raise ValueError(f"Unknown bias argument: {bias}")

            random_choices = random.choices(options, k=int(SF_Array.size / miso_config.Nt))

            time = np.arange(0, miso_config.Nc, 1)
            choice_index = 0

            for r in range(miso_config.nr_start, miso_config.nr_stop):
                for g in range(0, miso_config.Ng):
                    for e in range(0, miso_config.Ne):
                        for m in range(0, miso_config.Nm):
                            for t in range(0, miso_config.Nt):
                                mean = miso_config.parameter_dict['MISO2_Lifetimes'].Values[r, e, m, g, t]
                                stddev = miso_config.parameter_dict['MISO2_Lifetimes_deviation'].Values[r, e, m, g, t]
                                random_choice = random_choices[choice_index]
                                choice_index += 1
                                if random_choice == "low":
                                    mean = mean * 0.7
                                elif random_choice == "high":
                                    mean = mean * 1.3
                                input_args = (tuple([mean]), tuple([stddev]), t)
                                if input_args not in cache:
                                    lt = {'Type': self.parametrization,
                                          'Mean': [mean],
                                          'StdDev': [stddev]}
                                    sf = dsm.DynamicStockModel(
                                        t=time, lt=lt).compute_sf()[:, t]
                                    SF_Array[r, e, m, g, :, t] = sf
                                    cache[input_args] = sf
                                else:
                                    SF_Array[r, e, m, g, :, t] = cache[input_args]
                                np.fill_diagonal(SF_Array[r, e, m, g, :, :], 1)

        logger.info(f"Creating lifetime functions with cache and distribution {self.parametrization} and"
                    f"randomization {randomization}")
        logger.debug(f"Nr start {miso_config.nr_start}, nr stop {miso_config.nr_stop}")

        SF_Array = np.zeros((miso_config.Nr, miso_config.Ne,
                             miso_config.Nm, miso_config.Ng,
                             miso_config.Nc, miso_config.Nc))

        if cache is None:
            cache = {}

        if randomization:
            calculate_lifetimes_random(bias=bias)
        else:
            calculate_lifetimes()

        logger.info("Calculated SF Array")

        if SF_Array[SF_Array < 0].any():
            error_msg = "Survival function contains negative values"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if np.isnan(SF_Array).any():
            error_msg = "Survival function contains nan values"
            logger.error(error_msg)
            raise ValueError(error_msg)
        return SF_Array

    def get_sf_array(self, nr_start, nr_stop):
        """
        Returns subset of SF_Array.

        Args:
            nr_start(int): Start region index.
            nr_stop(int): Stop region index.

        Returns:
            SF_Array(np.array): Sliced view of SF_Array

        Raises:
            ValueError: If region indexes not within range of SF_Array.
        """

        logger.info(f"Returning SF array with index {nr_start} to {nr_stop} ")
        valid_nr_range = range(self.nr_start, self.nr_stop)

        if nr_start not in valid_nr_range or nr_stop-1 not in valid_nr_range:
            error_msg = "Tried to subset SF Array with illegal region index"
            logger.error(error_msg)
            raise ValueError(error_msg)

        return self.SF_Array[nr_start:nr_stop, ...]


def split_survival_functions(miso_survival_functions, index):
    """
    Returns a new MISOSurvivalFunctions object with a view of subset SF array.

    Since SF_Array should never be modified in-place, it is okay to return a view here.

    Args:
        miso_survival_functions(MISOSurvivalFunctions): Original object.
        index(int): Index to be subset, first dimension of sf array (region).

    Returns:
        split_sf(MISOSurvivalFunctions): New object split at index dimension.
    """
    logger.info(f"Splitting surival functions index: {index}")
    split_sf = MISO2SurvivalFunctions()

    split_sf.nr_start = miso_survival_functions.nr_start
    split_sf.nr_stop = miso_survival_functions.nr_stop
    split_sf.miso_config_id = miso_survival_functions.miso_config_id
    split_sf.config_path = miso_survival_functions.config_path
    split_sf.parametrization = miso_survival_functions.parametrization

    split_sf.SF_Array = miso_survival_functions.SF_Array[index, ...]
    split_sf.SF_Array = split_sf.SF_Array[np.newaxis, ...]
    # we have to add an empty dimension to keep data frame shape intact. there is probably a more elegant way to do this
    return split_sf


def load_sfs_from_folder(sfs_path, key="id"):
    """
    Tries to load all .pickle files in a directory as MISO2SurvivalFunctions and returns them as a list.

    Args:
        sfs_path(str): A filepath
        key(str): One of "id" (default) or "filename"

    Returns:
         configs(dict): A list of MISO2Config objects.
    """

    restored_sfs = {}
    sf_filenames = glob.glob(os.path.join(sfs_path, "*.pickle"))
    logger.info(f"Loading sfs from path: {sfs_path}")
    logger.info(f"Filenames: {sf_filenames}")
    for filename in sf_filenames:
        try:
            split_sf_path = os.path.join(sfs_path, filename)
            with open(split_sf_path, 'rb') as f:
                sf = pickle.load(f)
            if not isinstance(sf, MISO2SurvivalFunctions):
                raise AttributeError(f"Wrong filetype: {type(sf)}")

            if key == "id":
                restored_sfs[sf.miso_config_id] = sf
            elif key == "filename":
                restored_sfs[filename] = sf
            else:
                raise AttributeError
        except (IOError, AttributeError) as error:
            logger.error("Something happend while trying to load a survival function")
            logger.exception(error)
    return restored_sfs


def save_split_sfs_to_folder(path, sf_name, sf):
    """
    Splits a SF object into multiple objects, one per region, and saves them to a folder.

    Individual files receive a running index. Note that a corresponding miso config id must be set manually.

    Args:
         path(str): Folder path to save to
         sf_name(str): Filename of the config.
         sf(MISO2SurvivalFunctions): SF to split
    """
    end = sf.nr_stop
    for i in range(0, end):
        new_filename = "split_" + str(i) + "_" + sf_name + ".pickle"
        logger.info(f"split survival function {new_filename}")
        new_sf = split_survival_functions(sf, i)
        new_sf.save_to_pickle(filename=new_filename, folder=path)
