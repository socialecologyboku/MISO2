#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 19:33:13 2022

Monte Carlo object for systematic bias, in which all values are increased / decreased systematically
across samples so that stocks are maximised / minimised.

@author: bgrammer
"""

import collections
import logging as logger
import numpy as np
import odym.ODYM_Classes as msc
from model.config.MISO2_monte_carlo import MISO2MonteCarlo


class MISO2MonteCarloSystematic(MISO2MonteCarlo):
    """
    This implementation of the MISO2MonteCarlo class will introduce a systematic bias in the randomized variables.
    Samples of all values of all parameters will be drawn either from a value below or above the mean, so the entirety
    of the input data will either be systematically over- or underestimated to maximise overall stock returns.

    Attributes:
        mc_state(dict): Dictionary with parameter names as key and their Monte Carlo state as bool.
        mc_deque_lower(deque): Deque containing randomized version of parameter dicts with below mean values.
        mc_deque_higher(deque): Deque containing randomized version of parameter dicts with higher than mean values.
        random_state(int): Set this value if you want reproducible values. Is repeated for all batches \
        of mc parameter queue. Defaults to None.
    """

    def __init__(self):
        super().__init__()
        self.return_higher = False
        self.mc_deque_lower = collections.deque()
        self.mc_deque_higher = collections.deque()

    def get_bias(self, mode):
        """
        Return bias of next or last sample handed out by config.

        Args:
            mode(string): One of "next" or "last"

        Raises:
             ValueError: If mode is unknown
        """
        if self.return_higher is True:
            if mode == "next":
                return "higher"
            if mode == "last":
                return "lower"
        elif self.return_higher is False:
            if mode == "next":
                return "lower"
            if mode == "last":
                return "higher"

        raise ValueError(f"Unknown state of return_higher: {self.return_higher} with mode: {mode}")

    def update_parameter_randomization(self, new_state, parameter_list=None):
        """
        Updates the randomization state for all or specified list of parameters.

        This will reset the MISO2MonteCarlo objects queue of randomized dicts (if there was one).

        Args:
            new_state(bool): New state.
            parameter_list(list): List of parameter names.

        """

        states = self.mc_state

        if parameter_list:
            for parameter in parameter_list:
                if parameter in states:
                    states[parameter] = new_state
                else:
                    logger.warning(f"Cannot update parameter {parameter} randomization state,\
                                   because it does not exist")
        else:
            for key in states:
                states[key] = new_state

        self.mc_deque_lower.clear()
        self.mc_deque_higher.clear()

    def set_random_state(self, random_state):
        """
        Set the random state of the Monte Carlo configuration.

        Note that this will reset the existing Deque of randomized variables.

        Args:
            random_state(int): The seed of the random state.

        """

        logger.info(f"Setting new seed as random state: {random_state}")
        self.random_state = random_state
        self.mc_deque_lower.clear()
        self.mc_deque_higher.clear()

    def _fill_mc_parameter_queue(self, parameter_dict, uncertainty_settings, sample_size):
        """
        Creates a deque of randomized parameter_dicts.

        Args:
            parameter_dict(dict): Dictionary of :ref:`ODYM_Classes.Parameters`.
            uncertainty_settings(dict): Dictionary of uncertainty settings.
            sample_size(int): Number of parameter dicts that are randomized and stored in the deque.

        Returns:
            mc_deque_lower(deque), mc_deque_higher(deque): Two deques of randomized parameter_dicts.

        """
        logger.info(f"Creating systematic biased randomized parameter queue with size {sample_size}")

        lower_samples = {}
        higher_samples = {}

        parameters_to_randomize = {}

        flip_high_low = {
            "MISO2_WasteRate_recov_p5", "MISO2_WasteRate_unrecov_p5",
            "MISO2_WasteRate_recov_p7", "MISO2_WasteRate_unrecov_p7",
            "MISO2_WasteRate_recov_p9", "MISO2_WasteRate_unrecov_p9"}

        if sample_size % 2 != 0:
            sample_size += 1
        half_sample_size = int(sample_size/2)

        for k, v in parameter_dict.items():
            if self.mc_state[k]:
                logger.info(f"Uncertainty enabled for {k}, returning randomised values")
                parameters_to_randomize[k] = v
            else:
                logger.info(f"Uncertainty disabled for {k}, returning original values")
                lower_samples[k] = np.repeat(np.expand_dims(v.Values, axis=0), repeats=half_sample_size, axis=0)
                higher_samples[k] = np.repeat(np.expand_dims(v.Values, axis=0), repeats=half_sample_size, axis=0)

        for k, v in parameters_to_randomize.items():
            # as normal
            if v.Uncert is None:
                logger.warning(f"Parameter {k} is set to be randomized, but there are \
                                   no uncertainty values present. Returning original values")
                lower_samples[k] = np.repeat(np.expand_dims(v.Values, axis=0), repeats=half_sample_size, axis=0)
                higher_samples[k] = np.repeat(np.expand_dims(v.Values, axis=0), repeats=half_sample_size, axis=0)
                continue

            if uncertainty_settings[k]["Type"] == 'Allocation':

                logger.info(f"Parameter {k} is allocation (needs to sum up to 1 over sectors, \
                                    using Dirichlet distribution")
                lower_samples[k] = self._random_from_dirichlet(
                    base=v.Values, alphas=v.Uncert, sample_size=half_sample_size)
                higher_samples[k] = self._random_from_dirichlet(
                    base=v.Values, alphas=v.Uncert, sample_size=half_sample_size)
                # uncertainties for allocations are already scaled alphas,
            else:
                logger.info(f"Randomizing parameter {k} according to uncertainty data")
                uncert_values_lower = []
                uncert_values_higher = []
                for x, y in zip(v.Uncert, v.Values.flat):
                    if x is None:
                        uncert_values_lower.append([y] * half_sample_size)
                        uncert_values_higher.append([y] * half_sample_size)
                    else:
                        uncert_sample_lower, uncert_sample_higher = \
                            self._split_samples_lower_higher(x, half_sample_size)
                        uncert_values_lower.append(uncert_sample_lower)
                        uncert_values_higher.append(uncert_sample_higher)

                new_shape = (half_sample_size,) + v.Values.shape

                uncert_values_lower = np.array(uncert_values_lower, np.float64).T.reshape(new_shape)
                uncert_values_higher = np.array(uncert_values_higher, np.float64).T.reshape(new_shape)

                if k in flip_high_low:
                    lower_samples[k] = uncert_values_higher
                    higher_samples[k] = uncert_values_lower
                else:
                    lower_samples[k] = uncert_values_lower
                    higher_samples[k] = uncert_values_higher

        mc_deque_lower = collections.deque()
        mc_deque_higher = collections.deque()

        for i in range(0, half_sample_size):
            parameter_dict_copy_lower = {}
            parameter_dict_copy_higher = {}

            for k, v in parameter_dict.items():
                new_param_lower = msc.Parameter(Name=v.Name, ID=v.ID, UUID=v.UUID,
                                                P_Res=v.P_Res, MetaData=v.MetaData, Indices=v.Indices,
                                                Values=lower_samples[k][i],
                                                Uncert=None)
                new_param_higher = msc.Parameter(Name=v.Name, ID=v.ID, UUID=v.UUID,
                                                 P_Res=v.P_Res, MetaData=v.MetaData, Indices=v.Indices,
                                                 Values=higher_samples[k][i],
                                                 Uncert=None)
                parameter_dict_copy_lower[k] = new_param_lower
                parameter_dict_copy_higher[k] = new_param_higher

            mc_deque_lower.append(parameter_dict_copy_lower)
            mc_deque_higher.append(parameter_dict_copy_higher)

        return mc_deque_lower, mc_deque_higher

    def randomize_model_input(self, parameter_dict, uncertainty_settings, sample_size):
        """
        Returns a randomized version of the Parameter dictionary.

        Randomization variables are taken from the Uncert variable of the Parameter object to create two deque of half
        sample_size randomized parameter_dicts, one for lower and one for higher bias. If the Deque already exists and
        contains data, a parameter_dict will be drawn from it and returned.

        Args:
            parameter_dict(dict): :class:`ODYM Parameter objects <ODYM_Classes.Parameter>` with name as keys.
            uncertainty_settings(dict): Types and limits (min,max) of Parameters.
            sample_size(int): Number of sampled values that will be added to the randomization deque if it is empty.

        Returns:
            parameter_dict_mc(dict): Modified dict of ODYM Parameters.
        """

        logger.info("Randomizing model input")

        if self.mc_deque_higher and self.return_higher:
            self.return_higher = False
            parameter_dict_mc = self.mc_deque_higher.pop()
            logger.info("Returning randomized dict from higher deque")
        elif self.mc_deque_lower and not self.return_higher:
            self.return_higher = True
            parameter_dict_mc = self.mc_deque_lower.pop()
        else:
            logger.info(f"Deque is empty, creating new {sample_size} randomized parameter dicts")
            if self.random_state:
                logger.info(f"Random seed for draw: {self.random_state} ")

            self.mc_deque_lower, self.mc_deque_higher = self._fill_mc_parameter_queue(
                parameter_dict, uncertainty_settings, sample_size)

            parameter_dict_mc = self.mc_deque_higher.pop()
            self.return_higher = False

        return parameter_dict_mc

    def _split_samples_lower_higher(self, stats_array_dict, half_sample_size):
        """
        Returns two sample array of lower and higher samples for given stats array dict.

        Args:
            stats_array_dict(dict): Dictionary of stat arrays
            half_sample_size(int): Number of samples to generate per array

        Returns:
            lower_samples(np.ndarray)
            higher_samples(np.ndarray)

        Raises:
            AttributeError: If stats array dict does not support the operation.

        """
        if "Mean" in stats_array_dict:
            mean = stats_array_dict["Mean"]
            if np.isclose(mean, 0.0):
                return np.zeros(half_sample_size), np.zeros(half_sample_size)
        elif "Min" in stats_array_dict and "Max" in stats_array_dict:
            min_val = stats_array_dict["Min"]
            max_val = stats_array_dict["Max"]
            if np.isclose(min_val, 0.0) and np.isclose(max_val, 0.0):
                return np.zeros(half_sample_size), np.zeros(half_sample_size)
            mean = (min_val+max_val)/2
        else:
            raise AttributeError("No way to create mean from stats array dict")

        new_values = self._random_from_stats_array(stats_array_dict, sample_size=half_sample_size*4)

        if not isinstance(new_values, np.ndarray):
            new_values = np.array(new_values)

        if np.any(new_values):
            lower_mean = new_values[new_values < mean]
            higher_mean = new_values[new_values >= mean]
        elif np.all(np.isclose(new_values, new_values[0])):
            lower_mean = new_values
            higher_mean = new_values
        else:
            raise NotImplementedError("Unknown case in biased randomisation")

        if len(lower_mean) < half_sample_size or len(higher_mean) < half_sample_size:
            raise ValueError("Illegal state: Could not split sample array into two arrays of necessary size")

        lower_samples = lower_mean[0:half_sample_size]
        higher_samples = higher_mean[0:half_sample_size]
        return lower_samples, higher_samples
