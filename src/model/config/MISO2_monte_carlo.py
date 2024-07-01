#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 19:33:13 2022

Monte Carlo module to return a queue of randomised values for a ODYM parameter dictionary.

@author: bgrammer
"""

import collections
import logging as logger
from scipy import stats
import numpy as np
import odym.ODYM_Classes as msc


class MISO2MonteCarlo:
    """
    An object of this class is injected into a MISO2Config object for MC simulation.

    The object transforms each individual ODYM Parameter values to a randomized version depending on the
    information given in the Uncert attribute. For this to work it needs first to transform the uncertainty information
    parsed by ODYM (a list of stats-array-like semicolon separated strings) into a list of distribution dictionaries
    for easier transformation.

    Warning: Random state is at the moment meant for testing purposes only, not for production use.

    Attributes:
        mc_state(dict): Dictionary with parameter names as key and their Monte Carlo state as bool.
        index
        mc_deque(deque): Deque containing randomized version of parameter dicts. \
            Will be filled automatically when needed.
        random_state(int): Set this value if you want reproducible values. Is repeated for all batches \
            of mc parameter queue. Defaults to None.
    """

    def __init__(self):
        self.mc_state = None
        self.index = None
        self.mc_deque = collections.deque()
        self.random_state = None

    def get_bias(self, mode):
        """
        Return bias config.
        """
        return "none"

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
            logger.info(f"Updating mc state for parameters: {parameter_list}")
            for parameter in parameter_list:
                if parameter in states:
                    states[parameter] = new_state
                else:
                    logger.warning(f"Cannot update parameter {parameter} randomization state,\
                                   because it does not exist")
        else:
            for key in states:
                states[key] = new_state

        self.mc_deque.clear()

    def get_parameter_randomization(self):
        """
        Returns true if at least one parameter is set to be randomized, false otherwise.
        """

        logger.info("Getting randomization state")
        states = self.mc_state
        for state in states.values():
            if state:
                return True
        return False

    def set_random_state(self, random_state):
        """
        Set the random state of the Monte Carlo configuration.

        Note that this will reset the existing Deque of randomized variables.

        Args:
            random_state(int): The seed of the random state.

        """

        logger.info(f"Setting new seed as random state: {random_state}")
        self.random_state = random_state
        self.mc_deque.clear()

    def set_parameter_names(self, parameter_names):
        """
        Initialise Parameter names that can be randomised.

        All parameters MC state is initially set to False.

        Args:
            parameter_names(dict): Name of parameters for regulating Monte Carlo state.
        """

        self.mc_state = dict.fromkeys(parameter_names, False)

    def _random_from_dirichlet(self, base, alphas, sample_size=1):
        """
        Transforms a given base array of 'remgt' index shape with Dirichlet distributions draws in place.

        Vectors will be taken over the sector index. All zero input returns all zero output. Single-value input or
         a base input with a value over 0.95 will result in the base input to be returned, e.g. no randomisation
            will take place.

        Args:
            base(np.array): Numpy array of parameter base values. Quantiles are drawn from here.
            alphas(np.array): Numpy array of alpha parameter. Parametrises each quantile.
            sample_size(int): Number of samples to be drawn.
        Returns:
            new_values(np.array): An ndarray of Dirichlet samples of shape (sample_size,base.shape).
        """

        random_state = self.random_state
        new_values = np.repeat(np.expand_dims(np.zeros_like(base), axis=0), repeats=sample_size, axis=0)
        remgt = base.shape
        # Values array has an excess year compared to the data (for last buffer year)
        # so we cover only t-1 years
        for r in range(0, remgt[0]):
            for e in range(0, remgt[1]):
                for m in range(0, remgt[2]):
                    for t in range(0, remgt[4]):
                        quantiles = base[r, e, m, :, t]
                        if (quantiles == 0).all():
                            break

                        alpha = alphas[r, e, m, :, t]
                        alpha_idx = np.where(alpha > 0)
                        active_alpha = alpha[alpha_idx]

                        if active_alpha.size == 1 or np.any(quantiles > 0.95):
                            result_array = np.tile(quantiles, sample_size).reshape((sample_size,) + quantiles.shape)
                        else:
                            result_array = np.zeros((sample_size,) + alpha.shape, dtype=float)
                            dirichlet_samples = stats.dirichlet(active_alpha).rvs(
                                sample_size, random_state=random_state)
                            result_array[:, alpha_idx] = dirichlet_samples[:, np.newaxis, ]

                        new_values[:, r, e, m, :, t] = result_array

        return new_values

    def _random_from_stats_array(self, distribution, sample_size=1):
        """
        Returns a list of randomized variables for the given sample size.

        Note that this function does not work on the Dirichlet distribution, since this requires
        an input vector of related values.

        Args:
            distribution(dict): Dictionary with type and parameters of a distribution.
            sample_size(int): Number of random variables to be returned. Defaults to 1.

        Returns:
            r(list): A list of randomized variables.

        Raises:
            ValueError: If distribution name is unknown.

        """
        random_state = self.random_state

        if "Mean" in distribution:
            if distribution["Mean"] < 0.0:
                old_mean = distribution["Mean"]
                distribution["Mean"] = 0
                logger.warning(f"A data entry below 0 ({old_mean}) was found in the uncertainty data. \
                               This is invalid input data and needs to be corrected.")
                return [0] * sample_size

        try:
            if distribution["Type"] == 'Fixed':
                r = [distribution['Mean']] * sample_size
            elif distribution["Type"] == 'Lognormal':
                if distribution["Mean"] <= 0:
                    return [0] * sample_size
                if distribution['Scale'] <= 0:
                    return [distribution["Mean"]] * sample_size
                mu_x = distribution["Mean"]
                var_x = distribution["Scale"]*mu_x
                # scale should be CV
                mu_lognormal = distribution["Mean"]**2 / (np.sqrt(distribution["Mean"]**2+var_x**2))
                # this is defined as np.log(mu_x**2/np.sqrt(mu_x**2+var_x**2))
                # but scipy.lognorm expects np.exp(mu_x), so we omit log here

                std_lognormal = np.sqrt(np.log(1+(var_x**2/mu_x**2)))
                # scipy.lognorm expects std / sqrt of var

                r = stats.lognorm(s=std_lognormal, loc=0, scale=mu_lognormal).rvs(
                    size=sample_size, random_state=random_state)
            elif distribution["Type"] == 'Normal':
                if distribution["Mean"] <= 0:
                    return [0] * sample_size
                if distribution["Scale"] <= 0:
                    return [distribution["Mean"]] * sample_size
                if np.isnan(distribution['Min']):
                    # if no minimum is set, we use normal distribution
                    r = stats.norm(loc=distribution['Mean'],
                                   scale=distribution['Scale']).rvs(
                        size=sample_size, random_state=random_state)
                else:
                    # if there is a minimum, we use trunc norm
                    if np.isnan(distribution['Max']):
                        distribution['Max'] = np.Inf
                    a = (distribution['Min'] - distribution['Mean']) / distribution['Scale']
                    b = (distribution['Max'] - distribution['Mean']) / distribution['Scale']
                    r = stats.truncnorm(a=a, b=b, loc=distribution['Mean'], scale=distribution['Scale']).rvs(
                        size=sample_size, random_state=random_state)

            elif distribution["Type"] == 'Uniform':
                if distribution['Min'] < 0:
                    logger.warning("Received uniform minimum value below zero, setting to zero")
                    distribution["Min"] = 0

                r = stats.uniform(loc=distribution['Min'],
                                  scale=distribution['Max'] - distribution['Min']).rvs(
                    size=sample_size, random_state=random_state)

            # elif distribution["Type"] == 'Weibull':
            #     r = stats.weibull_min(loc = distribution['Loc'],
            #                                             scale = distribution['Scale'],
            #                                             c = distribution['Shape']).rvs(size = sample_size)
            # elif distribution["Type"] == 'Beta':
            #     r = stats.beta(a = distribution['Mean'] * (distribution['Mean'] \
            #                                                * (1-distribution['Mean']) / distribution['Scale']**2 - 1),
            #                           b = (1 - distribution['Mean']) * (distribution['Mean'] \
            #                           * (1-distribution['Mean']) / \
            #                               distribution['Scale']**2 - 1)).rvs(size = sample_size)

            else:
                error_msg = f"Received an uncertainty name with unknown distribution: {distribution['Type']}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        except ValueError as e:
            logger.error(repr(e))
            logger.error(f"An error occurred, invalid arguments for stats array {distribution}")
            logger.error("Returning zero values array")
            return [0] * sample_size

        return r

    def randomize_model_input(self, parameter_dict, uncertainty_settings, sample_size):
        """
        Returns a randomized version of the Parameter dictionary.

        Randomization variables are taken from the Uncert variable of the Parameter object to \
            create a Deque of sample_size randomized parameter_dicts.
        If the Deque already exists and contains data, a parameter_dict will be drawn from it \
            and returned.

        Args:
            parameter_dict(dict): :class:`ODYM Parameter objects <ODYM_Classes.Parameter>` with name as keys.
            uncertainty_settings(dict): Types and limits (min,max) of Parameters.
            sample_size(int): Number of sampled values that will be added to the randomization deque if it is empty.

        Returns:
            parameter_dict_mc(dict): Modified dict of ODYM Parameters.
        """

        logger.info("Randomizing model input")

        if self.mc_deque:
            parameter_dict_mc = self.mc_deque.pop()
            logger.info(f"Returning randomized dict from deque, remaining dicts: {len(self.mc_deque)}")
        else:
            logger.info(f"Deque is empty, creating new {sample_size} randomized parameter dicts")
            if self.random_state:
                logger.info(f"Random seed for draw: {self.random_state} ")

            self.mc_deque = self._fill_mc_parameter_queue(
                parameter_dict,
                uncertainty_settings,
                sample_size)

            parameter_dict_mc = self.mc_deque.pop()

        return parameter_dict_mc

    def _fill_mc_parameter_queue(self, parameter_dict, uncertainty_settings, sample_size):
        """
        Creates a deque of randomized parameter_dicts.

        Args:
            parameter_dict(dict): Dictionary of :ref:`ODYM_Classes.Parameters`.
            uncertainty_settings(dict): Dictionary of uncertainty settings.
            sample_size(int): Number of parameter dicts that are randomized and stored in the deque.

        Returns:
            mc_deque(deque): A deque of randomized parameter_dicts.

        """
        logger.info(f"Creating randomized parameter queue with size {sample_size}")

        samples = {}
        parameters_to_randomize = {}

        for k, v in parameter_dict.items():
            if self.mc_state[k]:
                logger.info(f"Uncertainty enabled for {k}, returning randomised values")
                parameters_to_randomize[k] = v
            else:
                logger.info(f"Uncertainty disabled for {k}, returning original values")
                samples[k] = np.repeat(np.expand_dims(v.Values, axis=0), repeats=sample_size, axis=0)

        for k, v in parameters_to_randomize.items():
            if v.Uncert is None:
                logger.warning(f"Parameter {k} is set to be randomized, but there are \
                               no uncertainty values present. Returning original values")
                samples[k] = np.repeat(np.expand_dims(v.Values, axis=0), repeats=sample_size, axis=0)
                continue

            if uncertainty_settings[k]["Type"] == 'Allocation':
                logger.info(f"Parameter {k} is allocation (needs to sum up to 1 over sectors, \
                                using Dirichlet distribution")
                samples[k] = self._random_from_dirichlet(base=v.Values, alphas=v.Uncert, sample_size=sample_size)
                # uncertainties for allocations are already scaled alphas
            else:
                logger.info(f"Randomizing parameter {k} according to uncertainty data")
                uncert_values = [[y] * sample_size if x is None else self._random_from_stats_array(
                    x, sample_size=sample_size)
                                 for x, y in zip(v.Uncert, v.Values.flat)]
                new_shape = (sample_size,) + v.Values.shape
                uncert_values = np.array(uncert_values, np.float64).T.reshape(new_shape)
                samples[k] = uncert_values

        mc_deque = collections.deque()

        for i in range(0, sample_size):
            parameter_dict_copy = {}
            for k, v in parameter_dict.items():
                new_param = msc.Parameter(Name=v.Name, ID=v.ID, UUID=v.UUID,
                                          P_Res=v.P_Res, MetaData=v.MetaData, Indices=v.Indices,
                                          Values=samples[k][i],
                                          Uncert=None)
                parameter_dict_copy[k] = new_param
            mc_deque.append(parameter_dict_copy)

        return mc_deque


def stats_array_to_dict(uncertainty_list, parameter_limit):
    """
    Transform a single stat array to a parametrized distribution.

    The Dirichlet distribution is not parts of stat-array definition and assigned custom code 13.
    Distributions Min and Max value are capped by the parameter_limits for Normal and Uniform
    distribution to avoid potential negative values and rates in excess of 1.

    Args:
        uncertainty_list(list): Stats array split into list.
        parameter_limit(dict): Dictionary with Min and Max hard limit.

    Returns:
        distribution(dict): Stats array as a dict.

    Raises:
        ValueError: if an uncertainty string with an unknown distribution is received.

    """

    # set uncertainty to None for zero values?

    if np.isclose(uncertainty_list[0], 1):
        distribution = {'Type': 'Fixed',
                        'Mean': uncertainty_list[1]}
    elif np.isclose(uncertainty_list[0], 2):
        distribution = {'Type': 'Lognormal',
                        'Mean': uncertainty_list[1],
                        'Scale': uncertainty_list[2]}
    elif np.isclose(uncertainty_list[0], 3):
        distribution = {'Type': 'Normal',
                        'Mean': uncertainty_list[1],
                        'Scale': uncertainty_list[2],
                        'Min': uncertainty_list[4],
                        'Max': uncertainty_list[5]}
    elif np.isclose(uncertainty_list[0], 4):
        distribution = {'Type': 'Uniform',
                        'Min': uncertainty_list[4],
                        'Max': uncertainty_list[5]}
    elif np.isclose(uncertainty_list[0], 8):
        distribution = {'Type': 'Weibull',
                        'Loc': uncertainty_list[1],
                        'Scale': uncertainty_list[2],
                        'Shape': uncertainty_list[3]}
    elif np.isclose(uncertainty_list[0], 10):
        distribution = {'Type': 'Beta',
                        'Mean': uncertainty_list[1],
                        'Scale': uncertainty_list[2]}
    elif np.isclose(uncertainty_list[0], 13):
        if len(uncertainty_list) < 3:
            alpha = 1
        else:
            alpha = uncertainty_list[2]

        distribution = {'Type': 'Dirichlet',
                        'Base': uncertainty_list[1],
                        'Alpha': alpha}
    else:
        error_msg = f"Received an uncertainty string with unknown distribution: {uncertainty_list}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if "Min" in distribution and distribution["Min"] < parameter_limit["Min"]:
        distribution["Min"] = parameter_limit["Min"]
    if "Max" in distribution and distribution["Max"] > parameter_limit["Max"]:
        distribution["Max"] = parameter_limit["Max"]

    return distribution


def transform_stats_array_to_dicts(parameter_dict, uncertainty_settings):
    """
    The function transforms stats arrays, as parsed by the ODYM module,
    into dictionaries readable by the Monte Carlo object. This replaces
    the values from the parameters Uncert entry. If all values in the Uncert
    variable are None, the Uncert variable itself will be set to None.

    Args:
        parameter_dict(dict): Dictionary of :class:`ODYM Parameter objects <ODYM_Classes.Parameter>`.
        uncertainty_settings(dict): Nested dict with settings of uncertainty values
    """
    for key, parameter in parameter_dict.items():
        if parameter is not None:
            if all(item is None for item in parameter.Uncert):
                logger.info(f"All uncertainties are None for parameter {key}, disabling uncertainty")
                parameter.Uncert = None
            elif uncertainty_settings[key]["Type"] == 'Allocation':
                logger.info(f"Parameter {key} is allocation parameter, no transformation of stats array")
                parameter.Uncert = np.array(
                    [0 if x is None else float(x) for x in parameter.Uncert]).reshape(parameter.Values.shape)
            else:
                logger.info(f"Transforming uncertainty stats arrays for: {key}")
                parameter_limit = uncertainty_settings[key]
                parameter.Uncert = [item if item is None else
                                    [float(x) for x in item.split(";")] for item in parameter.Uncert]
                parameter.Uncert = [x if x is None else stats_array_to_dict(
                    x, parameter_limit) for x in parameter.Uncert]
            # need to interfere here for alphas

        # float(x) returns NaN for non-number strings, which produces correct
        # results for our purposes
