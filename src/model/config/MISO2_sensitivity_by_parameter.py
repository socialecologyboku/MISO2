#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging as logger
import numpy as np
import odym.ODYM_Classes as msc
import operator


class MISO2SensitivityByParameter:
    def __init__(self, sensitivity_settings, cv_factor=1.0):
        self.sensitivity_settings = sensitivity_settings

        self.uncert_options = {}
        for group in sensitivity_settings.groupby("Test_cohorts"):
            self.uncert_options[group[0]] = group[1].set_index("Parameter_Name").to_dict(orient="index")
        self.active_randomization = None
        # name of active parameter group
        self.active_scenario = None
        self.cv_factor = cv_factor

    def available_parameter_groups(self):
        return self.uncert_options.keys()

    def update_parameter_randomization(self, new_parameter, new_scenario):
        """

        Set next parameter group to test. The Monte Carlo object will return a modified config of the specified type.

        Args:
             new_parameter(str): Name of parameter group to test next.
             new_scenario(str): One of "Low_scenario" or "High_scenario".
        """
        self.active_randomization = new_parameter
        allowed_scenarios = ["Low_scenario", "High_scenario"]
        if new_scenario not in allowed_scenarios:
            raise ValueError(f"Unknown scenario: {new_scenario}")

        self.active_scenario = new_scenario

    def get_bias(self, mode):
        """
        Return bias of next sample handed out by config.
        """
        return "sensitivity_by_parameter"

    def randomize_model_input(self, parameter_dict, uncertainty_settings, sample_size):
        active_options = self.uncert_options[self.active_randomization]
        # logger.info
        new_parameter_dict = {}

        for k, v in parameter_dict.items():
            if k in active_options:
                modified_values = modify_parameter(
                    parameter_name=k,
                    parameter_group=self.active_randomization,
                    test_options=active_options[k],
                    parameter_values=v.Values,
                    parameter_cvs=v.Uncert,
                    scenario=self.active_scenario,
                    uncertainty_settings=uncertainty_settings,
                    cv_factor=self.cv_factor
                )

                new_param = msc.Parameter(Name=v.Name, ID=v.ID, UUID=v.UUID,
                                          P_Res=v.P_Res, MetaData=v.MetaData, Indices=v.Indices,
                                          Values=modified_values, Uncert=v.Uncert
                                          )
                new_parameter_dict[k] = new_param
            else:
                new_parameter_dict[k] = v

        return new_parameter_dict

    def get_parameter_randomization(self):
        return True

    def transform_uncert_array(self, parameter_dict, uncertainty_settings):
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
                    logger.info(f"Parameter {key} is allocation parameter, not active in sensitivity parameter"
                                f"testing")
                    parameter.Uncert = None
                else:
                    logger.info(f"Transforming uncertainty stats arrays for: {key}")
                    parameter.Uncert = np.array(parameter.Uncert, dtype=np.float).reshape(parameter.Values.shape)
                    parameter.Uncert = np.nan_to_num(parameter.Uncert, copy=False, nan=0.0)


def modify_parameter(parameter_name, parameter_group, test_options, parameter_values,
                     parameter_cvs, scenario, uncertainty_settings, cv_factor):
    logger.info(f"Modifying parameter name: {parameter_name}, parameter group: {parameter_group}")
    ops = {"+": operator.add, "-": operator.sub}

    modified_values = parameter_values.copy()
    modifier = ops[test_options[scenario]]
    global_cv = test_options["Global_cv"]

    if global_cv != "Disabled":
        logger.info("Modifying by global cv values")
        modified_values *= (modifier(1, cv_factor*float(global_cv)))
    elif parameter_cvs is None:
        logger.warning(f"Parameter {parameter_name} set to be randomised, but no cv present")
        return parameter_values.copy()
    else:
        logger.info("Modifying by cvs")
        modified_values *= (modifier(1, cv_factor*np.array(parameter_cvs, dtype=float).reshape(parameter_values.shape)))

    min_value = uncertainty_settings[parameter_name]["Min"]
    max_value = uncertainty_settings[parameter_name]["Max"]

    modified_values[modified_values > max_value] = max_value
    # set all that exceed max value to max (should only happen for rates that exceed 1)

    modified_values[modified_values < min_value] = min_value
    # negative values, this should never happen

    return modified_values
