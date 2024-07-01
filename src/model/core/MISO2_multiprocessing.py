#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Multiprocessing functions for running MISO2 models.

Created on Mon Sep 26 10:45:08 2022

@author: bgrammer
"""

import logging as logger
from datetime import datetime
import os
import pathlib
import pickle

from model.config.MISO2_sensitivity_by_parameter import modify_parameter
from model.config.MISO2_survival_functions import MISO2SurvivalFunctions
from model.config.MISO2_logger import get_MISO2_logger
from model.output.MISO2_stats import MISO2Stats
from model.config import MISO2_system_definitions as msd
from model.core.MISO2_model import MISO2Model
from model.config.MISO2_config import MISO2Config
from itertools import product

from util.MISO2_exceptions import EnduseSummationError


def _run_model_wrapper(config, sf, run_model_arguments):
    """
    Wraps running a MISO2 model for multiprocessing pool functions.

    Args:
        config(MISOConfig): Config object
        sf(MISOSurvivalFunctions): SF
        run_model_arguments(dict): Run model arguments

    Returns:
        miso_output(MISOOutput): Output object
    """
    miso_model = MISO2Model(miso_config=config,
                            miso_survival_functions=sf)

    miso_model.define_mfa_system(msd.define_MFA_System_MISO2v7)

    miso_output = miso_model.run_model(estimate_aggregates=run_model_arguments["estimate_aggregates"],
                                       save_stock_cohorts=run_model_arguments["save_stock_cohorts"],
                                       save_last_year=run_model_arguments["save_last_year"])

    return miso_output, miso_model


def miso_run_pool_mc(config, run_model_arguments, save_to_file=None, sf=None, repetitions=1000,
                     logging_level=logger.WARNING, sf_randomization=True):
    """
    Function to be passed to a Multiprocessing.Pool for Monte Carlo parallel processing of single regions.

    Config and survival functions are meant to be of a single region. The config needs to be set up for MC
    randomization, with uncertainty values present. The save_to_file list specifies which outputs are to be saved.
    A new folder named after the output region is created in the output path and will hold the base file log and the
    MISO2Stats object (as JSON/parquet files) that aggregates mean and variances over all MC runs.
    The logfiles of the individual MC runs and the outputs (if they are saved) are located in subfolders
    created automatically within that directory, with each holding up to 500 runs.

    Be aware that if you run this function with a high number of repetitions, and you are saving all outputs,
    it will have a considerable runtime and consume significant amounts of disk space.

    Args:
        config(MISO2Config/str): Config for a single region. If a string is passed \
            instead of a MISO2Config object, it is interpreted as a path to a valid config.
        run_model_arguments(dict): Wraps "estimate_aggregates", "mc_batch_size".
        save_to_file(list): List of outputs to be saved. May include "all_outputs", "sf" and "result". \
            Defaults to None.
        sf(MISO2SurvivalFunctions): Survival functions for a single region. If None is passed to this argument, \
            the survival functions will be created.
        repetitions(int): Number of Monte Carlo repetitions. Defaults to 1000, hardcapped limit of 15.000.
        logging_level(logging.LEVEL): Level of console logging. Defaults to Warning.
        sf_randomization(bool): Randomize survival functions per run. True by default.

    Returns:
        miso_stats(MISO2Stats): Aggregated stats of all output runs.

    Raises:
        ValueError: When illegal number of repetitions are to be executed or mismatch in MC configuration
        TypeError: When wrong object types are passed to function.
    """

    if save_to_file is None:
        save_to_file = []

    if isinstance(config, str):
        with open(config, 'rb') as f:
            config = pickle.load(f)

    if "mc_batch_size" in run_model_arguments:
        config.mc_batch_size = run_model_arguments["mc_batch_size"]

    timestamp = datetime.now().strftime("%Y_%m_%d %H_%M_%S")
    region = config.master_classification["Countries"].Items[0]
    base_path = run_model_arguments["output_path"]
    base_output_folder = os.path.join(base_path, region)
    pathlib.Path(base_output_folder).mkdir(parents=True, exist_ok=True)

    get_MISO2_logger(
        log_filename='MISO2_log_' + region + "_" + timestamp + ".log",
        log_pathname=base_output_folder,
        file_level=logging_level,
        console_level=None)

    _check_run_pool_arguments(sf=sf, config=config, mc=True, repetitions=repetitions,
                              max_reps=15000)

    if sf is None and not sf_randomization:
        parametrization = run_model_arguments.get("sf_parametrization", "MISO2LogNormal")
        logger.info(f"Survival function missing, creating them with parametrization: {parametrization}")
        sf = MISO2SurvivalFunctions(
            miso_config=config, parametrization=parametrization)

    sf_cache = {}

    logger.info(f"Starting pool execution of region: {region} with {repetitions} repetitions")

    miso_stats = MISO2Stats()
    sub_output_folder = "r_0_" + region

    for i in range(0, repetitions):
        # create a new output folder for every 500 runs to not overwhelm filesystem
        if i % 500 == 0 and "all_outputs" in save_to_file:
            sub_output_folder = "r_" + str(i) + "_" + region
            sub_output_folder = os.path.join(base_output_folder, sub_output_folder)
            pathlib.Path(sub_output_folder).mkdir(parents=True, exist_ok=True)

        if sf_randomization:
            logger.info(f"Randomising survival functions with bias: {config.get_bias('next')}")
            sf = MISO2SurvivalFunctions(
                miso_config=config, parametrization=run_model_arguments.get("sf_parametrization", "MISO2LogNormal"),
                cache=sf_cache, randomization=True, bias=config.get_bias("next"))
        miso_output, miso_model = _run_model_wrapper(config, sf, run_model_arguments)
        miso_stats.add_miso_output(miso_output)

        # save intermediary results every 1000 runs as a backup
        if "result" in save_to_file and (i+1) % 1000 == 0:
            miso_stats.save_to_file(filetype="parquet",
                                    output_path=base_output_folder,
                                    filename="MISO2_stats_" + region)
        if "all_outputs" in save_to_file:
            miso_output.metadata["mc_run"] = i
            miso_output.save_to_file(filetype="parquet",
                                     output_path=sub_output_folder,
                                     filename="MISO2_output" + region + "_run_" + str(i))

    miso_stats.finalize()

    logger.info(f"Finished execution on region {region}")

    if "sf" in save_to_file:
        sf.save_to_pickle(folder=base_output_folder,
                          filename="MISO2_" + region + "_SF.pickle")
    if "result" in save_to_file:
        miso_stats.save_to_file(filetype="parquet",
                                output_path=base_output_folder,
                                filename="MISO2_stats_" + region)

    return miso_stats


def miso_run_pool(config, run_model_arguments, save_to_file=None, sf=None, logging_level=logger.WARNING):
    """
    Function to be passed to a Multiprocessing.Pool for parallel processing of single regions.

    Config and survival functions are meant to be of a single region. If logging or saving outputs is enabled, a new
    folder named after the output region is created in the output path.

    Args:
        config(MISO2Config/str): Config for a single region. If a string is passed instead of \
            a MISO2Config object, it is interpreted as a path to a valid config.
        run_model_arguments(dict): Wraps "output_path", "estimate_aggregates" options for model run.
        save_to_file(list): List of outputs to be saved. May include "sf" and "result".
        sf(MISO2SurvivalFunctions): Survival functions for a single region. \
            If None is passed to this argument, the survival functions will be created.
        logging_level(logging.LEVEL): Level of console logging. None will disable logging altogether. \
            Defaults to Warning.

    Returns:
        miso_output(MISO2Output): Result of the model run.

    Raises:
        ValueError: If mismatch in MC configuration is detected.
        TypeError: When wrong object types are passed to function.
    """
    if isinstance(config, str):
        with open(config, 'rb') as f:
            config = pickle.load(f)

    if save_to_file is None:
        save_to_file = []

    base_output_folder = None
    region = config.master_classification["Countries"].Items[0]
    timestamp = datetime.now().strftime("%Y_%m_%d %H_%M_%S")

    if save_to_file or logging_level:
        logger.info("Creating new output folder")
        base_path = run_model_arguments["output_path"]
        base_output_folder = os.path.join(base_path, region)
        pathlib.Path(base_output_folder).mkdir(parents=True, exist_ok=True)

    get_MISO2_logger(
        log_filename='MISO2_log_' + region + "_" + timestamp + ".log",
        log_pathname=base_output_folder,
        file_level=logging_level,
        console_level=None)

    if isinstance(config, str):
        with open(config, 'rb') as f:
            config = pickle.load(f)

    _check_run_pool_arguments(sf=sf,
                              config=config,
                              mc=False)

    logger.info(f"Starting pool execution of region: {region}")

    if sf is None:
        parametrization = run_model_arguments.get("sf_parametrization", "Normal")
        logger.info(f"Survival function missing, creating them with parametrization: {parametrization}")
        sf = MISO2SurvivalFunctions(miso_config=config, parametrization=parametrization)

    miso_output, miso_model = _run_model_wrapper(config, sf, run_model_arguments)

    if "sf" in save_to_file:
        sf.save_to_pickle(folder=base_output_folder,
                          filename="MISO2_" + region + "_SF.pickle")
    if "result" in save_to_file:
        miso_output.save_to_file(filetype="parquet",
                                output_path=base_output_folder,
                                filename="MISO2_output_" + region)

    if "save_debug_output" in run_model_arguments:
        if run_model_arguments["save_debug_output"]:
            miso_model.write_debug_to_xls(os.path.join(base_output_folder, region+"_debug.xlsx"))

    logger.info(f"Finished execution on region {region}")
    return miso_output


def _check_run_pool_arguments(sf, config, mc, repetitions=None, max_reps=None):
    """
    Checks validity of arguments for running pool.

    Checks for correct object types, legitimate range of nr, monte carlo sate and \
        reasonable number of MC repetitions.

    """
    if sf:
        if not isinstance(sf, MISO2SurvivalFunctions):
            error_msg = f"Passed non-MISO2SurvivalFunctions object to miso run pool: {type(sf)}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        if sf.nr_start != 0 and sf.nr_stop != 1:
            logger.warning("Passed a survival function with more than one region to a multiprocessing function. \
                           Only the first region will be processed and you are not using this pool method as intended.")

    if not isinstance(config, MISO2Config):
        error_msg = f"Passed non-MISO2Config object to miso run pool: {type(config)}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    if config.nr_start != 0 and config.nr_stop != 1:
        logger.warning("Passed a config with more than one region to a multiprocessing function. \
                       Only the first region will be processed and you are not using this pool method as intended.")

    config_mc_state = config.get_randomization_state()

    if config_mc_state != mc:
        error_msg = "Passed a non-randomized config to a Monte Carlo method or vice-versa. \
            This was probably not intentional."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if repetitions and max_reps:
        if repetitions > max_reps or repetitions < 1:
            error_msg = f"Passed illegal value {repetitions} as number of repetitions to method. \
                If you meant to run more than the hardcoded limit of {max_reps} repetitions, \
                    change the hardcoded maximum in the code"
            logger.error(error_msg)
            raise ValueError(error_msg)


def miso_run_pool_mc_batch(config, run_model_arguments, sf=None, repetitions=10000,
                           logging_level=logger.WARNING):
    """
    Simplified batch version that skips construction of MISO stats objects and only saves output files.

    Function to be passed to a Multiprocessing.Pool for Monte Carlo parallel processing of single regions.

    Config and survival functions are meant to be of a single region. The config needs to be set up for MC \
     randomization. The save_to_file list specifies which outputs are to be saved. A new folder named after the output \
      region is created in the output path and will hold the base file log and the MISO2Stats object \
       (as JSON/parquet files).
    The logfiles of the individual MC runs and the outputs (if they are saved) are located in subfolders \
        created automatically within that directory, with each holding up to 500 runs.

    Be aware that if you run this function with a high number of repetitions, it will have a considerable runtime and
    consume significant amounts of disk space.

    Args:
        config(MISO2Config/str): Config for a single region. If a string is passed \
            instead of a MISO2Config object, it is interpreted as a path to a valid config.
        run_model_arguments(dict): Wraps "estimate_aggregates", "mc_batch_size".
        sf(MISO2SurvivalFunctions): Survival functions for a single region. If None is passed to this argument, \
            the survival functions will be created.
        repetitions(int): Number of Monte Carlo repetitions. Defaults to 1000, hardcapped limit of 15.000.
        logging_level(logging.LEVEL): Level of console logging. Defaults to Warning.

    Returns:
        miso_stats(MISO2Stats): Aggregated stats of all output runs.

    Raises:
        ValueError: When illegal number of repetitions are to be executed or mismatch in MC configuration
        TypeError: When wrong object types are passed to function.
    """

    if isinstance(config, str):
        with open(config, 'rb') as f:
            config = pickle.load(f)

    if "mc_batch_size" in run_model_arguments:
        config.mc_batch_size = run_model_arguments["mc_batch_size"]

    timestamp = datetime.now().strftime("%Y_%m_%d %H_%M_%S")
    region = config.master_classification["Countries"].Items[0]
    base_path = run_model_arguments["output_path"]
    base_output_folder = os.path.join(base_path, region)
    pathlib.Path(base_output_folder).mkdir(parents=True, exist_ok=True)

    get_MISO2_logger(
        log_filename='MISO2_log_' + region + "_" + timestamp + ".log",
        log_pathname=base_output_folder,
        file_level=logging_level,
        console_level=None)

    _check_run_pool_arguments(sf=sf, config=config, mc=True, repetitions=repetitions,
                              max_reps=15000)

    sf_cache = {}

    logger.info(f"Starting pool execution of region: {region} with {repetitions} repetitions")

    sub_output_folder = "r_0_" + region

    for i in range(0, repetitions):
        if i % 500 == 0:
            sub_output_folder = "r_" + str(i) + "_" + region
            sub_output_folder = os.path.join(base_output_folder, sub_output_folder)
            pathlib.Path(sub_output_folder).mkdir(parents=True, exist_ok=True)

        logger.info(f"Randomising survival functions with bias: {config.get_bias('next')}")
        sf = MISO2SurvivalFunctions(
            miso_config=config, parametrization=run_model_arguments.get("sf_parametrization", "MISO2LogNormal"),
            cache=sf_cache, randomization=True, bias=config.get_bias("next"))
        miso_output, miso_model = _run_model_wrapper(config, sf, run_model_arguments)

        miso_output.metadata["mc_run"] = i
        miso_output.save_to_file(filetype="parquet",
                                 output_path=sub_output_folder,
                                 filename="MISO2_output" + region + "_run_" + str(i))

    logger.info(f"Finished execution on region {region}")


def miso_run_pool_sensitivity(config, run_model_arguments, logging_level=logger.WARNING):

    timestamp = datetime.now().strftime("%Y_%m_%d %H_%M_%S")
    region = config.master_classification["Countries"].Items[0]
    base_path = run_model_arguments["output_path"]
    base_output_folder = os.path.join(base_path, region)
    pathlib.Path(base_output_folder).mkdir(parents=True, exist_ok=True)

    get_MISO2_logger(
        log_filename='MISO2_log_' + region + "_" + timestamp + ".log",
        log_pathname=base_output_folder,
        file_level=logging_level,
        console_level=None)

    sf_cache = {}

    logger.info(f"Starting pool execution of region: {region} with sensitivity parameter testing")

    scenarios = ["Low_scenario", "High_scenario"]
    scenario_bias = {"Low_scenario": "lower", "High_scenario": "higher"}

    parameter_groups = config._monte_carlo.available_parameter_groups()

    results = {}

    for (parameter_group, scenario) in product(parameter_groups, scenarios):
        config.set_global_monte_carlo_state(parameter_group, scenario)

        if parameter_group in ["Lifetimes", "All"]:
            randomisation = True
        else:
            randomisation = False

        if parameter_group in ["Multiplier", "All"]:
            # This is a workaround, since this multiplier is not yet parsed as a proper parameter
            old_multipliers = config.multiplier_cementBitumen.copy()
            parameter_name = "Multiplier_CementBitumen"
            test_options = config._monte_carlo.uncert_options[
                config._monte_carlo.active_randomization]["Multiplier_CementBitumen"]

            config.multiplier_cementBitumen = modify_parameter(
                parameter_name=parameter_name, parameter_group=parameter_group,
                test_options=test_options, parameter_values=config.multiplier_cementBitumen,
                parameter_cvs=None, scenario=scenario, uncertainty_settings=config.uncertainty_settings,
                cv_factor=config._monte_carlo.cv_factor)

        sf = MISO2SurvivalFunctions(
            miso_config=config, parametrization=run_model_arguments.get("sf_parametrization", "MISO2LogNormal"),
            cache=sf_cache, randomization=randomisation, bias=scenario_bias[scenario])
        # we shortcut calculating new lifetimes in the config by using the bias operator of the SF functions
        # it has hardcoded +/- 30%, but we do want those values
        miso_output, miso_model = _run_model_wrapper(config, sf, run_model_arguments)
        results[(parameter_group, scenario)] = miso_output

        if parameter_group in ["Multiplier", "All"]:
            config.multiplier_cementBitumen = old_multipliers

    logger.info(f"Finished execution on region {region}")

    return results