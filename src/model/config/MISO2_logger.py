#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

src Logging routines

@author: bgrammer
"""

import os
import logging


def get_MISO2_logger(log_filename, log_pathname, file_level=None,
                     console_level=None, reset_handlers=True):
    """
    Configure the src logger.

    Convenience function for configuring the src loggers.

    Args:
        log_filename(str): Name of the log file.
        log_pathname(str): Path to save log file.
        file_level(level): Logging level. If None (default), no file log will be created.
        console_level(level): Logging level. If None (default), no console log will be created.
        reset_handlers(bool): If true, existing handlers will be removed before initialising new ones. Defaults to True.
    """

    if log_pathname and log_filename:
        log_file = os.path.join(log_pathname, log_filename)
    else:
        log_file = None
    logger = logging.getLogger()

    if reset_handlers:
        logger.handlers.clear()

    if file_level and console_level:
        logger.setLevel(min(file_level, console_level))
    elif file_level:
        logger.setLevel(file_level)
    elif console_level:
        logger.setLevel(console_level)

    console_log = None
    if console_level:
        console_log = logging.StreamHandler()
        console_log.setLevel(console_level)
        console_log_format = logging.Formatter('%(asctime)s %(levelname)s (%(filename)s <%(funcName)s>): %(message)s')
        console_log.setFormatter(console_log_format)
        logger.addHandler(console_log)

    file_log = None
    if file_level and log_file:
        file_log = logging.FileHandler(log_file, mode='a', encoding=None, delay=False)
        file_log.setLevel(file_level)
        file_log_format = logging.Formatter('%(asctime)s %(levelname)s (%(filename)s <%(funcName)s>): %(message)s\n')
        file_log.setFormatter(file_log_format)
        logger.addHandler(file_log)

    return logger, console_log, file_log
