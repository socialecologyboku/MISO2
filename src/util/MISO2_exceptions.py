#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 19:55:38 2022
@author: bgrammer
"""


class WasteRateError(Exception):
    """
    Error type that occurs when the recovered and non-recovered waste rate of two processes
    exceeds a total of 100%. This is either related to an error in the input data or result of
    Monte Carlo outliers.
    """


class MassBalanceError(Exception):
    """
    Error that occurs when the error in mass balance of a src model exceeds a certain
    threshold.
    """


class InputLimitError(Exception):
    """
    Error that occurs when minimum or maximum limit for input data is violated. Limits are
    parsed and set from the "MISO2_uncertainty_distribution_settings.xlsx" file.
    """


class ValueOverlapError(Exception):
    """
    Error that occurs when EoL and Total Recycling / Downcycling parameter values overlap.
    This is an illegal state for the model core.
    """


class EnduseSummationError(Exception):
    """
    Error that occurs when Enduse parameters do not sum up to exactly 1 (non-aggregates) \
        or 0 (aggregates).
    """
