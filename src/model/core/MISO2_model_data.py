#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:55:32 2022

@author: bgrammer & jstreeck
"""

import logging as logger
import numpy as np


class MISO2ModelData:
    """
    Data class that holds arrays needed for ongoing calculations within the MISO2 model.

    Attributes:
            total2primary (array-like or None): Placeholder for total to primary data.
            new_scrap (array-like or None): Placeholder for new scrap data.
            ratio_old_scrap (array-like or None): Placeholder for the ratio of old scrap.
            rem_waste_Cycling_noAggr (array-like or None): Placeholder for remaining waste cycling without aggregation.
            rec_virginAggr (array-like or None): Placeholder for recycled virgin aggregate data.
            downc_constr_mats (array-like or None): Placeholder for downcycled construction materials.
            GAS_split (array-like or None): Placeholder for GAS split data.
            cementBitumen_2_concreteAsphalt (array-like or None): Placeholder for cement and bitumen to concrete and
            asphalt conversion data.
            bal_4 (array-like or None): Placeholder for balance data (bal_4).
            bal_6 (array-like or None): Placeholder for balance data (bal_6).
            bal_8 (array-like or None): Placeholder for balance data (bal_8).
            bal_12 (array-like or None): Placeholder for balance data (bal_12).
            bal_13a (array-like or None): Placeholder for balance data (bal_13a).
            bal_13b (array-like or None): Placeholder for balance data (bal_13b).
            bal_13c (array-like or None): Placeholder for balance data (bal_13c).
            aggr_req_mg (array-like or None): Placeholder for aggregate requirement data.
            appar_consump_end_use (array-like or None): Placeholder for apparent consumption end-use data.
            trade_larger_req_g (int): Placeholder for trade larger requirement data, initialized to 0.
            new_scrap_recycled (array-like or None): Placeholder for new scrap recycled data.
    """

    __slots__ = "total2primary", "new_scrap", "ratio_old_scrap", "rem_waste_Cycling_noAggr", \
        "rec_virginAggr", "downc_constr_mats", "GAS_split", "cementBitumen_2_concreteAsphalt", \
        "bal_4", "bal_6", "bal_8", "bal_12", "bal_13a", "bal_13b", "bal_13c", "aggr_req_mg", \
        "cementBitumen_2_concreteAsphalt", "appar_consump_end_use", "trade_larger_req_g", "new_scrap_recycled"

    def __init__(self):
        self.total2primary = None
        self.new_scrap = None
        self.ratio_old_scrap = None
        self.rem_waste_Cycling_noAggr = None
        self.rec_virginAggr = None
        self.downc_constr_mats = None
        self.GAS_split = None
        self.cementBitumen_2_concreteAsphalt = None
        self.bal_4 = None
        self.bal_6 = None
        self.bal_8 = None
        self.bal_12 = None
        self.bal_13a = None
        self.bal_13b = None
        self.bal_13c = None

        self.appar_consump_end_use = None
        self.cementBitumen_2_concreteAsphalt = None
        self.aggr_req_mg = None
        self.trade_larger_req_g = 0
        self.new_scrap_recycled = None
        
    def create_data_structures(self, dimensions, selector_multiplier_materials):
        """
        Preallocate numpy arrays that are later used during the model calculation.

        Args:
            dimensions(dict): Dictionary with N dimensions
            selector_multiplier_materials(list): List with foundation multiplier material selectors.

        """
        logger.info("Creating data structures")

        self.total2primary = np.zeros((
            dimensions["Nr"], dimensions["Ne"], dimensions["Nm"], dimensions["Nt"]))
        # stores exogenous data for total material production
        self.new_scrap = np.zeros((
            dimensions["Nr"], dimensions["Ne"], dimensions["Nm"], dimensions["Nt"]))
        # estimate of absolute new scrap (assuming scrap trade and balancing at p12/p13
        # proportional to initial new scrap share on total scrap)
        self.ratio_old_scrap = np.zeros((
            dimensions["Nr"], dimensions["Ne"], dimensions["Nm"], dimensions["Nt"]))
        # estimate of ratio of old scrap (assuming scrap trade and balancing at p12/p13
        # proportional to initial old scrap share on total scrap)
        self.rem_waste_Cycling_noAggr = np.zeros((
            dimensions["Nr"], dimensions["Ne"], dimensions["Nm"], dimensions["Nt"]))
        # remaining waste (from F_12_13), after substracting cycling flows 13_20a/b
        self.rec_virginAggr = np.zeros((dimensions["Ne"], dimensions["Nm"]))
        # amount of virgin aggregates that is recycled
        self.downc_constr_mats = np.zeros((
            dimensions["Nr"], dimensions["Ne"], dimensions["Nm"], dimensions["Nt"]))
        # amount of bricks, concrete, asphalt that is downcycled
        
        self.new_scrap_recycled = np.zeros((
            dimensions["Nr"], dimensions["Ne"], dimensions["Nm"], dimensions["Nt"]))
        # dataframe to account recycled new scrap to distinguish from old scrap during recycling in MISO2 system

        self.GAS_split = np.zeros((
            dimensions["Nr"], dimensions["Ne"], dimensions["Nm"], dimensions["Ng"], dimensions["Nt"]))
        # stores F_9_10 after end-use split
        self.bal_4 = np.zeros((dimensions["Nr"], dimensions["Ne"], dimensions["Nm"], dimensions["Nt"]))
        # balancing item for process 4
        self.bal_6 = np.zeros((dimensions["Nr"], dimensions["Ne"], dimensions["Nm"], dimensions["Nt"]))
        # balancing item for process 6
        self.bal_8 = np.zeros((dimensions["Nr"], dimensions["Ne"], dimensions["Nm"], dimensions["Nt"]))
        # balancing item for process 8
        self.bal_12 = np.zeros((dimensions["Nr"], dimensions["Ne"], dimensions["Nm"], dimensions["Nt"]))
        # balancing item for process 12
        self.bal_13a = np.zeros((dimensions["Nr"], dimensions["Ne"], dimensions["Nm"], dimensions["Nt"]))
        # balancing item for process 13a (recycling)
        self.bal_13b = np.zeros((dimensions["Nr"], dimensions["Ne"], dimensions["Nm"], dimensions["Nt"]))
        # balancing item for process 13b (downcycling)
        self.bal_13c = np.zeros((dimensions["Nr"], dimensions["Ne"], dimensions["Nm"], dimensions["Nt"]))
        # balancing item for process 14 (total production vs. recycling)

        # selector needs to be passed as arguments
        self.aggr_req_mg = np.zeros((([
            len(selector_multiplier_materials), len(dimensions["endUse_aggregates"]),
            dimensions["Nt"]]))).reshape((
            len(selector_multiplier_materials), dimensions["Ng"], dimensions["Nt"], dimensions["Ne"]))
        # estimate on required aggregates for building and road foundations by material and end-use

        self.cementBitumen_2_concreteAsphalt = np.zeros(
            (dimensions["Nr"], dimensions["Ne"], dimensions["Nm"], dimensions["Ng"], dimensions["Nt"]))
        # dataframe for mixing sand and gravel with cement and bitumen to concrete and asphalt

        self.appar_consump_end_use = np.zeros(
            (dimensions["Nr"], dimensions["Ne"], dimensions["Nm"], dimensions["Ng"], dimensions["Nt"]))
        # dataframe to buffer apparent consumption of final products by end-use before assigning to MISO2 system
