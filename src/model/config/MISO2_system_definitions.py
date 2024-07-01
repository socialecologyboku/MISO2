#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:57:35 2022

src System definitions.

@author: JS, BP, BG
"""
import logging as logger
import numpy as np
import odym.ODYM_Classes as msc


def define_MFA_System_MISO2v7(MISO2, PrL_Number, PrL_Name, dimensions):
    """
    Function to define src MFA System v7 using ODYM.

    The function implements the src system definition v7 (:ref:`system definition`). The function contains the \
    model control parameters that are later used for error-checking in the MFA_system_control variable \
    of the MISO2Model object, such as which Stocks and Flows may contain all-zero values and the total \
    number of system stocks and flows.

    Args:
        MISO2(MFAsystem): ODYM MFASystem object which is loaded with src processes, stocks and flows.
        PrL_Number(list): List with PrL numbers.
        PrL_Name(list): List of PrL names.
        dimensions(dict): Dictionary with Nr,Ne,Nm,Nt and Ng dimensions as integers.

    Returns:
        MFA_system_control(dict): MFA system control parameters used for error checking and metadata creation. \
            Entries are "system_definition", number_stocks", "number_flows", "stock_names, "flow_names",
            "all_zero_ok", "no_data_yet" and "must_be_zero".
    """

    logger.info("Creating MISO system definition")
    # Add processes to system
    for p_id, m in enumerate(PrL_Number):
        MISO2.ProcessList.append(msc.Process(Name=PrL_Name[m], ID=p_id))

    # Define the flows and stocks of the system, and initialise their values:
    # process 1: domestic extraction
    remt_index = 'r,e,m,t'
    remgt_index = 'r,e,m,g,t'
    remt_zero_array = np.zeros((
        dimensions["Nr"], dimensions["Ne"], dimensions["Nm"], dimensions["Nt"]))
    remgt_zero_array = np.zeros((
        dimensions["Nr"], dimensions["Ne"], dimensions["Nm"], dimensions["Ng"], dimensions["Nt"]))

    MISO2.FlowDict['F_0_1'] = msc.Flow(
        Name='domestic extraction crude materials', P_Start=0, P_End=1,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_1_11a'] = msc.Flow(
        Name='waste crude materials, recoverable', P_Start=1, P_End=11,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_1_11b'] = msc.Flow(
        Name='waste crude  materials, unrecoverable', P_Start=1, P_End=11,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_1_2'] = msc.Flow(
        Name='domestic extraction raw materials', P_Start=1, P_End=2,
        Indices=remt_index, Values=remt_zero_array.copy())

    # Process 2: trade raw materials
    # F_1_2' under process1
    MISO2.FlowDict['F_2_15'] = msc.Flow(
        Name='export raw materials', P_Start=2, P_End=15,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_15_2'] = msc.Flow(
        Name='import raw materials', P_Start=15, P_End=2,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_2_3'] = msc.Flow(
        Name='apparent consumption raw materials', P_Start=2, P_End=3,
        Indices=remt_index, Values=remt_zero_array.copy())

    # Process 3: refining and smelting to raw products
    # F_2_3' under process2
    MISO2.FlowDict['F_3_11a'] = msc.Flow(
        Name='waste raw materials, recoverable', P_Start=3, P_End=11,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_3_11b'] = msc.Flow(
        Name='waste raw  materials, unrecoverable', P_Start=3, P_End=11,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_3_4a'] = msc.Flow(
        Name='primary production raw products (exog)', P_Start=3, P_End=4,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_3_4b'] = msc.Flow(
        Name='primary production raw products (calculated from exog. total production',
        P_Start=3, P_End=4,
        Indices=remt_index, Values=remt_zero_array.copy())

    # process 4: trade raw products
    # F_3_4a/b' under process3
    MISO2.FlowDict['F_20_4a'] = msc.Flow(
        Name='domestically recycled raw products in year t', P_Start=20, P_End=4,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_20_4b'] = msc.Flow(
        Name='domestically downcycled raw products in year t', P_Start=20, P_End=4,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_4_16'] = msc.Flow(
        Name='export raw products', P_Start=4, P_End=16,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_16_4'] = msc.Flow(
        Name='import raw products', P_Start=16, P_End=4,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_4_5'] = msc.Flow(
        Name='apparent consumption raw products', P_Start=4, P_End=5,
        Indices=remt_index, Values=remt_zero_array.copy())

    # process 5: fabrication of semi-manufactured products
    # F_4_5' under process4
    MISO2.FlowDict['F_5_11a'] = msc.Flow(
        Name='waste raw products, recoverable', P_Start=5, P_End=11,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_5_11b'] = msc.Flow(
        Name='waste raw products, unrecoverable', P_Start=5, P_End=11,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_5_6'] = msc.Flow(
        Name='production semi-manufactured products', P_Start=5, P_End=6,
        Indices=remt_index, Values=remt_zero_array.copy())

    # process 6: trade semi-manufactured products
    # F_5_6' under process5
    MISO2.FlowDict['F_6_17'] = msc.Flow(
        Name='export semi-manufactured products', P_Start=6, P_End=17,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_17_6'] = msc.Flow(
        Name='import semi-manufactured products', P_Start=17, P_End=6,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_6_7'] = msc.Flow(
        Name='apparent consumption semi-manufactured products', P_Start=6, P_End=7,
        Indices=remt_index, Values=remt_zero_array.copy())

    # process 7: manufacture of final products
    # F_6_7' under process6
    MISO2.FlowDict['F_7_11a'] = msc.Flow(
        Name='waste semi-manufactured products, recoverable', P_Start=7, P_End=11,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_7_11b'] = msc.Flow(
        Name='waste semi-manufactured products, unrecoverable', P_Start=7, P_End=11,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_7_8'] = msc.Flow(
        Name='production final products', P_Start=7, P_End=8,
        Indices=remt_index, Values=remt_zero_array.copy())

    # process 8: trade of final products
    # F_7_8' under process7
    MISO2.FlowDict['F_8_18'] = msc.Flow(
        Name='export final products', P_Start=8, P_End=18,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_18_8'] = msc.Flow(
        Name='import final products', P_Start=18, P_End=8,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_8_9'] = msc.Flow(
        Name='apparent consumption final products', P_Start=8, P_End=9,
        Indices=remt_index, Values=remt_zero_array.copy())

    # process 9: gross additions to stock of final products by end-use
    # F_8_9' under process8
    MISO2.FlowDict['F_9_11a'] = msc.Flow(
        Name='waste final products, recoverable', P_Start=9, P_End=11,
        Indices=remgt_index, Values=remgt_zero_array.copy())

    MISO2.FlowDict['F_9_11b'] = msc.Flow(
        Name='waste final products, unrecoverable', P_Start=9, P_End=11,
        Indices=remgt_index, Values=remgt_zero_array.copy())

    # #process 9: gross additions to stock of final products by end-use
    # #'F_8_9' under process8
    # FlowDict['F_9_11a'] = msc.Flow(
    #     Name = 'waste final products, recoverable', P_Start = 9, P_End = 11,
    #     Indices = 'r,e,m,t', Values= np.zeros((
    #         dimensions["Nr"],dimensions["Ne"],dimensions["Nm"],dimensions["Nt"])))

    # FlowDict['F_9_11b'] = msc.Flow(
    #     Name = 'waste final products, unrecoverable', P_Start = 9, P_End = 11,
    #     Indices = 'r,e,m,t', Values= np.zeros((
    #         dimensions["Nr"],dimensions["Ne"],dimensions["Nm"],dimensions["Nt"])))

    MISO2.FlowDict['F_9_10'] = msc.Flow(
        Name='gross additions to stock of final products', P_Start=9, P_End=10,
        Indices=remgt_index, Values=remgt_zero_array.copy())

    # Process 10: use
    # F_9_10' under process9
    MISO2.StockDict['S_10'] = msc.Stock(
        Name='in-use stock by cohort', P_Res=10, Type=0,
        Indices='r,e,m,g,t,c', Values=np.zeros((
            dimensions["Nr"], dimensions["Ne"], dimensions["Nm"], dimensions["Ng"],
            dimensions["Nt"], dimensions["Nt"])))

    MISO2.FlowDict['F_10_11'] = msc.Flow(
        Name='EoL outflows, collected', P_Start=10, P_End=11,
        Indices=remgt_index, Values=remgt_zero_array.copy())

    # process 11: collection of EoL & waste materials
    # All inputs in former processes 1-10
    # with this definition we assume, that only recoverable waste is traded-->
    MISO2.FlowDict['F_11_12'] = msc.Flow(
        Name='waste, recoverable', P_Start=11, P_End=12,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_11_14'] = msc.Flow(
        Name='waste, unrecoverable', P_Start=11, P_End=14,
        Indices=remt_index, Values=remt_zero_array.copy())

    # process 12: trade of waste materials (recoverable)
    # F_11_12' under process11
    MISO2.FlowDict['F_12_19'] = msc.Flow(
        Name='export waste, recoverable', P_Start=12, P_End=19,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_19_12'] = msc.Flow(
        Name='import waste, recoverable', P_Start=19, P_End=12,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_12_13'] = msc.Flow(
        Name='domestic supply waste, recoverable', P_Start=12, P_End=13,
        Indices=remt_index, Values=remt_zero_array.copy())

    # process 13: recycling and downcycling of recoverable waste to raw products
    # F_11_12a/b under process 4
    MISO2.FlowDict['F_13_14'] = msc.Flow(
        Name='waste, recoverable but not recovered', P_Start=13, P_End=14,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_13_20a'] = msc.Flow(
        Name='domestically recycled raw products in year t+1', P_Start=13, P_End=20,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_13_20b'] = msc.Flow(
        Name='domestically downcycled raw products in year t+1', P_Start=13, P_End=20,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_13_20c'] = msc.Flow(
        Name='recycled / downcycled aggregate supply (virgin & downcycled)', P_Start=13, P_End=20,
        Indices=remt_index, Values=remt_zero_array.copy())

    # process 14: final waste management
    MISO2.FlowDict['F_14_0'] = msc.Flow(
        Name='final waste', P_Start=14, P_End=0,
        Indices=remt_index, Values=remt_zero_array.copy())

    # process 20: time buffer for recycled and downcycled materials
    MISO2.StockDict['S_20a'] = msc.Stock(
        Name='stocking of recycled materials for one year', P_Res=20, Type=0,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.StockDict['S_20b'] = msc.Stock(
        Name='stocking of downcycled materials for one year', P_Res=20, Type=0,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.StockDict['S_20c'] = msc.Stock(
        Name='stocking of recycled / downcycled aggregate supply for one year (virgin & downcycled)',
        P_Res=20, Type=0,
        Indices=remt_index, Values=remt_zero_array.copy())

    MISO2.FlowDict['F_20_4c'] = msc.Flow(
        Name='recycled / downcycled aggregate supply (virgin & downcycled)', P_Start=20, P_End=4,
        Indices=remt_index, Values=remt_zero_array.copy())

    MFA_system_control = {
        "system_definition": "MISO2v7",
        "number_stocks": len(MISO2.StockDict),
        "number_flows": len(MISO2.FlowDict),
        "flow_names": list(MISO2.FlowDict.keys()),
        "stock_names": list(MISO2.StockDict.keys()),
        "all_zero_ok": ['F_0_1', 'F_1_11a', 'F_1_11b', 'F_1_2', 'F_2_15',
                        'F_15_2', 'F_2_3', 'F_3_11a', 'F_3_11b',
                        'F_13_20c', 'F_20_4c', "S_20a", "S_20b", "S_20c"],
        "no_data_yet": ['F_9_11a'],
        "must_be_zero": ["F_13_20c", "F_20_4c"]
        }

    MISO2.Initialize_FlowValues()
    # Assign empty arrays to flows according to dimensions.
    MISO2.Initialize_StockValues()
    # Assign empty arrays to flows according to dimensions.
    MISO2.Consistency_Check()
    # Check whether flow value arrays match their indices, etc. See method documentation.

    logger.info("MISO system definition successfully created")
    return MFA_system_control
