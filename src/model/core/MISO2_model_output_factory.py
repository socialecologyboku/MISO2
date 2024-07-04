#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Logic for aggregating and saving outputs of a MISO2 model object.

Created on Mon Feb 13 11:49:49 2023

@author: bgrammer
"""

import logging as logger
import pandas as pd
import numpy as np
from model.output.MISO2_output import MISO2Output


class MISO2ModelOutputFactory:
    """
    Contains the output creation logic of a MISO2Model.

    The individual methods allow for the creation of Pandas Dataframes that hold control, debug and output information.

    Args:
        model_data(MISO2ModelData): Holds arrays for ongoing calculations.
        mfa_system(MFAsystem): ODYM MFAsystem object.

    Attributes:
        miso_model_data(MISO2ModelData): Holds arrays for ongoing calculations.
        MISO2(MFAsystem): ODYM MFAsystem object.

    """

    __slots__ = "MISO2", "miso_model_data"

    def __init__(self, mfa_system, model_data):
        self.MISO2 = mfa_system
        self.miso_model_data = model_data

    def create_mass_balance_system(self, r, material_index, time_index, year_index):
        """
        Creates mass balance system from MISO2 data objects.

        Args:
            r(int): Regional index.
            material_index(pd.Index): Index of material names
            time_index(pd.Index): Index of time (in array position)
            year_index(pd.Index): Index of years (in actual years)

        Returns:
            mass_bal_system(pd.Dataframe): Mass balance system Dataframe with 
            system-wide (aggregated over processes) balances per material and year
        """
        logger.info("Checking mass balances total")
        # MANUAL BALANCE:
        # system-wide mass-balance over all years, materials,
        # processes (all exog inputs - stock - all outputs)
        remt_idx = np.s_[r, 0, :, year_index["min"]:year_index["max"]]
        r_stock = pd.DataFrame(
            np.einsum('emgtc->mt',
                      self.MISO2.StockDict['S_10'].Values[r, :, :, :, year_index["min"]:year_index["max"]]),
            index=material_index,
            columns=time_index)

        all_inputs = self.MISO2.FlowDict['F_3_4a'].Values[remt_idx] \
                     + self.MISO2.FlowDict['F_3_4b'].Values[remt_idx] \
                     + self.MISO2.FlowDict['F_20_4a'].Values[remt_idx] \
                     + self.MISO2.FlowDict['F_20_4b'].Values[remt_idx] \
                     + self.MISO2.FlowDict['F_20_4c'].Values[remt_idx] \
                     + self.MISO2.FlowDict['F_16_4'].Values[remt_idx] \
                     + self.MISO2.FlowDict['F_17_6'].Values[remt_idx] \
                     + self.MISO2.FlowDict['F_18_8'].Values[remt_idx] \
                     + self.MISO2.FlowDict['F_19_12'].Values[remt_idx]

        all_outputs = self.MISO2.FlowDict['F_14_0'].Values[remt_idx] \
                      + self.MISO2.FlowDict['F_4_16'].Values[remt_idx] \
                      + self.MISO2.FlowDict['F_6_17'].Values[remt_idx] \
                      + self.MISO2.FlowDict['F_8_18'].Values[remt_idx] \
                      + self.MISO2.FlowDict['F_12_19'].Values[remt_idx] \
                      + self.MISO2.FlowDict['F_13_20a'].Values[remt_idx] \
                      + self.MISO2.FlowDict['F_13_20b'].Values[remt_idx] \
                      + self.MISO2.FlowDict['F_13_20c'].Values[remt_idx]

        nas = r_stock - r_stock.T.shift(1).T
        nas.iloc[:, year_index["min"]] = r_stock.iloc[:, year_index["min"]]
        # due to shift we have to reset first year to original values

        bal = self.miso_model_data.bal_4[remt_idx] \
              + self.miso_model_data.bal_6[remt_idx] \
              + self.miso_model_data.bal_8[remt_idx] \
              + self.miso_model_data.bal_12[remt_idx]
        # b_13a_balancing  + r_bal13b + r_bal14 - these two elements not included here; in comparison to
        # balance items 4-12 bal13a/b do not change the mass balance as they just limit
        # the amount of recycled material if not enough new+old scrap is available

        mass_bal_system = all_inputs - all_outputs - nas + bal
        # calculate the sum of all years, except for last year as for this model cannot run entirely

        return mass_bal_system

    def create_mass_balance_process(self, output_dfs):
        """
        Creates mass balances per process and adds balancing items outside ODYM system definition. Aggregation takes
        place over years.

        The method assumes that output dataframes with the correct names are supplied.

        Args:
            output_dfs(dict): Dictionary of Pandas Dataframes.

        Raises:
            KeyError: When a Dataframe necessary for constructing the system is missing.
        """
        # ! implement so that we use ODYM to check 1) mass balance of all processes (total, per year?) +
        # double-checking of yearly and total mass-balances

        logger.info("Checking mass balances odym")
        # initial mass balance checks with ODYM function
        try:
            Bal = self.MISO2.MassBalance()
            logger.debug(f"Dimensions of balance: {Bal.shape}")
            # dimensions of balance are: time step x process x chemical element
            # reports the sum of all absolute balancing errors by process.
            Bal_1 = Bal[:, :, 0]
            # reports the sum of balancing errors by process (with negatives
            # sum over all processes
            F_14_0_waste_final = output_dfs["F_14_0_waste_final"]
            F_3_4a_prod_primary_raw_prod_exog = output_dfs["F_3_4a_prod_primary_raw_prod_exog"]
            F_3_4b_prod_primary_raw_prod_endog = output_dfs["F_3_4b_prod_primary_raw_prod_endog"]
            b_4_balancing = output_dfs["b_4_balancing"]
            b_6_balancing = output_dfs["b_6_balancing"]
            b_8_balancing = output_dfs["b_8_balancing"]
            b_12_balancing = output_dfs["b_12_balancing"]
            S_10_stock_total = output_dfs["S_10_stock_total"]
            F_16_4_IM_raw_prod = output_dfs["F_16_4_IM_raw_prod"]
            F_17_6_IM_semis = output_dfs["F_17_6_IM_semis"]
            F_18_8_IM_finals = output_dfs["F_18_8_IM_finals"]
            F_4_16_EX_raw_prod = output_dfs["F_4_16_EX_raw_prod"]
            F_6_17_EX_semis = output_dfs["F_6_17_EX_semis"]
            F_19_12_IM_waste_recov = output_dfs["F_19_12_IM_waste_recov"]
            F_8_18_EX_finals = output_dfs["F_8_18_EX_finals"]
            F_12_19_EX_waste_recov = output_dfs["F_12_19_EX_waste_recov"]
            F_13_20a_prod_recycle = output_dfs["F_13_20a_prod_recycle"]

            # ! should split this up in slices
            var_outside_ODYM_balance = np.array([
                -F_14_0_waste_final.sum().sum(),
                0, 0, F_3_4a_prod_primary_raw_prod_exog.values.sum().sum()
                + F_3_4b_prod_primary_raw_prod_endog.values.sum().sum(),
                b_4_balancing.sum().sum(), 0, b_6_balancing.sum().sum(), 0, b_8_balancing.sum().sum(),
                0, -S_10_stock_total.iloc[:, -1].sum(), 0, b_12_balancing.sum().sum(), -0, -0, 0,
                F_16_4_IM_raw_prod.sum().sum() - F_4_16_EX_raw_prod.sum().sum(),
                F_17_6_IM_semis.sum().sum() - F_6_17_EX_semis.sum().sum(),
                F_18_8_IM_finals.sum().sum() - F_8_18_EX_finals.sum().sum(),
                F_19_12_IM_waste_recov.sum().sum() - F_12_19_EX_waste_recov.sum().sum(),
                - F_13_20a_prod_recycle.iloc[:, -1].sum().sum()]).reshape((-1, 1))

            # ext_bal1 = Bal_1[:-1,:].sum(axis=0).reshape((21,1)) + var_outside_ODYM_balance
            ext_bal1 = Bal_1[:-1, :].sum(axis=0).reshape((-1, 1)) + var_outside_ODYM_balance

            mass_bal_process = pd.DataFrame(ext_bal1)

        except KeyError as key_error:
            logger.error("Could not find data output necessary to create mass balance")
            logger.error(repr(key_error))
            raise key_error

        return mass_bal_process

    def create_mass_balance_process_annual(self, output_dfs, Nt):
        """
        Creates mass balances per process and adds balancing items outside ODYM system definition for each year.

        The method assumes that output dataframes with the correct names are supplied.

        Args:
            output_dfs(dict): Dictionary of Pandas Dataframes.
            Nt(int): Time dimension

        Raises:
            KeyError: When a Dataframe necessary for constructing the system is missing.
        """
        # double-checking of yearly and total mass-balances

        logger.info("Checking mass balances odym")
        try:
            Bal = self.MISO2.MassBalance()   # initial mass balance checks with ODYM function

            logger.debug(f"Dimensions of balance: {Bal.shape}")
            # dimensions of balance are: time step x process x chemical element
            # reports the sum of all absolute balancing errors by process.
            Bal_1 = Bal[:, :, 0]
            # reports the sum of balancing errors by process (with negatives sum over all processes)
            F_14_0_waste_final = output_dfs["F_14_0_waste_final"]
            F_3_4a_prod_primary_raw_prod_exog = output_dfs["F_3_4a_prod_primary_raw_prod_exog"]
            F_3_4b_prod_primary_raw_prod_endog = output_dfs["F_3_4b_prod_primary_raw_prod_endog"]
            b_4_balancing = output_dfs["b_4_balancing"]
            b_6_balancing = output_dfs["b_6_balancing"]
            b_8_balancing = output_dfs["b_8_balancing"]
            b_12_balancing = output_dfs["b_12_balancing"]
            S_10_stock_total = output_dfs["S_10_stock_total"]
            F_16_4_IM_raw_prod = output_dfs["F_16_4_IM_raw_prod"]
            F_17_6_IM_semis = output_dfs["F_17_6_IM_semis"]
            F_18_8_IM_finals = output_dfs["F_18_8_IM_finals"]
            F_4_16_EX_raw_prod = output_dfs["F_4_16_EX_raw_prod"]
            F_6_17_EX_semis = output_dfs["F_6_17_EX_semis"]
            F_19_12_IM_waste_recov = output_dfs["F_19_12_IM_waste_recov"]
            F_8_18_EX_finals = output_dfs["F_8_18_EX_finals"]
            F_12_19_EX_waste_recov = output_dfs["F_12_19_EX_waste_recov"]
            F_13_20a_prod_recycle = output_dfs["F_13_20a_prod_recycle"]
            F_13_20a_prod_downcycle = output_dfs["F_13_20b_prod_downcycle"]

            var_outside_ODYM_balance = np.row_stack([
                -F_14_0_waste_final.sum(axis=0), np.zeros((1, Nt)), np.zeros((1, Nt)),
                F_3_4a_prod_primary_raw_prod_exog.values.sum(axis=0) + F_3_4b_prod_primary_raw_prod_endog.values.sum(
                    axis=0),
                b_4_balancing.sum(axis=0), np.zeros((1, Nt)), b_6_balancing.sum(axis=0), np.zeros((1, Nt)),
                b_8_balancing.sum(axis=0), np.zeros((1, Nt)),
                S_10_stock_total.sum(axis=0).shift(1).replace(np.nan, 0) - S_10_stock_total.sum(axis=0),
                np.zeros((1, Nt)), b_12_balancing.sum(axis=0), -np.zeros((1, Nt)), -np.zeros((1, Nt)),
                np.zeros((1, Nt)), F_16_4_IM_raw_prod.sum(axis=0) - F_4_16_EX_raw_prod.sum(axis=0),
                F_17_6_IM_semis.sum(axis=0) - F_6_17_EX_semis.sum(axis=0),
                F_18_8_IM_finals.sum(axis=0) - F_8_18_EX_finals.sum(axis=0),
                F_19_12_IM_waste_recov.sum(axis=0) - F_12_19_EX_waste_recov.sum(axis=0),
                F_13_20a_prod_recycle.sum(axis=0).shift(1).replace(np.nan, 0) - F_13_20a_prod_recycle.sum(axis=0)
                + F_13_20a_prod_downcycle.sum(axis=0).shift(1).replace(np.nan, 0) - F_13_20a_prod_downcycle.sum(
                    axis=0)]).T

            odym_bal = Bal_1[:-1, :]

            mass_bal_process_annual = odym_bal + var_outside_ODYM_balance

        except KeyError as key_error:
            logger.error("Could not find data output necessary to create mass balance")
            logger.error(repr(key_error))
            raise key_error

        return mass_bal_process_annual, odym_bal, var_outside_ODYM_balance

    def create_output_dfs(self, r, material_index, time_index, sector_index, year_index, save_stock_cohorts=False):
        """
        Creates the model's output dataframes.

        Args:
            r(int): Regional index.
            material_index(pd.Index): Index of material names
            time_index(pd.Index): Index of time (in array position)
            sector_index(pd.Index): Index of sectors.
            year_index(pd.Index): Index of years (in actual years)
            save_stock_cohorts(bool): Indicates if stock cohorts should be saved. Defaults to false.

        Return:
            output_dfs(dict): Dictionary of Pandas Dataframes.
        """
        output_dfs = {}
        logger.info("Creating output dataframes")

        remt_idx = np.s_[r, :, :, year_index["min"]:year_index["max"]]
        remgt_idx = np.s_[r, :, :, :, year_index["min"]:year_index["max"]]

        # p4
        output_dfs["F_3_4a_prod_primary_raw_prod_exog"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_3_4a'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_3_4b_prod_primary_raw_prod_endog"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_3_4b'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_20_4a_prod_recycle"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_20_4a'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_20_4b_prod_downcycl"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_20_4b'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_20_4c_supply_aggregates"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_20_4c'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_16_4_IM_raw_prod"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_16_4'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_4_16_EX_raw_prod"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_4_16'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["b_4_balancing"] = pd.DataFrame(
            np.einsum('emt->mt', self.miso_model_data.bal_4[remt_idx]),
            index=material_index,
            columns=time_index)

        # p5
        output_dfs["F_4_5_AC_raw_prod"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_4_5'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_5_11a_waste_recov_raw_prod"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_5_11a'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_5_11b_waste_unrecov_raw_prod"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_5_11b'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        # p6
        output_dfs["F_5_6_prod_semis"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_5_6'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_17_6_IM_semis"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_17_6'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_6_17_EX_semis"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_6_17'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["b_6_balancing"] = pd.DataFrame(
            np.einsum('emt->mt', self.miso_model_data.bal_6[remt_idx]),
            index=material_index,
            columns=time_index)

        # p7
        output_dfs["F_6_7_AC_semis"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_6_7'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_7_11a_waste_recov_semis"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_7_11a'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_7_11b_waste_unrecov_semis"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_7_11b'].Values[remt_idx]),
            index=material_index,
            columns=time_index)
        # p8

        output_dfs["F_7_8_prod_finals"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_7_8'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_18_8_IM_finals"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_18_8'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_8_18_EX_finals"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_8_18'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["b_8_balancing"] = pd.DataFrame(
            np.einsum('emt->mt', self.miso_model_data.bal_8[remt_idx]),
            index=material_index,
            columns=time_index)

        # p9
        output_dfs["F_8_9_AC_finals"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_8_9'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_9_11a_waste_recov_finals"] = pd.DataFrame(
            np.einsum('emgt->mt', self.MISO2.FlowDict['F_9_11a'].Values[remgt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_9_11b_waste_unrecov_finals"] = pd.DataFrame(
            np.einsum('emgt->mt', self.MISO2.FlowDict['F_9_11b'].Values[remgt_idx]),
            index=material_index,
            columns=time_index)

        # p10

        output_dfs["F_9_10_GAS"] = pd.DataFrame(
            np.einsum('emgt->mt', self.MISO2.FlowDict['F_9_10'].Values[remgt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["S_10_stock_total"] = pd.DataFrame(
            np.einsum('emgtc->mt', self.MISO2.StockDict['S_10'].Values[remgt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_10_11_supply_EoL_waste"] = pd.DataFrame(
            np.einsum('emgt->mt', self.MISO2.FlowDict['F_10_11'].Values[remgt_idx]),
            index=material_index,
            columns=time_index)

        # p11
        output_dfs["F_11_12_waste_recov_domest"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_11_12'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_11_14_waste_unrecov"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_11_14'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        # p12
        output_dfs["F_12_13_supply_waste_recov"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_12_13'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_19_12_IM_waste_recov"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_19_12'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_12_19_EX_waste_recov"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_12_19'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["b_12_balancing"] = pd.DataFrame(
            np.einsum('emt->mt', self.miso_model_data.bal_12[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["b_13c_balancing"] = pd.DataFrame(
            np.einsum('emt->mt', self.miso_model_data.bal_13c[remt_idx]),
            index=material_index,
            columns=time_index)

        # p13 & p14
        output_dfs["F_13_14_waste_recov_unused"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_13_14'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_14_0_waste_final"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_14_0'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_13_20a_prod_recycle"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_13_20a'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_13_20b_prod_downcycle"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_13_20b'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["F_13_20c_supply_aggregates"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.FlowDict['F_13_20c'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        # Additional parameters of interest
        output_dfs["r_downcycled_constr"] = pd.DataFrame(
            np.einsum('emt->mt', self.miso_model_data.downc_constr_mats[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["r_waste_afterRecycling"] = pd.DataFrame(
            np.einsum('emt->mt', self.miso_model_data.rem_waste_Cycling_noAggr[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["cementBitumen_2_concreteAsphalt"] = pd.DataFrame(
            np.einsum('emgt->mt', self.miso_model_data.cementBitumen_2_concreteAsphalt[remgt_idx]).reshape(
                -1, len(time_index)),
            index=material_index,
            columns=time_index)

        output_dfs["ratio_old_scrap"] = pd.DataFrame(
            np.einsum('emt->mt', self.miso_model_data.ratio_old_scrap[remt_idx]),
            index=material_index,
            columns=time_index)

        output_dfs["new_scrap_recycled"] = pd.DataFrame(
            np.einsum('emt->mt', self.miso_model_data.new_scrap_recycled[remt_idx]),
            index=material_index,
            columns=time_index)
        
        output_dfs["new_scrap_supply"] = pd.DataFrame(
            np.einsum('emt->mt', self.miso_model_data.new_scrap[remt_idx]),
            index=material_index,
            columns=time_index)

        # Dataframes that include dimension g (sector/end-use)

        multi_index = pd.MultiIndex.from_product([material_index, sector_index])
        mgt = np.einsum('emgtc->mgt', self.MISO2.StockDict['S_10'].Values[remgt_idx])

        output_dfs["S10_stock_enduse"] = pd.DataFrame(
            mgt.reshape(-1, len(time_index)),
            index=multi_index,
            columns=time_index)

        output_dfs["F_9_10_GAS_enduse"] = pd.DataFrame(
            np.einsum('emgt->mgt', self.MISO2.FlowDict['F_9_10'].Values[remgt_idx]).reshape(-1, len(time_index)),
            index=multi_index,
            columns=time_index)

        output_dfs["F_10_11_supply_EoL_waste_enduse"] = pd.DataFrame(
            np.einsum('emgt->mgt', self.MISO2.FlowDict['F_10_11'].Values[remgt_idx]).reshape(-1, len(time_index)),
            index=multi_index,
            columns=time_index)

        if save_stock_cohorts:
            remgtt_idx = np.s_[r, :, :, :, year_index["min"]:year_index["max"], year_index["min"]:year_index["max"]]

            multi_index_time = pd.MultiIndex.from_product([material_index, sector_index, time_index])

            mgtc = np.einsum('emgtc->mgtc', self.MISO2.StockDict['S_10'].Values[remgtt_idx])

            output_dfs["S10_stock_enduse_cohorts"] = pd.DataFrame(
                mgtc.reshape(-1, len(time_index)),
                index=multi_index_time,
                columns=time_index)

        return output_dfs

    def create_debug_dfs(self, r, material_index, multiplier_material_index, time_index, year_index, sector_index):
        """
        Creates output dataframes that are used for debugging and error checking.

        Args:
            r(int): Regional index.
            material_index(pd.Index): Index of material names
            time_index(pd.Index): Index of time (in array position)
            year_index(pd.Index): Index of years (in actual years)
            multiplier_material_index(pd.Index): Index of multiplier materials.
            sector_index(pd.Index): Index of sectors (by name)

        Returns:
            debug_output_dfs(dict): Dictionary of pd. Dataframes.

        """
        debug_output_dfs = {}

        remt_idx = np.s_[r, :, :, year_index["min"]:year_index["max"]]
        # remgt_idx = np.s_[r, :, :, :, year_index["min"]:year_index["max"]]

        debug_output_dfs["stock_S20a"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.StockDict['S_20a'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        debug_output_dfs["stock_S20b"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.StockDict['S_20b'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        debug_output_dfs["stock_S20c"] = pd.DataFrame(
            np.einsum('emt->mt', self.MISO2.StockDict['S_20c'].Values[remt_idx]),
            index=material_index,
            columns=time_index)

        debug_output_dfs["b_13a_balancing"] = pd.DataFrame(
            np.einsum('emt->mt', self.miso_model_data.bal_13a[remt_idx]),
            index=material_index,
            columns=time_index)

        debug_output_dfs["b_13b_balancing"] = pd.DataFrame(
            np.einsum('emt->mt', self.miso_model_data.bal_13b[remt_idx]),
            index=material_index,
            columns=time_index)

        aggr_req_mg_index = pd.MultiIndex.from_product([multiplier_material_index, sector_index])

        debug_output_dfs["aggr_req_mg"] = pd.DataFrame(
            np.einsum('mgt->mgt',
                      self.miso_model_data.aggr_req_mg[:, :, year_index["min"]:year_index["max"], 0]).reshape(
                -1, len(time_index)),
            index= aggr_req_mg_index,
            columns=time_index)

        # # selector needs to be passed as arguments
        # self.aggr_req_mg = np.zeros((([
        #     len(selector_multiplier_materials), len(dimensions["endUse_aggregates"]),
        #     dimensions["Nt"]]))).reshape((
        #     len(selector_multiplier_materials), dimensions["Ng"], dimensions["Nt"], dimensions["Ne"]))
        # estimate on required aggregates for building and road foundations by material and end-use

        return debug_output_dfs

    def create_miso_output(self, region, mfa_system_control, run_options, unique_id, output_dfs):
        """
        Creates a miso_output object loaded with the specified metadata and dataframes.

        The method will also run a self-check of the output object. Indices of the dataframes are
        assumed to be correctly set.

        Args:
            region(str): Name of the region
            mfa_system_control(dict): Dictionary of model's system control parameter.
            run_options(dict): Dictionary of model's run options.
            unique_id(UUID): The MISO configs unique id.
            output_dfs(dict): Dictionary of pandas dataframe that are added to the output.

        Returns:
            miso_output(MISO2Output): The MISO2Output object of the model run.
        """

        logger.info("Creating MISO output object")
        miso_output = MISO2Output(mfa_system_control=mfa_system_control,
                                  run_options=run_options,
                                  region=region,
                                  config_id=str(unique_id))

        for output_name, output_df in output_dfs.items():
            miso_output.add_result(name=output_name, df=output_df, description="result")

        miso_output.check_output_values()

        return miso_output
