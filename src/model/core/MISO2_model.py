#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Core logic of calculating a MISO2 model.

Created on Wed Feb 16 20:57:53 2022

@author: jstreeck & bgrammer
"""

import logging as logger
import numpy as np
import pandas as pd
import odym.ODYM_Classes as msc
import odym.dynamic_stock_model as dsm
from model.core.MISO2_model_data import MISO2ModelData
from model.core.MISO2_model_output_factory import MISO2ModelOutputFactory
from preprocessing.checks.MISO2_checks import check_enduse_values, check_exclusive_values, check_limits, \
    check_waste_ratios, check_mutually_exclusive_arrays
from util.MISO2_exceptions import WasteRateError, InputLimitError, ValueOverlapError, EnduseSummationError
from util.MISO2_functions import nb_any, equalize_enduses, set_arrays_exclusive_in_order


class MISO2Model:
    """
    Contains the logic for calculating a MISO2 Model.

    Args:
        miso_config(MISO2Config): Model configuration. Should not be written to from the model object.
        miso_survival_functions(MISO2SurvivalFunctions): Precomputed survival functions.
        nr_start(int): Region start index. This needs only be set when the MISO2Config data contains \
            more than one region and the data needs to be subset for computation. Defaults to 0.
        nr_stop(int): Region stop index. Defaults to 1.
        balance_error_tolerance(float): Error tolerance for mass balancing. Defaults to 1e-5.

    Attributes:
        miso_config(MISO2Config): Model configuration. Should not be written to from the model object.
        sf_array(MISO2SurvivalFunctions): Precomputed survival functions.
        balance_error_tolerance(float): Error tolerance for mass balancing.
        MISO2(MFAsystem): ODYM MFAsystem object.
        miso_model_data(MISO2ModelData): Holds arrays for ongoing calculations.
        MFA_system_control(dict): Used for error checking. Holds control values of system, \
            such as information about expected zero and non-zero values of MFA system after model run.

    Raises:
        AttributeError: If UUID of MISO2Config and MISO2SurvivalFunctions do not match up.
    """

    # requirements:
    # 1. no trade of aggr_4concr / _4asph allowed (aggregates only traded for aggr_virgin)
    # 2. trade of aggr_virgin only allowed at p4
    # 3. trade of aggr_downcycl not allowed
    # 4. input material flow data is not to contain information on aggregates virgin or downcycled
    # (as these estimated in aggregates gap)

    __slots__ = "miso_config", "MISO2", "miso_model_data", "_debug_output", \
        "balance_error_tolerance", "sf_array", "nr_start", "nr_stop", "MFA_system_control", "mass_bal_process", \
        "mass_bal_system", "mass_bal_process_annual", "odym_bal", "var_outside_ODYM_balance"

    def __init__(self, miso_config, miso_survival_functions, nr_start=0,
                 nr_stop=1, balance_error_tolerance=1e-5):

        valid_nr_range = range(miso_config.nr_start, miso_config.nr_stop)

        if nr_start not in valid_nr_range or nr_stop - 1 not in valid_nr_range:
            error_msg = f"Nr start: {nr_start} or nr stop: {nr_stop} passed to MISO2Model not within \
                range of MISO2Configs({miso_config.nr_start},{miso_config.nr_stop}) values"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if miso_config.unique_id == miso_survival_functions.miso_config_id:
            self.sf_array = miso_survival_functions.get_sf_array(nr_start, nr_stop)
        else:
            error_msg = "Aborted construction of MISO model since IDs of config \
                              and survival functions do not match. \
                            The survival functions were likely generated from a different config"
            logger.error(error_msg)
            raise AttributeError(error_msg)

        self.miso_config = miso_config
        self.nr_start = nr_start
        self.nr_stop = nr_stop

        self.MISO2 = msc.MFAsystem(Name='MISO2_Global',
                                   Geogr_Scope='Global',
                                   Unit='Mt',
                                   ProcessList=[],
                                   FlowDict={},
                                   StockDict={},
                                   ParameterDict=self.miso_config.get_parameter_dict(nr_start, nr_stop),
                                   Time_Start=miso_config.model_time_start,
                                   Time_End=miso_config.model_time_end,
                                   IndexTable=miso_config.index_table,
                                   Elements=miso_config.index_table.loc['Element'].Classification.Items)

        self.miso_model_data = self._create_data_structures(miso_config)

        self.balance_error_tolerance = balance_error_tolerance

        self.MFA_system_control = None

        self._debug_output = {}

        self.mass_bal_system = None
        self.mass_bal_process = None
        self.mass_bal_process_annual = None
        self.odym_bal = None
        self.var_outside_ODYM_balance = None

    def _create_data_structures(self, miso_config):
        """
        Wrapper for creating model data structures.

        Important: Due to legacy reasons, we construct arrays only for exactly one region.

        Args:
            miso_config(MISO2Config)

        Returns:
            model_data(MISO2ModelData)
        """
        model_data = MISO2ModelData()
        model_data.create_data_structures(dimensions={"Nr": 1,
                                                      "Ne": miso_config.Ne,
                                                      "Nt": miso_config.Nt,
                                                      "Ng": miso_config.Ng,
                                                      "Nm": miso_config.Nm,
                                                      "endUse_aggregates": miso_config.endUse_aggregates},
                                          selector_multiplier_materials=miso_config.selectors["multiplier_materials"])
        return model_data

    def define_mfa_system(self, mfa_system_initializer):
        """
        Load MFA system into the MISO2 model.

        Since the model core logic relies on access to the correct parameter names to
        model a system, the model logic is at the moment customized towards the most recent MFA
        system definition. In future versions, this should be replaced with a mapping of MFA systems to appropriate
        model logic and output functions.
        The MFA System initializer sets the control variables of the system (MFA_system_control) for
        later error checking and metadata creation.

        Args:
            mfa_system_initializer(func): A MISO2_system_definitions function.

        """
        logger.info("Initialising MFA System")
        self.MFA_system_control = mfa_system_initializer(MISO2=self.MISO2,
                                                         PrL_Number=self.miso_config.PrL_Number,
                                                         PrL_Name=self.miso_config.PrL_Name,
                                                         dimensions={"Nr": 1,
                                                                     "Ne": self.miso_config.Ne,
                                                                     "Nt": self.miso_config.Nt,
                                                                     "Ng": self.miso_config.Ng,
                                                                     "Nm": self.miso_config.Nm})

    def run_model(self, estimate_aggregates=True, save_stock_cohorts=True, save_last_year=False):
        """
        Run the MISO2 model.

        The parameters for the model run are taken from the models nr_start and nr_stop values.
        A start and stop region index can be specified, but the user needs to
        make sure that this index is covered by the MISO2Configs data and the MISO2SurvivalFunctions
        SF_Array.

        Args:
            estimate_aggregates(bool): If True, foundation aggregates will be estimated. Defaults to True.
            save_stock_cohorts(bool): If True, stock cohorts will be added to output. \
                They considerably increase the output size. Defaults to False.
            save_last_year(bool): If True, the last year of the model run will be included in \
                the output. Values of the last year may contain nonsensical or NaN values and \
                    are usually not desired in the output, but may be useful for debugging.

        Returns:
            miso_output(MISO2Output): Results of the model run.

        Raises:
            WasteRateError: If waste ratios exceed 1.
            InputLimitError: If any data point violates parameter hard limits.
        """

        ##########################################################

        #                    Error checks

        ##########################################################
        exclusive_parameters = [
            ("MISO2_EoLRateRecycling", "MISO2_TotalRateRecycling"),
            ("MISO2_EoLAbsoluteRecycling", "MISO2_TotalAbsoluteRecycling"),
            ("MISO2_EoLAbsoluteDowncycling", "MISO2_EoLRateDowncycling"),
            ("MISO2_EoLAbsoluteRecycling", "MISO2_EoLRateRecycling"),
            ("MISO2_TotalAbsoluteRecycling", "MISO2_TotalRateRecycling")]
        # ("MISO2_EoLAbsoluteDowncycling", "MISO2_TotalAbsoluteDowncycling"),
        # ("MISO2_TotalAbsoluteDowncycling", "MISO2_TotalRateDowncycling")
        # ("MISO2_EoLRateDowncycling", "MISO2_TotalRateDowncycling")
        # no data yet

        waste_hierarchy = [
            ("MISO2_TotalAbsoluteRecycling", self.MISO2.ParameterDict["MISO2_TotalAbsoluteRecycling"].Values),
            ("MISO2_EoLAbsoluteRecycling", self.MISO2.ParameterDict["MISO2_EoLAbsoluteRecycling"].Values),
            ("MISO2_TotalRateRecycling", self.MISO2.ParameterDict["MISO2_TotalRateRecycling"].Values),
            ("MISO2_EoLRateRecycling", self.MISO2.ParameterDict["MISO2_EoLRateRecycling"].Values),
        ]

        downcycling_exclusive = {
            "rate": self.MISO2.ParameterDict["MISO2_EoLRateDowncycling"].Values,
            "absolute": self.MISO2.ParameterDict["MISO2_EoLAbsoluteDowncycling"].Values,
            "material_position": self.miso_config.material_position
        }
 
        set_arrays_exclusive_in_order(waste_hierarchy)

        try:
            check_mutually_exclusive_arrays(waste_hierarchy, "normal")
            # in normal mode, simply checks arrays in successive order
            check_mutually_exclusive_arrays(downcycling_exclusive, "downcycling")
            # logic for this is hardcoded, dict arguments must match what the function expects

            check_waste_ratios(
                self.MISO2.ParameterDict)
            check_limits(
                self.MISO2.ParameterDict,
                self.miso_config.uncertainty_settings)
            check_exclusive_values(
                self.MISO2.ParameterDict,
                exclusive_parameters)
        except (WasteRateError, InputLimitError, ValueOverlapError) as error:
            logger.exception(error)
            raise

        enduse_selectors = self.miso_config.selectors["selector_NoAggr"]
        # indices for materials that should have enduses at the start of a model run
        non_enduse_selectors = self.miso_config.selectors["selector_Aggr"]
        # indices for materials whose enduses get calculated during a model run, should be initialised by zero
        selector_multiplier_materials = self.miso_config.selectors["multiplier_materials"]

        # not making a copy if we get into check logic
        try:
            check_enduse_values(
                self.MISO2.ParameterDict["MISO2_EndUseShares"],
                enduse_selectors, non_enduse_selectors, 0.001)
            enduse_shares = self.MISO2.ParameterDict["MISO2_EndUseShares"].Values.copy()

        except EnduseSummationError as error:
            logger.error("Enduses do not sum up to 1. Trying to repair this by scaling the enduse arrays."
                         "This may have been caused by only slight offsets, but there could also be grave errors in \
                         the data. You should manually verify this.")
            logger.exception(error)
            equalized_enduses = equalize_enduses(self.MISO2.ParameterDict["MISO2_EndUseShares"].Values,
                                                 enduse_selectors)
            self.MISO2.ParameterDict["MISO2_EndUseShares"].Values = equalized_enduses.copy()
            check_enduse_values(
                self.MISO2.ParameterDict["MISO2_EndUseShares"],
                enduse_selectors, non_enduse_selectors, 0.001)
            enduse_shares = equalized_enduses

        ##########################################################

        #                    MISO-2 dynamic stock-flow MODEL CORE

        ##########################################################

        logger.info("Assigning data to MFA System and solving")

        # Start of MODEL CORE
        for r in range(0, 1):
            dimensions = {"Ne": self.miso_config.Ne,
                          "Nt": self.miso_config.Nt,
                          "Ng": self.miso_config.Ng,
                          "Nm": self.miso_config.Nm
                          }
            # array to document requirement for aggregates; needs to be initialized within
            # regions loop; per end-use and material
            cementBitumen_2_concreteAsphalt = self.miso_model_data.cementBitumen_2_concreteAsphalt
            # array to save F_8_9 split by end-use
            appar_consump_end_use = self.miso_model_data.appar_consump_end_use
            # array to save aggregate requirements by material and end-use
            aggr_req_mg = self.miso_model_data.aggr_req_mg

            # multiplier_cementBitumen incoming from exogenous input data (see module 'config' script MISO2_config.py)
            multiplier_cementBitumen = np.expand_dims(
                self.miso_config.multiplier_cementBitumen.copy(), axis=0)
            # expand dimension to one since ODYM expects a dummy element in array shape
            # requires more dimensions / copy of data for > 1 elements
            
            #!
            endUse_aggregates = self.miso_config.endUse_aggregates
            # selector for materials that are aggregates
            selector_Aggr = self.miso_config.selectors["selector_Aggr"]
            # selector for materials that aren't aggregates
            selector_NoAggr = self.miso_config.selectors["selector_NoAggr"]
            # separator combined above selectors
            separator_OthAggr = self.miso_config.selectors["separator_OthAggr"]
            # selector for materials subject to downcycling
            selector_downc_mats = self.miso_config.selectors["selector_downc_mats"]
            # array to save excess trade in aggregates not required for use
            trade_larger_req_g = self.miso_model_data.trade_larger_req_g
            
            # buffer for timestep at p20
            buffer_flows_stocks = {
                "F_20_4a": "S_20a",
                "F_20_4b": "S_20b",
                "F_20_4c": "S_20c"
            }
            buffer_stocks_flows = {
                "S_20a": "F_13_20a",
                "S_20b": "F_13_20b",
                "S_20c": "F_13_20c"
            }
            
            # core calculations
            for t in range(0, dimensions["Nt"]):
                year = self.miso_config.model_time_start + t
                
                #! new scrap cycle still constant over years and regions???
                # globally set new scrap cycling rates per material (constant over all years and regions)
                new_scrap_cycle = np.expand_dims(self.miso_config.get_new_scrap_cycle(year), axis=0)
                # start year-loop with deriving recycled/downcycled material from last year from buffer S20
                self.derive_cycled_mat_from_last_year(r, t, buffer_flows_stocks)
                # Assign primary production and total production from input data and adjust to flow definitions 
                self.P4_generate_production_raw_products(r, selector_NoAggr, t)

                # LOOP twice:
                # once over all materials except aggregates (required to calculate
                # flow F_9_10 of bricks, concrete, asphalt for estimating aggregate requirements),
                # and once over aggregates 4concr + 4asph + virgin + downcycled
                for mats in separator_OthAggr:
                    # Add raw product trade, calculate apparent consumption and balance P4.
                    self.P4_add_raw_product_trade(r, mats, t)
                    # Fabricate semi-finished products via subtracting recoverable (a) & unrecoverable (b) waste during fabrication
                    self.P5_fabricate_semi_finished_products(r, mats, t)
                    # Add trade of semi-finished products, calculate apparent consumption and balance P6
                    self.P6_trade_semi_finished_products(r, mats, t)
                    # Manufacture final products via subtracting recoverable (a) & unrecoverable (b) waste during manufacturing
                    self.P7_manufacture_final_products(r, mats, t)
                    # Add trade of final products, calculate apparent consumption and balance P8
                    self.P8_trade_final_products(r, mats, t)
                    # Introduce end-use shares
                    appar_consump_end_use[r, :, mats, :, t] = \
                        self.P9_1_introduce_enduse_shares(r, mats, t, enduse_shares)
                    # calculate waste factors over supply chain
                    waste_cascade_dict = self.calc_waste_cascades(r, t)
                    # change of material categories: cement & bitumen + sand & gravel --> concrete & asphalt
                    self.P9_2_change_material_categories(r, t, cementBitumen_2_concreteAsphalt,
                                                         appar_consump_end_use, multiplier_cementBitumen,
                                                         waste_cascade_dict)
                    # Set P9 standard flows F9_11, recoverable (a) & unrecoverable (b) waste during fabrication
                    self.P9_2_recov_unrecov_waste_flows(r, mats, t, cementBitumen_2_concreteAsphalt,
                                                        selector_Aggr, trade_larger_req_g)
                    # estimate net trade of aggregates at p9
                    net_trade_aggr_p9 = self.calc_net_trade_aggr_p9(r, t, waste_cascade_dict)
                    
                    # CONDITIONAL: only for materials that are no aggregates
                    # Aggregate Gap: estimate demand of aggregates, appraise supply from recycled and \
                        # downcycled aggregates (from last year) and calc. recycled/downcycled use and additional required virgin (in current year)
                    if mats == selector_NoAggr and estimate_aggregates:
                        trade_larger_req_g = self.P9_3_aggregate_gap(r, t, enduse_shares, endUse_aggregates,
                                                                     selector_downc_mats, selector_multiplier_materials, 
                                                                     aggr_req_mg, waste_cascade_dict, net_trade_aggr_p9)

                    # now that we know the actual amount of cycled aggregates, we can calculate final waste flows
                    # of last year
                    # total supply of recoverable waste (13_14), less recycled material (13_20a),
                    # less downcycled asphalt, bricks, concrete (which is flow 13_20b downcycl.
                    # aggregates split to these materials)
                    last_year_idx = np.s_[r, :, :, t - 1]
                    this_year_idx = np.s_[r, :, :, t]

                    self.MISO2.FlowDict['F_13_14'].Values[last_year_idx] = \
                        self.MISO2.FlowDict['F_12_13'].Values[last_year_idx] \
                        - self.MISO2.FlowDict['F_13_20a'].Values[last_year_idx] \
                        - self.miso_model_data.downc_constr_mats[this_year_idx]

                    self.MISO2.FlowDict['F_13_14'].Values[last_year_idx][
                        np.isclose(self.MISO2.FlowDict['F_13_14'].Values[last_year_idx], 0.0, rtol=0, atol=1e-10)] = 0
                    # this subtracts near-equal floats that should zero out, which often produces
                    # rounding errors, so we reset to zero here

                    # final waste = total supply of scrap/waste materials  \
                    # - recycled materials (except aggr_downcycl; F_13_20a) \
                    # - downcycled materials (downc_constr_mats positions asphalt, bricks, conrete) \
                    # - recycled aggr_downcycl (downc_constr_mats position aggr_downcycl)
                    
                # Dynamic Stock Model - calculation of stock by cohort and EoL outflows
                # modifies F_10_11, S_10 inplace
                self.P10_Dynamic_Stock_Model(r, dimensions)
                # Waste collection, calculate collected potentially recoverable new scrap and old scrap and total
                self.P11_waste_collection(r, t)
                # Trade of potentially recoverable waste
                self.P12_trade_recoverable_waste(r, t)
                # recycling and downcycling
                self.P13_recycling_downcycling(r, t, selector_downc_mats, new_scrap_cycle)
                # Final waste disposal
                self.P14_final_waste_disposal(r, t)
                # time buffer for cycled materials
                self.P20_buffer_transfer(r, t, buffer_stocks_flows)

                ###########################################################################
                # Write output ##

            # CREATE METADATA AND INDICES

            [region, run_options, years, year_index, material_index, multiplier_material_index, time_index,
             sector_index] \
                = self._create_output_metadata(r, estimate_aggregates, save_last_year)

            output_factory = MISO2ModelOutputFactory(self.MISO2, self.miso_model_data)

            output_dfs = output_factory.create_output_dfs(r=r,
                                                          material_index=material_index,
                                                          time_index=time_index,
                                                          sector_index=sector_index,
                                                          year_index=year_index,
                                                          save_stock_cohorts=save_stock_cohorts)

            debug_dfs = output_factory.create_debug_dfs(r=r,
                                                        material_index=material_index,
                                                        multiplier_material_index=multiplier_material_index,
                                                        time_index=time_index,
                                                        year_index=year_index,
                                                        sector_index=sector_index)

            debug_dfs["mass_bal_system"] = output_factory.create_mass_balance_system(
                r=r,
                material_index=material_index,
                time_index=time_index,
                year_index=year_index)

            self.mass_bal_process = output_factory.create_mass_balance_process(
                output_dfs=output_dfs)

            self.mass_bal_process_annual, self.odym_bal, self.var_outside_ODYM_balance = \
                output_factory.create_mass_balance_process_annual(
                    output_dfs=output_dfs, Nt=dimensions["Nt"] - 1)

            self.mass_bal_system = output_factory.create_mass_balance_system(
                r=r,
                material_index=material_index,
                time_index=time_index,
                year_index=year_index)

            # ERROR CHECKS ###
            self._check_mass_bal_system(debug_dfs["mass_bal_system"])

            self._check_balances_odym(self.mass_bal_process)

            self._check_balances_odym_annual(self.mass_bal_process_annual)

            self._check_MFA_system()

            self._check_aggr_requirement_met(F_9_10_GAS=output_dfs["F_9_10_GAS"],
                                             r=r,
                                             period=years["max"] - years["min"])

            # CREATE OUTPUT OBJECT ###

            # output_dfs = {**output_dfs, **debug_dfs}

            miso_output = output_factory.create_miso_output(
                region=region,
                mfa_system_control=self.MFA_system_control,
                unique_id=self.miso_config.unique_id,
                run_options=run_options,
                output_dfs=output_dfs)

            self._debug_output = debug_dfs

            return miso_output

    def _create_output_metadata(self, r, estimate_aggregates, save_last_year):
        """
        Create Pandas indices and metadata for miso output.

        Args:
            r(int): Region index
            estimate_aggregates(bool): Run option
            save_last_year(bool): If last year of data should be saved

        Returns:
            region(String): Name of region
            run_options(dict): Dictionary of model run options
            years(dict): Dictionary of starting and end year
            year_index(dict): Dictionary with starting and end year index
            material_index(pd.Index): Pandas index of materials
            time_index(pd.Index): Pandas index of years
            sector_index(pd.Index): Pandas index of enduse sectors

        """
        region = self.miso_config.master_classification["Countries"].Items[r]
        run_options = {"estimate_aggregates": estimate_aggregates,
                       "monte_carlo_class": str(type(self.miso_config._monte_carlo)),
                       "bias": self.miso_config.get_bias("last")}

        years = {"min": self.miso_config.model_time_start,
                 "max": self.miso_config.model_time_end}

        year_index = {"min": 0,
                      "max": self.miso_config.Nt - 1}
        if save_last_year:
            year_index.max = self.miso_config.Nt
            years["max"] = self.miso_config.model_time_end + 1

        material_index = pd.Index(data=self.miso_config.material_position.keys(), name='material')
        multiplier_material_index = pd.Index(
            data=[k for k, v in self.miso_config.material_position.items()
                  if v in self.miso_config.selectors["multiplier_materials"]], name='material')
        time_index = pd.Index(data=list(range(years["min"], years["max"])), name='time')
        sector_index = pd.Index(data=self.miso_config.master_classification["End_use_sectors"].Items, name="sector")

        return [region, run_options, years, year_index, material_index, multiplier_material_index, time_index, sector_index]

    def _check_mass_bal_system(self, mass_bal_system):
        """
        Checks mass balances as system-wide sum.

        Args:
            mass_bal_system(pd.Dataframe): Pandas dataframe of mass balance systems.
        """
        logger.info("Checking mass balance system")
        bal_sum = mass_bal_system.iloc[:, :].sum().sum()
        # Original code omitted last year in sums
        logger.debug(f"Sum of in, out, stock, bal: {bal_sum}")

        if not np.isclose(bal_sum, 0.0, atol=self.balance_error_tolerance):
            logger.warning("WARNING: Difference in balance check exceeds tolerance")
            logger.warning(f"Value: {bal_sum}, Tolerance: {self.balance_error_tolerance}")

    def _check_balances_odym(self, mass_bal_process):
        """
        Checks ODYM balances per process and emits warning if they exceed tolerance.

        Args:
            mass_bal_process(pd.Dataframe): Pandas dataframe containing balance per process

        """
        logger.info("Checking balances per process via ODYM")
        ext_bal1 = mass_bal_process

        if not np.isclose(ext_bal1.sum(), 0.0, atol=self.balance_error_tolerance):
            logger.error("WARNING: Difference in long-term process balance check exceeds tolerance")
            logger.error(f"Value: {ext_bal1.sum()}, Tolerance: {self.balance_error_tolerance}")
            non_zero_ext_bal = np.transpose(np.nonzero(~np.isclose(ext_bal1, 0)))
            logger.error(f"Index of non-zero process elements: {non_zero_ext_bal}")

    def _check_balances_odym_annual(self, mass_bal_process_annual):
        """
        Checks ODYM balances per process and year and emits warning if they exceed tolerance.
        Note: for process 10 and process 20 (time buffer), checks run over two consecutive years.

        Args:
            mass_bal_process_annual(pd.Dataframe): Pandas dataframe containing balance per process

        """
        logger.info("Checking balances per process and year via ODYM")
        ext_bal1 = mass_bal_process_annual

        if not np.allclose(ext_bal1, np.zeros_like(ext_bal1), atol=self.balance_error_tolerance):
            logger.error("WARNING: Difference in yearly process balance check exceeds tolerance")
            logger.error(f"Value: {ext_bal1}, Tolerance: {self.balance_error_tolerance}")
            non_zero_ext_bal = np.transpose(np.nonzero(~np.isclose(ext_bal1, 0)))
            logger.error(f"Index of non-zero process elements: {non_zero_ext_bal}")

    def _check_total_balance_per_year(self, period, mass_bal_system):
        """
        Checks mass balance system is zero (within fp precision) for each year.

        Raises a warning if any year exceeds the tolerance.

        Args:
            period(int): Check runs from 0 to this index.
            mass_bal_system(pd.Dataframe): Pandas dataframe that contains the mass balance system.

        """
        logger.info("Checking mass balances per year")

        for t in range(0, period):
            balance_sum = mass_bal_system.iloc[:, t].sum()

            if not np.isclose(balance_sum, 0.0, atol=self.balance_error_tolerance):
                logger.error("WARNING: Difference in balance check exceeds tolerance")
                logger.error(f"Value: {sum}, Tolerance: {self.balance_error_tolerance}")

    def _check_aggr_requirement_met(self, F_9_10_GAS, r, period):
        """
        Checks if the required amount of aggr_virgin + aggr_downcycl is met at process 9 for each year.

        The method will check the respective enduse aggregates items of the MISO2 model and the config.
        Raises a warning if any year exceeds the tolerance.

        Args:
            F_9_10_GAS(Pd.DataFrame): Call elements for checks from this object
            r(int): Region index
            period(int): Check runs from 0 to this index.

        """
        logger.info("Checking if aggr requirement at p9 met per year")

        selector_multiplier_materials = self.miso_config.selectors["multiplier_materials"]
        aggr_req_mg = np.zeros((([
            len(selector_multiplier_materials), len(self.miso_config.endUse_aggregates),
            self.miso_config.Nt])))
        aggr_req_mg = aggr_req_mg.reshape((
            len(selector_multiplier_materials), self.miso_config.Ng, self.miso_config.Nt, self.miso_config.Ne))

        for t in range(0, period):
            for key, value in self.miso_config.endUse_aggregates.items():
                # estimate aggregates requirement per construction material m and end-use g
                if value:
                    aggregate_multiplier \
                        = self.MISO2.ParameterDict[('MISO2_' + value)].Values[r, :, selector_multiplier_materials, t]
                else:
                    aggregate_multiplier = 0

                pos_aggr = list(self.miso_config.endUse_aggregates).index(key)
                aggr_req_mg[:, list(self.miso_config.endUse_aggregates).index(key), t] \
                    = self.MISO2.FlowDict['F_9_10'].Values[r, :, selector_multiplier_materials, pos_aggr, t] \
                    * aggregate_multiplier
                    
        required = np.einsum('mgte->mt', aggr_req_mg)[:, : -1]
        
        pos_aggr_cycle = [self.miso_config.material_position.get('aggr_virgin'),
                          self.miso_config.material_position.get('aggr_downcycl')]
        supplied = F_9_10_GAS.iloc[pos_aggr_cycle, :]

        match_dema_suppl_per_year = required.sum(axis=0) - supplied.sum(axis=0)
        for t in range(0, period):
            item = match_dema_suppl_per_year.values[t]

            if not np.isclose(item, 0.0, atol=self.balance_error_tolerance):
                logger.error("WARNING: Difference in balance check exceeds tolerance")
                logger.error(f"Value: {item}, Tolerance: {self.balance_error_tolerance}")

    def _check_MFA_system(self):
        """
        Basic error checking for MFA system.

        The function checks if the number of stocks and flow remained unchanged after the MFA system initialisation
        and that all Stocks and Flows that should contain values do have positive entries and are not zero and vice
        versa. The MFA system control variables are set by the MISO2_system_definition function.
        """

        logger.info("Checking MFA System")

        expected_stocks = self.MFA_system_control["number_stocks"]
        model_stocks = len(self.MISO2.StockDict)
        if model_stocks != expected_stocks:
            logger.error(f"After model run, MISO models MFA system contains {model_stocks} stocks, \
                              but system definition only set {expected_stocks}. This means that a Stock \
                                  was added to the dictionary after the initial system definition")

        expected_flows = self.MFA_system_control["number_flows"]
        model_flows = len(self.MISO2.FlowDict)
        if model_flows != expected_flows:
            logger.error(f"After model run, MISO models MFA system contains {model_flows} flows, but \
                              system definition only set {expected_flows}. This means that a Flow was added \
                                  to the dictionary after the initial system definition")

        all_zero_ok = self.MFA_system_control["all_zero_ok"]
        no_data_yet = self.MFA_system_control["no_data_yet"]
        must_be_zero = self.MFA_system_control["must_be_zero"]
        zeros = all_zero_ok + no_data_yet + must_be_zero

        for k, v in self.MISO2.FlowDict.items():
            if k not in zeros:
                if np.equal(v.Values, np.zeros_like(v.Values)).all():
                    logger.error(f"Flow {k} contains only zero values")
                neg_nan = v.Values[v.Values < 0]

                if neg_nan.size > 0:
                    logger.error(f"Flow {k} contains {neg_nan.size} negative or nan-values")

            if k in must_be_zero:
                if not np.equal(v.Values, np.zeros_like(v.Values)).all():
                    logger.error(f"Flow {k} contains values, but should only be zeros")
            # this could be wrong for last year data

        for k, v in self.MISO2.StockDict.items():
            if k not in zeros:
                if np.equal(v.Values, np.zeros_like(v.Values)).all():
                    logger.error(f"Stock {k} contains only zero values ")

                neg_nan = v.Values[v.Values < 0]

                if neg_nan.size > 0:
                    logger.error(f"Stock {k} contains {neg_nan.size} negative or nan-values")

                # this could be wrong for last year data
            if k in must_be_zero:
                if not np.equal(v.Values, np.zeros_like(v.Values)).all():
                    logger.error(f"Stock {k} contains values, but should only be zeros")

    def derive_cycled_mat_from_last_year(self, r, t, buffer_flows_stocks):
        """
        Start year-loop with deriving recycled/downcycled material from last year from buffer S20

        S_20a = recycled material use (except for aggregates, these are listed in S_20c)
        S_20c = maximal supply of recycled virgin aggregates and downcycled aggregates from asphalt, \
            concrete, bricks (use determined from supply in p9)
        S_20 should be zero afterward.

        Args:
            r(int): Region index.
            t(int): Time index.
            buffer_flows_stocks(dict): Dictionary mapping flows to stocks,

        """
        this_year_idx = np.s_[r, :, :, t]
        prev_year_idx = np.s_[r, :, :, t - 1]

        for flow, stock in buffer_flows_stocks.items():
            self.MISO2.FlowDict[flow].Values[this_year_idx] = self.MISO2.StockDict[stock].Values[prev_year_idx]
            # subtract outflow from P20 from stock S20 (should be zero afterwards)
            self.MISO2.StockDict[stock].Values[prev_year_idx] \
                = self.MISO2.StockDict[stock].Values[prev_year_idx] \
                - self.MISO2.FlowDict[flow].Values[this_year_idx]

    def calc_waste_cascades(self, r, t):
        """
        Calculate waste cascade factors throughout processes 5-7-9 to adjust backward calculations for aggregates \
        in P9.2, P9.3. (i.e. add the losses waste occurring throughout processes 5-7-9 to calculate original flows)

        Args:
            r(int): Region index.
            t(int): Time index.
        """

        remt_idx = np.s_[r, :, :, t]

        # calculate a waste cascade factor for waste occurring through p5 and p7 to
        # calculate the aggregates for asphalt and concrete
        # that need to be supplied in p3 to cater for demand in p9
        waste_cascade_factor_p57 \
            = (1 - (self.MISO2.ParameterDict['MISO2_WasteRate_recov_p5'].Values[remt_idx]
                    + self.MISO2.ParameterDict['MISO2_WasteRate_unrecov_p5'].Values[remt_idx])) \
            * (1 - (self.MISO2.ParameterDict['MISO2_WasteRate_recov_p7'].Values[remt_idx]
                    + self.MISO2.ParameterDict['MISO2_WasteRate_unrecov_p7'].Values[remt_idx]))

        waste_cascade_factor_p579 \
            = (1 - (self.MISO2.ParameterDict['MISO2_WasteRate_recov_p5'].Values[remt_idx]
                    + self.MISO2.ParameterDict['MISO2_WasteRate_unrecov_p5'].Values[remt_idx])) \
            * (1 - (self.MISO2.ParameterDict['MISO2_WasteRate_recov_p7'].Values[remt_idx]
                    + self.MISO2.ParameterDict['MISO2_WasteRate_unrecov_p7'].Values[remt_idx])) \
            * (1 - (self.MISO2.ParameterDict['MISO2_WasteRate_recov_p9'].Values[remt_idx]
                    + self.MISO2.ParameterDict['MISO2_WasteRate_unrecov_p9'].Values[remt_idx]))

        waste_cascade_factor_p79 \
            = (1 - (self.MISO2.ParameterDict['MISO2_WasteRate_recov_p7'].Values[remt_idx]
                    + self.MISO2.ParameterDict['MISO2_WasteRate_unrecov_p7'].Values[remt_idx])) \
            * (1 - (self.MISO2.ParameterDict['MISO2_WasteRate_recov_p9'].Values[remt_idx]
                    + self.MISO2.ParameterDict['MISO2_WasteRate_unrecov_p9'].Values[remt_idx]))

        waste_cascade_factor_p9 \
            = (1 - (self.MISO2.ParameterDict['MISO2_WasteRate_recov_p9'].Values[remt_idx]
                    + self.MISO2.ParameterDict['MISO2_WasteRate_unrecov_p9'].Values[remt_idx]))

        waste_cascade_dict = {'waste_cascade_factor_p57': waste_cascade_factor_p57,
                              'waste_cascade_factor_p579': waste_cascade_factor_p579,
                              'waste_cascade_factor_p79': waste_cascade_factor_p79,
                              'waste_cascade_factor_p9': waste_cascade_factor_p9}

        return waste_cascade_dict

    def calc_net_trade_aggr_p9(self, r, t, waste_cascade_dict):
        """
        Calculate net trade of aggregates at process p9 (which requires subtraction of waste losses depending on at
        which part of supply chain trade occurs)

        Args:
            r(int): Region index.
            t(int): Time index.
            waste_cascade_dict(dict): Waste cascade dictionary
        """
        position_aggr = [self.miso_config.material_position.get('aggr_virgin'),
                         self.miso_config.material_position.get('aggr_downcycl')]
        aggr_pos_idx = np.s_[r, :, position_aggr, t]
        waste_aggr_idx = np.s_[:, position_aggr]

        # call parameter dictionary here as src system trade flows for mats = aggregates \
        # only instated on 2nd round of loop
        # reshape matches shape of waste_cascade_factors to that of indexed trade flows
        # (if index changes, change reshape)

        net_trade_aggr_p9 \
            = self.MISO2.ParameterDict['MISO2_Import_p4'].Values[aggr_pos_idx] \
            * waste_cascade_dict.get('waste_cascade_factor_p579')[waste_aggr_idx].reshape(len(position_aggr), 1) \
            - self.MISO2.ParameterDict['MISO2_Export_p4'].Values[aggr_pos_idx] \
            * waste_cascade_dict.get('waste_cascade_factor_p579')[waste_aggr_idx].reshape(len(position_aggr), 1) \
            + self.MISO2.ParameterDict['MISO2_Import_p6'].Values[aggr_pos_idx] \
            * waste_cascade_dict.get('waste_cascade_factor_p79')[waste_aggr_idx].reshape(len(position_aggr), 1) \
            - self.MISO2.ParameterDict['MISO2_Export_p6'].Values[aggr_pos_idx] \
            * waste_cascade_dict.get('waste_cascade_factor_p79')[waste_aggr_idx].reshape(len(position_aggr), 1) \
            + self.MISO2.ParameterDict['MISO2_Import_p8'].Values[aggr_pos_idx] \
            * waste_cascade_dict.get('waste_cascade_factor_p9')[waste_aggr_idx].reshape(len(position_aggr), 1) \
            - self.MISO2.ParameterDict['MISO2_Export_p8'].Values[aggr_pos_idx] \
            * waste_cascade_dict.get('waste_cascade_factor_p9')[waste_aggr_idx].reshape(len(position_aggr), 1)

        return net_trade_aggr_p9

    def P4_generate_production_raw_products(self, r, selector_NoAggr, t):
        """
        Assign primary production and total production from input data and adjust to flow definitions.

        Primary production F_3_4a or total production total2primary from input data. The latter adjusted to
        F_3_4b via re/downcycling flows 20_4a/b, F_3_4a and F_3_4b are exclusive per material, meaning that either
        total or primary production is reported per material

        Args:
            r(int): Region index.
            selector_NoAggr(list): List of non-aggregate indices.
            t(int): Time index.
        """
        yearly_idx = np.s_[r, :, :, t]
        non_aggr_idx = np.s_[r, :, selector_NoAggr, t]

        self.MISO2.FlowDict['F_3_4a'].Values[non_aggr_idx] \
            = self.MISO2.ParameterDict['MISO2_Production_p3_primary'].Values[non_aggr_idx]

        self.miso_model_data.total2primary[non_aggr_idx] \
            = self.MISO2.ParameterDict['MISO2_Production_p3_total'].Values[non_aggr_idx]

        # from total production total2primary, subtract secondary production in
        # F_20_4a/b to arrive at primary production F_3_4b
        matIndex_total = np.where(
            self.miso_model_data.total2primary[r, :, :, :].sum(axis=2) == 0, 0, 1)
        # some yearly values are empty, which makes total2primary negative; therefore clip these years in next line
        # result: subtract F_20_4a/b where total2primary is =! 0
        self.MISO2.FlowDict['F_3_4b'].Values[yearly_idx] \
            = (self.miso_model_data.total2primary[yearly_idx]
               - ((self.MISO2.FlowDict['F_20_4a'].Values[yearly_idx]
                   + self.MISO2.FlowDict['F_20_4b'].Values[yearly_idx])
                  * matIndex_total)).clip(0)

    def P4_add_raw_product_trade(self, r, mats, t):
        """
        Add raw product trade, calculate apparent consumption and balance P4.

        Args:
            r(int): Region index.
            mats(int): Material selector index.
            t(int): Time index.
        """
        # add raw product trade
        mats_idx = np.s_[r, :, mats, t]

        self.MISO2.FlowDict['F_4_16'].Values[mats_idx] \
            = self.MISO2.ParameterDict['MISO2_Export_p4'].Values[mats_idx]
        # exports of raw products

        self.MISO2.FlowDict['F_16_4'].Values[mats_idx] \
            = self.MISO2.ParameterDict['MISO2_Import_p4'].Values[mats_idx]
        # imports of raw products

        # AC raw products = prim prod (mat_x) + [tot_prod(mat_y) - sec_prod(mat_y)] + sec_prod + imports - exports
        self.MISO2.FlowDict['F_4_5'].Values[mats_idx] \
            = self.MISO2.FlowDict['F_3_4b'].Values[mats_idx] \
            + self.MISO2.FlowDict['F_3_4a'].Values[mats_idx] \
            + self.MISO2.FlowDict['F_16_4'].Values[mats_idx] \
            - self.MISO2.FlowDict['F_4_16'].Values[mats_idx] \
            + self.MISO2.FlowDict['F_20_4a'].Values[mats_idx] \
            + self.MISO2.FlowDict['F_20_4b'].Values[mats_idx]

        # balance p4 output in case of exports > (dom. production + imports) and write mismatch to balancing item bal_4
        self.MISO2.FlowDict['F_4_5'].Values[mats_idx], self.miso_model_data.bal_4[mats_idx] \
            = self.balance_process_indexed(self.MISO2.FlowDict['F_4_5'].Values, mats, r, t)

    def P5_fabricate_semi_finished_products(self, r, mats, t):
        """
        Fabricate semi-finished products
        #recoverable (a) & unrecoverable (b) waste during fabrication

        Args:
            r(int): Region index.
            mats(int): Material selector index.
            t(int): Time index.
        """
        mats_idx = np.s_[r, :, mats, t]

        self.MISO2.FlowDict['F_5_11a'].Values[mats_idx] \
            = self.MISO2.FlowDict['F_4_5'].Values[mats_idx] \
            * self.MISO2.ParameterDict['MISO2_WasteRate_recov_p5'].Values[mats_idx]
        self.MISO2.FlowDict['F_5_11b'].Values[mats_idx] \
            = self.MISO2.FlowDict['F_4_5'].Values[mats_idx] \
            * self.MISO2.ParameterDict['MISO2_WasteRate_unrecov_p5'].Values[mats_idx]

        # domestic production of semi-finished products = AC raw products - waste
        self.MISO2.FlowDict['F_5_6'].Values[mats_idx] \
            = self.MISO2.FlowDict['F_4_5'].Values[mats_idx] \
            - self.MISO2.FlowDict['F_5_11a'].Values[mats_idx] \
            - self.MISO2.FlowDict['F_5_11b'].Values[mats_idx]

    def P6_trade_semi_finished_products(self, r, mats, t):
        """
        Add trade of semi-finished products, calculate apparent consumption and balance P6.

        Args:
            r(int): Region index.
            mats(int): Material selector index.
            t(int): Time index.
        """

        mats_idx = np.s_[r, :, mats, t]

        # exports of semi-finished products
        self.MISO2.FlowDict['F_6_17'].Values[mats_idx] = \
            self.MISO2.ParameterDict['MISO2_Export_p6'].Values[mats_idx]

        # imports of semi-finished products
        self.MISO2.FlowDict['F_17_6'].Values[mats_idx] = \
            self.MISO2.ParameterDict['MISO2_Import_p6'].Values[mats_idx]

        # AC semi-finished products = production + imports - exports (all of semi-finished products)
        self.MISO2.FlowDict['F_6_7'].Values[mats_idx] = \
            self.MISO2.FlowDict['F_5_6'].Values[mats_idx] \
            - self.MISO2.FlowDict['F_6_17'].Values[mats_idx] \
            + self.MISO2.FlowDict['F_17_6'].Values[mats_idx]

        # balance p6 output in case of exports > (dom. production + imports) and write mismatch to balancing item bal_6
        self.MISO2.FlowDict['F_6_7'].Values[mats_idx], self.miso_model_data.bal_6[mats_idx] = \
            self.balance_process_indexed(self.MISO2.FlowDict['F_6_7'].Values, mats, r, t)

    def P7_manufacture_final_products(self, r, mats, t):
        """
        Manufacture final products via subtracting recoverable (a) & unrecoverable (b) waste during manufacturing.

        Args:
            r(int): Region index.
            mats(int): Material selector index.
            t(int): Time index.
        """
        mats_idx = np.s_[r, :, mats, t]

        self.MISO2.FlowDict['F_7_11a'].Values[mats_idx] \
            = self.MISO2.FlowDict['F_6_7'].Values[mats_idx] \
            * self.MISO2.ParameterDict['MISO2_WasteRate_recov_p7'].Values[mats_idx]
        self.MISO2.FlowDict['F_7_11b'].Values[mats_idx] \
            = self.MISO2.FlowDict['F_6_7'].Values[mats_idx] \
            * self.MISO2.ParameterDict['MISO2_WasteRate_unrecov_p7'].Values[mats_idx]

        # domestic production of final products = AC semi-finished products - waste
        self.MISO2.FlowDict['F_7_8'].Values[mats_idx] \
            = self.MISO2.FlowDict['F_6_7'].Values[mats_idx] \
            - self.MISO2.FlowDict['F_7_11a'].Values[mats_idx] \
            - self.MISO2.FlowDict['F_7_11b'].Values[mats_idx]

    def P8_trade_final_products(self, r, mats, t):
        """
        Add trade of final products, calculate apparent consumption and balance P8

        Args:
            r(int): Region index.
            mats(int): Material selector index.
            t(int): Time index.

        """

        mats_idx = np.s_[r, :, mats, t]

        # exports of final products
        self.MISO2.FlowDict['F_8_18'].Values[mats_idx] \
            = self.MISO2.ParameterDict['MISO2_Export_p8'].Values[mats_idx]

        # imports of final products
        self.MISO2.FlowDict['F_18_8'].Values[mats_idx] \
            = self.MISO2.ParameterDict['MISO2_Import_p8'].Values[mats_idx]

        # AC final products = production + imports - exports (all of final products)
        self.MISO2.FlowDict['F_8_9'].Values[mats_idx] \
            = self.MISO2.FlowDict['F_7_8'].Values[mats_idx] \
            - self.MISO2.FlowDict['F_8_18'].Values[mats_idx] \
            + self.MISO2.FlowDict['F_18_8'].Values[mats_idx]

        # balance p8 output in case of exports > (dom. production + imports) and write mismatch to balancing item bal_8
        self.MISO2.FlowDict['F_8_9'].Values[mats_idx], self.miso_model_data.bal_8[mats_idx] \
            = self.balance_process_indexed(self.MISO2.FlowDict['F_8_9'].Values, mats, r, t)

    def P9_1_introduce_enduse_shares(self, r, mats, t, enduse_shares):
        """
        P9.1 Introduce end-use shares gross additions to stock (GAS) per end-use = (AC final products - waste)
        end-use shares (new dimensions r,e,m,g,t)

        Args:
            r(int): Region index.
            mats(int): Material selector index.
            t(int): Time index.
            enduse_shares(np.Array): Array of enduse shares
        """

        return np.einsum('em,emg->emg',
                         self.MISO2.FlowDict['F_8_9'].Values[r, :, mats, t],
                         enduse_shares[r, :, mats, :, t])

    def P9_2_change_material_categories(self, r, t, cementBitumen_2_concreteAsphalt,
                                        appar_consump_end_use, multiplier_cementBitumen, waste_cascade_dict):
        """
        P9.2 change of material categories: cement & bitumen + sand & gravel --> concrete & asphalt

       Args:
           r(int): Region index.
           t(int): Time index.
           cementBitumen_2_concreteAsphalt():
           appar_consump_end_use(np.Array): remgt array
           multiplier_cementBitumen():
           waste_cascade_dict(dict):
        """
        aggr_4concr_pos = self.miso_config.material_position.get('aggr_4concr')
        cement_pos = self.miso_config.material_position.get('cement')
        aggr_4asph_pos = self.miso_config.material_position.get('aggr_4asph')
        asphalt_pos = self.miso_config.material_position.get('asphalt')
        bitumen_pos = self.miso_config.material_position.get('bitumen')

        remgt_idx = np.s_[r, :, :, :, t]
        concrete_idx = np.s_[r, :, self.miso_config.material_position.get('concrete'), :, t]
        cement_idx = np.s_[r, :, cement_pos, :, t]
        asphalt_index = np.s_[r, :, asphalt_pos, :, t]
        bitumen_index = np.s_[r, :, bitumen_pos, :, t]
        aggr_4_concr_idx = np.s_[r, :, aggr_4concr_pos, t]
        aggr_4_asph_idx = np.s_[r, :, aggr_4asph_pos, t]

        # write AC of final products (F_8_9 assigned to appar_consump_end_use in top-layer) to new dataframe to change
        # material categories in there
        cementBitumen_2_concreteAsphalt[remgt_idx] = appar_consump_end_use[remgt_idx]

        # concrete = AC of final cement products (F_8_9[cement]) * concrete/cement ratio
        # + concrete flows in F_8_9 from concrete recycling and trade
        cementBitumen_2_concreteAsphalt[concrete_idx] \
            = cementBitumen_2_concreteAsphalt[cement_idx] \
            * multiplier_cementBitumen[:, cement_pos, :] \
            + appar_consump_end_use[concrete_idx]

        # asphalt = AC of final bitumen products (F_8_9[bitumen])* asphalt/bitumen ratio
        # + asphalt flows in F_8_9 from asphalt recycling and trade
        cementBitumen_2_concreteAsphalt[asphalt_index] \
            = cementBitumen_2_concreteAsphalt[bitumen_index] \
            * multiplier_cementBitumen[:, bitumen_pos, :] \
            + appar_consump_end_use[asphalt_index]

        # manipulate multipliers so that only the cement/bitumen that was transformed into concrete/asphalt
        # are set to zero in calc. of F_3_4a/b below
        # multiplier data only has values != 0 for end-uses that shall transform cement/bitumen --> concrete/asphalt
        # (these are == 1)
        delete_material_transfer = np.minimum(multiplier_cementBitumen, 1)

        # calculate aggregates that need to be supplied in p4 for making concrete and asphalt in p9
        # by subtracting cement/bitumen content and concrete/asphalt from recycling/trade from newly mixed concrete
        # and applying the waste cascade factor to cover aggregate losses from p4->p9

        self.MISO2.FlowDict['F_3_4a'].Values[aggr_4_concr_idx] = \
            (np.einsum('eg->e',
                       cementBitumen_2_concreteAsphalt[concrete_idx])
             - np.einsum('eg->e',
                         cementBitumen_2_concreteAsphalt[cement_idx]
                         * delete_material_transfer[:, cement_pos, :])
             - np.einsum('eg->e',
                         appar_consump_end_use[concrete_idx])) \
            / waste_cascade_dict.get('waste_cascade_factor_p57')[0, aggr_4concr_pos] \
            - self.MISO2.FlowDict['F_20_4a'].Values[aggr_4_concr_idx]

        self.MISO2.FlowDict['F_3_4a'].Values[aggr_4_asph_idx] = \
            (np.einsum('eg->e',
                       cementBitumen_2_concreteAsphalt[asphalt_index])
             - np.einsum('eg->e',
                         cementBitumen_2_concreteAsphalt[bitumen_index]
                         * delete_material_transfer[:, bitumen_pos, :])
             - np.einsum('eg->e',
                         appar_consump_end_use[asphalt_index])) \
            / waste_cascade_dict.get('waste_cascade_factor_p57')[0, aggr_4asph_pos] \
            - self.MISO2.FlowDict['F_20_4a'].Values[aggr_4_asph_idx]

        # all cement, bitumen and aggregates for end-uses with multipliers >1 were used
        # to make concrete and asphalt, so set these positions zero to reflect this transition
        calc_material_transfer = np.logical_not(delete_material_transfer)
        # this will flip all zeros to true and all values to false

        cementBitumen_2_concreteAsphalt[remgt_idx] \
            = cementBitumen_2_concreteAsphalt[remgt_idx] \
            * calc_material_transfer
        cementBitumen_2_concreteAsphalt[r, :, [aggr_4concr_pos, aggr_4asph_pos], :, t] = 0

    def P9_2_recov_unrecov_waste_flows(self, r, mats, t, cementBitumen_2_concreteAsphalt,
                                       selector_Aggr, trade_larger_req_g):
        """
        Set P9 standard flows F9_11, recoverable (a) & unrecoverable (b) waste during fabrication.

        Conditionally for aggregates, recov. waste = input per end-use * waste rate per end-use \
            + excess aggregate imports (can be used in next year)

        Args:
            r(int): Region index.
            t(int): Time index.
            mats(list): Material selector index.
            cementBitumen_2_concreteAsphalt(np.Array): apparent consumption of materials with bitumen/cement partially transformed
                                                        to asphalt, concrete (see P9_2_change_material_categories)
            selector_Aggr(list): List of aggregate selectors
            trade_larger_req_g(np.Array): array with non-zero values, if net imports of virgin aggregates larger than aggregate requirement
                                          (aggregate requirement & trade_larger_req_g estimated in first loop of 'mats' selector, considering only
                                          non-aggregate materials; and compared to virgin aggregate net imports in second loop of 'mats' selector;
                                          trade_larger_req_g is overflowing into recoverable waste flows, to be recycled next year; this is a contingency
                                          for the rare case that imports of virgin aggregates larger than aggregate requirement )                                        )

        """

        mats_idx = np.s_[r, :, mats, t]
        mats_sector_idx = np.s_[r, :, mats, :, t]
        aggr_virg_idx = np.s_[r, :, self.miso_config.material_position.get('aggr_virgin'), :, t]

        # assert that array shapes match; if not, model run aborted
        assert cementBitumen_2_concreteAsphalt[mats_sector_idx].shape == np.einsum('em,emg->emg',
                                                                                   self.MISO2.ParameterDict[
                                                                                       'MISO2_WasteRate_unrecov_p9'].Values[
                                                                                       mats_idx],
                                                                                   np.ones((len(mats), 1,
                                                                                            self.miso_config.Ng))).shape

        self.MISO2.FlowDict['F_9_11b'].Values[mats_sector_idx] \
            = cementBitumen_2_concreteAsphalt[mats_sector_idx] \
            * np.einsum('em,emg->emg',
                        self.MISO2.ParameterDict['MISO2_WasteRate_unrecov_p9'].Values[mats_idx],
                        np.ones((len(mats), 1, self.miso_config.Ng)))

        self.MISO2.FlowDict['F_9_11a'].Values[mats_sector_idx] = cementBitumen_2_concreteAsphalt[mats_sector_idx] \
            * np.einsum('em,emg->emg', self.MISO2.ParameterDict['MISO2_WasteRate_recov_p9'].Values[mats_idx],
                        np.ones((len(mats), 1, self.miso_config.Ng)))
        # np.einsum introduces end-use dimension to waste rates (equal waste rates assumed per end-use)

        # conditional:
        # if materials are aggregates, recov. waste = input per end-use
        # * waste rate per end-use + excess aggregate imports (can be used in next year)
        # ! assuming that an excess of imported virgin aggregates (trade_larger_req_g)
        # can be re-used in recycling of new scrap in next year
        # in case of reported absolute cycling flows, part may be re-used, remainder goes to final waste
        # in case of reported relative cycling rates, new scrap cycle rate needs to be 1 to make sure that all of
        # this is considered for use next year
        if mats == selector_Aggr:
            self.MISO2.FlowDict['F_9_11a'].Values[aggr_virg_idx] = \
                self.MISO2.FlowDict['F_9_11a'].Values[aggr_virg_idx] \
                + trade_larger_req_g

        self.MISO2.FlowDict['F_9_10'].Values[mats_sector_idx] \
            = cementBitumen_2_concreteAsphalt[mats_sector_idx] \
            - self.MISO2.FlowDict['F_9_11a'].Values[mats_sector_idx] \
            - self.MISO2.FlowDict['F_9_11b'].Values[mats_sector_idx]

    def P9_3_aggregate_gap(self, r, t, enduse_shares,
                           endUse_aggregates, selector_downc_mats,
                           selector_multiplier_materials, aggr_req_mg, waste_cascade_dict, net_trade_aggr_p9):
        """
        Aggregate Gap: estimate demand of aggregates, appraise supply from recycled and \
            downcycled aggregates and calc. additional required virgin

        Aggregate input (time steps inherent)
        Estimate demand per end-use relevant for aggregate use as specified in variable
        endUse_aggregates (e.g. buildings, roads, other construction)
        Conditional: only run aggregate gap once (when prior flows for all materials
        except aggregates have been calculated) enable or disable estimation of aggregate gap

        Args:
            r(int): Region index.
            t(int): Time index.
            enduse_shares(np.Array): something else
            endUse_aggregates(dict): Maps enduses (?) to aggregate multiplier values
            selector_downc_mats(list): List of aggregate selectors
            selector_multiplier_materials(list): List of materials with foundation aggregate multipliers
            aggr_req_mg():
            net_trade_aggr_p9():
            waste_cascade_dict(dict):

        """
        remt_idx = np.s_[r, :, :, t]
        aggr_virgin_pos = self.miso_config.material_position.get('aggr_virgin')
        aggr_downcycl_pos = self.miso_config.material_position.get('aggr_downcycl')
        selector_downc_idx = np.s_[r, :, selector_downc_mats, t]
        aggr_downcycl_idx = np.s_[r, :, aggr_downcycl_pos, t]
        aggr_virgin_idx = np.s_[r, :, self.miso_config.material_position.get('aggr_virgin'), t]

        # estimate aggregates requirement per construction material m and end-use g
        for key, value in endUse_aggregates.items():

            enduse_pos = list(endUse_aggregates).index(key)

            if value:
                aggregate_multiplier \
                    = self.MISO2.ParameterDict[('MISO2_' + value)].Values[r, :, selector_multiplier_materials, t]
            else:
                aggregate_multiplier = 0

            aggr_req_mg[:, enduse_pos, t] \
                = self.MISO2.FlowDict['F_9_10'].Values[r, :, selector_multiplier_materials, enduse_pos, t] \
                * aggregate_multiplier
                
        # sum up dimensions of aggregate demand
        # requirement per material and end-use
        aggr_req_g = np.einsum('emg->mg',
                               aggr_req_mg[:, :, t]).reshape(1, self.miso_config.Ng)

        # shares of aggregate requirements in end-uses; np.divide avoids division by zero (0)
        aggr_req_shares = np.divide(aggr_req_g, aggr_req_g.sum(), out=np.zeros_like(aggr_req_g),
                                    where=aggr_req_g.sum() != 0)
        # shares ensure that every end-use receives the amount of aggregate is requires

        # requirement per material at process 9 (total aggregates)
        aggr_req_p9 = np.einsum('mg->m',
                                aggr_req_g[:])
        # requirement of aggregates at process 9 considering net trade
        aggr_req_p9_trade = max(aggr_req_p9 - net_trade_aggr_p9.sum(), 0)

        # ! if trade in aggregates is larger than what is required at process 9,
        # the excess aggregate is assumed to be transferred from virgin aggregates to recycling in next year
        # ! assuming distribution of excess aggregates 1. among materials according to shares in net trade,
        # and 2. among end-uses according to requirements (aggr_req_shares)
        trade_larger_req = abs(min(aggr_req_p9 - net_trade_aggr_p9.sum(), 0))
        new_trade_larger_req_g = trade_larger_req * aggr_req_shares

        # calculate potential SUPPLY of recycled aggregates from recoverable waste supply of last year F_20_4c
        aggr_supply_rec = \
            self.MISO2.FlowDict['F_20_4c'].Values[aggr_virgin_idx].copy()

        # calculate potential SUPPLY of downcycled aggregates per material (asphalt, bricks, concrete)
        # from potential supply of these materials for downcycling from last year + recycled aggr_downcycled
        supply_downcycl_construction_m = \
            self.MISO2.FlowDict['F_20_4c'].Values[selector_downc_idx].copy()
        supply_recycled_aggr_downcycl = \
            self.MISO2.FlowDict['F_20_4c'].Values[aggr_downcycl_idx].copy()
        aggr_supply_downc = supply_downcycl_construction_m.sum() + supply_recycled_aggr_downcycl
        # total downcycled aggregate supply (material dimension summed up)

        # calculate USE of recycled virgin aggregate at process 9 (p9) = total aggregate requirement at p9 (aggr_req_p9) -
        # max (aggr_req_p9 - supply of recycled virgin aggregate * yield factor virgin aggregate, 0)

        self.miso_model_data.rec_virginAggr[:, aggr_virgin_pos] = \
            max((aggr_req_p9_trade
                 - max(aggr_req_p9_trade - aggr_supply_rec
                       * waste_cascade_dict.get('waste_cascade_factor_p579')[0, aggr_virgin_pos], 0)), 0)

        # calculate USE of recycled virgin aggregate at p3 = use of recycled virgin aggregate at p9 / yield
        # factor virgin aggregate
        self.MISO2.FlowDict['F_20_4a'].Values[aggr_virgin_idx] \
            = self.miso_model_data.rec_virginAggr[:, aggr_virgin_pos] \
            / waste_cascade_dict.get('waste_cascade_factor_p579')[0, aggr_virgin_pos]

        # calculate USE of downcycled aggregate at p3 = min (hypothetical downcycled requirement at p3 to
        # cover aggregate gap at p9, supply of downcycled aggregate) =
        # min ((aggr_req_p9_trade - use of recycled virgin aggregate at p9)/yield factor virgin
        # aggregate, supply of downcycled aggregate)
        self.MISO2.FlowDict['F_20_4b'].Values[aggr_downcycl_idx] = \
            max(min((aggr_req_p9_trade - self.miso_model_data.rec_virginAggr[:, aggr_virgin_pos])
                    / waste_cascade_dict.get('waste_cascade_factor_p579')[0, aggr_downcycl_pos], aggr_supply_downc), 0)

        # required extraction of virgin aggregate at p3 = (required extraction of virgin aggregate
        # at p9 / yield factor virgin aggregate) =
        # ((aggr_req_p9_trade -use of recycled virgin aggregate at p9 - use of downcycled
        # aggregate at p9)/ yield factor virgin aggregate)
        # use of downcycled aggregate at p9 = use of downcycled aggregate at p3 * yield factor downcycled aggregate
        virgin_gap_p3 = max(
            (aggr_req_p9_trade - self.miso_model_data.rec_virginAggr[:, aggr_virgin_pos]
             - self.MISO2.FlowDict['F_20_4b'].Values[aggr_downcycl_idx]
             * waste_cascade_dict.get('waste_cascade_factor_p579')[0, aggr_downcycl_pos])
            / waste_cascade_dict.get('waste_cascade_factor_p579')[0, aggr_virgin_pos], 0)

        # write required extraction of virgin aggregate to fill aggregate gap at p3 to src flow
        self.MISO2.FlowDict['F_3_4a'].Values[aggr_virgin_idx] = virgin_gap_p3

        # calculate USE of bricks, concrete, asphalt for downcycling
        # ! assuming use of each material according to shares in initial supply (variable supply_downcycl_construction_m)
        # ! assuming that downcycled aggregate requirement first catered from recycled aggr_downcycl
        # and only if insufficient downcycling of bricks, asphalt, concrete occurring
        self.miso_model_data.downc_constr_mats[selector_downc_idx] = \
            max(
                self.MISO2.FlowDict['F_20_4b'].Values[aggr_downcycl_idx]
                - supply_recycled_aggr_downcycl, 0) \
            * np.divide(
                supply_downcycl_construction_m,
                supply_downcycl_construction_m.sum(),
                out=np.zeros_like(supply_downcycl_construction_m),
                where=supply_downcycl_construction_m.sum() != 0)

        # calculate USE of recycled aggr_downcycl = total downcycling (F_20_4_b) - downcycling of bricks, asphalt, concrete
        # used to estimate final waste (for this we require to separate the use of downcycled construction minerals and recycled aggregates_downcycled)
        self.miso_model_data.downc_constr_mats[aggr_downcycl_idx] = \
            self.MISO2.FlowDict['F_20_4b'].Values[aggr_downcycl_idx] \
            - self.miso_model_data.downc_constr_mats[selector_downc_idx].sum()

        # assign the actually used aggregate recycling/downcycling flows (F_20_4x) in year to
        # recycling/downcycling in year t-1 (F_13_20x)
        self.MISO2.FlowDict['F_13_20a'].Values[r, :, aggr_virgin_pos, t - 1] \
            = self.MISO2.FlowDict['F_20_4a'].Values[aggr_virgin_idx].copy()

        self.MISO2.FlowDict['F_13_20b'].Values[r, :, :, t - 1] \
            = self.MISO2.FlowDict['F_20_4b'].Values[remt_idx]

        # set aggregate supply (F_13_20, F_20_4c) to zero, so it does not impair mass balance
        # (was just a helper flow to handle material use over time-step)
        self.MISO2.FlowDict['F_13_20c'].Values[r, :, :, t - 1] = 0
        self.MISO2.FlowDict['F_20_4c'].Values[remt_idx] = 0

        # last: set end-use shares for aggregates to the distribution of required aggregates per
        # end-uses (variable aggr_req_g; shares used to split F_9_10 to end-uses in next loop iteration)
        enduse_shares[r, :, aggr_virgin_pos, :, t] = \
            aggr_req_shares
        enduse_shares[r, :, aggr_downcycl_pos, :, t] = \
            aggr_req_shares

        return new_trade_larger_req_g

    def P10_Dynamic_Stock_Model(self, r, dimensions):
        """
        Dynamic Stock Model calculation of stock by cohort and EoL outflows

        Modifies F_10_11 / S_10 inplace.

        TO DO: See if the stock computation can be simplified. Currently running over all years every year.
        odym function .compute_s_c_inflow_driven() needs dimensions t,tc; if running per year these are not \
        given anymore. tbd

        Args:
            r(int): Region index
            dimensions(dict): Model dimensions

        """
        logger.debug("Calculating P10 dynamic stock model")

        for g in range(0, dimensions["Ng"]):
            for e in range(0, dimensions["Ne"]):
                for m in range(0, dimensions["Nm"]):
                    remgt_idx = np.s_[r, e, m, g, :]
                    remgtt_idx = np.s_[r, e, m, g, :, :]

                    # short-circuited any call is optimisation for relatively sparse arrays
                    if not nb_any(self.MISO2.FlowDict['F_9_10'].Values[remgt_idx]):
                        self.MISO2.FlowDict['F_10_11'].Values[remgt_idx] = 0
                        self.MISO2.StockDict['S_10'].Values[remgtt_idx] = 0
                    else:
                        DSM_MISO2 = dsm.DynamicStockModel(
                            t=np.arange(0, self.miso_config.Nt, 1),
                            i=self.MISO2.FlowDict['F_9_10'].Values[remgt_idx],
                            lt={})
                        DSM_MISO2.sf = self.sf_array[remgtt_idx]

                        Stock_by_cohort = DSM_MISO2.compute_s_c_inflow_driven()
                        self.MISO2.FlowDict['F_10_11'].Values[remgt_idx] \
                            = np.einsum('tc->t', DSM_MISO2.compute_o_c_from_s_c())
                        self.MISO2.StockDict['S_10'].Values[remgtt_idx] = Stock_by_cohort

    def P11_waste_collection(self, r, t):
        """
        Waste collection, calculate collected potentially recoverable new scrap and old scrap and total

        Args:
            r(int): Region index.
            t(int): Time index.
        """

        remt_idx = np.s_[r, :, :, t]
        remgt_idx = np.s_[r, :, :, :, t]

        new_scrap_pre_trade \
            = self.MISO2.FlowDict['F_5_11a'].Values[remt_idx] \
            + self.MISO2.FlowDict['F_7_11a'].Values[remt_idx] \
            + np.einsum('emg->em', self.MISO2.FlowDict['F_9_11a'].Values[remgt_idx])

        old_scrap_pre_trade = np.einsum('emg->em', self.MISO2.FlowDict['F_10_11'].Values[remgt_idx])

        # calculate ratio between old and new scrap to enable use of different recycling rates
        self.miso_model_data.ratio_old_scrap[remt_idx] = np.divide(
            old_scrap_pre_trade, (old_scrap_pre_trade + new_scrap_pre_trade),
            out=np.zeros_like(old_scrap_pre_trade), where=(old_scrap_pre_trade + new_scrap_pre_trade) != 0)

        self.MISO2.FlowDict['F_11_12'].Values[remt_idx] = old_scrap_pre_trade + new_scrap_pre_trade

        # calculate total unrecoverable waste/dissipation that is considered part of final waste
        self.MISO2.FlowDict['F_11_14'].Values[remt_idx] \
            = self.MISO2.FlowDict['F_5_11b'].Values[remt_idx] \
            + self.MISO2.FlowDict['F_7_11b'].Values[remt_idx] \
            + np.einsum('emg->em', self.MISO2.FlowDict['F_9_11b'].Values[remgt_idx])

    def P12_trade_recoverable_waste(self, r, t):
        """
        Trade of potentially recoverable waste

        Args:
            r(int): Region index.
            t(int): Time index.
        """
        remt_idx = np.s_[r, :, :, t]

        self.MISO2.FlowDict['F_12_19'].Values[remt_idx] = self.MISO2.ParameterDict['MISO2_Export_p12'].Values[remt_idx]
        # exports of waste
        self.MISO2.FlowDict['F_19_12'].Values[remt_idx] = self.MISO2.ParameterDict['MISO2_Import_p12'].Values[remt_idx]
        # imports of waste

        # total supply of potentially recoverable waste = domestic new scrap + domestic old scrap + imports - exports
        self.MISO2.FlowDict['F_12_13'].Values[remt_idx] \
            = self.MISO2.FlowDict['F_11_12'].Values[remt_idx] \
            + self.MISO2.FlowDict['F_19_12'].Values[remt_idx] \
            - self.MISO2.FlowDict['F_12_19'].Values[remt_idx]

        # balance p12 output in case of exports > (dom. production + imports)
        self.MISO2.FlowDict['F_12_13'].Values[remt_idx], self.miso_model_data.bal_12[remt_idx] \
            = self.balance_process(self.MISO2.FlowDict['F_12_13'].Values, r, t)

        # calculate absolute new scrap after trade and balancing (assuming old/new scrap ratio as in pre-trade scrap)
        self.miso_model_data.new_scrap[remt_idx] = self.MISO2.FlowDict['F_12_13'].Values[remt_idx] \
            * (np.ones_like(self.miso_model_data.ratio_old_scrap[remt_idx])
                - self.miso_model_data.ratio_old_scrap[remt_idx])

    def P13_recycling_downcycling(self, r, t, selector_downc_mats, new_scrap_cycle):
        """
        P13 recycling and downcycling

        start with flow F_12_13 = apparent supply of recoverable waste
        cycling data can differ in two dimensions: it can be in absolute flows (ABSOLUTE) or flows can be calculated by
        multiplying waste flows by cycling rates (RATE); and it can either refer to end-of-life cycling only (EoL),
        or to total cycling (TOTAL), which includes EoL and new scrap cycling for recycling, these four categories are
        exclusive per material and year (and thus additive), meaning that a material can either have absolute or rate
        recycling input data for either EoL or Total recycling for downcycling, only the exclusive (additive) categories
        absolute or rate recycling input data apply, as we assume that no new scrap is downcycled (and thus both cases
        always refer to EoL not Total cycling.

        Args:
            r(int): Region index.
            t(int): Time index.
            selector_downc_mats(list): Indices of downcycle materials
            new_scrap_cycle():

        """
        aggr_virgin_pos = self.miso_config.material_position.get('aggr_virgin')
        aggr_downcycl_pos = self.miso_config.material_position.get('aggr_downcycl')

        remt_idx = np.s_[r, :, :, t]
        remt_idx_next_year = np.s_[r, :, :, t+1]
        downc_mats_idx = np.s_[r, :, selector_downc_mats, t]

        # RECYCLING
        # total supply of new and old scrap from collection
        rec_supply_total = self.MISO2.FlowDict['F_12_13'].Values[remt_idx]

        # EOL ABSOLUTE recycling flows from exogenous data
        rec_abs_eol = self.MISO2.ParameterDict['MISO2_EoLAbsoluteRecycling'].Values[remt_idx]

        # EOL RATE recycling (according to materials that are identified to use recycling rates in rate_rec)
        rec_rate_eol = rec_supply_total \
            * self.miso_model_data.ratio_old_scrap[remt_idx] \
            * self.MISO2.ParameterDict['MISO2_EoLRateRecycling'].Values[remt_idx]

        # TOTAL ABSOLUTE recycling flows from exogenous data
        rec_abs_total = self.MISO2.ParameterDict['MISO2_TotalAbsoluteRecycling'].Values[remt_idx]

        # TOTAL RATE recycling flows from exogenous data
        rec_rate_total = rec_supply_total * self.MISO2.ParameterDict['MISO2_TotalRateRecycling'].Values[remt_idx]

        eol_recycl = rec_abs_eol + rec_rate_eol
        total_recycl = rec_abs_total + rec_rate_total

        # calculate new scrap/waste recycling
        add_new_scrap_recycl = np.logical_not(total_recycl) * self.miso_model_data.new_scrap[remt_idx] * new_scrap_cycle
        # ! this assumes that all zeros where no total cycling data is reported exhibit new scrap recycling
        # (as eol recycling requires addition of new scrap)
        # ! thus new scrap cycling is always considered although zero in total abs recycling could also mean zero recycling
        # might want to change that in the future
        
        recycling_all = eol_recycl + total_recycl + add_new_scrap_recycl
        # balancing, in case waste supply F_12_13 is smaller than recycling;
        # in that case limit cycling to total waste supply
        recycling_all_balanced = np.where(recycling_all > self.MISO2.FlowDict['F_12_13'].Values[remt_idx],
                                          self.MISO2.FlowDict['F_12_13'].Values[remt_idx], recycling_all)

        self.miso_model_data.bal_13a[remt_idx] = recycling_all - recycling_all_balanced
        
        if t < (self.miso_config.Nt-1):
            recycling_all_bal_total_prod = np.where(
                (recycling_all_balanced > self.MISO2.ParameterDict['MISO2_Production_p3_total'].Values[remt_idx_next_year]) &
                (self.MISO2.ParameterDict['MISO2_Production_p3_total'].Values[remt_idx_next_year] != 0),
                self.MISO2.ParameterDict['MISO2_Production_p3_total'].Values[remt_idx_next_year],
                recycling_all_balanced)
            self.miso_model_data.bal_13c[remt_idx] = recycling_all_balanced - recycling_all_bal_total_prod

        else:
            recycling_all_bal_total_prod = recycling_all_balanced
            
        # estimate new scrap recycling - we assume that new scrap is recycled first (instead of old scrap)
        # bound new scrap recycling so that not larger than recycling_all_bal_total_prod (= < total production in next year)
        # save recycled new scrap in model_data variable to add to model output        
        self.miso_model_data.new_scrap_recycled[remt_idx] = np.where(
                                    (self.miso_model_data.new_scrap[remt_idx] * new_scrap_cycle) > recycling_all_bal_total_prod,
                                    recycling_all_bal_total_prod, (self.miso_model_data.new_scrap[remt_idx] * new_scrap_cycle))
        
        # assign balanced total recycling to recycling flow variable          
        self.MISO2.FlowDict['F_13_20a'].Values[remt_idx] = recycling_all_bal_total_prod
        

        # DOWNCYCLING
        # calculate potential supply of downcycled aggregates from remainder after recycling
        # remainder after recycling = remaining waste from F_12_13, after subtracting recycling flow F_13_20a
        self.miso_model_data.rem_waste_Cycling_noAggr[remt_idx] = \
            self.MISO2.FlowDict['F_12_13'].Values[remt_idx] \
            - self.MISO2.FlowDict['F_13_20a'].Values[remt_idx]

        # ABSOLUTE downcycling
        downc_abs = self.MISO2.ParameterDict['MISO2_EoLAbsoluteDowncycling'].Values[remt_idx]

        # balancing, in case waste supply F_12_13 is smaller than exogenous absolute cycling data in abs_rec
        # in that case limit cycling to total waste supply
        # ! could also be reversed = waste supply is set to absolute data
        # waste might also come from deposit from past years
        downc_balanced = np.where(
            downc_abs.sum() > self.miso_model_data.rem_waste_Cycling_noAggr[downc_mats_idx].sum(),
            self.miso_model_data.rem_waste_Cycling_noAggr[downc_mats_idx].sum(), downc_abs.sum())

        self.miso_model_data.bal_13b[remt_idx] = downc_balanced - downc_abs

        # flow of downcycled aggregates split to asphalt, bricks, concrete which supply base materials for downcycling
        # (assuming shares in EoL supply)
        downc_balanced_split = downc_balanced * \
            np.divide(self.miso_model_data.rem_waste_Cycling_noAggr[downc_mats_idx],
                      self.miso_model_data.rem_waste_Cycling_noAggr[downc_mats_idx].sum(),
                      out=np.zeros_like(self.miso_model_data.rem_waste_Cycling_noAggr[downc_mats_idx]),
                      where=self.miso_model_data.rem_waste_Cycling_noAggr[downc_mats_idx].sum() != 0)

        # RATE downcycling
        downc_rate \
            = (self.miso_model_data.rem_waste_Cycling_noAggr[downc_mats_idx].reshape((len(selector_downc_mats), 1))
               * self.MISO2.ParameterDict['MISO2_EoLRateDowncycling'].Values[downc_mats_idx])

        self.MISO2.FlowDict['F_13_20b'].Values[downc_mats_idx] = downc_balanced_split + downc_rate

        # for aggregates, material recycling is only potential supply (partially to be recycled in next year)-->
        # transfer recycled aggregates (virgin & downcycled) to F_13_20c supply matrix
        # supply of recycled aggregates from recycling of virgin aggregates
        self.MISO2.FlowDict['F_13_20c'].Values[r, :, aggr_virgin_pos, t] = \
            self.MISO2.FlowDict['F_13_20a'].Values[r, :, aggr_virgin_pos, t].copy()
        # supply of downcycled aggregates from recycling of downcycled aggregates from EoL
        self.MISO2.FlowDict['F_13_20c'].Values[r, :, aggr_downcycl_pos, t] = \
            self.MISO2.FlowDict['F_13_20a'].Values[r, :, aggr_downcycl_pos, t].copy()
        # supply of asphalt, bricks, concrete for downcycling from EoL of these materials
        # ! assumption: no new scrap from these materials for downcycling as quasi non-existent in production processes
        self.MISO2.FlowDict['F_13_20c'].Values[downc_mats_idx] = \
            self.MISO2.FlowDict['F_13_20b'].Values[downc_mats_idx].copy()

        # set the transfer recycled/downcycled aggregates and asphalt,
        # bricks, concrete for downcycling from step before in 13_20a/b to zero (fully transferred)
        self.MISO2.FlowDict['F_13_20a'].Values[r, :, [aggr_virgin_pos, aggr_downcycl_pos], t] = 0
        self.MISO2.FlowDict['F_13_20b'].Values[downc_mats_idx] = 0

        # no total downcycling as downcycled new scrap quasi non-existent (assumed)

    def P14_final_waste_disposal(self, r, t):
        """
        Final waste disposal

        Calculate final waste flows to outside system boundaries; calculate for previous year, \
            as F_13_14 could only be determined after
        Calculating actual use of recycled and downcycled aggregate in current year from supply of previous year

        Args:
            r(int): Region index.
            t(int): Time index.

        """
        remt_prev_year_idx = np.s_[r, :, :, t - 1]
        self.MISO2.FlowDict['F_14_0'].Values[remt_prev_year_idx] \
            = self.MISO2.FlowDict['F_13_14'].Values[remt_prev_year_idx] \
            + self.MISO2.FlowDict['F_11_14'].Values[remt_prev_year_idx]

    def P20_buffer_transfer(self, r, t, buffer_stocks_flows):
        """
        Buffer transferring cycled materials (20a/b) and potential supply of aggregates, \
            virgin & downcycled from year t to t+1

        Args:
            r(int): Region index.
            t(int): Time index.
            buffer_stocks_flows(dict): Dictionary mapping buffer flows and stocks

        """
        remt_idx = np.s_[r, :, :, t]
        for stock, flow in buffer_stocks_flows.items():
            self.MISO2.StockDict[stock].Values[remt_idx] = self.MISO2.FlowDict[flow].Values[remt_idx].copy()

    def balance_process_indexed(self, process_output, material_selector, r, t):
        """
        Balance process of selected material.

        Set negative consumption values to value of previous year

        Args:
            process_output(np.Array): Output of a process.
            material_selector(int): Index of selected material.
            r(int): Region index.
            t(int): Time index.

        Returns:
            cons_balanced(np.Array):
            balance_item(np.Array): Difference between original and modified array
        """

        consumption = process_output[r, :, material_selector, t].copy()
        consumption[consumption < 0] = np.nan
        # set negatives to NaN
        cons_balanced = np.where(
            np.isnan(consumption), process_output[r, :, material_selector, t - 1], consumption)
        # exchange NaNs with value from previous year
        balance_item = cons_balanced - process_output[r, :, material_selector, t]
        return cons_balanced, balance_item

    def balance_process(self, process_output, r, t):
        """
        Balance process of all materials.

        set negative consumption values to value of previous year
        Args:
            process_output(np.Array): Output of a process.
            r(int): Region index.
            t(int): Time index.

        Returns:
            cons_balanced(np.Array):
            balance_item(np.Array): Difference between original and modified array
        """

        consumption = process_output[r, :, :, t].copy()
        consumption[consumption < 0] = np.nan
        # set negatives to NaN
        cons_balanced = np.where(np.isnan(consumption), process_output[r, :, :, t - 1], consumption)
        # exchange NaNs with value from previous year
        balance_item = cons_balanced - process_output[r, :, :, t]
        return cons_balanced, balance_item

    def write_debug_to_xls(self, filepath):
        """
        Writes the model debug dataframes as sheets to an XLS file at specified location. Use this after a finished
        model run to get diagnostic information on the model that is not usually saved to outputs, e.g. mass balances.

        Args:
            filepath(str): Path and name of debug file.
        """
        with pd.ExcelWriter(filepath) as writer:
            for key, value in self._debug_output.items():
                if isinstance(value, pd.DataFrame):
                    if len(key) > 31:
                        sheet_name = key.split()
                    else:
                        sheet_name = key[:31]
                    # Excel sheet names can't have more than 32 characters

                    value.to_excel(writer, sheet_name=sheet_name, index=True)
                else:
                    logger.warning(f"Could not write {key} to excel since it is no Pandas dataframe")
