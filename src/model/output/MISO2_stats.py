#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 17:44:34 2022

@author: bgrammer
"""

import logging as logger
from copy import deepcopy
import pandas as pd
import numpy as np
from model.output.MISO2_output import MISO2Output


class MISO2Stats(MISO2Output):
    """
    Contains summary statistics (sample size, mean and sample variance) over multiple MISO2Output objects.

    This is a child class of MISO2Output.

    Attributes:
        values(dict): A nested dictionary of Pandas dataframes. Keys are parameter names and result types
            ("mean", "ssq" and "s_var").
        metadata(dict): Metadata information, copied from first output file added.
        n(int): Number of aggregated src outputs.
    """
    __slots__ = ("n",)

    def __init__(self):
        super().__init__()
        self.n = 0

    def add_miso_output(self, miso_output, add="result"):
        """
        Appends values of a MISO2Output object to the summary statistics.

        The implementation uses Welford's online algorithm. Sample variance is not continuously updated,
        but needs to be explicitly called for with the method calculate_s_var.

        Args:
            miso_output(MISO2Output): MISO2Output object, whose "result" values are added to the running mean/var.
            add(str): Type of output that is added. Defaults to "result".
        """

        logger.info("Adding output sheet to MISO Stats")
        if not isinstance(miso_output, MISO2Output):
            error_msg = f"Tried to add object to miso stats that is not a miso output: " \
                        f"{type(miso_output)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Welford's online algorithm following https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        if self.n == 0:
            self.metadata = deepcopy(miso_output.metadata)
            self.metadata["Total_mc_runs"] = None
            self.metadata["Output_type"] = "MISO2Stats"

            for result, values_dict in miso_output.values.items():
                for result_type, dataframe in values_dict.items():
                    if result_type == add:
                        self.values[result] = {
                            "mean": dataframe.copy(),
                            "ssq": pd.DataFrame(
                                np.zeros_like(dataframe),
                                index=dataframe.index,
                                columns=dataframe.columns),
                            "s_var": pd.DataFrame(
                                np.zeros_like(dataframe),
                                index=dataframe.index,
                                columns=dataframe.columns)}
            self.n += 1
        else:
            self._check_valid_output(miso_output)
            self.n += 1
            for result, values_dict in miso_output.values.items():
                for result_type, dataframe in values_dict.items():
                    if result_type == add:
                        mean = self.values[result]["mean"]
                        # error be here?
                        delta = dataframe - mean
                        mean += delta / self.n
                        delta2 = dataframe - mean
                        self.values[result]["ssq"] += delta * delta2

    def _check_valid_output(self, output):
        """
        Check if region and run options of MISO2Output added to object match.

        Args:
            output(MISO2Output): Output to be checked.

        Raises:
            ValueError: When options do not match.

        """
        control_metadata = self.metadata
        output_metadata = output.metadata

        if control_metadata["Total_mc_runs"]:
            error_msg = "Tried to add output to finalized miso stats"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if control_metadata["Region"] != output_metadata["Region"]:
            error_msg = f"Tried to add output of different region to miso stats: \
                {control_metadata['Region']} vs {output_metadata['Region']}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        for option, value in control_metadata["Run_options"].items():
            if value != output_metadata["Run_options"][option] and option != "bias":
                # bias may be different due to systematic errors (upper or higher than mean)
                error_msg = "Tried to add output with different run options to miso stats"
                logger.error(error_msg)
                raise ValueError(error_msg)

    def finalize(self):
        """
        Finalize MISO2Stats object by calculating variance and setting correct indices.

        Since index values change, no further output can be added to object once it has been finalized.
        """

        self._calculate_s_var()
        self._set_correct_indices()
        self.metadata["Total_mc_runs"] = self.n

    def _calculate_s_var(self):
        """
        Calculates sample variance (n-1) from the sum of squares.

        This method is used to finalize variance for the given n and ssq once the variance is required. This avoids
        rounding errors in variance due to continued division during the aggregation.

        """
        logger.info("Calculating variance sample")
        for value in self.values.values():
            if value is not None:
                value["s_var"] = value["ssq"] / (self.n-1)

    def _set_correct_indices(self):
        """
        Changes dataframe indices to indicate correct data type ("mean", "s_var", "ssq")
        """
        for values_dict in self.values.values():
            for result_type, df in values_dict.items():
                idx_names = df.index.names
                df.reset_index(inplace=True)
                df["type"] = result_type
                df.set_index(idx_names, inplace=True)

    def calculate_consecutive_stock_summaries(self, stocks="S10_stock_enduse"):
        """
        Returns consecutive stock summarise by enduse.

        Stock name / enduse output needs to be present for this to work.

        Args:
            stocks(str): Name of stock by enduse parameter

        Returns:
            combined(pd.DataFrame): Dataframe with combined results and number of run as part of index.

        """
        if stocks not in self.values:
            raise ValueError(f"Could not find stock name {stocks} in results")
        n = self.n  # number of runs

        idx_names = self.values[stocks]["mean"].index.names

        ssq = self.values[stocks]["ssq"].reset_index()
        ssq["type"] = "s_var"
        ssq.set_index(idx_names, inplace=True)
        means = self.values[stocks]["mean"].reset_index()
        means["type"] = "mean"
        means.set_index(idx_names, inplace=True)

        s_var = ssq / n
        combined = pd.concat([means, s_var])
        combined["run"] = n
        combined = combined.set_index("run", append=True)

        return combined