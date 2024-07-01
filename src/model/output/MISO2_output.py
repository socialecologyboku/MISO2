#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 17:27:56 2022

@author: bgrammer
"""
import platform
import os
import pickle
import logging as logger
from datetime import datetime
import json
import numpy as np
import pandas as pd

import model.output.MISO2_output_constants as OutputConstants


class MISO2Output:
    """
    Contains the output of a src model run.

    Args:
        mfa_system_control(dict):
        run_options(dict):
        region(String): Name of the region.
        mc_run(int): Number of Monte Carlo iteration. Optional.

    Attributes:
        metadata(dict): Metadata
        values(dict): Pandas dataframes of shape ()
        data_constraints(dict):
    """
    __slots__ = "values", "metadata", "data_constraints"

    def __init__(self, mfa_system_control=None, run_options=None, region=None, mc_run=None, config_id=None):

        if mfa_system_control is not None and run_options is not None:
            logger.info("Creating output object with metadata")
            sys_info = platform.uname()
            self.metadata = {
                "MISO2ConfigID": config_id,
                "Output_type": "MISO2Output",
                "Region": region,
                "MC_run": mc_run,
                "Saved_filenames": [],
                "Created": datetime.now().strftime("%Y_%m_%d %H_%M_%S"),
                "Run_options": run_options,
                "MFA_system_definitions": mfa_system_control,
                "System_info": {
                    "system": sys_info[0],
                    "node": sys_info[1],
                    "version": sys_info[2],
                    "machine": sys_info[3],
                    "processor": sys_info[4]}
                }
        else:
            logger.info("Creating empty dummy output")
            self.metadata = {}

        self.data_constraints = {
            "must_be_zero": "",
            "no_data_yet": "r_effective_downcycling",
            "may_be_negative": "mass_bal_system"
            }

        self.values = {}

    # sf need to be saved separately
    def add_result(self, name, df, description="result"):
        """
        Add a dataframe to the MISO2Output object.

        Args:
            name(str): Name of output
            df(Pandas.DataFrame): DataFrame of results
            description(str): Type of result, e.g. "result" or "mean"

        """
        logger.info(f"Adding {name} to result values")
        new_index = []
        old_index = df.index.names

        if "type" not in old_index:
            df["type"] = description
            new_index += ["type"]
        if "region" not in old_index:
            df["region"] = self.metadata["Region"]
            new_index += ["region"]
        if "name" not in old_index:
            df["name"] = name
            new_index += ["name"]

        if "time" in old_index:
            df.reset_index(inplace=True)
            df.set_index(["type", "region", "name", "material", "sector", "time"], inplace=True)
        elif "sector" in old_index:
            df.reset_index(inplace=True)
            df.set_index(["type", "region", "name", "material", "sector"], inplace=True)
        else:
            df.reset_index(inplace=True)
            df.set_index(["type", "region", "name", "material"], inplace=True)

        # create new multiindex

        df.columns = df.columns.astype(str)
        # columns names are passed as ints from model, this cannot be saved to parquet

        if name in self.values:
            self.values[name][description] = df
        else:
            self.values[name] = {description: df}
        # add result, name and region as index

    def save_to_file(self, filetype, output_path="model_output_data", filename=None):
        """
        Convenience wrapper for saving MISO2Output to disk.

        Note that parquet will save to multiple files, grouped by array shape, xlsx will save
        to a one file per type of data (e.g. result or var) with dataframes as sheets, while \
            pickle will save the entire object. There is no option to restore a miso output \
                object from XLSX.

        Args:
            output_path(str): Path to save file to.
            filetype(str): Available options = "xlsx", "parquet", "pickle".
            filename(str): Defaults to "MISO2_output_Nr_<Nr>.<extension>". \
                The correct extension will be added automatically.

        Raises:
            AttributeError: If no legitimate filetype is provided.
        """
        if filename is None:
            filename = f"MISO2_output_{self.metadata['Region']}"

        logger.info(f"Saving file {filename} to {filetype}")
        if filetype == "parquet":
            self._combine_and_save_to_parquet(output_path=output_path, filename=filename)
        elif filetype == "pickle":
            self._save_to_pickle(output_path=output_path, filename=filename)
        elif filetype == "xlsx":
            self._save_to_excel(output_path=output_path, prefix_filename=filename)
        elif filetype == "csv":
            self._save_to_csv(output_path=output_path, prefix_filename=filename)
        else:
            error_msg = f"Not a legitimate filetype: {filetype}"
            raise AttributeError(error_msg)

    def _save_to_pickle(self, output_path, filename):
        with open(os.path.join(output_path, filename), 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_to_csv(self, output_path, prefix_filename):
        for result_type, dataframes in self.values.items():
            filename_result = os.path.join(output_path, prefix_filename + "_" + result_type)
            for key, value in dataframes.items():
                filename = filename_result + "_" + key + ".csv"
                logger.info(f"Saving {result_type} as {filename}")
                value.to_csv(filename)

    def _save_to_excel(self, output_path, prefix_filename):
        """
        Save the output's values dictionary into XLSX workbooks with dataframes as sheets.

        One workbook will be created for each type of result. Sheet names will be truncated to 31 characters
        due to Excels limitations.

        Args:
            output_path(str): Path to save to
            filename(str): Filename

        """
        logger.info("Saving MISO Output to Excel")

        for result_type, dataframes in self.values.items():
            filename = os.path.join(output_path, prefix_filename + "_" + result_type + ".xlsx")
            logger.info(f"Saving {result_type} as {filename}")

            with pd.ExcelWriter(filename) as writer:
                for key, value in dataframes.items():
                    if isinstance(value, pd.DataFrame):
                        if len(key) > 31:
                            sheet_name = key.split()
                        else:
                            sheet_name = key[:31]
                        # Excel sheet names can't have more than 32 characters

                        value.to_excel(writer, sheet_name=sheet_name, index=True)
                    else:
                        logger.warning(f"Could not write {key} to excel since it is no pd df")

    def restore_from_parquets(self, metadata_filename, folder_path):
        """
        Restore a MISO2Output object from a JSON holding the metadata.

        To use this function correctly, create an empty MISO2Output and use this function to load it with data.
        The validity of the JSON as metadata for an output file is checked via the existence of an "Output_type" key.

        Args:
            metadata_filename(str): Filename of the JSON to be loaded as metadata. Should include the file extension.
            folder_path(str): Path to the folder where JSON object is located.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the JSON exists, but is not valid MISO2Output metadata.
        """

        logger.info(f"Restoring MISO Output from Parquets: {metadata_filename}")

        metadata_filepath = os.path.join(folder_path, metadata_filename)
        if os.path.exists(metadata_filepath):
            logger.debug(f"Reading {metadata_filepath}")
            with open(metadata_filepath, encoding="utf-8") as file:
                loaded_metadata = json.load(file)
        else:
            error_msg = "Metadata json does not exist at this location"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if isinstance(loaded_metadata, dict) and "Output_type" in loaded_metadata.keys():
            self.metadata = loaded_metadata
        else:
            error_msg = "Tried to restore a MISO2Output from a JSON file that does not \
                appear to hold valid MISO2Output metadata"
            logger.error(error_msg)
            raise ValueError(error_msg)

        for filename in self.metadata["Saved_filenames"]:
            logger.debug(f"Reading file: {filename}")
            file_path = os.path.join(folder_path, filename)
            combined_df = pd.read_parquet(file_path)
            # loc by tuple

            original_df_names = combined_df.index.unique(level='name').values
            original_df_types = combined_df.index.unique(level='type').values

            # need two splits, by name and by type
            for type_name in original_df_types:
                idx = pd.IndexSlice
                df_by_type = combined_df.loc[idx[[type_name], ...]]
                for name in original_df_names:
                    idx2 = pd.IndexSlice
                    df_by_name = df_by_type.loc[idx2[:, :, [name], :]]
                    # format parameter name, type_name
                    if name in self.values:
                        self.values[name][type_name] = df_by_name
                    else:
                        self.values[name] = {type_name: df_by_name}

    def _combine_and_save_to_parquet(self, output_path, filename):
        """
        #TODO Missing docstring
        """
        df_shapes = {}

        for k, v in self.values.items():
            for result, values in v.items():
                if values.shape in df_shapes:
                    df_shapes[values.shape] += [self.values[k][result]]
                else:
                    df_shapes[values.shape] = [self.values[k][result]]

        self.metadata["Saved_filenames"] = []

        for df_list in df_shapes.values():
            combined_df = pd.concat(df_list)
            key = OutputConstants.map_index_to_output_type(combined_df)
            new_filename = filename + "_" + key + ".parquet"
            save_location = os.path.join(output_path, new_filename)
            combined_df.to_parquet(save_location)
            self.metadata["Saved_filenames"] += [new_filename]

        metadata_filename = "metadata_" + filename + ".json"
        save_location = os.path.join(output_path, metadata_filename)
        with open(save_location, "w", encoding="utf-8") as metadata_file:
            json.dump(obj=self.metadata, fp=metadata_file, indent=4)

    def equals_values(self, other_output, print_differences=False):
        """
        Compares equality of dataframe values between two miso outputs.

        Args:
            other_output(MISO2Output): Output to be compared.
            print_differences(bool): If true, non-equal parameters will be printed to console.

        Returns:
            all_equal(bool): Returns true if all values are np.allclose for all parameters.
        """

        all_equal = True

        if len(self.values) != len(other_output.values):
            all_equal = False
            if print_differences:
                print("Result types does not match")

        for parameter, result_type in self.values.items():
            for result, dataframe in result_type.items():
                df_equal = np.allclose(dataframe, other_output.values[parameter][result], equal_nan=True)
                if not df_equal:
                    all_equal = False

                if print_differences and not df_equal:
                    print(f"Parameter {parameter} values do not match for result type {result}")

        return all_equal

    def check_output_values(self):
        """
        Checks the dataframes in the values variable for negative and NaN values and
        gives a warning if any are found.
        """
        # this is fixed, right?
        for result, values in self.values.items():
            for descr, df in values.items():
                zeros = self.data_constraints["must_be_zero"] + self.data_constraints["no_data_yet"]

                if result not in zeros:
                    if result not in self.data_constraints["may_be_negative"]:
                        negative_values = df.values[df.values < 0]
                        if len(negative_values) > 0:
                            logger.warning(f"Output value {result} contains negative values in {descr}")
                    nan_values = df.isnull().values.any()
                    if nan_values:
                        nan_count = df.isnull().sum().sum()
                        logger.warning(f"Output value {result} in {descr} contains nan values: {nan_count}")
                if result in self.data_constraints["must_be_zero"]:
                    non_null = df.values[df.values != 0]
                    if len(non_null) > 0:
                        logger.warning(f"Output value {result} contains non-zero values in {descr}, \
                                       but should be zero")


def save_batch_outputs(output_dict, folder_path, batch_number):
    for bias, output_by_sample_number in output_dict.items():
        # combine values of outputs with sample number
        for sample_number, miso_output in output_by_sample_number.items():
            pass
        subfolder = os.path.join(folder_path, bias)
        raise NotImplementedError
    # first key bias, second key sample number