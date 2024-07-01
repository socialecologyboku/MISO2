#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:20:55 2022

@author: bgrammer
"""

import os
import logging as logger
from copy import deepcopy
from pathlib import Path
import json
import dask.dataframe as dd
import pandas as pd
import numpy as np
from model.output.MISO2_output import MISO2Output
from model.output.MISO2_stats import MISO2Stats
import model.output.MISO2_output_constants as OutputConstants


class MISO2Database:
    """
    Aggregate object for MISO2Outputs.

    Stores regional outputs into larger dataframes and provides convenience wrapper to subset the data.
    Only outputs with the same system definition and run options can be added to a database object.
    It is possible to add both MISO2Outputs and MISO2Stats, although the latter needs to be finalized.
    The output types that are aggregated are determined by the shape and index of the MISO2Output's data,
    with series having only one column, enduse containing a sector index, and cohorts containing both a time
    and a sector index. All other dataframes are added as regular outputs.

    Attributes:
        _values(dict): Holds dataframes for four output types ("series", "enduse", "cohorts", "outputs").
        _metadata(dict): Metadata of database object. Contains references for MFA system definition \
            and run options as well as original outputs and stats metadata and saved file names.

    """
    __slots__ = "_values", "_metadata", "_cohorts"

    def __init__(self):
        self._values = {
            "series": None,
            "enduse": None,
            "outputs": None
            }

        self._cohorts = []

        self._metadata = {
            "MFA_system_definitions": None,
            "Saved_filenames": dict.fromkeys(self._values.keys(), None),
            "outputs_metadata": {},
            "stats_metadata": {},
            "Run_options": None
            }

    def create_database_from_output_files(self, directory, output_type="output"):
        """
        Convenience wrapper to fill a MISO2 database from outputs on disk.

        The function will read in all .json metadata in the directory and all subdirectories

        Note that there are some validity checks in place to prevent adding outputs with different system definitions
        and run options, but these do not cover all possible scenarios of garbled input.
        Make sure that the specified directory only contains MISO2Outputs and MISO2Stats that you really want to add to
        the database.

        Args:
            directory(str): Path to folder which is to be read in.
            output_type(str): One of "output" or "stats"

        """

        logger.info(f"Creating database from output files in directory {directory}")

        miso_outputs = []
        json_paths = Path(directory).rglob('*.json')
        logger.info(f"Found JSONs: {list(json_paths)}")
        for json_path in json_paths:
            try:
                logger.debug(f"Trying to read file at path: {json_path}")
                if output_type == "output":
                    miso_output = MISO2Output()
                    miso_output.restore_from_parquets(str(json_path.name), str(json_path.parent))
                    miso_outputs.append(miso_output)
                elif output_type == "stats":
                    miso_stats = MISO2Stats()
                    miso_stats.restore_from_parquets(str(json_path.name), str(json_path.parent))
                    miso_outputs.append(miso_stats)
            except FileNotFoundError:
                logger.error(f"No JSON could actually be found at the given path {json_path.name}. \
                             Since this function globs all JSON from the directory, \
                                 this error should never be raised at this point.")
            except ValueError:
                logger.error(f"Tried to construct a MISO2Output from invalid metadata in the context \
                             of adding it to a MISO2Database. Skipping the file {json_path.name}")

        self.add_miso_outputs(miso_outputs)

    def add_miso_outputs(self, miso_outputs, save_cohorts=True, db_folder=None):
        """
        Add MISO2Output files to database.

        If not yet set, control references for MFA system definition and run options
        are taken from the first output in the list. If enduse cohorts are present, they
        will be saved as parquet into an external folder to be lazily loaded as a Dask dataframe.
        Please note that is not written for efficiency, since it is expected to be a one-time operation. A large number
        of output files might lead to significant processing time.

        Args:
            miso_outputs(list): List of MISO2Outputs.
            save_cohorts(bool): If True (default), cohorts, if any are present in the outputs, \
                will be saved as parquets in the db folder.
            db_folder(str): Relative path folder name where parquets are saved. This will override \
                any existing associated database folder of the object. Defaults "miso_database".

        Raises:
            TypeError: If trying to add non-MISO2Output or an empty input list.
            ValueError: If output MFA system definition and run options do not match database control values.
        """

        logger.info("Adding miso_outputs to database")

        if db_folder is None:
            db_folder = os.path.join(OutputConstants.OUTPUT_FILEPATH, "miso_database")

        if not miso_outputs:
            error_msg = "Passed an empty list to the function"
            logger.error(error_msg)
            raise TypeError(error_msg)

        for output in miso_outputs:
            self._check_output_validity(output)
            region = output.metadata["Region"]

            for parameter, result_types in output.values.items():
                if result_types != "ssq":
                    for result_type, values in result_types.items():
                        key = OutputConstants.map_index_to_output_type(values)
                        logger.debug(f"Adding {key}: {parameter}, {result_type}")

                        if key == "cohorts" and save_cohorts:
                            self._export_df_to_file(values, region, "cohorts", db_folder)
                        elif self._values[key] is None:
                            self._values[key] = values.copy()
                        else:
                            self._values[key] = pd.concat([self._values[key], values])

            if isinstance(output, MISO2Stats):
                logger.info("Adding MISO Stats object to database")
                self._metadata["stats_metadata"][region] = deepcopy(output.metadata)
            elif isinstance(output, MISO2Output):
                logger.info("Adding MISO output object to database")
                self._metadata["outputs_metadata"][region] = deepcopy(output.metadata)
            else:
                logger.error(f"Tried to add a MISO2Output object of a (child)type that \
                             cannot be handled: {type(output)}.")

            if self._metadata["MFA_system_definitions"] is None:
                self._metadata["MFA_system_definitions"] = deepcopy(output.metadata["MFA_system_definitions"])
                self._metadata["Run_options"] = deepcopy(output.metadata["Run_options"])

    def _export_df_to_file(self, df, file_name, file_type, db_folder):
        logger.info(f"Saving {file_type} to db folder {db_folder}")

        os.makedirs(db_folder, exist_ok=True)

        new_filename = file_name + "_" + file_type + ".parquet"
        save_location = os.path.join(db_folder, new_filename)
        df.to_parquet(save_location)
        self._cohorts.append(save_location)

    def _check_output_validity(self, output):
        """
        Checks if added MISO2Output is valid object and if run options and MFA system \
            definitions match to database control values.

        Args:
            output(MISO2Output): Object to be checked.

        Raises:
            TypeError: If trying to add non-MISO2Output.
            ValueError: If output MFA system definition and run options do not match database control values.

        """
        logger.info("Checking output validity")
        if not isinstance(output, MISO2Output):
            error_msg = f"Tried to add non-MISO2Output object to database: {type(output)}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        control_definition = self._metadata["MFA_system_definitions"]
        output_definition = output.metadata["MFA_system_definitions"]

        if isinstance(control_definition, dict):
            logger.debug("Checking if output system definitions match database")
            for parameter, value in control_definition.items():
                if output_definition[parameter] != value:
                    error_msg = f"MFA system definitions do not match for {parameter}: \
                        database value {value} vs output value {output_definition[parameter]}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

        control_options = self._metadata["Run_options"]
        output_options = output.metadata["Run_options"]
        if isinstance(control_options, dict):
            for parameter, value in control_options.items():
                if output_options[parameter] != value:
                    error_msg = f"Run options do not match for {parameter}: database value \
                        {value} vs output value {output_options[parameter]}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

    def save_to_file(self, file_type, output_path, export_cohorts=False, filename="MISO2_data"):
        """
        Wrapper method for saving database to file.

        Cohorts are already saved as parquets, so they will not be exported via save in that format.

        Args:
            file_type(str): One of "xls", "parquet" and "csv"
            output_path(str): Output folder
            export_cohorts(bool): Export cohorts. These files may be excessively large. Defaults to false.
            filename(str): Filename prefix of output files
        """
        logger.info(f"Saving to filetype: {file_type}")
        if file_type == "xls":
            raise NotImplementedError("Saving to excel is not fully supported, since the dask components of this"
                                      "class and larger pd Dataframes cannot be exported in reasonable time."
                                      "It is recommended you export to csv instead and open the files in Excel."
                                      "If you really want to do this, call the private _save_to_excel method on"
                                      "this object")
            # self._save_to_excel(output_path, filename)

        if file_type == "parquet":
            self._save_to_parquet(output_path, filename)
        elif file_type == "csv":
            self._save_to_csv(output_path, filename, export_cohorts)
        else:
            logger.error(f"Cannot handle filetype: {file_type}")
            raise AttributeError

    def _save_to_parquet(self, output_path, filename="miso_database"):
        """
        Saves the database object into JSON (metadata) and parquet (dataframes) files on disk.

        The filenames are saved into the object's metadata before dumping, so it can later be restored.

        Note that the filenames in the metadata of this object are overwritten with the parquet \
            filenames of the last save.

        Args:
            output_path(str): Folder paths where objects are to be saved.
            filename(str): Metadata filename for database. Metadata is saved with .JSON suffix, \
                dataframes with a "_$datatype.parquet" suffix.

        """
        logger.info("Saving database to parquet")

        for result_type, df in self._values.items():
            if isinstance(df, pd.DataFrame):
                new_filename = filename + "_" + result_type + ".parquet"
                save_location = os.path.join(output_path, new_filename)
                df.to_parquet(save_location)
                self._metadata["Saved_filenames"][result_type] = new_filename
            else:
                logger.warning(f"Non-dataframe object in result dictionary: {df}")
        metadata_filename = filename + ".json"

        save_location = os.path.join(output_path, metadata_filename)

        with open(save_location, "w", encoding="utf-8") as metadata_file:
            json.dump(obj=self._metadata, fp=metadata_file, indent=4)

    def _save_to_excel(self, output_path, prefix_name, export_cohorts=False):
        """
        Save contents of object into Excel files and csv (cohorts).

        Dask DataFrame does not have to_excel method yet.

        Args:
            output_path(str): Path to output folder. Will be created if not yet existent.
            prefix_name(str): Prefix to both output folder and filenames.

        """
        os.makedirs(output_path, exist_ok=True)
        logger.info("Saving to excel / csv")
        for result_type, dataframe in self._values.items():
            if isinstance(dataframe, pd.DataFrame):
                if result_type == "enduse":
                    file_name = prefix_name + "_" + result_type + ".csv"

                    dataframe.to_csv(os.path.join(output_path, file_name))
                else:
                    file_name = prefix_name + "_" + result_type + ".xlsx"

                    dataframe.to_excel(os.path.join(output_path, file_name), engine="openpyxl", index=True)
                logger.info(f"Saved {result_type} as {file_name}")

        if export_cohorts:
            cohorts = self.get_cohorts()
            cohorts_file_name = prefix_name + "_cohorts-*.csv"
            cohorts.to_csv(os.path.join(output_path, cohorts_file_name))

    def restore_from_parquet(self, folder_path, metadata_filename):
        """
        Restore a MISO2Database object from saved files.

        Args:
            folder_path(str): Location of files.
            metadata_filename(str): Name of metadata JSON file.

        Raises:
            FileNotFoundError: When metadata file does not exist at given location.
        """
        logger.info("Restoring database from parquet")

        metadata_filepath = os.path.join(folder_path, metadata_filename)
        file_exists = os.path.exists(metadata_filepath)
        if file_exists:
            logger.debug("Reading file")
            with open(metadata_filepath, encoding="utf-8") as file:
                self._metadata = json.load(file)
        else:
            error_msg = "Metadata json does not exist at this location"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        for result_type, filename in self._metadata["Saved_filenames"].items():
            if filename:
                try:
                    logger.debug(f"Reading file: {filename}")
                    file_path = os.path.join(folder_path, filename)
                    df = pd.read_parquet(file_path)
                    self._values[result_type] = df
                except FileNotFoundError:
                    logger.error(f"Could not find {filename} that is given in database metadata")

    def get_subset(self, output_type="outputs", result_type=slice(None), region=slice(None),
                   parameter=slice(None), material=slice(None), sector=slice(None),
                   years=slice(None), drop_levels=False):
        """
        Convenience method to subset the multi-index database and return a copy of the result.

        Single value arguments can be passed as strings, while multi-value arguments need to be passed as lists or
        tuples. All arguments default to slice(None), returning the entire index if not otherwise specified.
        Use the get_categories() method to see which index values are available.

        Args:
            output_type(string): One of "series", "enduses", "cohorts" or "outputs" (the default).
            result_type(str/list): The type of result to be returned. One of e.g. "result", "mean", "s_var".
            region(str/list): The region, e.g. "Austria" or "World".
            parameter(str/list): The name of the output parameter.
            material(str/list): The name of the material.
            sector(str/list): The name of the sector.
            years(tuple): Start and end year to be returned. The upper bound is included.
            drop_levels(bool): Remove single-value levels of subset when querying.

        Returns:
            df(Pandas.DataFrame/Pandas.Series): A subset of a dataframe.

        Raises:
            ValueError:

        """
        logger.info("Getting subset copy of database")

        df = self._values[output_type]
        idx = pd.IndexSlice

        if years == slice(None):
            years_slice = years
        else:
            year_from = years[0]
            year_to = years[1]

            if year_to < year_from:
                error_msg = f"Cannot return time series with end date {year_to} smaller than start date {year_from}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            years_slice = slice(str(year_from), str(year_to))

        if output_type == "outputs":
            df = df.loc[idx[result_type, region, parameter, material], years_slice]
        elif output_type == "series":
            df = df.loc[idx[result_type, region, parameter, material], years_slice]
        elif output_type == "enduse":
            df = df.loc[idx[result_type, region, parameter, material, sector], years_slice]
        elif output_type == "cohorts":
            df = df.loc[idx[result_type, region, parameter, material, sector, years_slice], years_slice]

        df = df.copy()

        if drop_levels and isinstance(df.index, pd.MultiIndex):
            df.index = df.index.remove_unused_levels()
            index_drops = []
            for i, index in enumerate(df.index.names):
                logger.debug(f"Dropping {index} from index")
                if len(df.index.levels[i]) == 1:
                    index_drops.append(i)
            df = df.droplevel(level=index_drops)

        return df

    def get_cohorts(self, cohort_paths=None):
        """
        Returns a Dask dataframe of the enduse cohorts from parquet.

        Args:
            cohort_paths(str/list/glob): Path, list of path or glob from where to read in cohorts.
                If None (default), the database will try to read in cohorts from its associated filepaths.

        Returns:
            dask_cohorts(Dask.DataFrame): Dask DataFrame with cohorts.
        """

        if len(self._cohorts) == 0:
            logger.warning("No cohorts associated with this database")

        if cohort_paths is None:
            cohort_paths = self._cohorts

        valid_paths = []

        for file_path in cohort_paths:
            if not os.path.isfile(file_path):
                logger.warning(f"File path: {file_path} does not exist")
            else:
                valid_paths.append(file_path)

        dask_cohorts = dd.read_parquet(cohort_paths)

        return dask_cohorts

    def get_categories(self, df_type=None):
        """
        Return unique index values of the database.

        Args:
            df_type(string): Specific dataframe type to be returned. One of "series","enduse","cohorts","outputs".
                If None (default), all categories are returned.

        Returns:
            indices(dict): Dictionary of available categories.
        """

        if df_type is None:
            indices = dict.fromkeys(self._values, dict())
            for result_type, dataframe in self._values.items():
                if isinstance(dataframe, pd.DataFrame):
                    for name, levels in zip(dataframe.index.names, dataframe.index.levels):
                        indices[result_type][name] = list(levels)
                    years = np.array(dataframe.columns.values, dtype=int)
                    indices[result_type]["time"] = [np.min(years), np.max(years)]
        else:
            indices = {}
            dataframe = self._values[df_type]
            if dataframe is not None:
                for name, levels in zip(dataframe.index.names, dataframe.index.levels):
                    indices[name] = list(levels)
                years = np.array(dataframe.columns.values, dtype=int)
                indices["time"] = [np.min(years), np.max(years)]
        return indices

    def _save_to_csv(self, output_path, prefix_name, export_cohorts=False):
        """
        Save contents of object into Excel files and csv (cohorts).

        Dask DataFrame does not have to_excel method yet.

        Args:
            output_path(str): Path to output folder. Will be created if not yet existent.
            prefix_name(str): Prefix to both output folder and filenames.

        """
        os.makedirs(output_path, exist_ok=True)
        logger.info("Saving to csv")
        for result_type, dataframe in self._values.items():
            if dataframe is not None:
                file_name = prefix_name + "_" + result_type + ".csv"
                if isinstance(dataframe, pd.DataFrame):
                    dataframe.to_csv(os.path.join(output_path, file_name))
                logger.info(f"Saved {result_type} as {file_name}")

        if export_cohorts:
            cohorts = self.get_cohorts()
            cohorts_file_name = prefix_name + "_cohorts-*.csv"
            cohorts.to_csv(os.path.join(output_path, cohorts_file_name))
