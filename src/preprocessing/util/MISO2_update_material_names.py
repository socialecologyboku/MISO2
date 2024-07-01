#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper script to change material names in parameter input files.
Names in config / master need to be changed manually.

Created on Wed Jul 20 16:34:09 2022

@author: bgrammer
"""
import os
import glob
import logging as logger
import pandas as pd
import numpy as np

import preprocessing.util.MISO2_input_constants as input_constants
from util.MISO2_file_manager import regex_material_name


def main():
    """
    Exemplary usage
    """
    data_dir = os.path.join(os.pardir, "model_input_data", "USA")
    new_names_file = os.path.join(os.pardir, "documentation", "Material Names final.xlsx")
    update_material_names(data_dir, new_names_file)


def update_material_names(data_dir, new_names_file, version="_v1"):
    """
    Update input files with new names. Attention: version numbers of input files needs to match
    """

    new_names = pd.read_excel(new_names_file)

    names_only = new_names.loc[:, ("material_names_old", "material_names_new")]
    logger.info(names_only)
    conversion_dict = pd.Series(names_only.material_names_new.values, index=names_only.material_names_old).to_dict()
    logger.info(conversion_dict)
    parameters = list(
        pd.read_excel(data_dir+input_constants.UNCERTAINTY_DISTRIBUTION_FILENAME,
                      sheet_name='main').Parameter_Name)

    for parameter in parameters:
        try:
            file_name = data_dir+parameter+version+".xlsx"
            parameter_values = pd.read_excel(file_name, sheet_name=input_constants.VALUES_SHEET_NAME)
            parameter_values[input_constants.M_INDEX] = parameter_values.Material.map(conversion_dict)
            with pd.ExcelWriter(file_name, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                parameter_values.to_excel(writer, sheet_name=input_constants.VALUES_SHEET_NAME, index=False)

            logger.info(f"Updated: {file_name}")
        except FileNotFoundError:
            pass


def add_index_from_filename(data_dir, write_back=False):
    """
    Add Element column and material index (from filename) to excel sheets.
    Filenames must be of format uncertainty_<material>.xlsx

    """
    uncertainties_glob = os.path.join(data_dir, "*.xlsx")

    logger.info("Checking uncertainty files:")

    uncertainties_files = glob.glob(uncertainties_glob)
    logger.info(uncertainties_files)

    updated_dfs = {}
    for file_name in uncertainties_files:
        material_file = pd.read_excel(file_name, sheet_name=None)

        material_name = regex_material_name(file_name)
        updated_dfs[material_name] = {}
        for sheet_name, sheet in material_file.items():
            if input_constants.R_INDEX in sheet.columns.tolist():
                sheet[input_constants.E_INDEX] = "Element1"
                sheet[input_constants.M_INDEX] = material_name
                year_columns = np.arange(1900, 2017)
                column_names = [input_constants.R_INDEX, input_constants.E_INDEX, input_constants.M_INDEX] \
                    + list(year_columns)
                sheet = sheet[column_names]

                if write_back:
                    with pd.ExcelWriter(file_name, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                        sheet.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    updated_dfs[material_name][sheet_name] = sheet

    return updated_dfs
