#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of constants related to ODYM/src input files.

Created on Sat Mar  4 14:35:03 2023

@author: bgrammer
"""
#################################################

# Name of ODYM/MISO input cover sheet, if any

COVER_SHEET_NAME = "Cover"

#################################################

# Column headers of Indices according to ODYM parsing logic and input files

R_INDEX = "MISO2_country"
E_INDEX = "Element"
G_INDEX = "End_use_sector"
M_INDEX = "Material"
T_INDEX = "Time"
REMT_INDEX = [R_INDEX, E_INDEX, M_INDEX]
REMGT_INDEX = [R_INDEX, E_INDEX, M_INDEX, G_INDEX]

#################################################

# Dummy element name that needs to be present in ODYM parsing logic

DUMMY_ELEMENT = "Element1"

#################################################
# ODYM / Classification column headers

ODYM_REGION = ("Countries", R_INDEX)
ODYM_MATERIAL = ("SB_Material", M_INDEX)
ODYM_END_USE = ("End_use_sectors", G_INDEX)
ODYM_ELEMENT = ("Element", E_INDEX)

#################################################

# Config filenames. These need to be provided with src input datasets

CLASSIFICATION_FILENAME = "MISO2_classification_properties.xlsx"
UNCERTAINTY_DISTRIBUTION_FILENAME = "MISO2_uncertainty_distribution_settings.xlsx"
COUNTRY_CORRESPONDENCE_FILENAME = "MISO2_country_region_correspondence.xlsx"
UNCERTAINTY_SCORE_LOOKUP_FILENAME = "MISO2_uncertainty_scores.xlsx"
METADATA_FILENAME = "MISO2_metadata_template.xlsx"
#################################################

# Classification properties sheet names

MULTIPLIER_CEMENT_BITUMEN = "Multiplier_CementBitumen"
CLASSIFICATION_MATERIAL_SHEET = "SB_Material"
CLASSIFICATION_INDEX_COLUMN = "Item"

#################################################

# Sheet names for Values and uncertainties in input xls files

VALUES_SHEET_NAME = "values"
UNCERTAINTY_SHEET_NAME = "values_uncert"
COUNTRY_CORRESPONDENCE_SHEETNAME = "values"
CV_SHEET_NAME = "cv"
#################################################

# Column names for input creation files

UNCERTAINTY_SCORE_CATEGORIES_COLUMNS = ["Data_reliability", "Data_completeness", "Correlation_temporal",
                                        "Correlation_spatial", "Correlation_other"]
UNCERTAINTY_SCORE_TEMPORAL_COLUMN = "Correlation_temporal"
UNCERTAINTY_EXPERT_JUDGEMENT_COLUMN = "Expert_judgement"
UNCERTAINTY_ALL_SCORES = UNCERTAINTY_SCORE_CATEGORIES_COLUMNS + [UNCERTAINTY_EXPERT_JUDGEMENT_COLUMN]

REGION_COLUMN = "Region_Classification"
REGION_VALUE_COLUMN = "Region"
VALUE_COLUMN = "Value"
UNCERTAINTY_SCORE_COLUMN = "Uncertainty"
UNCERTAINTY_YEAR_START = "Year_from"
UNCERTAINTY_YEAR_STOP = "Year_to"
UNCERTAINTY_SOURCE_VALUES = [
    UNCERTAINTY_YEAR_START, UNCERTAINTY_YEAR_STOP, VALUE_COLUMN,
    UNCERTAINTY_EXPERT_JUDGEMENT_COLUMN] \
    + UNCERTAINTY_SCORE_CATEGORIES_COLUMNS
NOT_PARSE_COLUMN = "Not_parse_flag"
UNCERTAINTY_SOURCE_DATA_TYPES = {UNCERTAINTY_YEAR_START: "Int64", UNCERTAINTY_YEAR_STOP: "Int64",
                                 VALUE_COLUMN: "float", NOT_PARSE_COLUMN: "bool"}
for score in UNCERTAINTY_ALL_SCORES:
    UNCERTAINTY_SOURCE_DATA_TYPES[score] = "Int64"
#################################################
