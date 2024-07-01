#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of output constants.

Work in progress.

@author: bgrammer
"""

OUTPUT_FILEPATH = "model_output_data"
RESULT_TYPES = ["series", "cohorts", "enduse", "outputs"]

ENDUSE_INDEX = ["type", "region", "name", "material", "sector"]
COHORT_INDEX = ["type", "region", "name", "material", "time"]
OUTPUT_INDEX = ["type", "region", "name", "material"]
SERIES_INDEX = ["type", "region", "name", "material"]


def map_index_to_output_type(values):
    """
    Map index of a pd DataFrame to an MISO2 output type.

    Args:
        values(pd.DataFrame): A Pandas Dataframe

    Returns:
        key(String): corresponding result type
    """
    indices = values.index.names
    if values.shape[1] == 1:
        key = "series"
    elif "time" in indices:
        key = "cohorts"
    elif "sector" in indices:
        key = "enduse"
    else:
        key = "outputs"

    return key
