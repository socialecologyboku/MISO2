#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Collection of paths and filenames that can be processed by MISO2FileManager.

@author: bgrammer
"""

from enum import Enum


class MISO2FileTypes(Enum):
    """
    Collection of paths and filenames that can be processed by MISO2FileManager.
    """
    DATA_DIR = 1
    CONFIG = 2
    UNCERTAINTY_DISTRIBUTION = 3
    CLASSIFICATION = 4
    PARAMETER_FILE_LIST = 5
    PARAMETER_FILE_DICT = 6
    COUNTRY_REGION_CORRESPONDENCE = 7
    CV_LOOKUP = 8
    ADDITIONAL_DATA_DIR = 9
    UNCERTAINTY_VALUES_DIR = 10
    INPUT_SOURCES_DIR = 11
    METADATA_TEMPLATE = 12