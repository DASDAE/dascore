"""Constants used throughout obsplus."""
from pathlib import Path
from typing import TypeVar, Union

import numpy as np
import pandas as pd

PatchType = TypeVar("PatchType", bound="dascore.Patch")

SpoolType = TypeVar("SpoolType", bound="dascore.Spool")

# Bump this to force re-downloading of all data file
DATA_VERSION = "0.0.0"

# Types dascore can convert into time representations
timeable_types = Union[int, float, str, np.datetime64, pd.Timestamp]

# Number types
numeric_types = Union[int, float]

# Expected Keys in the Summary Dictionaries
SUMMARY_KEYS = ("format", "min_time", "max_time", "min_distance", "max_distance")

# expected fiber attributes
DEFAULT_DIMS = ("time", "distance")

# expected DAS attributes
REQUIRED_DAS_ATTRS = ("d_time", "d_distance")

# The smallest value an int64 can rep. (used as NaT by datetime64)
MININT64 = np.iinfo(np.int64).min

# The largest value an int64 can rep
MAXINT64 = np.iinfo(np.int64).max

# types used to represent paths
path_types = Union[str, Path]

# One second in numpy timedelta speak
ONE_SECOND = np.timedelta64(1, "s")

# One nanosecond
ONE_NANOSECOND = np.timedelta64(1, "ns")

# One second with a precision of nano seconds
ONE_SECOND_IN_NS = np.timedelta64(1_000_000_000, "ns")


# The expected attributes for the Patch
DEFAULT_PATCH_ATTRS = {
    "d_time": np.NaN,
    "d_distance": np.NaN,
    "data_type": "DAS",
    "data_units": "",
    "category": "",
    "time_min": np.datetime64("NaT"),
    "time_max": np.datetime64("NaT"),
    "time_units": "s",
    "distance_min": np.NaN,
    "distance_max": np.NaN,
    "distance_units": "m",
    "network": "",
    "station": "",
    "instrument_id": "",
    "history": lambda: [],
    "dims": "",
    "tag": "",
}

# Methods FileFormatter needs to support
FILE_FORMATTER_METHODS = ("read", "write", "get_format", "scan")

# A set of attributes which are used in Patch equality checks.
COMPARE_ATTRS = {
    "d_time",
    "d_distance",
    "data_type",
    "category" "time_min",
    "time_max",
    "distance_min",
    "distance_max",
    "instrument_id",
}

# Large and small np.datetime64[ns] (used when defaults are needed)
SMALLDT64 = np.datetime64(MININT64 + 5_000_000_000, "ns")
LARGEDT64 = np.datetime64(MAXINT64 - 5_000_000_000, "ns")

# Required shared attributes to merge patches together
PATCH_MERGE_ATTRS = ("network", "station", "dims", "data_type", "category")


# A map from the unit name to the code used in numpy.timedelta64
NUMPY_TIME_UNIT_MAPPPING = {
    "hour": "h",
    "minute": "m",
    "second": "s",
    "millisecond": "ms",
    "microsecond": "us",
    "nanosecond": "ns",
    "picosecond": "ps",
    "femtosecond": "fs",
    "attosecond": "as",
    "year": "Y",
    "month": "M",
    "week": "W",
    "day": "D",
}
