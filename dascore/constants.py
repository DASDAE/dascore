"""Constants used throughout obsplus."""
from typing import TypedDict, TypeVar, Union

import numpy as np
import pandas as pd

PatchType = TypeVar("PatchType", bound="dascore.Patch")

StreamType = TypeVar("StreamType", bound="dascore.Stream")

# Bump this to force re-downloading of all data file
DATA_VERSION = "0.0.0"

# Types dascore can convert into time representations
timeable_types = Union[int, float, str, np.datetime64, pd.Timestamp]

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


class PatchSummaryDict(TypedDict):
    """The expected minimum attributes for a Patch attrs."""

    d_time: np.timedelta64
    d_distance: float
    data_type: str
    category: str
    time_min: np.datetime64
    time_max: np.datetime64
    distance_min: float
    distance_max: float
    instrument_id: str
    dims: str
    tag: str


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
    "label": "",
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
