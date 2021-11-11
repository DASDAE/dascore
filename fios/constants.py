"""Constants used throughout obsplus."""
from typing import TypedDict, TypeVar, Union

import numpy as np

import fios

PatchType = TypeVar("PatchType", bound="fios.Patch")
StreamType = TypeVar("StreamType", bound="fios.Stream")


# Bump this to force re-downloading of all data file
DATA_VERSION = "0.0.0"

# Types fios can convert into time representations
timeable_types = Union[int, float, str]

# Expected Keys in the Summary Dictionaries
SUMMARY_KEYS = ("format", "min_time", "max_time", "min_distance", "max_distance")

# expected fiber attributes
REQUIRED_FIBER_DIMS = ("time", "distance")

# expected DAS attributes
REQUIRED_DAS_ATTRS = ("dt", "dx")

# The smallest value an int64 can rep. (used as NaT by datetime64)
MININT64 = np.iinfo(np.int64).min

# The largest value an int64 can rep
MAXINT64 = np.iinfo(np.int64).max


class PatchSummaryDict(TypedDict):
    """The expected minimum attributes for a Patch attrs."""

    dt: np.timedelta64
    dx: float
    data_type: str
    category: str
    time_min: np.datetime64
    time_max: np.datetime64
    distance_min: float
    distance_max: float
    instrument_id: str


# The expected attributes for the Trace2D
DEFAULT_PATCH_ATTRS = {
    "dt": np.NaN,
    "dx": np.NaN,
    "data_type": "",
    "category": "",
    "time_min": np.datetime64("NaT"),
    "time_max": np.datetime64("NaT"),
    "distance_min": np.NaN,
    "distance_max": np.NaN,
    "network": "",
    "station": "",
    "instrument_id": "",
    "deployment_id": "",
    "history": lambda: [],
}

# A set of attributes which are used in Patch equality checks.
COMPARE_ATTRS = {
    "dt",
    "dx",
    "data_type",
    "category" "time_min",
    "time_max",
    "distance_min",
    "distance_max",
    "instrument_id",
    "deployment_id",
}

# Large and small np.datetime64[ns] (used when defaults are needed)
SMALLDT64 = np.datetime64(MININT64 + 5_000_000_000, "ns")
LARGEDT64 = np.datetime64(MAXINT64 - 5_000_000_000, "ns")
