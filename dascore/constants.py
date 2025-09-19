"""Constants used throughout obsplus."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol, TypeVar, runtime_checkable

import numpy as np
import pandas as pd

import dascore as dc

PatchType = TypeVar("PatchType", bound="dc.Patch")

SpoolType = TypeVar("SpoolType", bound="dc.BaseSpool")


@runtime_checkable
class ExecutorType(Protocol):
    """Protocol for Executors that DASCore can use."""

    def map(self, func, iterables, **kwargs):
        """Map function for applying concurrency of some flavor."""


# Bump this to force re-downloading of all data file
DATA_VERSION = "0.0.0"

# Types dascore can convert into time representations
timeable_types = int | float | str | np.datetime64 | pd.Timestamp
opt_timeable_types = None | timeable_types

# Number types
numeric_types = int | float

# The smallest value an int64 can rep. (used as NaT by datetime64)
MININT64 = np.iinfo(np.int64).min

# The largest value an int64 can rep
MAXINT64 = np.iinfo(np.int64).max

# types used to represent paths
path_types = str | Path

# One second in numpy timedelta speak
ONE_SECOND = np.timedelta64(1, "s")

# One nanosecond
ONE_NANOSECOND = np.timedelta64(1, "ns")

# one billion
ONE_BILLION = 1_000_000_000

# One second with a precision of nano seconds
ONE_SECOND_IN_NS = np.timedelta64(ONE_BILLION, "ns")

# Float printing precision
FLOAT_PRECISION = 3

# Valid strings for "datatype" attribute
VALID_DATA_TYPES = (
    "",  # unspecified
    "velocity",
    "strain_rate",
    "phase",
    "phase_difference",
    "phase_rate",
    "strain",
    "temperature",
    "temperature_gradient",
)

# Valid categories (of instruments)
VALID_DATA_CATEGORIES = ("", "DAS", "DTS", "DSS")

max_lens = {
    "path": 120,
    "file_format": 15,
    "tag": 100,
    "network": 12,
    "station": 12,
    "dims": 40,
    "file_version": 9,
    "experiment_id": 50,
    "instrument_id": 50,
    "data_type": 20,
    "data_category": 4,
}

# Methods FileFormatter needs to support
FILE_FORMATTER_METHODS = ("read", "write", "get_format", "scan")

# These attributes are the default to ignore when determine if patches
# can be merged or broadcast together.
DEFAULT_ATTRS_TO_IGNORE = ("history", "dims")

# Large and small np.datetime64[ns] (used when defaults are needed)
SMALLDT64 = np.datetime64(MININT64 + 5_000_000_000, "ns")
LARGEDT64 = np.datetime64(MAXINT64 - 5_000_000_000, "ns")

# Required shared attributes to merge patches together
PATCH_MERGE_ATTRS = ("network", "station", "dims", "data_type", "data_category")

# Level of progress bar
PROGRESS_LEVELS = Literal["standard", "basic", None]

# Options for handling specific warnings
WARN_LEVELS = Literal["warn", "raise", None]

# A map from the unit name to the code used in numpy.timedelta64
NUMPY_TIME_UNIT_MAPPING = {
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

# description of samples argument
samples_arg_description = """
If True, the values in kwargs and step represent samples along a
dimension. Must be integers. Otherwise, values are assumed to have
same units as the specified dimension, or have units attached.
"""

attr_conflict_description = """
Indicates how to handle conflicts in attributes other than those
indicated by dim (eg tag, history, station, etc). If "drop" simply
drop conflicting attributes, or attributes not shared by all models.
If "raise" raise an
[AttributeMergeError](`dascore.exceptions.AttributeMergeError`] when
issues are encountered. If "keep_first", just keep the first value
for each attribute.
"""


select_values_description = """
Any dimension name can be passed as key, and the values can be:
    - a Slice or a tuple of (min, max) for that dimension. 
      `None` and ... both indicate open intervals.
    - an array of values to select, which must be a subset of the 
      coordinate array.
    - an array of booleans of the same length as the coordinate where
      `True` indicates values to keep. 
"""

check_behavior_description = """
check_behavior
    Indicates what to do when an incompatible patch is found in the
    spool. `None` will silently skip any incompatible patches,
    'warn' will issue a warning and then skip incompatible patches,
    'raise' will raise an
    [`IncompatiblePatchError`](`dascore.exceptions.IncompatiblePatchError`)
    if any incompatible patches are found.
"""


# Rich styles for various object displays.
dascore_styles = dict(
    np_array_threshold=100,  # max number of elements to show in array
    patch_history_array_threshold=10,  # max elements of array in hist str.
    dc_blue="blue",
    dc_red="red",
    dc_yellow="yellow",
    default_coord="bold",
    coord_range="bold green",
    coord_monotonic="bold grey",
    coord_array="bold orange",
    coord_degenerate="bold red",
    coord_non="bold red",
    units="bright blue",
    dtypes="bright black",
    keys="grey50",
    # these are for formatting date times
    ymd="blue",
    hms="green",
    dec="green",
)


_AGG_FUNCS: Mapping[str, Callable] = MappingProxyType(
    {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "min": np.nanmin,
        "max": np.nanmax,
        "sum": np.nansum,
        "std": np.nanstd,
        "first": partial(np.take, indices=0),
        "last": partial(np.take, indices=-1),
    }
)

DIM_REDUCE_DOCS = """
dim_reduce
    How to reduce the dimensional coordinate associated with the 
    aggregated axis. Can be the name of any valid aggregator, a callable,
    "empty" (the default) which returns a length 1 partial coord, or 
    "squeeze" which drops the coordinate. For dimensions with datetime 
    or timedelta datatypes, if the operation fails it will automatically 
    be applied to the coordinates converted to floats then the output 
    converted back to the appropriate time type. 
"""
