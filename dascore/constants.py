"""Constants used throughout obsplus."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from functools import partial
from pathlib import Path
from types import EllipsisType, MappingProxyType
from typing import Any, Literal, Protocol, TypeVar, get_args, runtime_checkable

import numpy as np
import pandas as pd

import dascore as dc
from dascore.compat import UPath

PatchType = TypeVar("PatchType", bound="dc.Patch")

SpoolType = TypeVar("SpoolType", bound="dc.BaseSpool")


@runtime_checkable
class ExecutorType(Protocol):
    """Protocol for Executors that DASCore can use."""

    # Two positional-only parameters, which is exactly what DASCore calls
    # this with. A named `iterables` plus a **kwargs catch-all excluded
    # ThreadPoolExecutor and ProcessPoolExecutor, which provide neither;
    # spelling the second parameter as *iterables instead would demand
    # arbitrarily many iterables and so exclude single-iterable clients.
    # The annotations are what stop a map() of the wrong shape (say one
    # taking an int) from satisfying this and failing at runtime.
    def map(self, fn: Callable, iterable: Iterable, /) -> Iterable:
        """Map function for applying concurrency of some flavor."""


# Bump this to force re-downloading of all data file
DATA_VERSION = "0.0.0"

# Types dascore can convert into time representations
timeable_types = int | float | str | np.datetime64 | pd.Timestamp
opt_timeable_types = None | timeable_types

# A (start, stop) selection range. Either end may be `...` to leave that
# side open, which is why these are not simply tuples of the value type.
time_select_type = tuple[
    opt_timeable_types | EllipsisType,
    opt_timeable_types | EllipsisType,
]
float_select_type = tuple[float | EllipsisType | None, float | EllipsisType | None]

# The `_attrs`/`_coords` namespace selectors accepted by select. Either a
# mapping of name -> selector (the general form) or a name/collection of
# names tagging which bare kwargs belong to that namespace.
namespace_select_type = Mapping[str, Any] | str | Iterable[str] | None

# Number types
numeric_types = int | float

# The smallest value an int64 can rep. (used as NaT by datetime64)
MININT64 = np.iinfo(np.int64).min

# The largest value an int64 can rep
MAXINT64 = np.iinfo(np.int64).max

# types used to represent paths
path_types = str | Path | UPath

# Protocols served over HTTP. The one spelling of this set; HDF5 tuning and
# the remote-cache downloader both key off it.
http_protocols = ("http", "https")

# Remote protocols that should use DASCore's smaller HDF5 readahead blocks.
# h5py performs many small metadata reads while opening files, and some S3-like
# backends default to large readahead chunks that overfetch remote data.
remote_hdf5_tuned_protocols = ("s3", "s3a", "s3n", *http_protocols)

# One second in numpy timedelta speak
ONE_SECOND = np.timedelta64(1, "s")

# One nanosecond
ONE_NANOSECOND = np.timedelta64(1, "ns")

# one billion
ONE_BILLION = 1_000_000_000

# One second with a precision of nano seconds
ONE_SECOND_IN_NS = np.timedelta64(ONE_BILLION, "ns")

# Valid strings for "datatype" attribute
DataType = Literal[
    "",  # unspecified
    "velocity",
    "strain_rate",
    "phase",
    "phase_difference",
    "phase_rate",
    "strain",
    "temperature",
    "temperature_gradient",
    "brillouin_spectrum",
    "fourier_transform",
    "amplitude_spectrum",
    "power_spectrum",
    "power_spectral_density",
    "frequency_band_energy",
    "stalta",
    "kurtosis",
    "envelope",
    "correlation",
    "tau_p",
    "dispersion",
    "phase_weighted_stack",
    "otdr",
]
VALID_DATA_TYPES = get_args(DataType)

# Valid categories (of instruments)
DataCategory = Literal["", "DAS", "DTS", "DSS"]
VALID_DATA_CATEGORIES = get_args(DataCategory)

max_lens = {
    "tag": 100,
    # Four codes of at most 12 characters plus separators, with headroom.
    "acquisition_key": 64,
    "dims": 40,
    "data_type": 32,
    "data_category": 4,
}

# Observing-system facts a reader may put in patch attrs. Every name is a
# field of the inventory's Acquisition, or of its Interrogator when dotted,
# so a value read from a file header and the same value enriched from an
# inventory are one attr rather than two spellings of one. The units are the
# inventory's: seconds, hertz, and meters. Readers convert at the parse
# boundary instead of shipping a companion units attr.
# Tested against the models in tests/test_core/test_attrs.py.
INVENTORY_ATTRS = (
    "closed_fiber_loop",
    "firmware_version",
    "gauge_length",
    "interrogator.instrument_type",
    "interrogator.manufacturer",
    "interrogator.model",
    "interrogator.name",
    "interrogator.serial_number",
    "interrogator_port",
    "pulse_rate",
    "pulse_width",
    "sample_rate",
    "software_version",
    "spatial_interval",
)

# Methods FileFormatter needs to support
FILE_FORMATTER_METHODS = ("read", "write", "get_format", "scan")

# These attributes are the default to ignore when determine if patches
# can be merged or broadcast together.
DEFAULT_ATTRS_TO_IGNORE = ("history", "dims")

# Large and small np.datetime64[ns] (used when defaults are needed)
SMALLDT64 = np.datetime64(MININT64 + 5_000_000_000, "ns")
LARGEDT64 = np.datetime64(MAXINT64 - 5_000_000_000, "ns")

# Required shared attributes to merge patches together
PATCH_MERGE_ATTRS = ("acquisition_key", "dims", "data_type", "data_category")

# Storage provenance: where a patch's bytes live rather than where its
# signal came from. The spool owns these and no reader may put them in
# patch attrs, since a patch merged from three files has no single answer.
# The pre-rename spellings are listed too: a patch carrying one of those is
# making the same claim under the old name.
STORAGE_PROVENANCE_ATTRS = (
    "source_path",
    "source_format",
    "source_version",
    "path",
    "file_format",
    "file_version",
)

# Level of progress bar
PROGRESS_LEVELS = Literal["standard", "basic", None]

# Options for handling specific warnings. "ignore" and None both mean
# "do nothing"; warn_or_raise has always accepted either.
WARN_LEVELS = Literal["warn", "raise", "ignore", None]

# The actions warnings.simplefilter and warnings.filterwarnings accept.
# Spelled out because the standard library's alias for them is stub-only.
# "all" is deliberately absent: Python only began accepting it in 3.14, and
# it raises on the 3.11-3.13 interpreters this project also supports.
WARNING_ACTIONS = Literal["default", "error", "ignore", "always", "module", "once"]

# A map from the unit name to the code used in numpy.timedelta64. The codes
# are spelled out in the annotation because numpy's unit parameter accepts
# only those literals, not str.
NUMPY_TIME_UNIT_MAPPING: Mapping[
    str,
    Literal["h", "m", "s", "ms", "us", "ns", "ps", "fs", "as", "Y", "M", "W", "D"],
] = {
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
indicated by dim (eg tag, history, acquisition_key, etc). If "drop" simply
drop conflicting attributes, or attributes not shared by all models.
If "raise" raise an
[AttributeMergeError](`dascore.exceptions.AttributeMergeError`] when
issues are encountered. If "keep_first", just keep the first value
for each attribute.
"""


select_values_description = """
Any dimension name can be passed as key, and the values can be:
    - a tuple of (min, max) for that dimension, or an equivalent slice.
      `None` and ... both indicate open intervals.
    - an integer, when `samples=True`, to select a single row or column.
    - an array of values to select, which must be a subset of the
      coordinate array.
    - an array of booleans of the same length as the coordinate where
      `True` indicates values to keep. This form does not support
      `samples=True`.
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
    dc_blue="blue",
    dc_red="red",
    dc_yellow="yellow",
    default_coord="bold",
    coord_range="bold green",
    coord_monotonic="bold grey",
    coord_segmented="bold cyan",
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


DEFAULT_COLORMAPS = {
    "frequency_band_energy": "Spectral_r",
    "stalta": "RdGy_r",
    "kurtosis": "gnuplot2",
    "envelope": "viridis",
    "correlation": "RdBu_r",
    "tau_p": "magma",
    "dispersion": "turbo",
    "phase_weighted_stack": "viridis",
    "fourier_transform": "magma",
    "power_spectral_density": "turbo",
    "power_spectrum": "turbo",
    "amplitude_spectrum": "turbo",
    "strain_rate": "RdBu_r",
    "strain": "seismic",
    "velocity": "viridis",
    "phase": "twilight_shifted",
    "phase_difference": "bone",
    "phase_rate": "seismic",
    "temperature": "coolwarm",
    "temperature_gradient": "RdYlBu_r",
    "otdr": "viridis",
}
