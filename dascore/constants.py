"""Constants used throughout DASCore."""

from __future__ import annotations

import textwrap
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

SpoolType = TypeVar("SpoolType", bound="dc.Spool")


@runtime_checkable
class ExecutorType(Protocol):
    """Protocol for Executors that DASCore can use."""

    # Two positional-only parameters, which is how DASCore calls this.
    # A named `iterables` plus **kwargs excluded ThreadPoolExecutor and
    # ProcessPoolExecutor, and *iterables would exclude single-iterable
    # clients. The annotations are what stop a map() of the wrong shape
    # from satisfying this and failing at runtime.
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
    "displacement",
    "velocity",
    "acceleration",
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

# Large and small np.datetime64[ns] (used when defaults are needed)
SMALLDT64 = np.datetime64(MININT64 + 5_000_000_000, "ns")
LARGEDT64 = np.datetime64(MAXINT64 - 5_000_000_000, "ns")

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
PROGRESS_LEVELS = Literal["standard", "basic", None, False]

progress_description = """
progress
    Controls the progress bar. "standard" produces the standard progress
    bar. "basic" is a simplified version with lower refresh rates, best
    for high-latency environments, and None (or False) disables the
    progress bar.
""".strip()

# Options for handling specific warnings. One spelling of "do nothing":
# "ignore", which every policy argument in the library also spells.
WARN_LEVELS = Literal["warn", "raise", "ignore"]

# What `enrich` does about a name the inventory leaves undefined: the warn
# levels, spelled by reference so the two sets cannot drift apart, plus the
# fourth answer only this question has -- fill the missing marker.
ON_MISSING = Literal[WARN_LEVELS, "null"]

# The actions warnings.simplefilter and warnings.filterwarnings accept.
# Spelled out because the standard library's alias for them is stub-only.
# "all" is deliberately absent: Python only began accepting it in 3.14, and
# it raises on the 3.12-3.13 interpreters this project also supports.
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
Indicates how to handle attributes which hold conflicting values across
the patches being combined (eg data_type, data_units, custom attrs). A
missing value (None, NaN, "") is a value like any other: it equals
another missing one and nothing else, so a patch which never stated an
attribute conflicts with one which did. History and the ids are never
compared. If "raise" (default) raise an
[AttributeMergeError](`dascore.exceptions.AttributeMergeError`) for
conflicting values. If "drop", omit the conflicting attributes from the
output. If "keep_first", keep the first patch's value of each.
"""


select_values_description = """
Any dimension name can be passed as key, and the values can be:
    - a tuple of (min, max) for that dimension, or an equivalent slice.
      `None` and ... both indicate open intervals, as does an infinite
      bound pointing away from the data, eg `(min, np.inf)`.
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
    spool. 'ignore' will silently skip any incompatible patches,
    'warn' will issue a warning and then skip incompatible patches,
    'raise' will raise an
    [`IncompatiblePatchError`](`dascore.exceptions.IncompatiblePatchError`)
    if any incompatible patches are found.
"""

# Enrichment parameters, shared by Patch.enrich and Spool.enrich so the
# two cannot describe the same arguments differently.

enrich_attrs_description = """
attrs
    True (the default) to copy the observing-system facts the inventory
    is authoritative for, a tuple of names to copy exactly those, or
    False to copy none. The blanket form excludes `data_type`,
    `data_category`, and `data_units`, which describe the data as it
    now stands, and `sample_rate` and `spatial_interval`, which the
    patch's own coordinates already state; naming one restores the
    as-acquired value.
""".strip()

enrich_coords_description = """
coords
    True (the default) to add the geometry axes and label groups of
    the resolved optical path, a tuple of names to add exactly those, or
    False to add none. Names may be `distance` for optical distance, one
    of the axes the inventory's CRS names, a label group, or a qualified
    track field such as `coupling.medium`.
""".strip()

enrich_on_missing_description = """
on_missing
    What to do when an explicitly requested name is one the inventory does
    not define: "raise" (the default), "warn" to say so and leave it off,
    "ignore" to leave it off silently, or "null" to fill the
    dtype-appropriate missing marker so the name is present either way.
    Blanket requests copy what is applicable and never trigger it, and
    per-channel coverage gaps are always missing values rather than errors.
""".strip()

# One paragraph, not two: a blank line inside a parameter description
# reads as the start of the next parameter when the API docs are built.
enrich_conflict_description = f"""
conflict
{textwrap.indent(attr_conflict_description.strip(), "    ")}
    Enrichment combines the inventory's values with the patch's own, so
    the default `keep_first` lets the inventory win and re-enriching is
    a refresh. `raise` is the misresolution guard: a header disagreeing
    with the resolved acquisition usually means the `acquisition_key`
    resolved to the wrong place.
""".strip()


# Rich styles for various object displays. Every value has to be one rich can
# parse: it resolves an unparsable style to a blank one rather than raising,
# so a misspelling here does not fail, it silently stops coloring.
dascore_styles = dict(
    dc_blue="blue",
    dc_red="red",
    dc_yellow="yellow",
    default_coord="bold",
    coord_range="bold green",
    coord_monotonic="bold grey50",
    coord_segmented="bold cyan",
    coord_array="bold dark_orange",
    coord_non="bold red",
    units="bright_blue",
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
