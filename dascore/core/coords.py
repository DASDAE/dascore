"""Machinery for coordinates.

See ['Coordinate Internals'](`dascore/docs/notes/coordinate_internals.qmd`) for the
current coord-family and string-coordinate design notes.

Every numeric coordinate is a [`NumericND`](`dascore.core.coords.NumericND`):
a table of runs which holds ranges, exact grids, segmented coordinates,
monotonic arrays, unsorted arrays and N-D arrays alike. The four classes it
was once split across are removed; build a coordinate with
[`get_coord`](`dascore.core.coords.get_coord`), or one of ``NumericND``'s
``from_*`` builders, and ask it what it holds: ``evenly_sampled`` for what a
range was, ``runs_count`` for how many runs it holds, and ``holes`` for
whether any of them starts past where the one before ended.
"""

from __future__ import annotations

import abc
import copy
import datetime
import itertools
import math
import re
from collections.abc import Mapping, Sequence, Sized
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from types import EllipsisType, MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NoReturn, Self, cast, overload

import numpy as np
import pandas as pd
from pydantic import (
    Field,
    field_validator,
    model_validator,
)
from rich.text import Text

import dascore as dc
from dascore.compat import array, is_array
from dascore.constants import _AGG_FUNCS, DIM_REDUCE_DOCS, dascore_styles
from dascore.core._run_kernels import (
    _FLOAT_INDEX_MAX,
    _INT64_MAX,
    SOURCE_ID,
    FloatKernel,
    TickKernel,
    _record_dtype,
    _rows,
    _same_labels,
    _tick_bounds,
    _ticked,
    float_rows,
    float_terms,
    get_kernel,
)
from dascore.exceptions import CoordError, ParameterError
from dascore.models import (
    ArrayLike,
    DascoreBaseModel,
    UnitQuantity,
)
from dascore.units import (
    Quantity,
    Unit,
    convert_units,
    get_factor_and_unit,
    get_quantity,
    get_quantity_str,
    percent,
    units_match,
)
from dascore.utils.array import (
    _coerce_text_array,
    _is_text_coercible_array,
    hash_array,
)
from dascore.utils.display import (
    RichRepr,
    get_nice_text,
    range_texts,
    rate_text,
    span_text,
)
from dascore.utils.docs import compose_docstring, get_docstring
from dascore.utils.gaps import DEFAULT_TOLERANCE, GapTolerance
from dascore.utils.identity import H, new_id
from dascore.utils.misc import (
    _get_nullish,
    _maybe_array_to_slice,
    _maybe_unpack,
    _to_slice,
    _validate_sample_values,
    all_close,
    all_diffs_close_enough,
    cached_method,
    get_middle_value,
    glob_to_regex,
    is_strictly_monotonic,
    iterate,
    sanitize_range_param,
)
from dascore.utils.time import (
    dtype_time_like,
    is_datetime64,
    is_timedelta64,
    to_float,
    to_int,
)

# Values for min/max/step. The CoordSummary validator coerces these to match
# the summary dtype (datetime64, float, etc.) so they are open-ended here.
min_max_type = Any
step_type = Any

CoordKind = Literal["string", "empty", "single", "array", "range"]

# Cheap zero for comparing against timedelta steps; building it through
# dc.to_timedelta64(0) in hot paths costs ~10x more than reusing this.
_TD64_ZERO = np.timedelta64(0, "ns")


@cache
def _second_quantity():
    """Cache the 'second' quantity time-like coords are normalized to."""
    return get_quantity("s")


def ensure_consistent_dtype(value, name, dtype):
    """Ensure the values are consistent with dtype."""
    # For some reason all ints are getting converted to floats using default
    # pydantic type validation. This just fixes this manually.
    # TODO: See if this is needed in a few version after pydantic 2.1.1
    if pd.isnull(value):
        return value
    elif np.issubdtype(dtype, np.datetime64):
        if name == "step":
            value = dc.to_timedelta64(value)
        else:
            value = dc.to_datetime64(value)
    elif np.issubdtype(dtype, np.timedelta64):
        value = dc.to_timedelta64(value)
    # convert numpy numerics back to python
    elif np.issubdtype(dtype, np.floating):
        value = float(value) if value is not None else np.nan
    elif np.issubdtype(dtype, np.integer):
        value = int(value)
    return value


def _is_translation_equivariant(func, data):
    """Return True if shifting inputs shifts the reduced output equally."""
    # This is a bit heavy/magic, but needed for generic support.
    valid = data[~pd.isnull(data)]
    if not len(valid):
        return True
    valid = valid[:32]
    shift = 1.0
    with np.errstate(all="ignore"):
        try:
            base = np.asarray(func(valid))
            shifted = np.asarray(func(valid + shift))
        except Exception:
            return True
    expected = base + shift
    try:
        return bool(np.allclose(shifted, expected, equal_nan=True))
    except TypeError:
        return True


def _reduce_time_like(func, data):
    """Reduce datetime/timedelta data relative to a reference value."""
    data = np.asarray(data)
    valid = data[~pd.isnull(data)]
    if not valid.size:
        return np.atleast_1d(_get_nullish(data.dtype))

    # Some reducers cannot operate directly on time-like dtypes. If direct
    # reduction fails, or returns only nulls despite valid input, fall back to
    # reducing offsets from a valid reference value.
    if func not in {np.mean, np.nanmean}:
        with suppress(TypeError, ValueError, OverflowError):
            out = np.atleast_1d(func(data))
            # Return direct reductions that produce at least one non-null value.
            if not np.all(pd.isnull(out)):
                return out

    ref = valid[0]
    # Reducers like std over absolute times are not semantically time points,
    # but this preserves the previous dim_reduce behavior for equivariant reducers.
    delta_float = dc.to_float(data - ref)
    reduced = dc.to_timedelta64(func(delta_float))
    out = ref + reduced if _is_translation_equivariant(func, delta_float) else reduced
    return np.atleast_1d(out)


def _validate_new_length(length) -> int:
    """Ensure a requested coordinate length is a non-negative integer."""
    # bool is an int subclass; True/False are never a sensible length.
    if isinstance(length, bool) or not isinstance(length, int | np.integer):
        msg = f"change_length requires an integer length, not {length!r}."
        raise ParameterError(msg)
    if length < 0:
        msg = f"change_length requires a non-negative length, not {length}."
        raise ParameterError(msg)
    return int(length)


def _get_dtype(value, dtype):
    """Get the data type based on the first argument."""
    if dtype is not None and dtype != "":
        return str(dtype)
    value = type(value)
    return str(np.dtype(value))


def _conformed(value, dtype: np.dtype):
    """
    The value as the given dtype, or unchanged where that would lose it.

    Conforming is only ever a change of spelling. A coordinate whose
    metadata does not fit the dtype it declares — a partial one stating
    an integer dtype and a fractional start — would otherwise have the
    difference truncated away, and two coordinates which are not equal
    would share an identity.
    """
    if pd.isnull(value):
        return value
    with suppress(TypeError, ValueError, OverflowError):
        original = np.asarray(value)
        converted = original.astype(dtype)
        if converted.astype(original.dtype) == original:
            return converted
    return value


def _scalar_dtype(dtype: np.dtype, name: str) -> np.dtype:
    """
    The dtype a coordinate's own scalar is conformed to before hashing.

    The coordinate's dtype, at its own precision — a coordinate keeping
    picoseconds must not have them rounded away, or two coordinates a
    picosecond apart would share an identity. A step is the duration
    between values, so it takes the matching time unit rather than the
    time kind itself.
    """
    if dtype.kind not in "mM":
        return dtype
    unit = np.datetime_data(dtype)[0]
    return np.dtype(f"timedelta64[{unit}]") if name == "step" else dtype


def _legacy_grid_runs(data: Mapping, dtype) -> np.ndarray | None:
    """
    The one run a document's exact-grid fields state, or None.

    Before the run table the exact grid of a range was three fields beside
    the envelope (see `_EXACT_GRID_FIELDS`). A summary written that way --
    by an out-of-tree reader, or an older DASCore -- still states a run, so
    it is read as one rather than dropped for a whole-tick approximation.
    """
    terms = [data.get(x) for x in _EXACT_GRID_FIELDS]
    if all(x is None for x in terms):
        return None
    num = int(terms[0]) if terms[0] is not None else 1
    den = int(terms[1]) if terms[1] is not None else 1
    offset = int(terms[2] or 0)
    if den < 1 or not num:
        msg = f"An exact grid needs a step and a positive denominator, got {terms}."
        raise CoordError(msg)
    low, high = data.get("min"), data.get("max")
    if _is_null(low) or _is_null(high):
        msg = "An exact grid states a run only beside both of its bounds."
        raise CoordError(msg)
    dtype = normalize_coord_dtype(dtype)
    ticked = _ticked(dtype)
    start = high if num < 0 else low
    length = data.get("len")
    if length is None:
        ends = np.asarray([low, high])
        if ticked:
            # A tick label is the floor of its ideal position, so the last
            # sample inside the envelope is the largest k with
            # floor(k * num / den) <= span.
            span = abs(_to_tick(high) - _to_tick(low))
            length = int(((span + 1) * den - 1) // abs(num)) + 1
        else:
            span = abs(float(ends[1]) - float(ends[0]))
            length = int(span * den // abs(num)) + 1
    if not ticked:
        # the fraction an older float range stated is the step it divides to
        return float_rows(dtype, [float(np.asarray(start))], [int(length)], num / den)
    return _rows(dtype, [_to_tick(start)], [int(length)], [num], [den], [offset])


class CoordSummary(DascoreBaseModel):
    """
    A summary for coordinates.

    Provides enough information for indexing coordinates and creating range
    coordinates.
    """

    # Defaulted because the before-validator below derives it from min
    # whenever it is absent, so requiring it misdescribes the constructor.
    dtype: str = ""
    min: min_max_type
    max: min_max_type
    step: step_type | None = None
    units: UnitQuantity | None = None
    dims: tuple[str, ...] = ()
    len: int | None = None
    data_id: str | None = None
    # The exact grid of one evenly sampled run, in ticks; None otherwise.
    step_numerator: int | None = None
    step_denominator: int | None = None
    origin_offset: int | None = None
    # Each run's summary, in order, for a coordinate of several runs, so an
    # index can see holes inside a patch; None otherwise, including past
    # _MAX_SUMMARY_RUNS runs. Left out of the repr, which it would swamp.
    runs: tuple[CoordSummary, ...] | None = Field(default=None, repr=False)

    @property
    def is_exact_grid(self) -> bool:
        """Return True when the summary states an exact integer grid."""
        return self.step_numerator is not None

    @property
    def is_range_like(self) -> bool:
        """Return True when the summary can reconstruct an evenly sampled coord."""
        return not pd.isnull(self.step)

    @model_validator(mode="before")
    @classmethod
    def get_correct_dtype_cast_values(cls, data: Any) -> Any:
        """Ensure the correct dtype is provided and value conform to it."""
        # Any mapping, not just a dict: dtype has a default now, so a mapping
        # this skipped would quietly produce an empty one rather than being
        # derived from min. Copied because the input need not be mutable.
        if isinstance(data, Mapping):
            data = dict(data)
            min_val = data["min"]
            dtype = _get_dtype(min_val, data.get("dtype"))
            data["dtype"] = str(dtype).split("[")[0]
            for name in ["min", "max", "step"]:
                val = data.get(name)
                data[name] = ensure_consistent_dtype(val, name, dtype)
        return data

    @model_validator(mode="after")
    def _derive_dtype_if_unset(self) -> Self:
        """Fill in a dtype the before-validator never saw.

        That validator only fires for mapping input, so attribute-based
        validation (``from_attributes=True``) would otherwise keep the
        empty default, which the indexer treats as an unsupported coord.
        """
        if not self.dtype:
            dtype = _get_dtype(self.min, None)
            # Conform the values too, so this path agrees with the mapping one
            # instead of deriving a dtype the values then contradict. Confined
            # to the unset case on purpose: conforming on every validation
            # measured ~50% of construction cost, and a summary is built per
            # coordinate while indexing. An attribute input that *does* carry a
            # dtype is therefore still left alone, as it always has been.
            for name in ("min", "max", "step"):
                value = ensure_consistent_dtype(getattr(self, name), name, dtype)
                object.__setattr__(self, name, value)
            object.__setattr__(self, "dtype", str(dtype).split("[")[0])
        return self

    def to_coord(self) -> BaseCoord:
        """
        Rebuild the coordinate the summary states.

        A summary carrying a run per run rebuilds them all, holes and all,
        so long as each states a grid: labels no grid states stay in the
        file the summary came from. Otherwise only the envelope is known,
        and only an evenly sampled one rebuilds a range.
        """
        if (runs := self.runs) is not None and len(runs) > 1:
            return concat_coords(*(x.to_coord() for x in runs), units=self.units)
        fields = {
            "min": self.min,
            "max": self.max,
            "len": self.len,
            "step_numerator": self.step_numerator,
            "step_denominator": self.step_denominator,
            "origin_offset": self.origin_offset,
        }
        if self.dtype:
            dtype = normalize_coord_dtype(self.dtype)
            if (rows := _legacy_grid_runs(fields, dtype)) is not None:
                return NumericND.from_rows(rows, dtype=dtype, units=self.units)
        if not self.is_range_like:
            msg = "Cannot convert summary which is not evenly sampled to coord."
            raise CoordError(msg)
        step = self.step
        assert step is not None  # is_range_like above rules out a null step
        # this is a reverse coord
        if np.sign(step) == -1:
            start, stop = self.max, self.min + step
        else:
            start, stop = self.min, self.max + step
        return get_coord(start=start, stop=stop, step=step, units=self.units)


@cache
def _get_coord_filter_validators(dtype):
    """Get filter validators for a given input type."""

    def _is_sub_dtype(dtype1, dtype2):
        """Helper function to get sub dtypes."""
        # uncomment these if validators that aren't numpy types are needed.
        # with suppress(TypeError):
        #     if issubclass(dtype1, dtype2):
        if np.issubdtype(dtype1, dtype2):
            return True
        return False

    # A list of dtype, func for validating/coercing single filter inputs.
    validators = (
        (pd.Timestamp, dc.to_datetime64),
        (np.datetime64, dc.to_datetime64),
        (pd.Timedelta, dc.to_timedelta64),
        (np.timedelta64, dc.to_timedelta64),
    )

    out = []
    for cls, func in validators:
        if _is_sub_dtype(dtype, cls):
            out.append(func)
    return tuple(out)


def get_compatible_values(val, dtype):
    """
    Get values compatible with dtype.

    This will essentially perform any type conversions needed to go from
    one dtype to another. It is useful for handling datetime conversions.

    Parameters
    ----------
    val
        The values to convert.
    dtype
        A numpy compatible datatype or string.
    """
    validators = _get_coord_filter_validators(dtype)
    for func in validators:
        if val is not None:
            val = func(val)
    return val


@dataclass(frozen=True)
class Missing:
    """
    The grid positions a coordinate has no sample at.

    Built by [`missing`](`dascore.core.coords.BaseCoord.missing`) from run
    arithmetic, so it costs one entry per hole rather than one per
    position; `positions` spells the labels out, up to a limit.

    Examples
    --------
    >>> from dascore.core.coords import get_coord
    >>> coord = get_coord(data=[1, 3, 4, 10, 11, 12], step=1)
    >>> missing = coord.missing()
    >>> missing.count, missing.complete
    (6, False)
    >>> list(missing.iter_runs())
    [(np.int64(2), np.int64(2)), (np.int64(5), np.int64(9))]
    >>> missing.positions().tolist()
    [2, 5, 6, 7, 8, 9]
    """

    step: Any
    # each hole as (first missing label, last missing label, how many)
    runs: tuple[tuple[Any, Any, int], ...]
    dtype: Any = None

    @property
    def count(self) -> int:
        """How many grid positions have no sample."""
        return sum(n for _, _, n in self.runs)

    @property
    def complete(self) -> bool:
        """Whether every grid position between the first and last has a sample."""
        return not self.runs

    def iter_runs(self):
        """Yield each hole as its first and last missing label."""
        for first, last, _ in self.runs:
            yield first, last

    def positions(self, limit: int = 10_000) -> np.ndarray:
        """
        Every missing label, or raise when there are more than ``limit``.

        A long outage on a fast grid is millions of positions; the runs say
        as much without allocating them.
        """
        if self.count > limit:
            msg = (
                f"{self.count} missing positions exceed the limit of {limit}; "
                "raise the limit or use iter_runs."
            )
            raise ParameterError(msg)
        if not self.runs:
            return np.array([], dtype=self.dtype)
        return np.concatenate(
            [first + np.arange(n) * self.step for first, _, n in self.runs]
        )

    def __str__(self):
        holes = " | ".join(
            f"[{first}]" if n == 1 else f"[{first} … {last}]"
            for first, last, n in self.runs
        )
        return f"Missing(step={self.step})  missing {self.count}\n  {holes}"


def _hole(before, step, count: int) -> tuple:
    """A hole of ``count`` positions after ``before``, both ends from one anchor."""
    first = before + step
    return (first, first + (count - 1) * step, count)


def _discontinuity_frame(rows, kind: str, tolerance) -> pd.DataFrame:
    """
    The discontinuities frame from ``(index, before, after, expected)`` rows.

    ``expected`` is the spacing a run expects after itself (None when it
    states none); ``excess`` is the spacing beyond it. ``kind="gaps"``
    keeps the rows the tolerance calls gaps.
    """
    columns = ["index", "before", "after", "delta", "excess"]
    df = pd.DataFrame(rows, columns=["index", "before", "after", "expected"])
    # signed even for unsigned labels, which would wrap under subtraction
    df["delta"] = [_diffs([b, a])[0] for b, a in zip(df["before"], df["after"])]
    df["excess"] = [
        np.nan if pd.isnull(expected) else abs(delta) - abs(expected)
        for delta, expected in zip(df["delta"], df["expected"])
    ]
    if kind == "gaps":
        stated = df["expected"].notna().to_numpy()
        keep = np.zeros(len(df), dtype=bool)
        if stated.any():
            delta = df["delta"].to_numpy()[stated]
            step = df["expected"].to_numpy()[stated]
            keep[stated] = tolerance.is_gap(delta, step)
        df = df[keep]
    return df[columns].reset_index(drop=True)


class BaseCoord(RichRepr, DascoreBaseModel, abc.ABC):
    """
    Coordinate interface.

    Coordinates are used to manage labels and indexing along a single
    data dimension.

    Coordinates should usually be created with
    [get_coords](`dascore.core.coords.get_coord`) rather than using the class
    directly.
    """

    units: UnitQuantity = None
    step: Any = None
    # Every coord has a shape; each subclass derives it in a before-validator
    # from the values or range it was built with. The default exists only
    # because those validators are invisible to type checkers, which would
    # otherwise want shape passed at every construction site.
    shape: tuple[int, ...] = ()
    dtype: Any = None

    if TYPE_CHECKING:
        # Every coord exposes its values, but the array-backed coords store
        # them in a pydantic field while the rest compute them in a property.
        # Pydantic refuses to let a field shadow an inherited property (and a
        # field here would make values a required init argument), so the
        # shared interface is only declared for type checkers.
        @property
        def values(self) -> ArrayLike:
            """The coordinate's values."""

    # What a class of coordinate is, rather than what an instance holds, so
    # they are class variables: a private attribute is a per-instance slot
    # pydantic fills on every construction, and a coordinate is built on
    # every patch operation.
    _style_name: ClassVar[str] = "default_coord"
    _evenly_sampled: ClassVar[bool] = False
    _sorted: ClassVar[bool] = False
    _reverse_sorted: ClassVar[bool] = False
    _partial: ClassVar[bool] = False

    @model_validator(mode="before")
    @classmethod
    def _check_time_units(cls, data: Any) -> Any:
        """Ensure time units are s if dtype is time-like."""
        if isinstance(data, dict):
            # This handles the coord range case.
            is_timey = False
            if start := data.get("start"):
                is_timey = is_timedelta64(start) or is_datetime64(start)
            elif (values := data.get("values")) is not None:
                is_timey = dtype_time_like(values)
            if is_timey and data.get("units") != (quant := get_quantity("s")):
                data["units"] = quant
        return data

    @field_validator("shape", mode="before")
    @classmethod
    def _validate_shape_to_tuple(cls, value):
        """Ensure shape is a tuple."""
        # This also allows shape to be an int.
        return tuple(iterate(value))

    def convert_units(self, units) -> Self:
        """
        Convert from one unit to another. Set units if None are set.

        A coordinate already carrying exactly these units -- magnitude
        included, so `100 cm` is not `m` -- returns itself, letting a
        caller detect a conversion with nothing to do by identity.
        """
        if units_match(self.units, units):
            return self
        return self._convert_units(units)

    def _convert_units(self, units) -> Self:
        """
        Perform the conversion; callers normally screen out no-op requests.

        Concrete rather than abstract so that a subclass written against
        the older API, where `convert_units` was the abstract method,
        still instantiates. DASCore's own classes are held to it by
        `TestUnitNoOps.test_every_coord_class_implements_the_hook`.
        """
        msg = f"{type(self).__name__} does not implement unit conversion."
        raise NotImplementedError(msg)

    def _get_value_index(self, coord_array, values_to_find):
        """Get the indices were values occur in array, account for duplicates."""
        # We check insertion order from both sides to catch duplicate values.
        inds_left = np.searchsorted(coord_array, values_to_find, side="left")
        inds_right = np.searchsorted(coord_array, values_to_find, side="right")
        if np.all(inds_right == (inds_left + 1)):  # Quick path for no duplicates.
            return inds_left
        # Each left right pair now needs to form a range so we include all
        # elements in between.
        ar = np.stack([inds_left, inds_right], axis=-1)
        inds = np.concatenate([np.arange(x[0], x[1]) for x in ar])
        return inds

    def _order_by_value_array(self, values_to_find):
        """Select values based on an array of values."""
        coord_array = self.values
        # First simply filter arg values to only include those in the index
        values_to_find = values_to_find[np.isin(values_to_find, coord_array)]
        # Handle fast cases for sorted and reverse sorted coords.
        if self.sorted:
            inds = self._get_value_index(coord_array, values_to_find)
            return self[inds], inds
        if self.reverse_sorted:
            # Need to_float here because datetime can't be multiplied by -1.
            inds = self._get_value_index(
                -to_float(coord_array), -to_float(values_to_find)
            )
            return self[inds], inds
        # Sort the array, then find insertion points, and map
        # back to pre-sorted indices.
        argsort = np.argsort(coord_array)
        sorted_coord_array = coord_array[argsort]
        sorted_inds = self._get_value_index(sorted_coord_array, values_to_find)
        inds = argsort[sorted_inds]
        return self[inds], inds

    def _order_by_sample_array(self, array):
        """Select based on index values."""
        if not np.issubdtype(array.dtype, np.integer):
            msg = "Using an array input for select with samples requires integer dtype."
            raise CoordError(msg)
        # Filter out bad indices
        array = array[np.abs(array) < len(self)]
        return self[array], array

    def _select_by_value_array(self, array):
        """Select values based on an array of values."""
        values = self.values
        # First simply filter arg values to only include those in the index
        valid_values = np.isin(values, array)
        return self[valid_values], valid_values

    def _select_by_sample_array(self, array):
        """Select based on index values."""
        if not np.issubdtype(array.dtype, np.integer):
            msg = "Using an array input for select with samples requires integer dtype."
            raise CoordError(msg)
        # Filter out bad indices
        if self.ndim != 1:
            msg = "Select only works on 1D coords."
            raise CoordError(msg)
        inds = np.arange(len(self))
        valid_values = np.isin(inds, array)
        return self[valid_values], valid_values

    def _select_by_array(self, arg, samples=False, relative=False):
        """Select based on arg being an array."""
        if samples:
            return self._select_by_sample_array(arg)
        if np.issubdtype(getattr(arg, "dtype", None), np.bool_):
            return self[arg], arg
        arg = self._get_compatible_value(arg, relative=relative)
        return self._select_by_value_array(arg)

    def _select_by_samples(self, arg):
        """Select using samples."""
        _validate_sample_values(arg)
        reductions = _to_slice(arg)
        new = self[reductions]
        return new, reductions

    @abc.abstractmethod
    def select(
        self, args, relative=False, samples=False
    ) -> tuple[BaseCoord, slice | ArrayLike]:
        """
        Returns an entity that can be used in a list for numpy indexing
        and selected coord.
        """

    def order(
        self, array, relative=False, samples=False
    ) -> tuple[BaseCoord, slice | ArrayLike]:
        """
        Order coordinate according to array values or samples.

        Parameters
        ----------
        array
            A numpy array of values in coordinate or (if samples)
            indices.
        relative
            If True, the values are relative to the start or end of coordinate.
        samples
            If True, the array is of dtype in and refers to samples in the
            coordinate.
        """
        array = np.atleast_1d(array)
        if samples:
            coord, inds = self._order_by_sample_array(array)
        else:
            array_compat = self._get_compatible_value(array, relative=relative)
            coord, inds = self._order_by_value_array(array_compat)
        return coord, _maybe_array_to_slice(inds, len(self))

    def align_to(
        self, other: BaseCoord
    ) -> tuple[BaseCoord, BaseCoord, slice | ArrayLike, slice | ArrayLike]:
        """
        Align the coordinate to another coordinate.

        This returns two new coordinates which share values as well indexer's
        needed to align corresponding arrays.

        Parameters
        ----------
        other
            The other coordinate.
        """

        def valid_non_coord(coord1, coord2):
            lens = {len(x) for x in [coord1, coord2]}
            # For compatibility one coord must have length 1 or
            # coords must be same length.
            if not (1 in lens or len(lens) == 1):
                msg = (
                    "Non coordinates must be the same length as coordinate "
                    "or length 1 for broadcasting to work."
                )
                raise CoordError(msg)

        if self == other:
            return self, other, slice(None), slice(None)
        if self.ndim != 1:
            msg = "can only align 1D coords."
            raise CoordError(msg)
        if isinstance(self, CoordPartial) or isinstance(other, CoordPartial):
            valid_non_coord(self, other)
            return self, other, slice(None), slice(None)
        data1, data2 = self.values, other.values
        intersection = np.intersect1d(data1, data2)
        coord1, slice1 = self.order(intersection)
        coord2, slice2 = other.order(intersection)
        return coord1, coord2, slice1, slice2

    @overload
    def __getitem__(self, item: int | np.integer) -> Any: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray) -> Self: ...

    @overload
    def __getitem__(self, item: tuple) -> Any: ...

    @abc.abstractmethod
    # Left unannotated on purpose. An int index yields a bare value, so the
    # honest return contains Any, which would absorb the overloads above
    # rather than let the checker verify them.
    def __getitem__(self, item):
        """Index the coord; slices return a new coord, int indices a value."""

    @cached_method
    def __len__(self):
        """Total number of elements."""
        return self.shape[0]

    def _repr_fields(self) -> tuple[tuple[str, Text, bool], ...]:
        """
        The facts this coordinate states, in the order it states them.

        One source for the line a terminal prints and the row a panel
        draws, so the two cannot come to hold different facts. A field
        the coordinate has nothing to say for is left out rather than
        stated blank.

        This is the hook a subclass states extra facts through. A
        manager builds its rows from these rather than from each
        coordinate's rendered line, so facts added by overriding
        ``__rich__`` alone would show on the coordinate and nowhere it
        is held.
        """
        fields: list[tuple[str, Text, bool]] = []
        # What the values are measured in, said on each of them rather
        # than in a field of its own: how far the fiber runs and what
        # that is measured in are one fact, and reading the second of
        # them off the end of the line is not how it is read. A time
        # states its units in the way it is written -- an instant, a
        # step of "0.0005s" -- and says nothing here.
        stated = None if dtype_time_like(self.dtype) else self._unit_str

        def measured(value: Text) -> Text:
            """The value, and what it is measured in where it says."""
            if not stated:
                return value
            return value + Text(f" {stated}", dascore_styles["units"])

        # Drawn as one range: the two ends state the same blocks, and
        # the second states only what the first did not already.
        near, far = range_texts(self.min(), self.max())
        if not pd.isnull(self.min()):
            fields.append(("min", measured(near), True))
        if not pd.isnull(self.max()):
            fields.append(("max", measured(far), True))
        # How far the two ends lie apart, which they do not otherwise
        # say: a fiber run from 1212.4 m to 1636.7 m is 424.3 m of it.
        if not (pd.isnull(self.min()) or pd.isnull(self.max())):
            if (span := span_text(self.min(), self.max(), stated)) is not None:
                # The brackets say what it is, so a line does not
                # need the label a column heading gives it.
                fields.append(("span", span, False))
        if not pd.isnull(self.step):
            step = measured(get_nice_text(self.step))
            if (rate := rate_text(self.step)) is not None:
                step = step + rate
            fields.append(("step", step, True))
        # A coordinate which states no value has nothing to hang its
        # units on, and they are a fact of it either way.
        if stated and not fields:
            fields.append(("units", get_nice_text(stated, style="units"), True))
        fields.append(("shape", get_nice_text(self.shape), True))
        fields.append(("dtype", get_nice_text(self.dtype), True))
        return tuple(fields)

    @property
    def _rich_style(self) -> str:
        """The colour a coordinate of this kind is drawn in."""
        return dascore_styles[self._style_name]

    def __rich__(self):
        key_style = dascore_styles["keys"]
        base = Text("")
        base += Text(self.__class__.__name__, style=self._rich_style)
        base += Text("(")
        for label, value, labelled in self._repr_fields():
            base += Text(f" {label}: ", key_style) if labelled else Text(" ")
            base += value
        base += Text(" )")
        return base

    def __array__(self, dtype=None, copy=False):
        """Numpy method for getting array data with `np.array(coord)`."""
        return self.values

    def __hash__(self):
        """Disable Python hash semantics in favor of explicit ids."""
        msg = "Coordinates are not hashable; use `data_id` for stable IDs."
        raise TypeError(msg)

    def _hash_scalar(self, value, name: str = "start") -> tuple[str, str | None]:
        """
        Return a dtype-aware scalar hash token.

        The value is first conformed to the coordinate's own dtype, since
        an id names *values*, not how they were spelled: a range whose
        start was given as `0` holds the same coordinate as one given
        `0.0`, and a step of four milliseconds is the step of four
        million nanoseconds. Without this they would be stored under
        different identities and never deduplicate.
        """
        if value is None:
            return ("none", None)
        dtype = np.dtype(self.dtype) if self.dtype else None
        if dtype is not None:
            value = _conformed(value, _scalar_dtype(dtype, name))
        return ("scalar", hash_array(np.asarray([value])))

    def _class_identity(self) -> str:
        """Return a stable identifier for this coordinate's class."""
        cls = self.__class__
        return f"{cls.__module__}.{cls.__qualname__}"

    @abc.abstractmethod
    def _id_components(self) -> tuple[Any, ...]:
        """Return subclass-specific id components."""

    @property
    @cached_method
    def data_id(self) -> str:
        """
        Return the id of the values this coordinate holds.

        Exact: the class, the units as written, and the values in the
        coordinate's own units and dtype. The same range in metres and in
        centimetres selects differently, so they are two ids.

        Notes
        -----
        Ids are stable identifiers, not tolerant comparison, so
        coordinates which are approximately equal can have different ids.
        """
        payload = (
            self._class_identity(),
            self._unit_str,
            *self._id_components(),
        )
        # Built from strings, numbers and None alone, so it is its own encoding.
        return H("coord", payload, encoded=True)

    def _identity(self) -> tuple[str, str]:
        """Return the id this coordinate has as a parameter or in a content id."""
        return "coord", self.data_id

    @cached_method
    def min(self):
        """Return min value."""
        return self._min()

    @cached_method
    def max(self):
        """Return max value."""
        return self._max()

    @property
    def has_values(self) -> bool:
        """
        Whether the coordinate holds labels, rather than only a shape.

        False for a [`CoordPartial`](`dascore.core.coords.CoordPartial`),
        which states how many samples a dimension has, and perhaps their
        spacing, without saying where any of them is.
        """
        return not self._partial

    @property
    def _unit_str(self) -> str | None:
        """Return a unit string, or None for a coord carrying no units."""
        return get_quantity_str(self.units)

    @abc.abstractmethod
    def _min(self):
        """Returns (or generates) the array data."""

    @abc.abstractmethod
    def _max(self):
        """Returns (or generates) the array data."""

    @property
    @cached_method
    def ndim(self) -> int:
        """Return the number of dimensions in patch."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Return the size of the coordinate data."""
        # math rather than np.prod: the shape is a tuple of ints, and numpy
        # hands back an np.int64 (or a float 1.0 for the empty shape).
        return math.prod(self.shape)

    @property
    def step_exact(self) -> Fraction | None:
        """
        The exact spacing as a fraction in coordinate units, or None.

        A timedelta step is exact in seconds and an integer step in its own
        units; a float step has no exact form and gives None.
        """
        step = self.step
        if _is_null(step):
            return None
        if is_timedelta64(step):
            return Fraction(int(to_int(step)), _NS_PER_S)
        return Fraction(int(step)) if _is_int(step) else None

    @property
    def evenly_sampled(self) -> bool:
        """Returns True if the coord is evenly sampled."""
        return self._evenly_sampled

    @property
    def sorted(self) -> bool:
        """Returns True if the coord in sorted."""
        return self._sorted

    @property
    def reverse_sorted(self) -> bool:
        """Returns True if the coord in sorted in reverse order."""
        return self._reverse_sorted

    def set_units(self, units) -> Self:
        """Set new units on coordinates."""
        if units_match(self.units, units):
            return self
        new = dict(self)
        new["units"] = units
        return self.__class__(**new)

    def simplify_units(self) -> Self:
        """Simplify the coordinate units."""
        _, unit = get_factor_and_unit(self.units, simplify=True)
        return self.convert_units(unit)

    def coord_range(self, extend: bool = True):
        """
        Return a scaler value for the coordinate range (e.g., number of seconds).

        Parameters
        ----------
        extend
            If true, count the end of the range as max() + sample step. This
            can only work for evenly sampled coordinates.
        """
        if extend:
            # Only an evenly sampled coordinate has an exclusive end, and
            # the class which can be one answers for itself.
            msg = (
                "If extend is True, the coord_range can only be called on "
                f"evenly sampled coordinates but {self} is not."
            )
            raise CoordError(msg)
        return self.max() - self.min()

    @abc.abstractmethod
    def sort(self, reverse=False) -> tuple[BaseCoord, slice | ArrayLike]:
        """Sort the contents of the coord. Return new coord and slice for sorting."""

    def snap(self) -> BaseCoord:
        """
        Snap the coordinates to evenly sampled grid points.

        This will cause some loss of precision but often makes the data much
        easier to work with.
        """
        return self

    def fuse(self, tolerance=None, keep_step: bool = False) -> BaseCoord:
        """
        Return the simplest coordinate representing the same values.

        Unlike [`snap`](`dascore.core.coords.BaseCoord.snap`), which fits the
        whole coordinate to one grid, fuse re-fits runs a seam at a time and
        keeps a seam it cannot close without moving a value further than
        `tolerance` (an absolute distance here, not a count of steps).

        Parameters
        ----------
        tolerance
            The maximum amount any coordinate value may change. For time-like
            coordinates this is a timedelta (numeric values interpreted as
            seconds). None or 0 permit only exact (lossless) simplifications.
        keep_step
            If True, only re-fit at the segments' own step, so missing
            samples stay missing however large the tolerance. Merging uses
            this: a hole is data that is absent, not a slower sampling rate.

        Notes
        -----
        Most coordinates are already in their simplest form and return
        themselves. A coordinate of several runs re-fits them as evenly
        sampled grids wherever the fit error stays within tolerance,
        possibly collapsing to a single run.
        """
        return self

    def get_discontinuities(self, kind="all", tolerance=None) -> pd.DataFrame:
        """
        Return a dataframe describing discontinuities in the coordinate.

        Parameters
        ----------
        kind
            Either "all" (every segment boundary) or "gaps" (boundaries whose
            spacing exceeds the local sampling interval by more than
            `tolerance`).
        tolerance
            For kind="gaps", the excess spacing (beyond the expected sampling
            interval) required to report a boundary. For time-like
            coordinates numeric values are interpreted as seconds. Default 0.

        Notes
        -----
        The returned dataframe has columns: `index` (position of the first
        sample after the boundary), `before`, `after` (values on either side),
        `delta` (after - before) and `excess` (delta minus the expected local
        sampling interval, NaN when no sampling interval is defined).

        A monotonic array reports every spacing which is not its declared
        step (or its median spacing when it declares none); a segmented
        coordinate reports the seams between its runs; every other
        coordinate (a range, an unordered array, a partial or string
        coordinate) returns an empty dataframe. ``tolerance`` may also be a
        `dascore.utils.gaps.GapTolerance`.
        """
        if kind not in ("all", "gaps"):
            msg = f"kind must be 'all' or 'gaps', got {kind!r}"
            raise ParameterError(msg)
        tolerance = self._gap_tolerance(tolerance)
        return _discontinuity_frame(self._seams(), kind, tolerance)

    def _seams(self) -> list[tuple]:
        """The ``(index, before, after, expected)`` rows of every discontinuity."""
        return []

    def _gap_tolerance(self, tolerance) -> GapTolerance:
        """
        The gap tolerance a public argument spells.

        A number is an absolute excess in coordinate units (seconds for
        time), as [`fuse`](`dascore.core.coords.BaseCoord.fuse`)
        reads it; a quantity or timedelta converts to those units; a
        `GapTolerance` counting samples is returned unchanged, and one
        stating an excess has that excess converted likewise.
        """
        if isinstance(tolerance, GapTolerance):
            if tolerance.count is not None:
                return tolerance
            tolerance = tolerance.excess
        if tolerance is None:
            tolerance = 0
        stated = None
        if is_timedelta64(tolerance) and not dtype_time_like(self.dtype):
            # A timedelta against a numeric coordinate is a length in
            # seconds, which only the coordinate's units can place.
            stated = (to_float(tolerance), "s")
        elif hasattr(tolerance, "units"):  # pint quantity tolerances
            stated = (tolerance.magnitude, tolerance.units)
        if stated is not None:
            magnitude, from_units = stated
            target = "s" if dtype_time_like(self.dtype) else self.units
            # A tolerance is a DELTA, so it converts between two anchor
            # points: 20 degC of deviation is 36 degF, never 68.
            anchor = convert_units(0.0, target, from_units)
            tolerance = convert_units(magnitude, target, from_units) - anchor
        if dtype_time_like(self.dtype):
            tolerance = dc.to_timedelta64(tolerance)
            zero = dc.to_timedelta64(0)
        else:
            zero = 0
        if tolerance < zero:
            msg = "tolerance must not be negative."
            raise ParameterError(msg)
        return GapTolerance.absolute(tolerance)

    def missing(self) -> Missing:
        """
        The grid positions between the first and last sample with no sample.

        Missing is relative to a declared step: labels ``[0, 2, 4]`` fill a
        step-2 grid and miss positions 1 and 3 of a step-1 grid. A
        coordinate without a step cannot say and raises (an unordered array
        never carries one); so does a segmented coordinate whose runs share
        a step but meet off its grid.

        Examples
        --------
        >>> from dascore.core.coords import get_coord
        >>> coord = get_coord(data=[1, 3, 4, 10, 11, 12], step=1)
        >>> coord.missing().count
        6
        >>> get_coord(start=0, stop=10, step=1).missing().complete
        True
        """
        if _is_null(self.step):
            msg = "missing needs a declared step; this coordinate has none."
            raise CoordError(msg)
        return Missing(step=self.step, runs=tuple(self._holes()), dtype=self.dtype)

    def _holes(self) -> list[tuple]:
        """Each hole as ``(first missing label, last missing label, count)``."""
        return []

    @abc.abstractmethod
    def update_limits(self, min=None, max=None, step=None, **kwargs) -> BaseCoord:
        """
        Update the limits or sampling of the coordinates.

        If start and stop are defined a new step is determined and returned.
        Next, the step size is updated changing only the end. Then the start
        is updated changing the start/end. Then the end is updated changing
        the start/end.

        Parameters
        ----------
        min
            The new start of the coordinate.
        max
            The new stop of the coordinate.
        step
            New step for the coordinate
        **kwargs
            Any other attributes which are used to create new coordinate.

        Notes
        -----
        For an evenly sampled coordinate stop will be max + step.
        """

    def _update_data(self, data=None, values=None, **kwargs) -> BaseCoord:
        """The coordinate new labels state, or this one when none are given."""
        if data is None and values is None:
            return self
        data = values if data is None else data
        return get_coord(data=data, units=kwargs.get("units"))

    def new(self, **kwargs):
        """Update coordinate."""
        info = self.model_dump(exclude_unset=True, exclude_defaults=True)
        if "data" in kwargs:
            kwargs["values"] = kwargs.pop("data")
        if "values" in kwargs:
            info.pop("shape", None)

        info.update(kwargs)
        return get_coord(**info)

    def _get_index_values(self, indices):
        """The labels at these (possibly negative) sample indices."""
        return np.asarray(self.values)[np.asarray(indices)]

    def _get_compatible_value(self, value, relative=False):
        """
        Return values that are compatible with dtype/units of coord.

        This is used, for example, to coerce values in select tuple
        so direct comparison with coord values is possible.
        """
        # strip units and v
        if hasattr(value, "units"):
            mag, unit = value.magnitude, value.units
            if unit == percent:
                value = (mag / 100.0) * self.coord_range(extend=False)
                relative = True
            else:
                value = convert_units(value.magnitude, self.units, value.units)
        # if null or ... just return None
        if not is_array(value) and (pd.isnull(value) or value is Ellipsis):
            return None
        # special case for datetime/timedelta and relative
        if relative:
            # A relative offset into any time-like coord is a duration.
            if dtype_time_like(self.dtype):
                value = dc.to_timedelta64(value)
            value = self._get_relative_values(value)
        # apply validators. These can, eg, coerce to correct dtype.
        out = get_compatible_values(value, self.dtype)
        return out

    def _slice_degenerate(self, sliz):
        """
        Return bool indicating if the slice should yield degenerate
        (empty array).
        """
        start, stop = sliz.start, sliz.stop
        # check if slice is between samples
        between = start is not None and start == stop
        # check if slice is outside of range
        bad_start = start is not None and (start < 0 or start >= len(self))
        bad_stop = stop is not None and (stop <= 0)
        return between or bad_start or bad_stop

    def _get_slice_tuple(
        self,
        select: slice | EllipsisType | tuple[Any, Any] | None,
        relative=False,
    ) -> tuple[Any, Any]:
        """
        Get a tuple with (start, stop) and perform basic checks.

        Parameters
        ----------
        select
            An object for determining select range.
        relative
            If True, the select values are relative to the minimum
            (positive values) or maximum (negative values) of the
            coordinate.
        """
        select_tuple = sanitize_range_param(select)
        p1, p2 = (
            self._get_compatible_value(x, relative=relative) for x in select_tuple
        )
        # reverse order if needed to ensure p1 < p2. This needs to be
        # after the compatible value conversion in case pre-converted
        # values are different types.
        if p1 is not None and p2 is not None and p2 < p1:
            p1, p2 = p2, p1
        return p1, p2

    def _get_relative_values(self, value):
        """Get relative values based on start (pos) or stop (neg)."""
        pos = np.sign(value).astype(np.int_) >= 0
        if is_array(value):
            out = np.where(pos, self.min() + value, self.max() + value)
        else:
            out = self.min() + value if pos else self.max() + value
        return out

    def empty(self, axes=None) -> BaseCoord:
        """
        Empty out the coordinate.

        Parameters
        ----------
        axes
            The axis to empty, if None empty all.
        """
        if axes is None:
            new_shape = np.zeros(len(self.shape), dtype=np.int_)
        else:
            assert np.max(axes) <= (len(self) - 1)
            new_shape = np.asarray(self.shape)
            for ind in iterate(axes):
                new_shape[ind] = 0
        data = np.empty(tuple(new_shape), dtype=self.dtype)
        return get_coord(data=data)

    def index(self, indexer, axis: int | None = None) -> BaseCoord:
        """
        Index the coordinate and return new coordinate.

        Parameters
        ----------
        indexer
            Anything that can be used in numpy indexing.
        axis
            The axis along which to apply the indexer. If None,
            just apply indexer to numpy array.
        """
        if axis:
            ndims = len(self.shape)
            assert ndims >= (axis + 1)
            indexer = tuple(
                slice(None, None) if i != axis else indexer for i in range(ndims)
            )
        out = self[indexer]
        if isinstance(out, BaseCoord):
            if out.ndim:
                return out
            out = out.values[()]
        return get_coord(data=out, units=self.units)

    def to_summary(self, dims=()) -> CoordSummary:
        """Get the summary info about the coord."""
        return CoordSummary(
            min=self.min(),
            max=self.max(),
            # a summary with a step rebuilds a range; a step declared on
            # arrays or shared by runs describes their grid, not a range
            step=self.step if self.evenly_sampled else None,
            dtype=self.dtype,
            units=self.units,
            dims=dims,
            len=self.shape[0] if self.ndim else 1,
            data_id=self.data_id,
        )

    def update(self, **kwargs):
        """Update parts of the coordinate."""
        out = self
        info = self.model_dump()
        update_fields = {
            i: v for i, v in kwargs.items() if not all_close(v, info.get(i))
        }
        units = update_fields.pop("units", None)
        _ = update_fields.pop("dtype", None)
        if update_fields:
            out = out.update_limits(**update_fields)._update_data(**update_fields)
        if units is not None:
            out = out.convert_units(units)
        return out

    def get_sample_count(self, value, samples=False, enforce_lt_coord=False) -> int:
        """
        Return the number of samples represented by a value.

        This is calculated by dividing the value by dt and rounding up.
        Therefore, the output will always be greater or equal to 1.

        Parameters
        ----------
        value
            The value (supports units).
        samples
            If True, value is already in units of samples.
        enforce_lt_coord
            If True, raise an error if the number of samples obtained exceeds
            the length of the coordinate.
        """
        if self.ndim != 1:
            msg = "get sample count only works for 1D coords."
            raise CoordError(msg)
        if not self.evenly_sampled:
            msg = "Coordinate is not evenly sampled, can't get sample count."
            raise CoordError(msg)
        if samples:
            if not isinstance(value, int | np.integer):
                msg = "When samples==True values must be integers."
                raise ParameterError(msg)
            samples = int(value)
        else:
            compat_val = self._get_compatible_value(value, relative=True)
            duration = compat_val - self.min()
            ratio = self._samples_in(duration)
            if np.issubdtype(self.dtype, np.floating):
                nearest = np.round(ratio)
                # Adding a relative float value to coord.min() and subtracting
                # it back can introduce a small cancellation error. Snap only
                # those float-coordinate ratios that fall within that drift.
                tol = 10 * abs(np.spacing(self.min()) / self.step)
                if abs(ratio - nearest) <= tol:
                    ratio = nearest
            samples = math.ceil(ratio)
        if enforce_lt_coord and samples > len(self):
            msg = (
                f"value of {value} with samples={samples} results in a window "
                f"larger than coordinate length of {len(self)}."
            )
            raise ParameterError(msg)
        return samples

    def _samples_in(self, duration):
        """How many steps a duration spans, unrounded."""
        # The magnitude of the step, because a window of 30 m is thirty
        # samples whether the coordinate counts up or down. A descending
        # coordinate has a negative step, and dividing by it signed would
        # make the count negative.
        return duration / np.abs(self.step)

    def _out_of_bounds_indices(self, array) -> np.ndarray:
        """The positions values past either end of an evenly sampled coord map to."""
        return ((array - self.min()) / self.step).astype(np.int64)

    def _get_index(self, value, forward=True):
        """
        Get the index a value would occupy in the coordinate.

        Overridden by the coords that index by value. Unordered arrays
        have no such position, and string coords deliberately keep out of
        positional semantics (see _raise_string_coord_error).
        """
        msg = f"{type(self).__name__} does not support indexing by value."
        raise CoordError(msg)

    def get_next_index(
        self, value, samples=False, allow_out_of_bounds=False, relative=False
    ) -> np.ndarray | np.integer:
        """
        Get the index a value would have in a coordinate.

        A sized value yields an array of indices; anything else yields a
        single index, which is a numpy integer rather than a builtin int.

        This returns the "next" rather than the closest, index if the exact
        value is not contained by the index.

        Parameters
        ----------
        value
            The value which could be contained by the coordinate.
        samples
            If True, value refers to samples (ie an index) of coord.
        allow_out_of_bounds
            If True, allow the value to be out of bounds of the coordinate
            and just return an index referring to the end
            (len(coords) - 1) or beginning (0).
        relative
            If True, the provided values are relative to the start (if positive)
            or end (if negative) of the coordinate.

        Examples
        --------
        >>> from dascore.core import get_coord
        >>> coord = get_coord(start=0, stop=10, step=1)
        >>> # Find the index for a value contained by the coordinate.
        >>> assert coord.get_next_index(1) == 1
        >>> # The next (not closest) index is return for value not in coord.
        >>> assert coord.get_next_index(2.000001) == 3
        """
        if not self.sorted:
            msg = f"Coords must be sorted to use get_next_index, {self} is not."
            raise CoordError(msg)
        input_array_like = isinstance(value, Sized)
        array = np.atleast_1d(value)

        # handle samples
        if samples:
            min_val, max_val = 0, len(self) - 1
            array = array.astype(np.int64)
            # account for negative indexing (wrap around) only when not allowing OOB
            if not allow_out_of_bounds:
                wrap_around = array < 0
                array[wrap_around] = array[wrap_around] + max_val + 1
        else:
            min_val, max_val = self.min(), self.max()
            array = self._get_compatible_value(array, relative=relative)
        # handle out of bounds cases
        is_gt, is_lt = array > max_val, array < min_val
        if not allow_out_of_bounds and np.any(is_gt | is_lt):
            msg = f"Value: {array} is out of bounds for {self}"
            raise ValueError(msg)

        # If allow_out_of_bounds and we have out of bounds values,
        # compute actual indices for evenly sampled coords
        if allow_out_of_bounds and np.any(is_gt | is_lt):
            # For samples mode, just return the raw indices (no clamping)
            if samples:
                return array if input_array_like else array[0]
            # For absolute mode with evenly sampled coords, compute index from value
            if hasattr(self, "step") and self.step is not None:
                indices = self._out_of_bounds_indices(array)
                return indices if input_array_like else indices[0]

        # Clamp values to bounds for backward compatibility when not out of bounds
        array[is_gt] = max_val
        array[is_lt] = min_val
        # samples should already have the answer, just return
        if samples:
            return array if input_array_like else array[0]
        # otherwise get forward and backward inds
        forward_index = self._get_index(array, forward=True)
        back_index = self._get_index(array, forward=False)
        bad_for_index = pd.isnull(forward_index) | forward_index == -9999
        forward_index[bad_for_index] = back_index[bad_for_index]
        return forward_index if input_array_like else forward_index[0]

    def approx_equal(self: BaseCoord, other: BaseCoord) -> bool:
        """
        Return True if the coordinates are approximately equal.

        This is a tolerant comparison helper. It is intentionally distinct
        from `data_id`, which is stricter and intended for stable ids.

        Parameters
        ----------
        other
            Another coordinate.
        """
        if self is other:
            return True
        if self.shape != other.shape:
            return False
        non_coords = [self._partial, other._partial]
        if all(non_coords):
            return self == other
        if any(non_coords):
            return False
        return all_close(self.values, other.values)

    def change_length(self, length: int) -> Self:
        """
        Adjust the length of the coordinate by changing the end value.

        This is useful for floating point coordinates who frequently suffer
        from off by one errors.

        Note: Not all coordinates implement this method.

        Parameters
        ----------
        length
            The output length. Must be a non-negative integer.

        Raises
        ------
        ParameterError
            If length is not a non-negative integer.
        """
        msg = f"Coordinate type {self.__class__} does not implement change_length"
        raise NotImplementedError(msg)

    @compose_docstring(dim_reduce=DIM_REDUCE_DOCS)
    def reduce_coord(self, dim_reduce="empty"):
        """
        Get a reduced coordinate.

        This is used to get a coordinate after aggregating along a dimension.

        Parameters
        ----------
        {dim_reduce}
        """
        if dim_reduce == "empty":
            # Preserve concrete single-sample coords; only synthesize a partial
            # coord when reduction actually collapses a longer coordinate.
            if len(self) == 1:
                return self
            new_coord = get_coord(shape=(1,), units=self.units, dtype=self.dtype)
        elif dim_reduce == "squeeze":
            return None
        else:
            func = dim_reduce if callable(dim_reduce) else _AGG_FUNCS.get(dim_reduce)
            if func is None:
                msg = "dim_reduce must be 'empty', 'squeeze' or valid aggregator."
                raise ParameterError(msg)
            coord_data = self.values
            if dtype_time_like(coord_data):
                result = _reduce_time_like(func, coord_data)
            else:
                result = func(self.values)
            new_coord = self.update(data=result)
        return new_coord


class CoordPartial(BaseCoord):
    """
    A coordinate which only contains partial information.
    """

    # Redeclared without a default: a partial coord is nothing but its
    # shape, and it is the one coord which cannot re-derive it on the way
    # back from a model_dump(exclude_defaults=True).
    shape: tuple[int, ...]
    start: Any = np.nan
    stop: Any = np.nan
    step: Any = np.nan
    _style_name: ClassVar[str] = "coord_non"
    _partial: ClassVar[bool] = True

    @field_validator("start", "stop", "step", mode="before")
    @classmethod
    def _validate_nullish_to_nan(cls, value, info):
        """Ensure nullish values are actually set as NaN"""
        if pd.isnull(value):
            return np.nan
        return value

    def __getitem__(self, item):
        # Index the broadcast view without allocating the full coordinate.
        selected = self.values[item]
        return self.__class__(shape=selected.shape, units=self.units, dtype=self.dtype)

    def _max(self):
        """Dummy funct to do nothing but raise."""
        return self.stop

    def _min(self):
        return self.start

    def update(self, **kwargs):
        """No values to change so update can just call new."""
        return self.new(**kwargs)

    # update_limits is spelled out rather than aliased to update so it keeps
    # the signature its base declares. It must forward only what the caller
    # supplied: a None reaching _validate_nullish_to_nan would overwrite the
    # stored start, stop or step with nan.
    def update_limits(self, min=None, max=None, step=None, **kwargs) -> BaseCoord:
        """No values to limit, so only what was passed is applied."""
        limits = {"min": min, "max": max, "step": step}
        passed = {i: v for i, v in limits.items() if v is not None}
        return self.update(**passed, **kwargs)

    def _convert_units(self, units) -> Self:
        """Convert scalar metadata units, or set units if none exist."""
        out = self.model_dump(exclude_unset=True, exclude_defaults=True)
        out["units"] = units
        if self.units is None or dtype_time_like(self.dtype):
            return self.__class__(**out)
        for name in ("start", "stop", "step"):
            value = getattr(self, name)
            out[name] = (
                value
                if pd.isnull(value)
                else convert_units(value, to_units=units, from_units=self.units)
            )
        return self.__class__(**out)

    def sort(self, reverse=False):
        """Sort dummy array. Does nothing."""
        return self, slice(None, None)

    def __len__(self):
        return self.shape[0]

    @property
    def values(self):
        """Return the internal data. Same as values attribute."""
        null_val = np.asarray(_get_nullish(self.dtype))
        # NaN is a plain python float, so a narrower floating dtype has to
        # be asked for by name or the values contradict the recorded dtype.
        if self.dtype is not None and np.dtype(self.dtype).kind == "f":
            null_val = null_val.astype(self.dtype)
        data = np.broadcast_to(null_val, self.shape)
        return data

    def _check_order_and_select(self, relative, samples):
        """Check that samples is True and relative false else raise msg."""
        if relative or not samples:
            msg = (
                "UnCoord does not support relative and samples must be True "
                "for both select and order methods."
            )
            raise CoordError(msg)

    def select(
        self, args, relative=False, samples=False
    ) -> tuple[BaseCoord, slice | ArrayLike]:
        """
        Select new values inside coord.

        For partial, samples==True or raise.
        """
        # Need to ensure relative is used OR the select has no effect.
        try:
            self._check_order_and_select(relative, samples)
        except CoordError as e:
            if not is_array(args):
                args = self._get_slice_tuple(args, relative=False)
                # Check if the select has no effect and return self or raise.
                if all(pd.isnull(x) for x in args):
                    return self, slice(None)
            raise e
        if is_array(args):
            return self._select_by_array(args, relative=relative, samples=samples)
        return self._select_by_samples(args)

    @compose_docstring(doc=get_docstring(BaseCoord.order))
    def order(
        self, array, relative=False, samples=False
    ) -> tuple[BaseCoord, slice | ArrayLike]:
        """
        {doc}.
        """
        self._check_order_and_select(relative, samples)
        return super().order(array, relative=relative, samples=samples)

    @compose_docstring(doc=get_docstring(BaseCoord.change_length))
    def change_length(self, length: int) -> Self:
        """
        {doc}
        """
        if self.ndim != 1:
            msg = "change_length only works on 1D coords."
            raise CoordError(msg)
        # A shape-only coord is always partial, so this really is Self; the
        # factory's declared BaseCoord return is just wider than the case.
        return cast("Self", get_coord(shape=(_validate_new_length(length),)))

    def to_summary(self, dims=()) -> CoordSummary:
        """Get the summary info about the coord."""
        return CoordSummary(
            min=np.nan,
            max=np.nan,
            step=np.nan,
            dtype=self.dtype,
            units=None,
            dims=dims,
            data_id=self.data_id,
        )

    def _id_components(self) -> tuple[Any, ...]:
        """Return the scalar payload identifying partial coords."""
        return (
            self.shape,
            str(np.dtype(self.dtype)),
            self._hash_scalar(self.start, "start"),
            self._hash_scalar(self.stop, "stop"),
            self._hash_scalar(self.step, "step"),
        )


_EXACT_GRID_FIELDS = ("step_numerator", "step_denominator", "origin_offset")
# The same three terms as the run table and the spool index name them.
_GRID_COLUMNS = ("num", "den", "offset")
# How many runs a summary states one by one before it keeps its envelope only.
_MAX_SUMMARY_RUNS = 256
# Nanoseconds per second: the tick of every exact time grid.
_NS_PER_S = 10**9
# The tick a datetime64 or timedelta64 reserves for NaT.
_NAT_TICK = np.iinfo(np.int64).min


def _fraction_step(step) -> Fraction | None:
    """A step given as a Fraction or (numerator, denominator) tuple, else None."""
    if isinstance(step, tuple):
        if len(step) != 2 or not step[1]:
            msg = f"A step given as a tuple is (numerator, denominator), got {step}."
            raise CoordError(msg)
        return Fraction(*step)
    return step if isinstance(step, Fraction) else None


def _is_null(value) -> bool:
    """Return True for None or a null scalar (NaN, NaT)."""
    if value is None:
        return True
    if _fraction_step(value) is not None:
        return False
    return bool(pd.isnull(_maybe_unpack(value)))


# Built once; a union spelled inside a test makes a type object per call.
_FLOATS = (float, np.floating)


def _is_int(value) -> bool:
    """Return True for a python or numpy integer (not a bool)."""
    value = _maybe_unpack(value)
    return isinstance(value, int | np.integer) and not isinstance(
        value, bool | np.bool_
    )


# A float spacing this close to a whole number of steps is on the grid;
# a float grid such as 0.1 cannot be held exactly, an off-grid label can.
_GRID_RTOL = 1e-6
# Run detection can turn jitter into many tiny runs. Keep one stored run when
# runs exceed this fraction and either the array is large or no run exceeds it.
_MIN_SEGMENT_GUARD_SIZE = 1_000
_MAX_SEGMENT_FRACTION = 0.1


def _declared_step(step, dtype):
    """
    A step declared on array values, as the scalar the values are measured in.

    A fraction may only be whole (labels on a fractional grid are floors of
    ideal positions, so they do not state it) and becomes seconds for time
    and an integer otherwise; a zero or non-finite step is no grid.
    """
    if (fraction := _fraction_step(step)) is not None:
        if fraction.denominator != 1:
            msg = (
                f"A fractional step ({fraction}) cannot be declared on array "
                "values; build the range with start, step, and shape instead."
            )
            raise CoordError(msg)
        whole = fraction.numerator
        step = (
            np.timedelta64(whole * _NS_PER_S, "ns") if dtype_time_like(dtype) else whole
        )
    magnitude = np.abs(np.asarray(step))[()]
    if not magnitude or not np.isfinite(to_float(magnitude)):
        msg = f"A declared step must be a finite non-zero spacing, got {step}."
        raise CoordError(msg)
    return step


def _diffs(values) -> np.ndarray:
    """Neighbour spacings, signed even for unsigned values."""
    values = np.asarray(values)
    if values.dtype.kind == "u":
        values = values.astype(np.int64)
    return np.diff(values)


def _on_grid(deltas, step) -> np.ndarray:
    """
    The whole number of steps in each spacing, raising when one is not whole.

    Ticks (time and integer) must divide exactly; floats within a relative
    tolerance of the step.
    """
    deltas, step = np.asarray(deltas), np.asarray(step)
    if deltas.dtype.kind in "mM":
        deltas, step = deltas.astype("timedelta64[ns]").astype(np.int64), _to_tick(step)
    if np.issubdtype(deltas.dtype, np.integer):
        counts, remainder = np.divmod(deltas, step)
        off = remainder != 0
    else:
        counts = np.round(deltas / step)
        off = np.abs(deltas - counts * step) > np.abs(step) * _GRID_RTOL
    if np.any(off):
        msg = f"Values are not on a grid of step {step}: spacing {deltas[off][0]}."
        raise CoordError(msg)
    return counts.astype(np.int64)


def _past_float_counting(values: np.ndarray) -> bool:
    """Whether these are integer labels a float64 cannot count one by one."""
    if not np.issubdtype(values.dtype, np.integer) or not values.size:
        return False
    return bool(np.max(np.abs(values.astype(np.float64))) > _FLOAT_INDEX_MAX)


def _keeps_step(segments, ascending: bool) -> bool:
    """
    Whether any seam between `segments` skips a position of their grid.

    A seam a whole number of steps wider than one is a hole: the samples
    for those positions are absent, and re-fitting the run to one range
    would spread the hole over it, reaching the last sample early and
    relabeling every sample after the hole. A seam which is *not* a whole
    number of steps is misalignment rather than absent data -- the jitter
    of labels rounded on their way to a file, or members trimmed where
    they overlapped -- and remains the tolerance's business.
    """
    for prev, nxt in itertools.pairwise(segments):
        step = prev.step if not _is_null(prev.step) else nxt.step
        if _is_null(step):
            # Neither side states a grid, so the seam skips no position of
            # one; whether it may close is the tolerance's business.
            continue
        before = prev.max() if ascending else prev.min()
        after = nxt.min() if ascending else nxt.max()
        steps = abs(after - before) / abs(step)
        whole = np.round(steps)
        if whole > 1 and abs(steps - whole) <= _GRID_RTOL * whole:
            return False
    return True


def _to_tick(value) -> int:
    """Return a time value as nanoseconds, or an integer value as itself."""
    value = _maybe_unpack(value)
    if is_datetime64(value) or is_timedelta64(value):
        # A nanosecond is the tick every time is counted in; anything a
        # nanosecond count cannot state is refused rather than rounded.
        return int(_as_ns(np.asarray(value)).view("int64")[()])
    if isinstance(value, float | np.floating):
        if not float(value).is_integer():
            msg = f"An integer coordinate cannot hold the non-integer value {value}."
            raise CoordError(msg)
        return int(value)
    try:
        tick = int(value)
    except (TypeError, ValueError) as e:
        msg = f"{value!r} is not an integer or time value."
        raise CoordError(msg) from e
    if not -_INT64_MAX <= tick < _INT64_MAX:
        # An unsigned label above the signed range has no tick to be.
        msg = f"{value} lies outside the int64 range a coordinate counts in."
        raise CoordError(msg)
    return tick


# --- the run table -----------------------------------------------------

# Float labels this close together meet: a boundary between two float grids
# cannot ask for equality of values which were never computed the same way.
_FLOAT_RTOL = 1e-12


def _as_dtype(value) -> np.dtype:
    """
    The coordinate dtype a scalar or array implies.

    Times are normalised to nanoseconds, which is the tick every run is
    counted in; every other numeric dtype is kept as it is, so a float32
    or int32 coordinate stays one however the table stores its record.
    """
    return _coord_dtype(np.asarray(value).dtype)


def _coord_dtype(dtype) -> np.dtype:
    """The coordinate dtype a dtype states, without an array to read it off."""
    dtype = normalize_coord_dtype(dtype)
    # Text and voids have no arithmetic the table can offer, so a row
    # naming one is the float placeholder a frame holds envelopes as.
    return np.dtype("float64") if dtype.kind in "SUV" else dtype


# Numpy's temporal units, coarsest first, so a unit can be told from a
# nanosecond by where it sits.
_TIME_UNITS = ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as")


def _out_of_ns(array: np.ndarray, target: np.dtype):
    """The label a nanosecond count cannot hold, or None."""
    flat = np.ravel(array)
    flat = flat[~np.isnat(flat)]
    if not flat.size:
        return None
    for value in (flat.min(), flat.max()):
        try:
            converted = np.asarray(value).astype(target)
        except (OverflowError, ValueError):
            return value
        # Older supported NumPy versions wrap overflowing temporal casts.
        if converted.astype(array.dtype) != value:
            return value
    return None


def _as_ns(values) -> np.ndarray:
    """
    Temporal labels counted in nanoseconds, refusing what one cannot hold.

    A nanosecond is the tick every time is counted in, and int64 spans it
    only from 1678 to 2262; a unit finer than a nanosecond can also name an
    instant between two ticks. Either way the conversion would change the
    label rather than restate it, so it is refused by name.
    """
    array = np.asarray(values)
    kind = "datetime64" if array.dtype.kind == "M" else "timedelta64"
    target = np.dtype(f"{kind}[ns]")
    if array.dtype == target or not array.size:
        return array.astype(target, copy=False)
    unit = np.datetime_data(array.dtype)[0]
    if _TIME_UNITS.index(unit) > _TIME_UNITS.index("ns"):
        out = array.astype(target)
        # NaT is never equal to itself, so it is not an off-grid label.
        off = (out.astype(array.dtype) != array) & ~np.isnat(array)
        if np.any(off):
            msg = f"{np.ravel(array[off])[0]} is not a whole number of nanoseconds."
            raise CoordError(msg)
        return out
    if (bad := _out_of_ns(array, target)) is not None:
        msg = f"{bad} lies outside the nanosecond range of a coordinate."
        raise CoordError(msg)
    return array.astype(target)


def _as_coord_values(values) -> np.ndarray:
    """Labels in the dtype a coordinate holds them in."""
    array = np.asarray(values)
    dtype = _as_dtype(array)
    if np.dtype(dtype).kind in "mM":
        return _as_ns(array)
    return array.astype(dtype, copy=False)


def _check_unsigned(values, dtype) -> None:
    """Refuse an unsigned label the signed tick range cannot hold."""
    if np.dtype(dtype).kind != "u" or not np.size(values):
        return
    top = np.max(values)
    if int(top) >= _INT64_MAX:
        # An unsigned label above the signed range has no tick to be.
        msg = f"{top} lies outside the int64 range a coordinate counts in."
        raise CoordError(msg)


def _as_ticks(values, dtype) -> np.ndarray:
    """Labels as integer ticks (nanoseconds, or the integers themselves)."""
    dtype = np.dtype(dtype)
    if dtype.kind in "mM":
        values = _as_ns(values)
    values = np.ascontiguousarray(values).astype(dtype, copy=False)
    # A time is already a count of nanoseconds; a narrower integer has to
    # be widened rather than reinterpreted.
    if dtype.kind in "mM":
        return values.view("int64")
    _check_unsigned(values, dtype)
    return values.astype("int64", copy=False)


def normalize_coord_dtype(dtype) -> np.dtype:
    """
    The dtype a run table counts in; a time is always in nanoseconds.

    A summary and a spool index row name a time dtype without its unit
    ("datetime64"), which is not the nanosecond dtype each run's start is
    a tick of; every other dtype is already what it says.
    """
    dtype = np.dtype(dtype)
    if dtype.kind == "M":
        return np.dtype("datetime64[ns]")
    if dtype.kind == "m":
        return np.dtype("timedelta64[ns]")
    return dtype


def _as_record(runs, dtype) -> np.ndarray:
    """
    Runs as the native record array a table of this dtype holds.

    Plain rows are cast to the record; a record read from a file of the
    other byte order is put in this machine's, since a run's hash reads the
    words as they lie.
    """
    record = _record_dtype(dtype)
    rows = np.asarray(runs) if isinstance(runs, np.ndarray) else None
    if rows is None or rows.dtype.names is None:
        # Plain rows, which may name their source or state a grid and stop.
        return runs_from_rows(runs, dtype)
    return rows if rows.dtype == record else rows.astype(record)


def runs_from_rows(rows, dtype) -> np.ndarray:
    """
    A run table from plain rows of ``(start, length, num, den, offset)``.

    The spelling a spool index row states a run in, so the index can hand
    its rows back to [`get_coord`](`dascore.core.coords.get_coord`)
    without knowing how the table stores them. A row may name the source
    its labels live in as a sixth field; one which does not states a grid.
    """
    record = _record_dtype(dtype)
    width = len(record.names or ())
    padded = [(*tuple(row), b"")[:width] for row in rows]
    return np.asarray(padded, record)


def _grid_run_stops(runs: np.ndarray, dtype) -> np.ndarray:
    """
    Each grid run's last label, in the spelling its start is stated in.

    A stored run's last label is in its own labels, which a bare table does
    not carry, so such a row states its start instead.
    """
    rows = np.asarray(runs)
    return get_kernel(dtype).labels(rows, slice(None), rows["length"] - 1)


def run_heads(runs: np.ndarray, dtype) -> np.ndarray:
    """
    Each run's first label, as a tick or a float.

    A tick row states it as its ``start``; a float row's ``start`` is the
    origin of its grid, so its first label has to be worked out.
    """
    return get_kernel(dtype).heads(np.asarray(runs))


def run_step(row, dtype):
    """One run's spacing as the envelope's scalar, or None for a stored run."""
    if not row["den"]:
        return None
    return get_kernel(dtype).step_of(row.item())


def _step_terms(step, dtype) -> tuple[int, int]:
    """
    A step of any spelling as the two grid terms a run row holds.

    For ticks the numerator and denominator are left as they were given: an
    origin offset is stated against this denominator, so reducing the step
    alone would move the phase. For floats the terms are the step's own
    bits and a stride of one.
    """
    if not _ticked(dtype):
        fraction = _fraction_step(step)
        value = float(fraction) if fraction is not None else _maybe_unpack(step)
        value = float(to_float(value)) if is_timedelta64(value) else float(value)
        if not math.isfinite(value):
            msg = f"A step must be a finite number, got {step}."
            raise CoordError(msg)
        return int(float_rows(dtype, [0.0], [1], [value])["num"][0]), 1
    if _fraction_step(step) is not None:
        if isinstance(step, tuple):
            num, den = int(step[0]), int(step[1])
        else:
            num, den = step.numerator, step.denominator
        if den < 1:
            msg = f"A step denominator must be positive, got {den}."
            raise CoordError(msg)
        # A fraction is given in coordinate units: seconds for a time.
        return (num * _NS_PER_S, den) if dtype_time_like(dtype) else (num, den)
    return _to_tick(_maybe_unpack(step)), 1


def _counts(lengths) -> np.ndarray:
    """
    Run lengths as the integer numpy counts repeats with.

    A run's length is an int64 column, which a 32 bit build will not take
    as a repeat count, so it is stated in the platform's own index type.
    """
    return np.asarray(lengths).astype(np.intp, copy=False)


def _continues(rows: np.ndarray, dtype) -> np.ndarray:
    """Whether each run begins where the run before would put its next sample."""
    before, after = rows[:-1], rows[1:]
    # Two stored runs are one run when the second reads on from where the
    # first stopped: the same source array, the same stride, and the window
    # picking up at the index the first left off at. Fusing them then moves
    # no label and asks nothing of the source, so no id is recomputed.
    stored = (before["den"] == 0) & (after["den"] == 0)
    stored &= before[SOURCE_ID] == after[SOURCE_ID]
    stored &= before["num"] == after["num"]
    stored &= after["offset"] == before["offset"] + before["num"] * before["length"]
    return stored | get_kernel(dtype).continues(rows)


def _canonical(rows: np.ndarray, dtype) -> np.ndarray:
    """Reduce grids, validate their range, and fuse exact continuations."""
    kernel = get_kernel(dtype)
    if len(rows) != 1:  # one row is already the table it states
        rows = rows[rows["length"] > 0]
    if len(rows) == 1:
        # The common case, where the vectorised body below is all overhead.
        row = rows[0].item()
        if row[1] > 0:
            reduced = kernel.reduced_one(row[:5])
            if reduced != row[:5]:
                rows = np.array([(*reduced, row[5])], rows.dtype)
            kernel.check_range(rows, dtype)
            return rows
        rows = rows[:0]
    if not len(rows):
        # The empty coordinate is a run of no samples, not an empty table,
        # so it still carries its dtype and concatenates away.
        return _rows(dtype, 0, [0], 0, 1, 0)
    rows = kernel.reduced(rows.copy())
    kernel.check_range(rows, dtype)
    heads = np.flatnonzero(np.concatenate([[True], ~_continues(rows, dtype)]))
    if len(heads) != len(rows):
        lengths = np.add.reduceat(rows["length"], heads)
        rows = rows[heads].copy()
        rows["length"] = lengths
    return rows


def _array_id(values: np.ndarray) -> str:
    """
    The content id of an array of labels.

    Labels of no fixed layout -- an object array of things which merely
    compare -- name no bytes to digest, so they take an id of their own
    rather than one another array could share by accident.
    """
    if values.dtype.hasobject:
        return new_id()
    return hash_array(values)


def _kept_sources(rows: np.ndarray, sources) -> Mapping[str, Any] | None:
    """
    The source arrays these rows still read, frozen against change.

    A table which has lost its last stored row over one source drops it;
    the rest are carried through by id, never re-hashed.
    """
    if not sources:
        return None
    used = set(np.unique(rows[SOURCE_ID][rows["den"] == 0]).tolist())
    kept = {k: v for k, v in sources.items() if k in used}
    if not kept:
        return None
    if isinstance(sources, MappingProxyType) and len(kept) == len(sources):
        return sources
    return MappingProxyType(kept)


def _union_sources(coords) -> dict:
    """
    One mapping covering every coordinate's sources.

    Equal ids are equal arrays, so the union needs no comparison and a
    repeated source is carried once.
    """
    out: dict = {}
    for coord in coords:
        if coord.sources:
            out.update(coord.sources)
    return out


def _fuse_float_neighbours(rows: np.ndarray, dtype) -> np.ndarray:
    """
    Restate a float run on the grid of the run before it, where that is exact.

    Two float grids of one spacing built apart -- each anchored at its own
    first label -- do not read as one, though every label of the second may
    be a label of the first's. The second is moved onto the first's grid
    only when that grid reproduces every one of its labels, bit for bit.
    """
    if _ticked(dtype) or len(rows) < 2:
        return rows
    out = rows.copy()
    target = np.dtype(dtype)
    for index in range(1, len(out)):
        before, row = out[index - 1], out[index]
        same = before["num"] == row["num"] and before["den"] == row["den"]
        if not (same and row["den"] != 0 and row["length"] > 0):
            continue
        moved = out[index : index + 1].copy()
        moved["start"] = before["start"]
        moved["offset"] = before["offset"] + before["length"] * abs(before["den"])
        k = np.arange(int(row["length"]), dtype=np.int64)
        labels = FloatKernel.labels(out, index, k).astype(target, copy=False)
        if FloatKernel._reproduces(moved, labels, target):
            out[index] = moved[0]
    return out


def _one_grid(rows: np.ndarray, dtype, num: int, den: int) -> bool:
    """Whether every run begins on the one grid these terms make."""
    if len(rows) < 2:
        return True
    return get_kernel(dtype).same_grid(rows, num, den)


def _declares(rows: np.ndarray, dtype, step) -> bool:
    """Whether a step declared beside the runs is a grid they all sit on."""
    try:
        num, den = _step_terms(step, dtype)
    except CoordError:
        return False
    return bool(num) and _one_grid(rows, dtype, num, den)


def _scalar_step(rows: np.ndarray, dtype):
    """
    The spacing every run shares, as the coordinate's scalar step, or None.

    A run of one sample states no spacing of its own, so among several runs
    only those holding two or more are asked; a lone run is taken at its
    word. The runs must also meet on the grid that spacing makes, or the
    spacing is not one the coordinate as a whole follows.
    """
    if len(rows) == 1:
        # One run states its own spacing, and meets no other run on it.
        _, length, num, den, _ = rows[0].item()[:5]
        if not length or not den:
            return None
        return _as_step(num, den, dtype)
    if not rows["length"].sum() or np.any(rows["den"] == 0):
        return None
    spacing = rows[rows["length"] > 1]
    if not len(spacing):
        return None
    num, den = spacing["num"], spacing["den"]
    if not (np.all(num == num[0]) and np.all(den == den[0])):
        return None
    if not _one_grid(rows, dtype, int(num[0]), int(den[0])):
        return None
    return _as_step(int(num[0]), int(den[0]), dtype)


def _as_step(num: int, den: int, dtype):
    """A run's grid terms as the scalar step: whole ticks, or a float."""
    step = get_kernel(dtype).step_of((0, 0, num, den, 0))
    return np.timedelta64(int(step), "ns") if dtype_time_like(dtype) else step


def _run_detection(
    anchors: np.ndarray, ticked: bool = True
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Where each run starts, which of them are grids, and how many were read.

    A spacing belongs to a run when a neighbouring spacing matches it, as
    DASCore's array-to-segments detection has it; the samples left over
    gather into stored runs. Where one run's spacing gives way to another's
    the two runs meet at a sample, which the earlier of them keeps: without
    that cut a change of rate would read as one grid and relabel every
    sample past it.

    Tick spacings match when they are equal. Float spacings of one grid
    differ by the rounding of their labels, so they match within a few
    units of that rounding; this only proposes the runs, and each is kept
    as a grid only if a grid reproduces its labels exactly.
    """
    # Neighbours are compared rather than subtracted where a difference
    # could leave int64; an overflowing spacing matches nothing.
    with np.errstate(over="ignore"):
        diffs = np.diff(anchors)
    if ticked:
        wrapped = (diffs > 0) != (anchors[1:] > anchors[:-1])
        equal = (diffs[:-1] == diffs[1:]) & ~wrapped[:-1] & ~wrapped[1:]
    else:
        # a NaN spacing matches nothing, so a null label ends a run
        finite = anchors[np.isfinite(anchors)]
        top = np.max(np.abs(finite)) if finite.size else 0.0
        with np.errstate(invalid="ignore"):
            equal = np.abs(diffs[:-1] - diffs[1:]) <= 8 * np.spacing(top)
    in_run = np.zeros(len(diffs), dtype=bool)
    in_run[1:] |= equal
    in_run[:-1] |= equal
    # A spacing which opens a run of its own where the spacing before it
    # closed one: the sample between them ends the earlier run.
    opens = np.zeros(len(diffs), dtype=bool)
    opens[1:] = in_run[1:] & in_run[:-1] & ~equal
    splits = np.flatnonzero(~in_run | opens) + 1
    block_starts = np.concatenate([[0], splits])
    block_lengths = np.diff(np.concatenate([block_starts, [len(anchors)]]))
    # A block of one sample states no grid; only the runs do.
    on_grid = block_lengths > 1
    heads = np.concatenate([[True], on_grid[1:] | on_grid[:-1]])
    return block_starts[heads], on_grid[heads], len(block_starts)


def _float_runs(values: np.ndarray, starts, lengths, on_grid, dtype, step=None):
    """
    Try the declared or neighboring grid, then fixed multiplication candidates.

    Keep labels stored when no candidate reproduces them exactly.
    """
    rows = float_rows(dtype, values[starts].astype(np.float64), lengths, 0.0, 0, 0)
    origin = None
    for index in np.flatnonzero(on_grid):
        first = int(starts[index])
        piece = values[first : first + int(lengths[index])]
        if origin is None and step is not None:
            origin = (float(piece[0]), float(step), 1)
        row = FloatKernel.fit(piece, dtype, origin=origin)
        if row is None:
            continue
        rows[index] = row[0]
        origin = (
            float(row["start"][0]),
            float(float_terms(row)[0][0]),
            int(row["den"][0]),
        )
    return rows


def _nearly_even(values: np.ndarray) -> bool:
    """Whether float labels are spaced evenly to within their own rounding."""
    with np.errstate(over="ignore", invalid="ignore"):
        diffs = np.diff(values.astype(np.float64, copy=False))
        slack = 8 * np.spacing(np.max(np.abs(values)))
        return bool(np.ptp(diffs) <= slack)


def _array_grid(values: np.ndarray, dtype) -> np.ndarray | None:
    """Recognize a single grid only after reproducing all supplied labels."""
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        return None
    if not is_strictly_monotonic(values):
        return None
    if _ticked(dtype):
        return _array_tick_grid(_as_ticks(values, dtype), dtype)
    if not _nearly_even(values):
        return None
    return FloatKernel.fit(values, dtype)


def _fitted_grid(values: np.ndarray, dtype) -> np.ndarray | None:
    """
    The even grid a monotonic array's spacings cluster around, or None.

    What `get_coord` has always counted as evenly sampled: labels whose
    spacings all sit within `all_diffs_close_enough`'s tolerance of their
    median restate one grid, and the grid then states them, so a label may
    move by a fraction of a step. `_array_grid` is tried first, so anything
    a grid reproduces exactly keeps its own labels.
    """
    if values.ndim != 1 or len(values) < 2 or not is_strictly_monotonic(values):
        return None
    if np.dtype(dtype).kind == "f" and np.dtype(dtype).itemsize != 8:
        # A row is counted in float64, which neither a wider float's labels
        # nor a narrower float's roundings of it are; fitting one would move
        # every label the coordinate holds.
        return None
    diffs = np.sort(_diffs(values))
    if diffs[0] != diffs[-1] and not all_diffs_close_enough(np.unique(diffs)):
        return None
    # A median which keeps the labels' own type, as the spacing of a grid
    # counted in whole ticks has to be.
    step = diffs[len(diffs) // 2]
    count = len(values)
    if _ticked(dtype):
        return _rows(dtype, [_to_tick(values[0])], [count], [_to_tick(step)], [1], [0])
    return float_rows(dtype, [float(values[0])], [count], float(step))


def _array_tick_grid(anchors: np.ndarray, dtype) -> np.ndarray | None:
    """Return the whole-tick grid stated by identical differences, or None."""
    span = int(anchors[-1]) - int(anchors[0])
    if abs(span) >= _INT64_MAX:
        return None
    diffs = np.diff(anchors)
    if diffs.max() != diffs.min():
        return None
    count = len(anchors)
    return _rows(dtype, anchors[0], [count], int(diffs[0]), 1, 0)


def _first_anchor(values: np.ndarray, dtype) -> float | int:
    """The first label of an array, as a tick or a float."""
    flat = np.ravel(values)[:1]
    return (_as_ticks(flat, dtype) if _ticked(dtype) else flat)[0]


def _guard_declines(sample_count: int, run_count: int) -> bool:
    """Whether detection would produce an overly fragmented table."""
    dense = sample_count >= _MIN_SEGMENT_GUARD_SIZE
    limit = _MAX_SEGMENT_FRACTION * sample_count
    return dense and run_count > limit


def _exact_dtype(start, stop, step, shape) -> np.dtype | None:
    """
    The dtype of the exact grid these inputs describe, or None for a float range.

    Nanosecond time and integers have a tick (a nanosecond, or one); a
    fraction step is only meaningful on such a grid. Coarser time units,
    floats, and integer spans that do not divide by their count keep the
    float representation, as they always have.
    """
    anchor = stop if _is_null(start) else start
    fraction = _fraction_step(step)
    ends = [np.asarray(x).dtype for x in (start, stop) if not _is_null(x)]
    if is_datetime64(anchor) or is_timedelta64(anchor):
        if fraction is not None:
            return np.dtype(f"{ends[0].kind}8[ns]")
        parts = (
            ends if _is_null(step) else [*ends, np.asarray(_maybe_unpack(step)).dtype]
        )
        try:
            dtype = np.result_type(*parts)
        except TypeError:  # not all time-like: the float validator says so
            return None
        return dtype if dtype.kind in "mM" and dtype.str.endswith("[ns]") else None
    if not (_is_int(anchor) and (_is_null(stop) or _is_int(stop))):
        if fraction is not None:
            msg = (
                "A fraction step requires a datetime64, timedelta64, or integer "
                f"coordinate, not {np.asarray(anchor).dtype}."
            )
            raise CoordError(msg)
        return None
    if _is_null(step):
        # Without a step the spacing comes from the span and count, and an
        # uneven division makes floats, as it always has.
        length = next(iter(iterate(shape))) if shape is not None else 0
        if len(ends) == 2 and length and (_to_tick(stop) - _to_tick(start)) % length:
            return None
        step = 1
    if fraction is None and not _is_int(step):
        return None
    tick = round(fraction) if fraction is not None else _maybe_unpack(step)
    return np.asarray(_maybe_unpack(anchor) + tick).dtype


def _grid_fields(start_tick: int, num: int, den: int, offset: int, count: int, dtype):
    """
    The stored fields of an exact grid, normalized and checked.

    The terms are reduced by their common divisor here; the run table then
    puts the grid in lowest terms (see `TickKernel.reduced`). The labels
    must fit the dtype: a grid which would wrap an int8, or whose
    arithmetic would leave int64, is refused.
    """
    g = math.gcd(num, den, offset)
    num, den, offset = num // g, den // g, offset // g
    dtype = np.dtype(dtype)
    stop_tick = start_tick + (offset + count * num) // den
    TickKernel._check_one((start_tick, count, num, den, offset), dtype)
    ticks = np.asarray([start_tick, stop_tick], dtype=np.int64).astype(dtype)
    step_tick = round(Fraction(num, den))
    step = np.timedelta64(step_tick, "ns") if dtype.kind in "mM" else step_tick
    return dict(
        start=ticks[0][()],
        stop=ticks[1][()],
        step=step,
        shape=(count,),
        dtype=dtype,
        step_numerator=num,
        step_denominator=den,
        origin_offset=offset,
    )


def _exact_fields(values, dtype: np.dtype) -> dict:
    """Resolve the exact grid a validator input describes."""
    start, stop, step = (
        None if _is_null(v := values.get(x)) else _maybe_unpack(v)
        for x in ("start", "stop", "step")
    )
    shape = values.get("shape")
    time_like = dtype.kind in "mM"
    start_tick = None if start is None else _to_tick(start)
    stop_tick = None if stop is None else _to_tick(stop)
    count = None
    if shape is not None:
        shape = tuple(iterate(shape))
        # `get_coord` answers a shape of any other rank with a partial
        # coordinate, so a range is only ever handed one axis.
        assert len(shape) == 1, "a range is built from a 1D shape"
        count = int(shape[0])
        assert count >= 1, "get_coord answers an empty shape with a partial coord"
    # The grid: a fraction step wins, then explicit grid fields, then a
    # scalar step, then the span divided by the count.
    offset = int(values.get("origin_offset") or 0)
    if (frac := _fraction_step(step)) is not None:
        if time_like:
            frac = frac * _NS_PER_S
        num, den = frac.numerator, frac.denominator
    elif values.get("step_numerator") is not None:
        num = int(values["step_numerator"])
        den = int(values.get("step_denominator") or 1)
    elif step is not None:
        num, den, offset = _to_tick(step), 1, 0
    else:
        # get_coord refuses fewer than three of start, stop, step and shape
        assert start_tick is not None and stop_tick is not None and count
        frac = Fraction(stop_tick - start_tick, count)
        num, den, offset = frac.numerator, frac.denominator, 0
    if den < 1:
        msg = "step_denominator must be positive."
        raise CoordError(msg)
    if 0 < abs(num) < den:
        msg = "A step smaller than one tick would repeat labels."
        raise CoordError(msg)
    if not 0 <= offset < den:
        msg = f"origin_offset must satisfy 0 <= offset < {den}, got {offset}."
        raise CoordError(msg)
    if count is None:
        assert start_tick is not None and stop_tick is not None
        if num == 0 or start_tick == stop_tick:
            count = 1
        else:
            span = stop_tick - start_tick
            if (span > 0) != (num > 0):
                msg = "Sign of step must match sign of stop - start"
                raise CoordError(msg)
            # Rounded to a tenth of a sample as a float, as the float range
            # always has, so a stop a hair past a sample does not add one.
            ratio = float(Fraction(span * den - offset, num))
            count = max(1, math.ceil(round(ratio, 1)))
    if start_tick is None:
        assert stop_tick is not None, "start or stop is present"
        # The ideal origin is count steps before stop; start is its floor.
        start_tick, offset = divmod(stop_tick * den - count * num, den)
    return _grid_fields(start_tick, num, den, offset, count, dtype)


def _float_fields(values) -> dict:
    """Resolve a float (or coarse-time) range from a validator input."""

    def _round_ratio(numerator, denominator, digits):
        """Round numerator/denominator, cheaply for scalars."""
        # Inputs are always scalar-like (multi-element arrays are rejected
        # by the pd.isnull check above) and rounding python floats is
        # ~10x faster than numpy scalars, hence the float conversion.
        ratio = _maybe_unpack(numerator / denominator)
        return round(float(ratio), digits)

    req_values = ("start", "stop", "step", "shape")
    _attrs = [values.get(x, None) for x in req_values]
    # get_coord refuses fewer than three of these before a range is built
    assert sum(not pd.isnull(x) for x in _attrs) >= 3
    # Now get start, stop, step from length, if provided.
    start, stop, step, shape = _attrs
    # A time is counted in nanoseconds, so it is restated in them before any
    # arithmetic: one no nanosecond count can hold is named here rather than
    # overflowing numpy's unit conversion below.
    start, stop, step = (
        _as_ns(np.asarray(x))[()] if (is_datetime64(x) or is_timedelta64(x)) else x
        for x in (start, stop, step)
    )
    if not pd.isnull(shape):
        shape = tuple(iterate(shape))
        assert len(shape) == 1, "a range is built from a 1D shape"
        length = shape[0]
        if pd.isnull(start):
            start = stop - step * length
        if pd.isnull(stop):
            stop = start + step * length
        if pd.isnull(step):
            step = (stop - start) / length
            # handle conversion to integer if other values are ints.
            if isinstance(start, int) and isinstance(stop, int):
                step = int(step) if np.isclose(np.round(step), step) else step

    zero = _TD64_ZERO if is_timedelta64(step) else 0
    if step != zero:
        span = _round_ratio(stop - start, step, 1)
        int_val = int(_maybe_unpack(np.ceil(span)))
        stop = start + step * int_val
    start_equal_stop = _maybe_unpack(start == stop)
    length = 1 if start_equal_stop else int(_round_ratio(stop - start, step, 0))
    # step should have the same sign as stop-start, see #321.
    # Compare signs via direct comparisons (rather than np.sign) since
    # np.sign(datetime64) returns a datetime64 which includes precision,
    # so even if the sign is the same, differing precision fails; direct
    # comparisons are also much cheaper than to_float conversions.
    diff = stop - start
    same_sign = ((step > zero) == (diff > zero)) & ((step < zero) == (diff < zero))
    if not same_sign:
        msg = "Sign of step must match sign of stop - start"
        raise CoordError(msg)
    # Note: dtype was a property before but it messed up model
    # serialization.
    return dict(
        start=start,
        stop=stop,
        step=step,
        shape=(length,),
        dtype=np.asarray(start + step).dtype,
        step_numerator=None,
        step_denominator=None,
        origin_offset=None,
    )


class NumericND(BaseCoord):
    """
    A numeric coordinate held as a table of runs.

    Each structured-array row stores a run's origin, length, grid terms, and
    phase. Integer and temporal grids use exact tick fractions; float grids
    use integer indices into a multiplication or division formula. A row
    with ``den == 0`` instead reads its labels out of a source array, which
    `sources` holds under the id in the row's ``source_id``. Runs partition
    the samples, and a hole is a run that does not continue its predecessor.
    Runs may be unordered or overlap; `sorted` and `reverse_sorted` describe
    their label order.

    Build one with [`get_coord`](`dascore.core.coords.get_coord`) rather than
    with the class directly.

    Parameters
    ----------
    runs
        The run table: a structured array with the fields ``start``,
        ``length``, ``num``, ``den``, ``offset`` and ``source_id``. A stored
        row reads ``source_id``'s array from ``offset``, ``length`` labels
        on, striding by ``num``.
    sources
        The label arrays the stored runs read, by array id; an immutable
        mapping, or None when no run is stored.

    Notes
    -----
    ``step`` is the spacing every run shares -- whole ticks for a time or
    an integer -- or None, so the library can keep reading it as a scalar;
    `step_exact` states a ticked one as a fraction of coordinate units.

    Examples
    --------
    >>> import numpy as np
    >>> from fractions import Fraction
    >>> from dascore.core.coords import NumericND, concat_coords
    >>>
    >>> t0 = np.datetime64("2020-01-01", "ns")
    >>> coord = NumericND.from_run(t0, Fraction(1, 1024), 2048)
    >>> assert coord.step == np.timedelta64(976562, "ns")
    >>> assert coord.step_exact == Fraction(1, 1024)
    >>>
    >>> # A second run after a hole; the rate is still one rate.
    >>> late = NumericND.from_run(t0 + np.timedelta64(4, "s"), Fraction(1, 1024), 8)
    >>> table = concat_coords(coord, late)
    >>> assert table.runs_count == 2 and table.holes
    >>> assert table.step == coord.step
    >>>
    >>> # Runs which continue each other fuse back into one.
    >>> assert concat_coords(coord[:100], coord[100:]) == coord
    """

    runs: Any = None
    # An immutable mapping of array id to the labels of the stored runs. Held
    # as Any and reached only through `_source_array`, so a later kind of
    # source (one which is read from a file) needs no change here.
    sources: Any = None

    if TYPE_CHECKING:
        # The before-validator below also takes the dtype, which pydantic
        # cannot see is a field of the base.
        def __init__(self, **data: Any) -> None: ...

    # --- construction

    @model_validator(mode="before")
    @classmethod
    def _as_table(cls, values: Any) -> Any:
        """
        Hold the runs as a record array of the coordinate's layout.

        The class states its own fields and nothing else; a start, a step,
        an array or a set of segments is read by
        [`get_coord`](`dascore.core.coords.get_coord`) and the ``from_*``
        builders, which is where a coordinate is made.
        """
        if not isinstance(values, dict):
            return values
        if values.get("runs") is None:
            msg = (
                "NumericND is built from its own run table; use get_coord, "
                "from_array, from_run, from_rows or concat_coords "
                f"to make one from {sorted(values)}."
            )
            raise CoordError(msg)
        return {**values, "runs": _as_record(values["runs"], values["dtype"])}

    @model_validator(mode="after")
    def _check_canonical(self) -> Self:
        """Refuse a directly constructed table that is not canonical."""
        rows = _canonical(self.runs, self.dtype)
        if not np.array_equal(rows, self.runs):
            msg = (
                "The run table is not in canonical form; build the coordinate "
                "with get_coord or NumericND.from_rows, which put it there."
            )
            raise CoordError(msg)
        return self

    @classmethod
    def _build(
        cls, dtype, rows, sources=None, units=None, step=None, shape=None
    ) -> Self:
        """
        Canonicalise a fresh table and hold it as a coordinate.

        ``step`` is the grid the labels are *declared* to sit on, kept only
        where the runs themselves state none (a stored run of jittered
        timestamps), so that `missing` answers the same on both sides of
        the dense-array guard. ``shape`` is the shape labels of their own
        state, which counting samples would flatten: an N-D run to one axis,
        and a rank-0 label to a coordinate of one sample it never was.
        """
        dtype = np.dtype(dtype)
        rows = _canonical(rows, dtype)
        lengths = rows["length"]
        total = int(lengths[0]) if len(lengths) == 1 else int(lengths.sum())
        shape = (total,) if shape is None else tuple(shape)
        if dtype.kind in "mM":
            # A time is measured in seconds whatever a caller states, as
            # `_check_time_units` holds every other coordinate class to.
            units = _second_quantity()
        derived = _scalar_step(rows, dtype)
        if derived is None and step is not None and _declares(rows, dtype, step):
            derived = step
        return cls.model_construct(
            _fields_set={"units", "step", "shape", "dtype", "runs", "sources"},
            units=get_quantity(units) if units is not None else None,
            step=derived,
            shape=shape,
            dtype=dtype,
            runs=rows,
            sources=_kept_sources(rows, sources),
        )

    @classmethod
    def _from_labels(
        cls, dtype, rows, labels, units=None, step=None, shape=None
    ) -> Self:
        """
        Hold a fresh table whose stored runs read ``labels``.

        The one place an array id is computed: every later coordinate --
        a slice, a reversal, a concatenation -- carries the id it is given
        here rather than hashing the labels again.
        """
        rows = np.asarray(rows).copy()
        stored = rows["den"] == 0
        if not np.any(stored):
            return cls._build(dtype, rows, None, units, step, shape)
        values = np.asarray(labels)
        key = _array_id(values).encode("ascii")
        rows[SOURCE_ID][stored] = key
        # Each stored row's window into the labels, in table order, which is
        # the order `from_array` and `_from_declared` lay them out in.
        starts = np.concatenate([[0], np.cumsum(np.where(stored, rows["length"], 0))])
        rows["offset"][stored] = starts[:-1][stored]
        rows["num"][stored] = 1
        return cls._build(dtype, rows, {key: values}, units, step, shape)

    @classmethod
    def from_rows(
        cls, runs, labels=None, dtype=None, units=None, step=None, sources=None
    ) -> Self:
        """
        Build from a run table.

        Parameters
        ----------
        runs
            A record array of runs, or anything which casts to one.
        labels
            The labels of the stored runs, concatenated in table order. The
            rows are re-pointed at one fresh source array holding them.
        dtype
            The coordinate dtype; taken from ``labels`` when not given.
        units
            The units of the labels.
        step
            The grid stored runs are declared to sit on.
        sources
            The source arrays the rows already name, by id, for a table
            whose stored runs carry their ``source_id``.
        """
        dtype = _as_dtype(labels) if dtype is None else _coord_dtype(dtype)
        rows = _as_record(runs, dtype)
        stored = (rows["den"] == 0) & (rows["length"] > 0)
        if labels is None and not sources and np.any(stored):
            # A stored run is nothing but its labels; filling it from its
            # start would silently repeat that one label.
            msg = "A stored run (den == 0) cannot be rebuilt without its labels."
            raise CoordError(msg)
        if labels is not None:
            values = np.asarray(labels).astype(dtype)
            shape = None if values.ndim == 1 else values.shape
            return cls._from_labels(dtype, rows, values, units, step, shape)
        return cls._build(dtype, rows, sources, units, step)

    @classmethod
    def from_run(
        cls, start, step, shape, origin_offset: int = 0, units=None, dtype=None
    ) -> Self:
        """
        Build one evenly sampled run.

        Parameters
        ----------
        start
            The first label.
        step
            The spacing: a number, a timedelta, a `Fraction`, or a
            ``(numerator, denominator)`` tuple in coordinate units.
        shape
            The sample count.
        origin_offset
            The phase of the first sample, in ``1 / denominator`` ticks.
        units
            The units of the labels.
        dtype
            The coordinate dtype; taken from ``start`` when not given. A
            float spacing on integer labels states one the start cannot.
        """
        if isinstance(shape, int | np.integer):
            count = int(shape)
        elif isinstance(shape, tuple) and len(shape) == 1:
            count = int(shape[0])
        else:
            count = int(np.prod(shape))
        dtype = _as_dtype(start) if dtype is None else _coord_dtype(dtype)
        num, den = _step_terms(step, dtype)
        start = _maybe_unpack(start)
        if _is_null(start):
            msg = f"A run needs a first label to start from, got {start}."
            raise CoordError(msg)
        if not _ticked(dtype):
            if np.dtype(dtype).itemsize > 8:
                # A row counts in float64, which a wider float's labels are
                # not; they are kept as the labels they are.
                labels = np.asarray(start) + np.arange(count) * np.asarray(step)
                return cls.from_array(labels.astype(dtype), units=units, detect=False)
            rows = _rows(dtype, [float(start)], [count], [num], [den], [0])
            return cls._build(dtype, rows, None, units)
        # Lowest terms before the row is built, phase with them: a step
        # given as a long fraction of seconds is a short one of ticks.
        offset = int(origin_offset)
        common = max(math.gcd(abs(num), den), 1)
        num, den, offset = num // common, den // common, offset // common
        if max(abs(num), den) >= _INT64_MAX:
            msg = f"A step of {step} has no fraction of ticks within int64."
            raise CoordError(msg)
        rows = _rows(dtype, [_to_tick(start)], [count], [num], [den], [offset])
        return cls._build(dtype, rows, None, units)

    @classmethod
    def from_array(
        cls,
        data,
        units=None,
        step=None,
        detect: bool = True,
        tolerance=None,
        fit: bool = False,
    ) -> Self:
        """
        Preserve labels, compressing recognized exact grids into runs.

        Detection checks whole-tick spacing and simple float multiplication
        grids. Unrecognized labels remain stored; highly fragmented arrays
        stay whole rather than splitting into many small runs.

        Parameters
        ----------
        data
            The labels.
        units
            The units of the labels.
        step
            Declared spacing for one-dimensional monotonic labels. Every
            spacing must span whole steps. The declaration survives stored
            representation.
        detect
            Whether to read runs out of the spacings. Without it the labels
            are kept whole as one stored run. Detected grid runs must
            reproduce every supplied label exactly.
        tolerance
            If not None, apply
            [`fuse`](`dascore.core.coords.BaseCoord.fuse`) with this
            tolerance to the result, re-fitting jittery runs and absorbing
            small gaps with bounded error.
        fit
            Whether monotonic labels which nearly restate one grid are read
            as that grid, moving them by a fraction of a step. This is what
            [`get_coord`](`dascore.core.coords.get_coord`) does with an
            array; the builders themselves keep every label.
        """
        if tolerance is not None:
            out = cls.from_array(data, units=units, step=step, detect=detect, fit=fit)
            return cast("Self", out.fuse(tolerance))
        values = _as_coord_values(data)
        dtype = _as_dtype(values)
        if not values.size:
            # No labels state no grid; an N-D emptiness keeps its shape.
            rows = _rows(dtype, 0, [0], 0, 1, 0)
            nd = None if values.ndim == 1 else values.shape
            return cls._build(dtype, rows, None, units, shape=nd)
        if np.dtype(dtype).kind not in "iufMm":
            # Labels the table has no arithmetic for are simply held; an
            # object which cannot be subtracted states no grid at all.
            rows = _rows(dtype, [0.0], [values.size], 0, 0, 0)
            shape = None if values.ndim == 1 else values.shape
            return cls._from_labels(dtype, rows, values, units, shape=shape)
        if step is not None:
            return cls._from_declared(values, dtype, step, units, detect=detect)
        if detect and (grid := _array_grid(values, dtype)) is not None:
            try:
                return cls._build(dtype, grid, None, units)
            except CoordError:
                # Valid labels can end at their dtype's limit even though
                # the grid's exclusive stop would overflow it.
                return cls.from_array(values, units=units, detect=False)
        if fit and (grid := _fitted_grid(values, dtype)) is not None:
            with suppress(CoordError):
                return cls._build(dtype, grid, None, units)
        if not detect or values.ndim != 1 or len(values) < 3:
            # Labels which are kept rather than read still have to be labels
            # this coordinate can answer about at all.
            _check_unsigned(values, dtype)
            rows = _rows(dtype, [_first_anchor(values, dtype)], [values.size], 0, 0, 0)
            shape = None if values.ndim == 1 else values.shape
            return cls._from_labels(dtype, rows, values, units, shape=shape)
        ticked = _ticked(dtype)
        anchors = _as_ticks(values, dtype) if ticked else values
        starts, on_grid, detected = _run_detection(anchors, ticked)
        if _guard_declines(len(values), detected):
            starts, on_grid = np.zeros(1, np.int64), np.zeros(1, bool)
        lengths = np.diff(np.concatenate([starts, [len(values)]]))
        limit = _MAX_SEGMENT_FRACTION * len(values)
        fragmented = detected > limit and int(lengths.max()) <= limit
        if ticked:
            follow = np.minimum(starts + 1, len(values) - 1)
            spacings = np.where(on_grid, anchors[follow] - anchors[starts], 0)
            num, den = spacings.astype(np.int64), np.where(on_grid, 1, 0)
            rows = _rows(dtype, anchors[starts], lengths, num, den, 0)
        else:
            # A float grid only holds the labels it reproduces exactly;
            # the runs it cannot keep theirs instead.
            rows = _float_runs(values, starts, lengths, on_grid, dtype)
        labels = values[np.repeat(rows["den"] == 0, _counts(rows["length"]))]
        try:
            out = cls._from_labels(dtype, rows, labels, units)
        except CoordError:
            # A detected run can reach the dtype limit even when the whole
            # array is not a grid. Its supplied labels remain valid.
            return cls.from_array(values, units=units, detect=False)
        if dtype_time_like(dtype) and fragmented:
            # Fractional-rate timestamps can alternate between neighbouring
            # tick spacings. Keep real gaps, but not their phantom seams.
            tolerance = GapTolerance.samples(DEFAULT_TOLERANCE)
            if out.get_discontinuities("gaps", tolerance).empty:
                return cls.from_array(values, units=units, detect=False)
        return out

    @classmethod
    def _from_declared(cls, values, dtype, step, units, detect: bool = True) -> Self:
        """Build from labels which are declared to sit on a grid of ``step``."""
        if values.ndim != 1:
            msg = "A declared step needs one-dimensional, monotonic values."
            raise CoordError(msg)
        step = _declared_step(step, dtype)
        magnitude = np.abs(np.asarray(step))[()]
        if len(values) > 1 and not is_strictly_monotonic(values):
            msg = "A declared step needs one-dimensional, monotonic values."
            raise CoordError(msg)
        signed = (
            step
            if len(values) < 2
            else magnitude
            if values[-1] > values[0]
            else -magnitude
        )
        anchors = _as_ticks(values, dtype) if _ticked(dtype) else values
        if len(values) < 2:
            num, den = _step_terms(signed, dtype)
            rows = _rows(dtype, anchors[:1], [len(values)], num, den, 0)
            return cls._build(dtype, rows, None, units, step=signed)
        counts = _on_grid(_diffs(values), signed)
        splits = np.flatnonzero(counts != 1) + 1
        starts = np.concatenate([[0], splits])
        if not detect or _guard_declines(len(values), len(starts)):
            # Too many runs to be worth detecting, or none asked for; the
            # labels are kept as they are, with the grid they declare.
            rows = _rows(dtype, anchors[:1], [len(values)], 0, 0, 0)
            return cls._from_labels(dtype, rows, values, units, step=signed)
        lengths = np.diff(np.concatenate([starts, [len(values)]]))
        # Runs of a single sample state no spacing of their own, so the
        # grid they were read against travels with them.
        if _ticked(dtype):
            rows = _rows(dtype, anchors[starts], lengths, _to_tick(signed), 1, 0)
            return cls._build(dtype, rows, None, units, step=signed)
        # Float labels within a rounding of the declared grid are on it;
        # a run is still a grid only where one reproduces it exactly.
        grid = np.ones(len(starts), dtype=bool)
        rows = _float_runs(values, starts, lengths, grid, dtype, step=signed)
        single = (rows["den"] == 0) & (lengths == 1)
        if np.any(single):
            # One label is its own grid, of the declared spacing.
            heads = values[starts[single]].astype(np.float64)
            rows[single] = float_rows(dtype, heads, lengths[single], signed)
        labels = values[np.repeat(rows["den"] == 0, _counts(rows["length"]))]
        return cls._from_labels(dtype, rows, labels, units, step=signed)

    # --- the run table

    @property
    def runs_count(self) -> int:
        """How many runs the table holds."""
        return len(self.runs)

    @property
    def segments(self) -> tuple[NumericND, ...]:
        """Each run as a coordinate of its own."""
        return tuple(self._run_view(i) for i in range(len(self.runs)))

    def _run_view(self, index: int) -> NumericND:
        """The run at ``index`` as a coordinate of one run."""
        rows = self.runs[index : index + 1].copy()
        return self._build(self.dtype, rows, self.sources, self.units, step=self.step)

    def __getstate__(self) -> dict:
        """The state pickle takes; a mapping proxy is not one of its types."""
        state = super().__getstate__()
        fields = state.get("__dict__") or {}
        if fields.get("sources") is not None:
            fields = {**fields, "sources": dict(fields["sources"])}
            state = {**state, "__dict__": fields}
        return state

    def __setstate__(self, state) -> None:
        """Freeze the sources again on the way back in."""
        super().__setstate__(state)
        if self.sources is not None:
            object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))

    def __deepcopy__(self, memo=None) -> Self:
        """Copy through the pickle state, whose sources are a plain mapping."""
        out = self.__class__.__new__(self.__class__)
        out.__setstate__(copy.deepcopy(self.__getstate__(), memo))
        return out

    @property
    def _ticks(self) -> bool:
        """Whether labels are whole ticks (times and integers) or floats."""
        return _ticked(self.dtype)

    @property
    def _foreign(self) -> tuple | None:
        """
        How another library stated these same labels, or None.

        A converter or reader leaves a ``(kind, *payload)`` note here, such
        as the tie points an XDAS file interpolates between, so that handing
        the coordinate straight back gives that library exactly what it
        gave. It is a note on this one object: no operation carries it to
        the coordinate it returns, and it is no part of a dump, of equality,
        or of an id. The labels are always the run table's.
        """
        return self._cache.get("foreign")

    def _note_foreign(self, kind: str, *payload) -> Self:
        """Leave a note of how another library stated these labels."""
        self._cache["foreign"] = (kind, *payload)
        return self

    @property
    def _kernel(self) -> type[TickKernel] | type[FloatKernel]:
        """The arithmetic the rows are read with; see `dascore.core._run_kernels`."""
        return get_kernel(self.dtype)

    @property
    @cached_method
    def _narrow(self) -> np.dtype | None:
        """The dtype the labels are held in, where a row's float64 is not it.

        Rows are counted in float64 and the coordinate rounds each label
        into its own dtype, so a lookup on a narrower float has to compare
        the rounded labels: the doubles a row makes are not the labels this
        coordinate hands out, and two of its labels can round closer
        together than the row's own spacing.
        """
        dtype = np.dtype(self.dtype)
        if dtype.kind != "f" or dtype.itemsize >= 8:
            return None
        return dtype

    @property
    @cached_method
    def _run_heads(self) -> np.ndarray:
        """The first label (tick or float) of each run."""
        return self._kernel.heads(self.runs)

    @property
    @cached_method
    def _sample_starts(self) -> np.ndarray:
        """The sample index each run begins at, with the total appended."""
        # Cached and shared by callers, so hand back a read-only array.
        return array(np.concatenate([[0], np.cumsum(self.runs["length"])]))

    @property
    @cached_method
    def _label_starts(self) -> np.ndarray:
        """Where each run's labels begin in `_flat_labels`, with the total."""
        stored = np.where(self.runs["den"] == 0, self.runs["length"], 0)
        return np.concatenate([[0], np.cumsum(stored)])

    @property
    def _stored_labels(self) -> bool:
        """Whether any run reads its labels from a source array."""
        return self.sources is not None

    def _source_array(self, source_id) -> np.ndarray:
        """
        The labels one source holds, flat.

        The only place a source's values are read, so what a source *is*
        stays behind this one call.
        """
        return np.ravel(np.asarray(self.sources[source_id]))

    def _run_labels(self, index: int) -> np.ndarray:
        """The window of its source one stored run reads."""
        row = self.runs[index]
        source = self._source_array(row[SOURCE_ID])
        offset, num, length = int(row["offset"]), int(row["num"]), int(row["length"])
        if num == 1:
            return source[offset : offset + length]
        return source[offset + num * np.arange(length, dtype=np.int64)]

    @property
    @cached_method
    def _flat_labels(self) -> np.ndarray:
        """Every stored run's labels, in table order, as ticks (or floats)."""
        stored = np.flatnonzero(self.runs["den"] == 0)
        parts = [self._run_labels(int(index)) for index in stored]
        flat = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return _as_ticks(flat, self.dtype) if self._ticks else flat

    @property
    def start(self):
        """The first label."""
        return self._from_anchor(self._run_heads[:1])[0][()]

    @property
    def stop(self):
        """One step past the last label, as a range states its end."""
        rows = self.runs
        row = rows[-1]
        if row["den"] == 0:
            last = self._run_ends[-1]
            return self._from_anchor(np.asarray([last]))[0][()]
        last, k = np.asarray([len(rows) - 1]), np.asarray([int(row["length"])])
        return self._from_anchor(self._kernel.labels(rows, last, k))[0][()]

    def _from_anchor(self, anchor) -> np.ndarray:
        """Anchors (ticks or floats) as labels of the coordinate dtype."""
        anchor = np.ascontiguousarray(anchor)
        dtype = np.dtype(self.dtype)
        if dtype.kind in "mM":
            return anchor.astype("int64", copy=False).view(dtype)
        return anchor.astype(dtype, copy=False)

    @property
    def step_exact(self) -> Fraction | None:
        """The exact spacing in coordinate units (seconds for time), or None."""
        rows = self.runs
        if not self._ticks or _scalar_step(rows, self.dtype) is None:
            # a float spacing is the double it is, which has no exact form
            return super().step_exact
        fraction = Fraction(int(rows["num"][0]), int(rows["den"][0]))
        return fraction / _NS_PER_S if dtype_time_like(self.dtype) else fraction

    @property
    def evenly_sampled(self) -> bool:
        """Whether the labels are one grid run of at least one sample."""
        rows = self.runs
        return len(rows) == 1 and bool(rows["den"][0]) and bool(rows["length"][0])

    @property
    def _exact(self) -> bool:
        """Whether the labels come from an integer grid of one run."""
        return self.evenly_sampled and self._ticks

    @property
    def _grid_terms(self) -> tuple[int, int, int]:
        """The numerator, denominator, and origin offset of an exact grid."""
        assert self._exact, "only an exact grid has terms"
        row = self.runs[0]
        return int(row["num"]), int(row["den"]), int(row["offset"])

    @property
    def _start_tick(self) -> int:
        """The first label as an integer tick."""
        return int(self.runs["start"][0])

    @property
    def _ideal_origin(self) -> Fraction:
        """The ideal origin of an exact grid, in ticks."""
        _, den, offset = self._grid_terms
        return Fraction(self._start_tick * den + offset, den)

    def _run_spacing(self, index: int):
        """The spacing a stored run is held to, in anchor space, or None.

        Its own labels say what one step looks like, unless the coordinate
        declares a step, which outranks them.
        """
        if not _is_null(self.step):
            return _to_tick(self.step) if self._ticks else float(self.step)
        first, last = self._label_starts[index], self._label_starts[index + 1]
        labels = self._flat_labels[first:last]
        if len(labels) < 2:
            return None  # one label states no spacing at all
        # Differenced before it is widened: a nanosecond tick past 2**53 has
        # no float of its own, but the spacings between such ticks do.
        median = float(np.median(np.abs(np.diff(labels))))
        return median if labels[-1] >= labels[0] else -median

    @property
    @cached_method
    def _hole_boundaries(self) -> np.ndarray:
        """Whether each boundary between runs opens a hole.

        A run which changes rate at the sample the last one stopped at is
        not a hole. A stored run states no next position, so the spacing its
        own labels keep stands in for one; the boundary is a hole where the
        next run starts more than one sample past its last label, counting
        samples the way `missing` does -- the distance over that spacing,
        rounded. Rounding is what keeps jitter from reading as a hole: a
        run whose own spacings vary states nothing finer than a sample.
        """
        rows = self.runs
        if len(rows) < 2:
            return np.zeros(0, dtype=bool)
        before = rows[:-1]
        heads = self._run_heads[1:]
        expected = self._kernel.labels(before, np.arange(len(before)), before["length"])
        if self._ticks:
            gap = expected != heads
        else:
            gap = ~np.isclose(expected, heads, rtol=_FLOAT_RTOL, atol=0.0)
        # A boundary one side of which is not a label at all -- a NaN or a
        # NaT among stored values -- states no distance, so no hole.
        ends = self._run_ends
        known = ~(
            pd.isnull(self._from_anchor(ends[:-1]))
            | pd.isnull(self._from_anchor(heads.copy()))
        )
        gap &= known
        for index in np.flatnonzero((before["den"] == 0) & known):
            spacing = self._run_spacing(int(index))
            if not spacing or not np.isfinite(spacing):
                gap[index] = False
                continue
            # differenced in the anchors' own type, which for a tick is an
            # exact integer where its float would have rounded
            delta = float(heads[index] - ends[index])
            gap[index] = round(delta / spacing) > 1
        return gap

    @property
    def holes(self) -> bool:
        """Whether any run begins past where the run before it would end."""
        return bool(np.any(self._hole_boundaries))

    # --- labels

    def _labels(self, indices) -> np.ndarray:
        """
        The labels at these sample indices.

        Indices outside the coordinate extend the grid of the run they fall
        nearest, which is what padding a range asks for.
        """
        indices = np.asarray(indices, np.int64)
        rows = self.runs
        ends = self._sample_starts[1:]
        run = np.clip(
            np.searchsorted(ends, indices.ravel(), side="right"), 0, len(rows) - 1
        )
        k = indices.ravel() - self._sample_starts[run]
        if self._ticks and k.size:
            # Past either end the grid carries on, which its own range check
            # never vouched for; a label int64 cannot hold is refused.
            outside = (k < 0) | (k > rows["length"][run])
            if np.any(outside):
                reach = np.abs(rows["start"][run][outside].astype(np.float64)) + np.abs(
                    k[outside].astype(np.float64)
                ) * np.abs(rows["num"][run][outside].astype(np.float64)) / np.maximum(
                    rows["den"][run][outside], 1
                )
                if np.any(reach >= _INT64_MAX):
                    msg = "A label that far outside the coordinate leaves int64."
                    raise CoordError(msg)
        out = self._kernel.labels(rows, run, k)
        if self._stored_labels:
            stored = rows["den"][run] == 0
            if np.any(stored):
                gather = self._label_starts[run[stored]] + k[stored]
                out[stored] = self._flat_labels[gather]
        return self._from_anchor(out).reshape(indices.shape)

    @property
    @cached_method
    def values(self):
        """The labels. Cached and shared, so the array is read-only."""
        rows = self.runs
        if not self.size and self.ndim != 1:
            # An N-D coordinate emptied along one axis keeps its shape,
            # which a run of no samples cannot state on its own.
            return array(np.empty(self.shape, dtype=self.dtype))
        if self._stored_labels and len(rows) == 1:
            flat = self._from_anchor(self._flat_labels)
            return array(flat if self.ndim == 1 else flat.reshape(self.shape))
        lengths = rows["length"]
        kernel = self._kernel
        if len(rows) == 1:
            count = int(lengths[0])
            k = np.arange(count, dtype=np.int64)
            return array(self._from_anchor(kernel.labels(rows, 0, k, reach=count)))
        total = int(lengths.sum())
        counts = _counts(lengths)
        k = np.arange(total, dtype=np.int64) - np.repeat(
            self._sample_starts[:-1], counts
        )
        run = np.repeat(np.arange(len(rows)), counts)
        out = kernel.labels(rows, run, k, reach=int(lengths.max()))
        if self._stored_labels:
            out[np.repeat(rows["den"] == 0, counts)] = self._flat_labels
        return array(self._from_anchor(out))

    def _get_index_values(self, indices):
        """The labels at these indices, counting negatives from the end."""
        indices = np.asarray(indices)
        return self._labels(np.where(indices < 0, indices + len(self), indices))

    # --- order and limits

    @property
    @cached_method
    def _run_ends(self) -> np.ndarray:
        """The last label (tick or float) of each run."""
        rows = self.runs
        out = self._kernel.labels(rows, np.arange(len(rows)), rows["length"] - 1)
        if self._stored_labels:
            stored = rows["den"] == 0
            out[stored] = self._flat_labels[self._label_starts[1:][stored] - 1]
        return out

    @property
    @cached_method
    def _direction(self) -> int:
        """1 for increasing labels, -1 for decreasing, 0 for neither."""
        rows = self.runs
        if self.ndim != 1 or np.dtype(self.dtype).kind not in "iufMm":
            # labels the table has no arithmetic for state no order
            return 0
        if not self.size:
            # Nothing to compare, so there is no order to read off.
            return 0
        grid = rows["den"] != 0
        num = rows["num"][grid]
        # A zero-step run is flat, so it goes either way.
        up, down = bool(np.all(num >= 0)), bool(np.all(num <= 0))
        if self._stored_labels and len(self._flat_labels) > 1:
            flat = self._flat_labels
            if np.dtype(self.dtype).kind in "mM" and np.any(flat == _NAT_TICK):
                # A missing time is not the smallest one: read as the tick
                # it is held as it would sort first and be handed back as
                # the minimum. NaN does this to itself, since it compares
                # false either way; NaT held as int64 does not.
                return 0
            inner = np.ones(len(flat) - 1, dtype=bool)
            cuts = np.unique(self._label_starts)
            inner[cuts[(cuts > 0) & (cuts < len(flat))] - 1] = False
            # compared, not subtracted: a difference can leave int64
            up &= bool(np.all((flat[1:] > flat[:-1])[inner]))
            down &= bool(np.all((flat[1:] < flat[:-1])[inner]))
        if len(rows) > 1:  # runs may overlap, which their boundaries show
            ends, starts = self._run_ends[:-1], self._run_heads[1:]
            up &= bool(np.all(ends < starts))
            down &= bool(np.all(ends > starts))
        return 1 if up else (-1 if down else 0)

    @property
    def sorted(self) -> bool:
        """Whether every label is greater than the one before."""
        return self._direction == 1

    @property
    def reverse_sorted(self) -> bool:
        """Whether every label is less than the one before."""
        return self._direction == -1

    def _min(self):
        if not self.size:
            return _get_nullish(self.dtype)
        if self.reverse_sorted:
            return self._from_anchor(self._run_ends[-1:])[0][()]
        # Labels in no order are read out; a missing one is not the
        # smallest label, it is no label at all.
        return self.start if self.sorted else np.nanmin(self.values)

    def _max(self):
        if not self.size:
            return _get_nullish(self.dtype)
        if self.reverse_sorted:
            return self.start
        if self.sorted:
            return self._from_anchor(self._run_ends[-1:])[0][()]
        return np.nanmax(self.values)

    def empty(self, axes=None) -> Self:
        """A coordinate of no samples, keeping this one's dtype and units."""
        if self.ndim > 1:
            # An N-D coordinate keeps its rank when it is emptied, whether
            # one axis was named or all of them were.
            shape = np.asarray(self.shape)
            for ind in iterate(axes) if axes is not None else range(self.ndim):
                shape[ind] = 0
            return self.from_array(
                np.empty(tuple(shape), dtype=self.dtype), units=self.units
            )
        rows = _rows(self.dtype, 0, [0], 0, 1, 0)
        return self._build(self.dtype, rows, None, self.units)

    # --- slicing

    def __getitem__(self, item):
        if self.ndim != 1:  # a stored N-D run is indexed as its labels are
            out = self.values[item]
            return self.from_array(out, units=self.units) if np.ndim(out) else out
        if isinstance(item, np.ndarray) and item.ndim == 0 and item.dtype.kind in "iu":
            item = int(item)
        if isinstance(item, int | np.integer):
            if item >= len(self) or item < -len(self):
                raise IndexError(f"{item} exceeds coord length of {self}")
            return self._get_index_values(item)[()]
        if isinstance(item, slice):
            start = None if item.start is ... else item.start
            end = None if item.stop is ... else item.stop
            indices = range(len(self))[slice(start, end, item.step)]
            if not len(indices):
                return self.empty()
            if indices == range(len(self)):  # every sample, in order
                return self
            return self._sliced(indices.start, indices.step, len(indices))
        indices = np.asarray(item) if isinstance(item, (list, np.ndarray)) else None
        if (
            indices is not None
            and indices.ndim == 1
            and (indices.dtype.kind in "biu" or (isinstance(item, list) and not item))
        ):
            if indices.dtype.kind == "b":
                if len(indices) != len(self):
                    raise IndexError("Boolean index must match the coordinate length.")
                indices = np.flatnonzero(indices)
            elif np.any(indices >= len(self)) or np.any(indices < -len(self)):
                raise IndexError("Index exceeds coordinate length.")
            out = self._get_index_values(indices.astype(np.int64))
        else:
            out = self.values[item]
        if not np.ndim(out):  # one label, not a coordinate of one
            return out
        # Only a step the runs do not state themselves is carried: it is
        # the grid the labels were declared to sit on, and it survives
        # only an order it can still be held against.
        keep = not _is_null(self.step) and np.ndim(out) == 1
        keep = keep and _scalar_step(self.runs, self.dtype) is None
        keep = keep and is_strictly_monotonic(out)
        # Read as any labels are, except that these are labels this
        # coordinate already held: a selection which is exactly evenly
        # sampled says so, and no label moves onto a grid it is merely near.
        return self.from_array(out, units=self.units, step=self.step if keep else None)

    def _sliced(self, first: int, stride: int, count: int) -> Self:
        """
        The coordinate of samples ``first, first + stride, ...`` (``count``).

        A single grid run may be sliced past either end, which extends the
        grid; padding a coordinate asks for exactly that.
        """
        if stride < 0:
            last = first + stride * (count - 1)
            return self._sliced(last, -stride, count)._reversed()
        rows = self.runs
        if len(rows) == 1 and rows["den"][0] != 0 and self._ticks:
            # One tick run, in python integers: it may be sliced far past
            # either end, where int64 has no room for the products.
            start, _, num, den, offset = rows[0].item()[:5]
            carry, offset = divmod(offset + first * num, den)
            if max(abs(num * stride), abs(start + carry)) >= _INT64_MAX:
                msg = f"Slicing from {first} by {stride} takes the run past int64."
                raise CoordError(msg)
            new = _rows(
                self.dtype, [start + carry], [count], [num * stride], [den], [offset]
            )
            return self._build(self.dtype, new, None, self.units, step=self.step)
        if len(rows) == 1 and rows["den"][0] != 0:
            new = self._kernel.sliced(rows, first, stride)
            new["length"] = count
            return self._build(self.dtype, new, None, self.units, step=self.step)
        starts = self._sample_starts
        # The first selected sample inside each run, and how many it holds,
        # computed per run rather than per sample.
        base = np.maximum(starts[:-1], first)
        picked = first + -((first - base) // stride) * stride
        limit = np.minimum(starts[1:], first + stride * count)
        counts = np.maximum(-(-(limit - picked) // stride), 0)
        keep = counts > 0
        k = picked[keep] - starts[:-1][keep]
        new = self._kernel.sliced(rows[keep], k, stride)
        new["length"] = counts[keep]
        if (stored := new["den"] == 0).any():
            # The window moved; its first label is the one it now starts on.
            flat = self._flat_labels
            heads = self._label_starts[:-1][keep][stored] + k[stored]
            new["start"][stored] = flat[heads]
        return self._build(self.dtype, new, self.sources, self.units, step=self.step)

    def _reversed(self) -> Self:
        """The same samples in the opposite order."""
        rows = self.runs
        new = self._kernel.reversed(rows)
        if (stored := rows["den"] == 0).any():
            new["start"][stored] = self._flat_labels[self._label_starts[1:][stored] - 1]
        step = self.step
        if step is not None and not self.evenly_sampled:
            step = -step if np.ndim(step) == 0 else step
        return self._build(self.dtype, new[::-1], self.sources, self.units, step=step)

    def index(self, indexer, axis: int | None = None) -> BaseCoord:
        """Index one-dimensional runs without expanding their unused labels."""
        if self.ndim != 1 or axis not in (None, 0, -1):
            return super().index(indexer, axis=axis)
        out = self[indexer]
        if isinstance(out, BaseCoord):
            return out
        # The labels picked are labels this coordinate already held, so
        # they are re-read rather than fitted to a grid.
        return self.from_array(np.asarray(out), units=self.units)

    def sort(self, reverse=False) -> tuple[BaseCoord, slice | np.ndarray]:
        """Sort the labels; return the sorted coordinate and the index to apply."""
        if (self.sorted and not reverse) or (self.reverse_sorted and reverse):
            return self, slice(None)
        if self.sorted or self.reverse_sorted:
            return self._reversed(), slice(None, None, -1)
        order = np.argsort(self.values)
        order = order[::-1] if reverse else order
        return self.from_array(self.values[order], units=self.units), order

    # --- lookup by value

    @property
    @cached_method
    def _reverse_view(self) -> Self:
        """The coordinate the other way round, for lookup on a sorted table."""
        return self._reversed()

    def _bound_tick(self, value, forward: bool) -> int:
        """The integer tick a query bound is equivalent to."""
        if is_datetime64(value) or is_timedelta64(value):
            return _to_tick(value)
        value = _maybe_unpack(value)
        if isinstance(value, _FLOATS):
            value = math.ceil(value) if forward else math.floor(value)
        return int(min(max(int(value), -_INT64_MAX), _INT64_MAX - 1))

    def _index_sorted(self, value, forward: bool) -> int:
        """
        The index a query value maps to, for a table of increasing labels.

        Forward: the first index whose label is at or past the value; else
        the last index whose label is at or before it. A value past either
        end of a grid gets the position that grid would give it, outside
        the coordinate, which is how the caller tells an open bound from a
        bound which merely lands on the last sample.
        """
        rows = self.runs
        if isinstance(value, _FLOATS) and not math.isfinite(value):
            return len(self) if value > 0 else -1
        anchor = self._bound_tick(value, forward) if self._ticks else float(value)
        heads = self._run_heads
        if self._narrow is not None:
            # the labels are these heads rounded again into a narrower float
            heads = heads.astype(self.dtype).astype(np.float64)
        if len(rows) == 1:  # one run needs no search to be found
            run = 0 if anchor >= heads[0] else -1
        else:
            run = int(np.searchsorted(heads, anchor, side="right")) - 1
        if run < 0:
            # Before every run. A first run which is a grid still says
            # where the value would sit; stored labels can only say that
            # the value is before the first of them.
            if rows["den"][0] == 0:
                return -1
            run = 0
        _, length, _, den, _ = rows[run].item()[:5]
        if den == 0:
            piece = self._flat_labels[
                self._label_starts[run] : self._label_starts[run] + length
            ]
            side = "left" if forward else "right"
            index = int(np.searchsorted(piece, anchor, side=side))
            k = index if forward else index - 1
            if not forward and index == length and anchor > piece[-1]:
                # Past every label the run holds, which the caller reads
                # as a bound the coordinate does not reach.
                k = length
        else:
            k = self._kernel.index_of(rows[run].item(), anchor, forward, self._narrow)
        base = 0 if run == 0 else int(self._sample_starts[run])
        k = int(k)
        grid = den != 0
        first, last = run == 0, run == len(rows) - 1
        if forward:
            if k >= length:  # past this run
                return base + k if last else int(self._sample_starts[run + 1])
            if k < 0:  # in the space before this run
                return base + k if (first and grid) else base
            return base + k
        if k >= length:
            return base + k if last else base + length - 1
        if k < 0:
            return base + k if (first and grid) else base - 1
        return base + k

    def _index_one(self, value, forward: bool):
        """The index one value maps to, on a table of either direction."""
        if self.reverse_sorted:
            index = self._reverse_view._index_sorted(value, not forward)
            return len(self) - 1 - index
        return self._index_sorted(value, forward)

    def _get_index(self, value, forward=True):
        """
        The index a query value maps to.

        Forward: the first index whose label is at or past the value in the
        coordinate's direction; else the last index whose label is at or
        before it. None when the value is null or lies past the open end.
        An array of values gives an array of indices, unclamped, as the
        range coordinate's does.
        """
        if (value := self._get_compatible_value(value)) is None:
            return None
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value[()]
        if not (self.sorted or self.reverse_sorted):
            msg = (
                "Lookup by value needs a sorted coordinate; select on an "
                "unsorted one matches its labels instead."
            )
            raise CoordError(msg)
        if isinstance(value, Sized):
            out = [self._index_one(x, forward) for x in np.atleast_1d(value)]
            return np.asarray(out, dtype=np.int64)
        out = self._index_one(value, forward)
        if (forward and out < 0) or (not forward and out >= len(self)):
            return None
        return out

    def _select_by_mask(self, args) -> tuple[BaseCoord, slice | ArrayLike]:
        """Select from an unsorted coordinate by matching its labels."""
        low, high = args
        values = self.values
        mask = np.ones(values.shape, dtype=bool)
        if low is not None:
            mask &= values >= low
        if high is not None:
            mask &= values <= high
        if not np.any(mask):
            return self.empty(), mask
        if np.all(mask):
            return self, slice(None, None)
        return get_coord(data=values[mask], units=self.units), mask

    def select(
        self, args, relative=False, samples=False
    ) -> tuple[BaseCoord, slice | ArrayLike]:
        """Select a value window; the result keeps only the runs inside it."""
        if is_array(args):
            return self._select_by_array(args, relative=relative, samples=samples)
        if samples:
            return self._select_by_samples(args)
        args = self._get_slice_tuple(args, relative=relative)
        if not self.size:  # nothing to keep, and no order to keep it by
            return self, slice(0, 0)
        if not (self.sorted or self.reverse_sorted):
            return self._select_by_mask(args)
        start = self._get_index(args[0], forward=self.sorted)
        stop = self._get_index(args[1], forward=self.reverse_sorted)
        if self.reverse_sorted:
            start, stop = stop, start
        start = None if start == 0 else start
        data = slice(start, (stop + 1) if stop is not None else stop)
        if self._slice_degenerate(data):
            return self.empty(), slice(0, 0)
        return self[data], data

    # --- identity

    def _id_components(self) -> tuple[Any, ...]:
        """The run table, ids of stored labels and all."""
        # Every stored run names its labels by a content id held in the
        # table, so hashing the table commits to the labels too and no
        # source array is read to say what this coordinate is.
        rows = self.runs
        table = rows.copy()
        single = rows["length"] == 1
        if np.any(single):
            # A run of one sample keeps a step no label shows, so its
            # spacing and phase are no part of what it is.
            table["start"][single] = self._run_heads[single]
            table["num"][single] = 0
            table["den"][single] = 1
            table["offset"][single] = 0
            table[SOURCE_ID][single] = b""
        components: tuple[Any, ...] = (
            str(np.dtype(self.dtype)),
            self.shape,
        )
        if not self._ticks and np.any(made := (rows["den"] != 0) & ~single):
            # A float row is not canonical -- one set of doubles can be
            # counted from more than one origin, and a slice keeps its
            # parent's -- so the labels it makes are its identity, and the
            # row itself says nothing.
            table["start"][made] = 0.0
            table["num"][made] = 0
            table["den"][made] = 1
            table["offset"][made] = 0
            labels = np.concatenate(
                [self._run_view(int(i)).values for i in np.flatnonzero(made)]
            )
            components += (("labels", hash_array(np.ascontiguousarray(labels))),)
        components += (hash_array(np.ascontiguousarray(table)),)
        # A grid the labels are declared to sit on, which they do not state
        # themselves, is part of what the coordinate is; how it was spelled
        # is not, so it is hashed as the coordinate's own scalar.
        if _scalar_step(self.runs, self.dtype) is None and not _is_null(self.step):
            components += (("step", self._hash_scalar(self.step, "step")),)
        return components

    def __eq__(self, other) -> bool:
        """
        Whether two coordinates hold the same labels in the same units.

        Equal tables are equal coordinates, which is the cheap answer. Two
        tables which differ may still hold the same labels -- a short run
        cannot show the phase of its grid, and one set of doubles can be
        counted from more than one origin -- so they are then compared label
        by label, exactly.
        [`approx_equal`](`dascore.core.coords.NumericND.approx_equal`) compares
        closely.
        """
        if not isinstance(other, NumericND):
            return False
        if self.shape != other.shape or self.units != other.units:
            return False
        if np.dtype(self.dtype) != np.dtype(other.dtype):
            return False
        if self.data_id == other.data_id:
            return True
        if not (self.size and self.ndim == 1):
            return False  # stored N-D labels are hashed as the labels they are
        # The two ends are cheap and settle nearly every unequal pair.
        ends = [0, len(self) - 1]
        if not _same_labels(self._labels(ends), other._labels(ends)):
            return False
        return _same_labels(self.values, other.values)

    __hash__ = BaseCoord.__hash__

    def approx_equal(self, other: BaseCoord) -> bool:
        """Whether two coordinates hold approximately the same labels."""
        if self is other:
            return True
        if self.shape != other.shape:
            return False
        if other._partial:
            return False
        if isinstance(other, NumericND) and np.array_equal(self.runs, other.runs):
            if not (self._stored_labels or other._stored_labels):
                return True
        return all_close(self.values, other.values)

    # --- updates

    def _translated(self, delta) -> Self:
        """Shift every label; the grids move with them."""
        rows = self.runs.copy()
        stored = self._stored_labels
        shape = None if self.ndim == 1 else self.shape
        if self._ticks:
            shift = _to_tick(delta)
            low, high = _tick_bounds(self.dtype)
            anchors = rows["start"]
            if stored:
                anchors = np.concatenate([anchors, self._flat_labels])
            # in python integers: a tick at the end of int64 and the bound
            # itself are one float64, which would let a wrap through
            if anchors.size:
                ends = [int(anchors.min()) + shift, int(anchors.max()) + shift]
                if min(ends) < low or max(ends) > high:
                    msg = f"Shifting by {delta} takes a label outside {self.dtype}."
                    raise CoordError(msg)
            rows["start"] += shift
            moved = self._from_anchor(self._flat_labels + shift) if stored else None
        else:
            rows["start"] += float(delta)
            moved = self._flat_labels + delta if stored else None
        if not stored:
            return self._build(self.dtype, rows, None, self.units, step=self.step)
        # Moved labels are new labels, so they become a source of their own.
        return self._from_labels(
            self.dtype, rows, moved, self.units, step=self.step, shape=shape
        )

    def _with_step(self, step) -> Self:
        """The same start and sample count on a new cadence."""
        assert self.evenly_sampled, "update_limits re-spaces only a single grid run"
        frac = _fraction_step(step)
        if frac is None:
            # A number beside a time coordinate is a duration in its units.
            step = get_compatible_values(step, type(self.step))
        whole = frac is None and (_is_int(step) or is_timedelta64(step))
        if self._ticks and not (frac is not None or whole):
            # A cadence of no whole ticks is no longer a tick grid; the
            # constructor picks the representation the new step needs.
            return cast(
                "Self",
                get_coord(
                    start=self.start, step=step, shape=self.shape, units=self.units
                ),
            )
        num, den = _step_terms(step, self.dtype)
        if self._ticks and 0 < abs(num) < den:
            # Only a ticked coordinate has a smallest spacing; a float one
            # is free to step by any fraction of its units.
            msg = "A step smaller than one tick would repeat labels."
            raise CoordError(msg)
        rows = _rows(self.dtype, self._run_heads, self.runs["length"], [num], [den], 0)
        return self._build(self.dtype, rows, None, self.units)

    @compose_docstring(doc=get_docstring(BaseCoord.change_length))
    def change_length(self, length: int) -> Self:
        """
        {doc}
        """
        length = _validate_new_length(length)
        if len(self) == length:
            return self
        if not self.evenly_sampled:
            msg = "Only a single grid run can change its length."
            raise NotImplementedError(msg)
        return self._sliced(0, 1, length)

    @compose_docstring(doc=get_docstring(BaseCoord.update_limits))
    def update_limits(self, min=None, max=None, step=None, **kwargs) -> BaseCoord:
        """{doc}."""
        if all(x is not None for x in [min, max, step]):
            msg = "At most two parameters can be specified in update_limits."
            raise ValueError(msg)
        given = sum(x is not None for x in (min, max, step))
        if not self.evenly_sampled and given > 1:
            # Only a grid can be re-spaced and re-anchored at once; a
            # coordinate holding its own labels takes one change at a time.
            if self.runs_count > 1:
                msg = "Cannot specify both min and max in update_limits."
                raise ParameterError(msg)
            msg = "At most one parameter can be specified in update_limits."
            raise ValueError(msg)
        if step is not None and not self.evenly_sampled and self.size:
            if self.runs_count > 1:
                msg = (
                    "Segmented coordinates have no single step; use fuse "
                    "or snap to get an evenly sampled coordinate first."
                )
                raise ParameterError(msg)
            # Stored labels have no grid to re-space, so one is fitted first.
            return self.snap().update_limits(step=step, **kwargs)
        out = self
        if min is not None and max is not None:
            # min is the new start, max the new exclusive stop, and the
            # count is kept; below a tick the labels become floats.
            min = get_compatible_values(min, self.dtype)
            max = get_compatible_values(max, self.dtype)
            span = _to_tick(max) - _to_tick(min) if self._ticks else 0
            if self._ticks and abs(frac := Fraction(span, len(self))) >= 1:
                rows = _rows(
                    self.dtype,
                    [_to_tick(min)],
                    [len(self)],
                    [frac.numerator],
                    [frac.denominator],
                    0,
                )
                out = self._build(self.dtype, rows, None, self.units)
            else:
                new_step = (max - min) / len(self)
                out = get_coord(start=min, stop=max, step=new_step, units=self.units)
            return out.new(**kwargs) if kwargs else out
        if step is not None:
            out = out._with_step(step)
        for bound, end in ((min, "min"), (max, "max")):
            if bound is None:
                continue
            try:
                shift = get_compatible_values(bound, self.dtype) - getattr(out, end)()
            except OverflowError:
                msg = f"{bound} is not a label a {self.dtype} coordinate can hold."
                raise CoordError(msg) from None
            out = out._translated(shift)
        return out.new(**kwargs) if kwargs else out

    def new(self, **kwargs) -> BaseCoord:
        """Update the coordinate; new labels or new range fields rebuild it."""
        data = kwargs.pop("data", None)
        data = kwargs.pop("values", None) if data is None else data
        units = kwargs.pop("units", None)
        step = kwargs.pop("step", None)
        units = self.units if units is None else units
        if data is not None:
            return get_coord(data=data, units=units, step=step)
        if (segments := kwargs.pop("segments", None)) is not None:
            return get_coord(segments=segments, units=units)
        fields = {x: kwargs.pop(x) for x in ("start", "stop", "shape") if x in kwargs}
        if fields or step is not None:
            # A range states itself in these, so a new one of them states a
            # new range; labels of their own have only their own step.
            if not self.evenly_sampled:
                return get_coord(data=self.values, units=units, step=step)
            info: dict[str, Any] = dict(
                start=self.start,
                stop=self.stop,
                shape=self.shape,
                units=units,
                step=self.step if step is None else step,
            )
            if self._exact and step is None:
                # The exact terms outrank the whole-tick step beside them.
                info.update(zip(_EXACT_GRID_FIELDS, self._grid_terms))
            info.update(fields)
            if "stop" in fields or step is not None:
                # A new end or cadence re-derives the count, as a range always has.
                info.pop("shape", None)
            return get_coord(**info)
        out = self.set_units(units)
        if kwargs:
            out = out.update_limits(**kwargs)
        return out

    def set_units(self, units) -> Self:
        """Set new units on the coordinate, leaving every label alone."""
        if units_match(self.units, units):
            return self
        shape = None if self.ndim == 1 else self.shape
        return self._build(
            self.dtype, self.runs, self.sources, units, step=self.step, shape=shape
        )

    def _convert_units(self, units) -> Self:
        """
        Convert units, or set units if none exist.

        Done a run at a time: a grid run converts its two ends and keeps
        its count, so only the labels a run actually stores are ever
        spelled out.
        """
        if dtype_time_like(self.dtype):  # time units are fixed
            return self
        if self.units is None or not self.size:
            return self.set_units(units)
        if self._ticks and np.any(self.runs["den"] > 1):
            msg = (
                "Cannot convert the units of an integer coordinate with a "
                f"fractional step ({self.step_exact}); the result is not an "
                "integer grid."
            )
            raise CoordError(msg)
        if self.runs_count > 1:
            return self._converted_table(units)
        if self.evenly_sampled:
            start = convert_units(self.start, to_units=units, from_units=self.units)
            stop = convert_units(self.stop, to_units=units, from_units=self.units)
            step = (stop - start) / len(self)
            return cast(
                "Self", get_coord(start=start, stop=stop, step=step, units=units)
            )
        values = convert_units(self.values, to_units=units, from_units=self.units)
        step = self.step
        if step is not None:
            # a step is a difference, so an affine unit's offset cancels
            anchor = convert_units(step * 0, units, self.units)
            step = convert_units(step, units, self.units) - anchor
        return self.from_array(values, units=units, step=step, detect=False)

    def _converted_table(self, units) -> Self:
        """
        Every run converted in one pass over the table's columns.

        A unit conversion is affine, so a label maps through it directly and
        a spacing -- a difference -- through its scale alone; taking the
        runs a column at a time rather than one coordinate at a time is
        what keeps a many-run conversion linear in more than name.
        """
        rows, old = self.runs, self.units
        grid = rows["den"] != 0
        anchors = self._from_anchor(self._run_heads.copy()).astype(np.float64)
        # Each grid run's exclusive end, so its new spacing comes from its
        # own two ends and its count -- the arithmetic a single run's
        # conversion does, done for every run at once.
        stops = self._kernel.labels(rows, np.arange(len(rows)), rows["length"])
        ends = np.where(grid, self._from_anchor(stops).astype(np.float64), anchors)
        moved = convert_units(
            np.concatenate([anchors, ends]), to_units=units, from_units=old
        )
        starts, stops = np.split(np.asarray(moved, dtype=np.float64), 2)
        spacings = np.zeros(len(rows), np.float64)
        spacings[grid] = (stops[grid] - starts[grid]) / rows["length"][grid]
        dtype = np.dtype(np.float64)
        new = float_rows(dtype, starts, rows["length"], spacings, grid.astype(np.int64))
        step = self.step
        if step is not None:
            # a step is a difference, so an affine unit's offset cancels
            zero = convert_units(step * 0, units, old)
            step = convert_units(step, units, old) - zero
        shape = None if self.ndim == 1 else self.shape
        if not self._stored_labels:
            return self._build(dtype, new, None, units, step=step, shape=shape)
        # Converted labels are other labels, so they become their own source.
        labels = convert_units(
            self._from_anchor(self._flat_labels), to_units=units, from_units=old
        )
        return self._from_labels(dtype, new, labels, units, step=step, shape=shape)

    def coord_range(self, extend: bool = True):
        """The span of the coordinate; extended, to its exclusive end."""
        if not extend:
            return self.max() - self.min()
        if not self.evenly_sampled:
            msg = (
                "If extend is True, the coord_range can only be called on "
                f"evenly sampled coordinates but {self} is not."
            )
            raise CoordError(msg)
        count = int(self.runs["length"][0])
        ends = self._kernel.labels(self.runs, np.zeros(2, np.int64), [0, count])
        return np.abs(self._from_anchor(ends)[1] - self._from_anchor(ends)[0])

    def _samples_in(self, duration):
        """How many steps a duration spans, unrounded."""
        if not self.evenly_sampled or not self._ticks:
            return super()._samples_in(duration)
        if dtype_time_like(self.dtype):
            ticks = Fraction(int(to_int(dc.to_timedelta64(duration))))
        else:
            ticks = Fraction(_maybe_unpack(np.asarray(duration)).item())
        row = self.runs[0]
        return ticks / abs(Fraction(int(row["num"]), int(row["den"])))

    def _out_of_bounds_indices(self, array) -> np.ndarray:
        """Positions past either end, from the grid."""
        if not self.evenly_sampled:
            return super()._out_of_bounds_indices(array)
        last = self._from_anchor(self._run_ends[-1:])[0][()]
        beyond = (array > last) if self.sorted else (array < last)
        return np.where(
            beyond, self._get_index(array, forward=False), self._get_index(array)
        ).astype(np.int64)

    # --- gaps, snapping, and simplification

    @staticmethod
    def _expected_step(seg) -> Any:
        """The expected next-sample spacing after a run, or None."""
        if not _is_null(seg.step):
            return seg.step
        if len(seg) > 1:
            values = seg.values
            return values[-1] - values[-2]
        return None

    def _expected_spacing(self):
        """The spacing a stored run is held to: its declared step, else the median."""
        if not _is_null(self.step):
            return self.step
        diffs = _diffs(self.values)
        # the median magnitude, signed with the labels' direction, so
        # either orientation judges the same spacings
        median = np.median(np.abs(diffs))
        return median if self.sorted else -median

    def _stored_seams(self) -> list[tuple]:
        """Every spacing of a lone stored run which is not the expected one."""
        if not (self.sorted or self.reverse_sorted):
            # Labels in no order state no expected spacing, so every one of
            # them would read as a seam; there is nothing to report.
            return []
        values = self.values
        if len(values) < 2:
            return []
        diffs = _diffs(values)
        expected = self._expected_spacing()
        seams = np.flatnonzero(diffs != expected)
        return [(int(i) + 1, values[i], values[i + 1], expected) for i in seams]

    def _seams(self) -> list[tuple]:
        """One row per seam between runs, expecting the run's own step after it."""
        rows = self.runs
        if len(rows) < 2:
            # A single stored run states no runs of its own, so its
            # spacings are read the way an array coordinate's were.
            stored = bool(len(rows)) and rows["den"][0] == 0 and self.ndim == 1
            return self._stored_seams() if stored else []
        offsets = self._sample_starts
        ends = self._from_anchor(self._run_ends)
        starts = self._from_anchor(self._run_heads.copy())
        out = []
        for num in range(1, len(rows)):
            before, after = ends[num - 1], starts[num]
            expected = self._expected_step(self._run_view(num - 1))
            out.append((int(offsets[num]), before, after, expected))
        return out

    def _steps_between(self, index: int, before, after) -> int:
        """How many grid positions of run ``index`` separate two of its labels."""
        row = self.runs[index].item()
        if self._ticks and row[3] > 1:
            # A fractional tick step does not divide a long span exactly:
            # dividing by the whole ticks it rounds to disagrees by a
            # sample after a few hours. The run's own rational terms count
            # the positions between two of its labels without rounding,
            # and in python integers, which cannot overflow.
            forward = self._kernel.index_of
            return forward(row, _to_tick(after), True) - forward(
                row, _to_tick(before), True
            )
        return int(_on_grid(np.asarray([after - before]), self.step)[0])

    def _holes(self) -> list[tuple]:
        """Each hole as ``(first missing label, last missing label, count)``."""
        step = self.step
        assert not _is_null(step), "missing() asks only a coordinate with a step"
        rows = []
        # Both columns are the whole table's, so they are taken once rather
        # than rebuilt on every trip round the runs.
        ends = self._from_anchor(self._run_ends)
        starts = self._from_anchor(self._run_heads.copy())
        for index, run in enumerate(self.segments):
            if index:
                before, after = ends[index - 1], starts[index]
                count = self._steps_between(index - 1, before, after) - 1
                if count:
                    rows.append(_hole(before, step, count))
            if run.runs["den"][0] == 0 and len(run) > 1:
                values = run.values
                counts = _on_grid(_diffs(values), step)
                rows.extend(
                    _hole(values[i], step, int(counts[i]) - 1)
                    for i in np.flatnonzero(counts > 1)
                )
        return rows

    @compose_docstring(doc=get_docstring(BaseCoord.snap))
    def snap(self) -> BaseCoord:
        """
        {doc}

        Notes
        -----
        The min/max of the coordinate remain unchanged; every interior label
        may move without bound. [`fuse`](`dascore.core.coords.BaseCoord.fuse`)
        re-fits a coordinate one run at a time instead, keeping the seams a
        tolerance does not cover.
        """
        if self.evenly_sampled or not self.size:
            return self
        if np.dtype(self.dtype).itemsize > 8:
            # a grid is counted in float64, which a wider float's labels
            # are not, so fitting one would move every one of them
            return self
        return self._snapped()

    def _snapped(self) -> BaseCoord:
        """The even grid between this coordinate's two ends."""
        one = self._get_compatible_value(1) - self._get_compatible_value(0)
        return _even_grid(
            self.min(), self.max(), len(self), self.reverse_sorted, self.units, one
        )

    def fuse(self, tolerance=None, keep_step: bool = False) -> BaseCoord:
        """
        Return the simplest coordinate representing the same values.

        Runs are greedily re-fit as evenly sampled grids; a fit is accepted
        only when no value moves by more than `tolerance`. With a
        sufficient tolerance a fully contiguous table collapses to one run.

        Parameters
        ----------
        tolerance
            The maximum amount any coordinate value may change. For
            time-like coordinates this is a timedelta (numeric values
            interpreted as seconds). None or 0 permit only exact
            simplifications.
        keep_step
            If True, a re-fit may not change the runs' declared step, so a
            hole stays a hole however large the tolerance.
        """
        if len(self.runs) < 2:
            # One run is already the simplest thing its labels can be: a
            # grid states itself, and stored labels have no seam to close.
            return self
        if not (self.sorted or self.reverse_sorted):
            return self
        tol = self._get_tolerance(tolerance)
        segments = self.segments
        result: list[NumericND] = []
        run = [segments[0]]
        run_fit = self._fit_run(run, tol, keep_step)
        for seg in segments[1:]:
            trial = [*run, seg]
            fit = self._fit_run(trial, tol, keep_step)
            if fit is not None:
                run, run_fit = trial, fit
            else:
                result.append(run_fit if run_fit is not None else run[0])
                run, run_fit = [seg], self._fit_run([seg], tol, keep_step)
        result.append(run_fit if run_fit is not None else run[0])
        return concat_coords(*result) if len(result) > 1 else result[0]

    def _get_tolerance(self, tolerance):
        """
        Coerce the tolerance to the dtype expected for value deviations.

        None is no bound at all, which is what an infinite count asks
        for: a finite excess cannot express it, and multiplying infinity
        by a step gives NaT rather than a bound to compare against.
        """
        if isinstance(tolerance, GapTolerance) and tolerance.count is not None:
            if not np.isfinite(tolerance.count):
                return None
            steps = [abs(x.step) for x in self.segments if not _is_null(x.step)]
            tolerance = tolerance.count * get_middle_value(steps) if steps else 0
        return self._gap_tolerance(tolerance).excess

    def _fit_run(self, run, tol, keep_step: bool = False) -> NumericND | None:
        """Fit a run of table runs to a single grid within tol, or None."""
        if len(run) == 1 and run[0].evenly_sampled:
            return run[0]
        n = sum(len(x) for x in run)
        if n < 2:
            return None
        ascending = self.sorted
        # The grid between the run's two ends is what snapping these same
        # labels gives, so it is taken from there rather than spelled out
        # again; the labels are concatenated either way.
        actual = np.concatenate([x.values for x in run])
        one = self._get_compatible_value(1) - self._get_compatible_value(0)
        low, high = np.min(actual), np.max(actual)
        candidate = _even_grid(low, high, n, not ascending, self.units, one)
        deviation = np.max(np.abs(candidate.values - actual))
        too_far = tol is not None and deviation > tol
        if too_far or (keep_step and not _keeps_step(run, ascending)):
            return None
        return cast("NumericND", candidate)

    # --- summary and display

    def to_summary(self, dims=()) -> CoordSummary:
        """Get the summary info about the coord, the grid it states included."""
        summary = super().to_summary(dims=dims)
        if self.runs_count > 1:
            if self.runs_count > _MAX_SUMMARY_RUNS:
                return summary
            runs = tuple(x.to_summary(dims=dims) for x in self.segments)
            return summary.model_copy(update={"runs": runs})
        row = self.runs[0]
        if not (self._ticks and row["den"]):
            # Only a grid of whole ticks states terms a summary can hold; a
            # float run's phase and stride have no field of their own, so it
            # travels as the envelope and step every summary carries.
            return summary
        return summary.model_copy(
            update=dict(
                step_numerator=int(row["num"]),
                step_denominator=int(row["den"]),
                origin_offset=int(row["offset"]),
            )
        )

    @property
    def _rich_style(self) -> str:
        """The colour a coordinate is drawn in, from the shape it has."""
        if self.runs_count > 1:
            return dascore_styles["coord_segmented"]
        if self.evenly_sampled:
            return dascore_styles["coord_range"]
        if self.ndim == 1 and (self.sorted or self.reverse_sorted):
            return dascore_styles["coord_monotonic"]
        return dascore_styles["coord_array"]

    def _repr_fields(self) -> tuple[tuple[str, Text, bool], ...]:
        """The facts of the table; an empty one has only its shape to state."""
        if not self.size:
            return (
                ("shape", get_nice_text(self.shape), True),
                ("dtype", get_nice_text(self.dtype), True),
            )
        fields = super()._repr_fields()
        rows = self.runs
        if self._exact and rows["den"][0] != 1:
            # The rounded step misstates a fractional grid, so it is exact.
            exact = cast("Fraction", self.step_exact)
            text = Text(f"{exact}")
            if dtype_time_like(self.dtype):
                rate = 1 / abs(exact)
                rate_str = str(rate) if rate.denominator == 1 else f"{float(rate):g}"
                text += Text(" s") + Text(f" ({rate_str} Hz)", dascore_styles["units"])
            elif self._unit_str:
                text += Text(f" {self._unit_str}", dascore_styles["units"])
            fields = tuple(
                ("step", text, True) if name == "step" else (name, value, labelled)
                for name, value, labelled in fields
            )
        if len(rows) == 1:
            return fields
        runs = ("runs", get_nice_text(len(rows)), True)
        return (*fields[:-2], runs, *fields[-2:])


# Built once; a union spelled inside the test below makes a type object
# on every constructor call.
_PY_DATE_TYPES = (datetime.datetime, datetime.date)


def _even_grid(low, high, count: int, reverse: bool, units, one) -> BaseCoord:
    """
    The evenly sampled coordinate of ``count`` labels between two ends.

    ``one`` is what a step of one looks like in the labels' own type, which
    is the spacing a single label is given.
    """
    if count == 1:
        step = one
    else:
        span = high - low
        if is_timedelta64(span):
            ticks = float(span.astype(np.int64)) / (count - 1)
            step = np.timedelta64(int(np.round(ticks)), "ns")
        else:
            step = span / (count - 1)
    if reverse:
        step, start = -step, high
        stop = low + step
    else:
        start, stop = low, high + step
    # Through get_coord, so a spacing the labels' own dtype cannot hold --
    # a fractional step on an integer coordinate -- comes back as the float
    # coordinate which can.
    return get_coord(start=start, stop=stop, step=step, units=units).change_length(
        count
    )


def _numpy_time(value):
    """A python datetime or timedelta as the numpy scalar it names."""
    if isinstance(value, _PY_DATE_TYPES):
        return dc.to_datetime64(value)
    if isinstance(value, datetime.timedelta):
        return dc.to_timedelta64(value)
    return value


def _range_table(values: dict) -> NumericND:
    """The one-run table a start/stop/step/shape input describes."""
    # A python datetime states an instant the same way numpy does; read as
    # one here, it lands on the nanosecond grid rather than in an object
    # array which has no tick at all. See #467.
    values = {**values, **{x: _numpy_time(values.get(x)) for x in ("start", "stop")}}
    if isinstance(values.get("step"), datetime.timedelta):
        values["step"] = dc.to_timedelta64(values["step"])
    get = values.get
    units = get("units")
    dtype = _exact_dtype(get("start"), get("stop"), get("step"), get("shape"))
    if dtype is None:
        fields = _float_fields(values)
        return NumericND.from_run(
            fields["start"],
            fields["step"],
            fields["shape"],
            units=units,
            dtype=fields["dtype"],
        )
    fields = _exact_fields(values, dtype)
    rows = _rows(
        fields["dtype"],
        [_to_tick(fields["start"])],
        [fields["shape"][0]],
        [fields["step_numerator"]],
        [fields["step_denominator"]],
        [fields["origin_offset"]],
    )
    return NumericND._build(fields["dtype"], rows, None, units)


def _check_exact_grid(stated: bool, *values) -> None:
    """Refuse an exact grid stated beside labels which state their own."""
    if stated and any(x is not None for x in values):
        msg = (
            "step_numerator, step_denominator and origin_offset state the "
            "grid of a range; they cannot be combined with values, segments "
            "or runs."
        )
        raise CoordError(msg)


def concat_tables(*coords: NumericND) -> NumericND:
    """
    Join run tables end to end into one table.

    A run which continues the one before it on the same grid, phase
    included, is fused into it, so equal samples give an equal coordinate
    however they were assembled. Anything else starts a new run.

    Parameters
    ----------
    *coords
        The coordinates to join; all must share a dtype and units.
    """
    first, *rest = coords
    if any(x.dtype != first.dtype for x in rest):
        msg = f"Runs must share a dtype, got {[str(x.dtype) for x in coords]}."
        raise CoordError(msg)
    if any(get_quantity(x.units) != get_quantity(first.units) for x in rest):
        msg = "Runs must share units."
        raise CoordError(msg)
    rows = _fuse_float_neighbours(np.concatenate([x.runs for x in coords]), first.dtype)
    # Equal ids are equal arrays, so the union needs no comparison and the
    # rows keep pointing where they already pointed.
    sources = _union_sources(coords)
    # A declared grid survives only when every coordinate states it; a run
    # which states none says the labels may sit anywhere.
    steps = [x.step for x in coords]
    stated = all(not _is_null(x) for x in steps) and len({str(x) for x in steps}) == 1
    step = steps[0] if stated else None
    if step is not None and not _declares(rows, first.dtype, step):
        # The runs as given do not all sit on the grid they declare, and
        # fusing adjacent stored runs would hide the junction that says so.
        step = None
    return first._build(first.dtype, rows, sources, first.units, step=step)


def _promoted(tables: list[NumericND]) -> list[NumericND]:
    """The tables in the one dtype they all fit, refusing mixed kinds."""
    kinds = {np.dtype(x.dtype).kind for x in tables}
    if len(kinds) > 1:
        # Width promotion within one dtype kind is lossless (i4+i8, f4+f8);
        # mixing kinds (int64 + float64) can silently alter values above
        # 2**53, so it is rejected outright.
        dtypes = {np.dtype(x.dtype) for x in tables}
        msg = f"Segments must share compatible dtypes, got {dtypes}."
        raise CoordError(msg)
    units = {get_quantity(x.units) for x in tables}
    if len(units) > 1:
        msg = "All segments must have the same units."
        raise CoordError(msg)
    dtype = np.result_type(*[np.dtype(x.dtype) for x in tables])
    if all(np.dtype(x.dtype) == dtype for x in tables):
        return tables
    return [x if np.dtype(x.dtype) == dtype else _widened(x, dtype) for x in tables]


def _widened(coord: NumericND, dtype) -> NumericND:
    """
    One table in a wider dtype, holding the labels it held.

    A row counts in float64 and the coordinate rounds each label into its
    own dtype, so a narrower float's labels are not the ones its row makes
    once that rounding goes. The row travels only where it still makes
    them; where it does not, the labels do, and the declared step stays with
    the row -- the promoted labels are a narrower float's roundings, which no
    longer sit on that step exactly.
    """
    out = NumericND.from_rows(
        coord.runs,
        sources=coord.sources,
        dtype=dtype,
        units=coord.units,
        step=coord.step,
    )
    if _same_labels(out.values, coord.values.astype(dtype)):
        return out
    return NumericND.from_array(
        coord.values.astype(dtype), units=coord.units, detect=False
    )


def _ordered(tables: list[NumericND]) -> list[NumericND]:
    """The tables in label order, refusing a mixed direction or an overlap."""
    multi = [x for x in tables if len(x) > 1]
    ascending = multi[0].sorted if multi else True
    for coord in multi:
        if not (coord.sorted if ascending else coord.reverse_sorted):
            msg = "All segments must be sorted in a consistent direction."
            raise CoordError(msg)
    # Sort on native values; float conversion would collapse ns datetimes.
    out = sorted(tables, key=lambda x: x.min(), reverse=not ascending)
    for prev, nxt in itertools.pairwise(out):
        good = nxt.min() > prev.max() if ascending else nxt.max() < prev.min()
        if not good:
            msg = (
                "Segments must be monotonic and non-overlapping; segment "
                f"({nxt.min()}, {nxt.max()}) overlaps or precedes "
                f"({prev.min()}, {prev.max()})."
            )
            raise CoordError(msg)
    return out


def concat_coords(*coords, units=None) -> BaseCoord:
    """
    Concatenate monotonic coordinates into a single coordinate.

    This operation is truth-preserving: no value is ever altered, and every
    boundary between inputs that does not continue exactly becomes a run of
    its own. Inputs which continue each other exactly fuse back into one
    run. Use [`fuse`](`dascore.core.coords.BaseCoord.fuse`) on the
    result for tolerance-bounded gap absorption.

    Parameters
    ----------
    *coords
        Coordinates to concatenate. Each must be a numeric coordinate
        ([`NumericND`](`dascore.core.coords.NumericND`)); empty ones are
        dropped. Inputs are ordered by their envelopes; they must share
        dtype kind, units, and sort direction, and must not overlap.
    units
        If provided, set (not convert) these units on the output.

    Examples
    --------
    >>> from dascore.core.coords import concat_coords, get_coord
    >>>
    >>> c1 = get_coord(start=0.0, stop=10.0, step=1.0)
    >>> c2 = get_coord(start=15.0, stop=25.0, step=1.0)
    >>> coord = concat_coords(c1, c2)
    >>> assert len(coord) == len(c1) + len(c2)
    """
    tables: list[NumericND] = []
    for coord in coords:
        if isinstance(coord, Mapping):  # model_dump round-trip payloads
            coord = get_coord(**coord)
        if not isinstance(coord, BaseCoord):
            msg = f"concat_coords requires coordinates, got {type(coord)}."
            raise CoordError(msg)
        if isinstance(coord, NumericND):
            if len(coord):
                tables.append(coord)
        elif coord.ndim and coord.size:
            msg = (
                "concat_coords only supports evenly sampled, monotonic, or "
                f"segmented coordinates, got {type(coord)}."
            )
            raise CoordError(msg)
    if units is not None:
        tables = [x.set_units(units) for x in tables]
    if not tables:
        msg = "concat_coords requires at least one non-empty coordinate."
        raise CoordError(msg)
    tables = _promoted(tables)
    if len(tables) == 1:
        return tables[0]
    return concat_tables(*_ordered(tables))


def _grid_pieces(coord: BaseCoord) -> list[tuple[int, NumericND]]:
    """
    The runs of consecutive grid positions, each with its source offset.

    A range is one run; an array declaring a step splits at its holes, and
    a lone sample without a step takes the other runs' step.
    """
    segments = coord.segments if isinstance(coord, NumericND) else (coord,)
    steps = [x.step for x in segments if not _is_null(x.step)]
    pieces, offset = [], 0
    for seg in segments:
        step = seg.step
        if _is_null(step) and len(seg) == 1 and steps:
            step = steps[0]
        if seg.evenly_sampled:
            pieces.append((offset, seg))
        elif isinstance(seg, NumericND) and not _is_null(step):
            values = seg.values
            counts = _on_grid(_diffs(values), step)
            edges = np.flatnonzero(counts != 1) + 1
            for first, stop in itertools.pairwise([0, *edges.tolist(), len(values)]):
                piece = get_coord(
                    start=values[first],
                    step=step,
                    shape=(stop - first,),
                    units=seg.units,
                )
                # floats pass _on_grid per spacing; the run must not drift
                drift = np.abs(piece.values - values[first:stop])
                if values.dtype.kind == "f" and np.max(drift) > abs(step) / 2:
                    msg = (
                        f"Values drift more than half a step from the grid of "
                        f"step {step}; use snap_coords before filling gaps."
                    )
                    raise CoordError(msg)
                pieces.append((offset + first, piece))
        else:
            msg = (
                "Filling gaps needs a coordinate with a declared step; this one "
                "has none. Use snap_coords or resample to put it on a grid first."
            )
            raise CoordError(msg)
        offset += len(seg)
    return pieces


def _same_step(first: NumericND, exact, other: NumericND) -> bool:
    """
    Whether a run shares the first run's step, `exact` being its exact form.

    Exactly for ticks; for floats, closely enough that the run drifts from
    the first run's grid by a negligible fraction of a step.
    """
    if first._exact and (other_exact := other.step_exact) is not None:
        return exact == other_exact
    ratio = float(other.step) / float(first.step)
    return bool(abs(ratio - 1) * max(len(other) - 1, 1) <= _GRID_RTOL)


def _grid_position(anchor: NumericND, label) -> int:
    """The position on the anchor's grid nearest a label."""
    if not anchor._exact:
        return int(np.round((label - anchor.start) / anchor.step))
    num, den, offset = anchor._grid_terms
    tick, start = _to_tick(label), anchor._start_tick
    after = (den * (tick - start) - offset) // num + 1
    # the labels either side as the integer ticks _labels casts to dtype,
    # so the comparison stays in Python integers and cannot wrap
    ticks = [start + (offset + pos * num) // den for pos in (after - 1, after)]
    return after - 1 if abs(ticks[0] - tick) <= abs(ticks[1] - tick) else after


def _max_missing(step: NumericND, coord: BaseCoord, limit, samples: bool):
    """The most missing positions a filled hole may have, or None for any."""
    if limit is None:
        return None
    if samples:
        if not _is_int(limit) or limit < 0:
            msg = f"A sample limit must be a non-negative integer, got {limit!r}."
            raise ParameterError(msg)
        return int(limit)
    tolerance = coord._gap_tolerance(limit)
    if tolerance.count is not None:
        msg = "Pass a count of missing samples with samples=True instead."
        raise ParameterError(msg)
    excess = tolerance.excess
    if step._exact and (exact := step.step_exact) is not None:
        if is_timedelta64(excess):
            # the limit was rounded to whole nanoseconds; allow that rounding
            excess = Fraction(2 * int(to_int(excess)) + 1, 2 * _NS_PER_S)
        return int(Fraction(excess) // abs(exact))
    return math.floor(float(excess) / abs(float(step.step)) * (1 + _GRID_RTOL))


def _fill_layout(
    coord: BaseCoord, limit=None, samples: bool = False
) -> tuple[BaseCoord, tuple[tuple[int, int, int], ...]] | None:
    """
    Place every run of a coordinate on one grid, filling the holes between.

    Returns the filled coordinate and, per run, its ``(source start, source
    stop, target start)``, or None when there is nothing to fill. `limit`
    and `samples` are read as in `Patch.fill_gaps`; holes past the limit
    stay as seams. An off-grid run moves to the nearest position, and two
    runs on one position raise.
    """
    if coord.evenly_sampled or len(coord) < 2:
        return None
    pieces = _grid_pieces(coord)
    first = pieces[0][1]
    first_exact = first.step_exact
    for _, piece in pieces[1:]:
        if not _same_step(first, first_exact, piece):
            msg = (
                f"Runs are sampled at different steps ({first.step} and "
                f"{piece.step}); resample them to one step before filling gaps."
            )
            raise CoordError(msg)
    max_missing = _max_missing(first, coord, limit, samples)
    # each group: its anchor run, its filled length, and its runs' blocks
    groups: list[list] = []
    for source, piece in pieces:
        stop = source + len(piece)
        if groups:
            anchor, length, blocks = groups[-1]
            position = _grid_position(anchor, piece.start)
            missing = position - length
            if missing < 0:
                msg = (
                    f"Samples near {piece.start} land on the same grid position "
                    f"as the ones before them, so the gap cannot be filled."
                )
                raise CoordError(msg)
            if max_missing is None or missing <= max_missing:
                blocks.append((source, stop, position))
                groups[-1][1] = position + len(piece)
                continue
        groups.append([piece, len(piece), [(source, stop, 0)]])
    if all(len(blocks) == 1 for *_, blocks in groups):
        return None
    coords, out, offset = [], [], 0
    for anchor, length, blocks in groups:
        if len(blocks) == 1:  # untouched: keep the labels as they are
            coords.append(coord[blocks[0][0] : blocks[0][1]])
        else:
            coords.append(anchor.change_length(length))
        out.extend((start, stop, offset + pos) for start, stop, pos in blocks)
        offset += length
    # a lone group is a range, which concat_coords returns unchanged
    new = concat_coords(*coords)
    return new, tuple(out)


def _get_coord_kind(
    data: ArrayLike | None = None,
    *,
    dtype=None,
    step=None,
    length: int | None = None,
    is_string: bool | None = None,
) -> CoordKind:
    """Return the shared internal coord-kind description."""
    if data is not None:
        if _is_text_coercible_array(data):
            return "string"
        size = int(np.size(data))
        if size == 0:
            return "empty"
        if size == 1:
            return "single"
        return "array"
    if length == 0:
        return "empty"
    # Metadata-only classification cannot prove object arrays are string-like;
    # callers must pass is_string=True explicitly for that case.
    if is_string is None and dtype not in (None, ""):
        is_string = np.dtype(dtype).kind in {"U", "S"}
    if is_string:
        return "string"
    if step is not None:
        return "range"
    return "array"


def _raise_string_coord_error(operation: str) -> NoReturn:
    """Raise a consistent error for unsupported string coord operations."""
    msg = f"String coordinates do not support {operation}."
    raise CoordError(msg)


class CoordString(BaseCoord):
    """A coordinate implementation for string/categorical values.

    See ['Coordinate Internals'](`dascore/docs/notes/coordinate_internals.qmd`) for the
    constraints that make string coords differ from numeric and time-like
    coords. Plain string selectors use exact matching unless they contain `*`,
    `?` or `[`, in which case they are read as globs, as SQLite reads them.
    Compiled regular expressions are also supported as explicit pattern
    selectors.
    """

    values: ArrayLike
    _style_name: ClassVar[str] = "coord_array"

    @model_validator(mode="before")
    @classmethod
    def _validate_values(cls, values):
        """Normalize inputs to a string/bytes numpy array."""
        # Pydantic's "before" validators usually receive the raw model payload
        # as a dict during normal construction, e.g. CoordString(values=...).
        # If some other shape is passed through, leave it alone and let the
        # standard model validation path decide whether it is acceptable.
        if not isinstance(values, dict):
            return values
        try:
            data = _coerce_text_array(values.get("values"))
        except ValueError as exc:
            raise CoordError(str(exc)) from exc
        # Keep CoordString internally unicode-backed and derive the remaining
        # model fields from the normalized array rather than trusting callers.
        values["values"] = data
        values["shape"] = data.shape
        values["dtype"] = data.dtype
        # String coordinates deliberately do not participate in unit or step-
        # based coord behavior, so force those fields to the null form here.
        values["units"] = None
        values["step"] = None
        return values

    def _convert_units(self, units) -> Self:
        """
        String coordinates cannot be converted between units.

        A request for no units is answered by the caller's guard, so
        anything reaching here is asking for real ones.
        """
        _raise_string_coord_error("unit conversion")

    def set_units(self, units) -> Self:
        """Reject setting units on string coordinates."""
        return self.convert_units(units)

    def _get_compatible_value(self, value, relative=False):
        """Normalize selectors without truncating longer string probes."""
        if relative:
            _raise_string_coord_error("relative selection")
        if not is_array(value) and (pd.isnull(value) or value is Ellipsis):
            return None
        if is_array(value):
            return np.asarray(value)
        return value

    def select(
        self, args, relative=False, samples=False
    ) -> tuple[BaseCoord, slice | ArrayLike]:
        """Select by exact values, wildcard patterns, regexes, samples, or masks."""
        if relative:
            _raise_string_coord_error("relative selection")
        if args is None:
            return self, slice(None)
        if is_array(args):
            return self._select_by_array(args, samples=samples)
        if samples:
            return self._select_by_samples(args)
        if isinstance(args, slice | tuple):
            _raise_string_coord_error("range selection")
        if isinstance(args, re.Pattern):
            mask = np.array([bool(args.search(value)) for value in self.values])
            return self._select_by_value_array(self.values[mask])
        if isinstance(args, str) and any(char in args for char in "*?["):
            pattern = glob_to_regex(args)
            mask = np.array([bool(pattern.match(value)) for value in self.values])
            return self._select_by_value_array(self.values[mask])
        values = np.asarray([args])
        return self._select_by_value_array(values)

    def coord_range(self, extend: bool = True):
        """String coordinates do not support range calculations."""
        _raise_string_coord_error("range operations")

    def sort(self, reverse=False):
        """Sort values lexicographically."""
        inds = np.argsort(self.values)
        if reverse:
            inds = inds[::-1]
        return self[inds], inds

    @property
    def sorted(self) -> bool:
        """Return True when values are lexicographically nondecreasing."""
        values = np.asarray(self.values).reshape(-1)
        if len(values) <= 1:
            return True
        return bool(np.all(values[:-1] <= values[1:]))

    @property
    def reverse_sorted(self) -> bool:
        """Return True when values are lexicographically nonincreasing."""
        values = np.asarray(self.values).reshape(-1)
        if len(values) <= 1:
            return False
        return bool(np.all(values[:-1] >= values[1:]))

    def update_limits(self, min=None, max=None, step=None, **kwargs) -> BaseCoord:
        """Reject numeric limit updates on string coords."""
        # Deliberately match BaseCoord's parameter names for API parity.
        unsupported_kwargs = set(kwargs) - {"data"}
        if any(value is not None for value in (min, max, step)) or unsupported_kwargs:
            _raise_string_coord_error("limit updates")
        return self

    def to_summary(self, dims=()) -> CoordSummary:
        """Return a lossy summary for string coordinates."""
        return CoordSummary(
            min=self.min(),
            max=self.max(),
            step=None,
            dtype=self.dtype,
            units=None,
            dims=dims,
            len=self.shape[0] if self.ndim else 1,
            data_id=self.data_id,
        )

    def _id_components(self) -> tuple[Any, ...]:
        """Return the array payload identifying string coords."""
        return (("array", hash_array(self.values)),)

    def __getitem__(self, item) -> Self:
        """Return a subset of the coordinate."""
        out = self.values[item]
        if np.ndim(out) == 0:
            return out
        return self.new(values=np.asarray(out, dtype=self.dtype))

    def _min(self):
        """Return lexicographic minimum."""
        values = np.asarray(self.values).reshape(-1)
        return None if not values.size else min(values)

    def _max(self):
        """Return lexicographic maximum."""
        values = np.asarray(self.values).reshape(-1)
        return None if not values.size else max(values)


def get_coord(
    *,
    # An int names a length, producing a partial coord of that shape.
    # Sequence is spelled out because ArrayLike does not cover a plain
    # list, which is accepted here and used throughout the tests.
    data: ArrayLike | np.ndarray | BaseCoord | Sequence | int | None = None,
    values: ArrayLike | np.ndarray | None = None,
    start=None,
    min=None,
    stop=None,
    max=None,
    step=None,
    units: Unit | Quantity | str | None = None,
    shape: int | tuple[int, ...] | None = None,
    dtype: str | np.dtype | None = None,
    segments: tuple[BaseCoord, ...] | list[BaseCoord] | None = None,
    runs=None,
    labels=None,
    sources=None,
    step_numerator: int | None = None,
    step_denominator: int | None = None,
    origin_offset: int | None = None,
) -> BaseCoord:
    """
    Return a coordinate from provided inputs.

    Parameters
    ----------
    data
        An array of labels, preserved exactly, or an integer specifying the
        length of a partial coordinate. A grid is inferred only when it
        reproduces every label; otherwise the labels are stored.
    values
        Alias for data.
    start
        The start value of the array, inclusive.
    min
        The minimum value, same as start.
    stop
        The stopping value of an array, exclusive.
    max
        Alias for stop; exclusive, like stop.
    step
        The sampling spacing of an array.
    units
        Indication of units.
    shape
        If an int or tuple, the output should be a partial coord of with
        this shape. Otherwise, leave unset.
    dtype
        Data type for coord. Often can be inferred from other arguments.
    segments
        A sequence of monotonic coordinates to concatenate into one
        coordinate (see [`concat_coords`](`dascore.core.coords.concat_coords`)).
        Cannot be combined with other value inputs.
    runs
        A run table (a record array with the fields ``start``, ``length``,
        ``num``, ``den`` and ``offset``) to rebuild a
        [`NumericND`](`dascore.core.coords.NumericND`) from exactly, as a
        spool index row or a dumped coordinate states it. Cannot be
        combined with other value inputs.
    labels
        The labels of ``runs``' stored runs, concatenated in table order.
    sources
        The arrays ``runs``' stored runs read, by array id, as a dumped
        coordinate states them. Given instead of ``labels``, so a table
        which already names its sources keeps their ids.
    step_numerator, step_denominator, origin_offset
        The exact grid of an integer or time range in ticks (see
        [`NumericND`](`dascore.core.coords.NumericND`)). Normally
        these come from a dumped coordinate; pass ``step`` as a `Fraction`
        or ``(numerator, denominator)`` tuple to state a fractional step.

    Notes
    -----
    See ['Coordinate Internals'](`dascore/docs/notes/coordinate_internals.qmd`) for
    dispatch and coord-family design notes.

    The following combinations of input parameters are typical:
        (start, stop, step)
        (data)
        (data, step) - useful for length 1 arrays.
        (values)
        (values, step) - useful for length 1 arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.core import get_coord
    >>>
    >>> # Create a coordinate from a start, stop, and range value.
    >>> range_coord = get_coord(start=1, stop=12, step=1)
    >>>
    >>> # Create an identical coordinate from an array.
    >>> array_coord = get_coord(data=np.arange(1, 12, 1))
    >>> # This array coord should return an identical coordinate
    >>> assert range_coord == array_coord
    >>>
    >>> # Coordinate from an array that is sorted, but not evenly sampled
    >>> array = np.sort(np.random.rand(20))
    >>> array_coord2 = get_coord(data=array)
    >>>
    >>> # Coordinate from random array
    >>> array = np.random.rand(20)
    >>> array_coord3 = get_coord(data=array)
    >>>
    >>> # Create a partial coordinate of a given shape
    >>> partial_coord = get_coord(shape=(10,))
    >>>
    >>> # A time coordinate sampled at exactly 1024 Hz (a fraction of a
    >>> # nanosecond) never drifts from its grid.
    >>> time = get_coord(
    ...     start=np.datetime64("2020-01-01"), step=(1, 1024), shape=(2048,)
    ... )
    >>> assert time.step_exact == 1 / 1024 and time[::2].step_exact == 1 / 512
    """

    def _check_data_compatibility(data, start, stop, step):
        """Ensure input combinations are valid."""
        if data is None:
            if any([start is None, stop is None, step is None]):
                msg = "When data is not defined, start, stop, and step must be."
                raise CoordError(msg)

    def _get_shape(shape):
        """Return proper shape tuple or None."""
        if shape is None or isinstance(shape, Sized):
            return shape
        return (shape,)

    def _get_array(data, values):
        """Get the array from either data or values."""
        if data is not None and values is not None:
            msg = "Cannot specify both data and values. Use only data."
            raise CoordError(msg)
        elif values is not None:
            data = values
        return data

    _check_exact_grid(
        any(x is not None for x in (step_numerator, step_denominator, origin_offset)),
        data,
        values,
        segments,
        runs,
    )
    if runs is not None:
        others = (data, values, start, min, stop, max, segments)
        if any(x is not None for x in others):
            msg = "runs cannot be combined with other coordinate value inputs."
            raise CoordError(msg)
        return NumericND.from_rows(
            runs, labels=labels, dtype=dtype, units=units, step=step, sources=sources
        )
    if segments is not None:
        # shape/dtype/step are derived fields of a coordinate, so they
        # legitimately appear alongside segments when round-tripping a
        # model_dump (e.g. through CoordManager); ignore them here.
        others = (data, values, start, min, stop, max)
        if any(x is not None for x in others):
            msg = "segments cannot be combined with other coordinate value inputs."
            raise CoordError(msg)
        out = concat_coords(*segments, units=units)
        if not _is_null(step) and not _is_null(out.step) and step != out.step:
            msg = f"step {step} contradicts the segments' step {out.step}."
            raise CoordError(msg)
        return out

    data = _get_array(data, values)
    shape = _get_shape(shape)
    grid = dict(
        step_numerator=step_numerator,
        step_denominator=step_denominator,
        origin_offset=origin_offset,
    )
    if data is None and shape is not None:
        attrs = dict(
            shape=shape, start=start, stop=stop, step=step, units=units, dtype=dtype
        )
        # A 1D shape with two of start/stop/step states a range, and so
        # does a start with the exact grid beside it; anything less (or
        # more dimensions) is a partial coord. A range that then fails to
        # validate is an error, not a partial.
        stated = sum(not _is_null(x) for x in (start, stop, step))
        stated += step_numerator is not None
        if len(shape) != 1 or shape[0] == 0 or stated < 2:
            return CoordPartial(**attrs)
        # A time no nanosecond count can state is refused by name; it is
        # not a range this cannot describe, it is not a coordinate at all.
        for bound in (start, stop, step):
            if is_datetime64(bound) or is_timedelta64(bound):
                _to_tick(bound)
        try:
            return _range_table(
                dict(
                    shape=shape, start=start, stop=stop, step=step, units=units, **grid
                )
            )
        except CoordError:
            # A float range that cannot be built is still a partial, as it
            # always was; an exact grid raises only for what is wrong.
            if _exact_dtype(start, stop, step, shape) is not None:
                raise
            return CoordPartial(**attrs)

    # maybe convert min/max to start stop.
    if start is None and min is not None:
        start = min
    if stop is None and max is not None:
        stop = max
    _check_data_compatibility(data, start, stop, step)
    # data array was passed; see if it is monotonic/evenly sampled
    if data is not None:
        # Handle attached units.
        if isinstance(data, dc.units.Quantity):
            data, maybe_units = data.magnitude, data.units
            units = units if units is not None else maybe_units
        if isinstance(data, (int | np.integer)):
            shape = _get_shape(data)
            attrs = dict(
                shape=shape, start=start, stop=stop, step=step, units=units, dtype=dtype
            )
            return CoordPartial(**attrs)
        if isinstance(data, BaseCoord):  # just return coordinate
            return data
        if not isinstance(data, np.ndarray):
            data = np.atleast_1d(data)
        kind = _get_coord_kind(data)
        if kind == "string":
            if units not in (None, ""):
                _raise_string_coord_error("unit conversion")
            if step not in (None, "") and not pd.isnull(step):
                _raise_string_coord_error("range operations")
            return CoordString(values=data)
        if kind == "empty":
            dtype = dtype or data.dtype
            return CoordPartial(shape=data.shape, units=units, step=step, dtype=dtype)
        # special case of len 1 array either get range, if step specified
        # or sorted monotonic array if not.
        elif kind == "single":
            if not _is_null(step):
                return NumericND.from_array(data, units=units, step=step)
            return NumericND.from_array(data, units=units, detect=False)
        if not _is_null(step):
            # a declared step is a claim about the grid, so the values are
            # read against it exactly rather than fitted
            return NumericND.from_array(data, units=units, step=step)
        if np.all(pd.isnull(data)):
            # The values say nothing, but their type still does: an array of
            # NaT came from datetimes and should stay datetimes, as the
            # empty case above also keeps. Only a kind whose null the
            # values can actually hold is recorded; an object array of
            # Nones has no null but NaN, so claiming "object" would state
            # a dtype which `values` then contradicts.
            if dtype is None and data.dtype.kind in "fmM":
                dtype = data.dtype
            return CoordPartial(
                shape=data.shape,
                units=units,
                start=start,
                stop=stop,
                step=step,
                dtype=dtype,
            )
        return NumericND.from_array(data, units=units, fit=True)
    else:
        return _range_table(
            dict(start=start, stop=stop, step=step, units=units, **grid)
        )
