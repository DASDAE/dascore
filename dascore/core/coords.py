"""Machinery for coordinates.

See ['Coordinate Internals'](`dascore/docs/notes/coordinate_internals.qmd`) for the
current coord-family and string-coordinate design notes.
"""

from __future__ import annotations

import abc
import itertools
import math
import re
from collections.abc import Mapping, Sequence, Sized
from contextlib import suppress
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import cache
from types import EllipsisType
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NoReturn,
    Self,
    cast,
    overload,
)
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import (
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from rich.text import Text

import dascore as dc
from dascore.compat import array, is_array
from dascore.constants import _AGG_FUNCS, DIM_REDUCE_DOCS, dascore_styles
from dascore.exceptions import CoordError, ParameterError
from dascore.models import (
    ArrayLike,
    DascoreBaseModel,
    FrozenDictType,
    UnitQuantity,
    sensible_model_equals,
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
from dascore.utils.gaps import GapTolerance
from dascore.utils.identity import H
from dascore.utils.mapping import FrozenDict
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
    # Each run's summary, in order, for a segmented coordinate, so an index
    # can see holes inside a patch; None otherwise, including past
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
            data["dtype"] = str(dtype)
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
            object.__setattr__(self, "dtype", str(dtype))
        return self

    def to_coord(self) -> BaseCoord:
        """Convert to an evenly sampled coord, if possible."""
        if not self.is_range_like:
            msg = "Cannot convert summary which is not evenly sampled to coord."
            raise CoordError(msg)
        step = self.step
        assert step is not None  # is_range_like above rules out a null step
        if (num := self.step_numerator) is not None:
            if self.len is None:
                msg = "An exact grid summary needs its length to rebuild a coord."
                raise CoordError(msg)
            return get_coord(
                start=self.min if num >= 0 else self.max,
                shape=(self.len,),
                step_numerator=num,
                step_denominator=self.step_denominator or 1,
                origin_offset=self.origin_offset or 0,
                units=self.units,
            )
        # this is a reverse coord
        if np.sign(step) == -1:
            start, stop = self.max, self.min + step
        else:
            start, stop = self.min, self.max + step
        # The scalars are kept at nanoseconds; the coord takes the recorded unit.
        if self.dtype and (dtype := np.dtype(self.dtype)).kind in "mM":
            start, stop = (
                np.asarray(start).astype(dtype)[()],
                np.asarray(stop).astype(dtype)[()],
            )
            unit, count = np.datetime_data(dtype)
            step = np.asarray(step).astype(f"m8[{count}{unit}]")[()]
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

    _rich_style = dascore_styles["default_coord"]
    _evenly_sampled = False
    _sorted = False
    _reverse_sorted = False
    _partial = False

    @model_validator(mode="before")
    @classmethod
    def _check_time_units(cls, data: Any) -> Any:
        """Ensure time units are s if dtype is time-like."""
        # Every coord states its dtype in a before-validator of its own,
        # which pydantic runs first, so the dtype is the one thing here
        # that can speak for values this has not seen.
        if isinstance(data, dict) and dtype_time_like(data.get("dtype")):
            if data.get("units") != (quant := get_quantity("s")):
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
        stated = None if dtype_time_like(self.dtype) else self.unit_str

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
            self.unit_str,
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
    def unit_str(self) -> str | None:
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

    @property
    def degenerate(self) -> bool:
        """Return True of the coord is degenerate."""
        shape = self.shape
        return not len(shape) or np.prod(shape) == 0

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
        if not self.evenly_sampled and extend:
            msg = (
                "If extend is True, the coord_range can only be called on "
                f"evenly sampled coordinates but {self} is not."
            )
            raise CoordError(msg)
        coord_range = self.max() - self.min()
        if extend:
            # Handle reverse sorted case
            coord_range += np.abs(self.step)
        return coord_range

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

        Unlike [`snap`](`dascore.core.coords.BaseCoord.snap`), which forces a
        uniform coordinate with unbounded interior error, fuse never moves
        any value by more than `tolerance`.

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
        themselves. A coordinate holding several runs re-fits them as evenly
        sampled grids wherever the fit error stays within tolerance, possibly
        collapsing to a single grid.
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
        coordinate with no step -- an unordered array, or runs which share
        none -- names no grid to be missing from, so nothing is missing.

        Examples
        --------
        >>> from dascore.core.coords import get_coord
        >>> coord = get_coord(data=[1, 3, 4, 10, 11, 12], step=1)
        >>> coord.missing().count
        6
        >>> get_coord(start=0, stop=10, step=1).missing().complete
        True
        >>> assert get_coord(data=[1.0, 2.5, 7.0]).missing().complete
        """
        holes = () if _is_null(self.step) else tuple(self._holes())
        return Missing(step=self.step, runs=holes, dtype=self.dtype)

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
        For an evenly sampled coord stop will be max + step.
        """

    def new(self, **kwargs):
        """Update coordinate."""
        info = self.model_dump(exclude_unset=True, exclude_defaults=True)
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
        # Absolute bounds describe an unordered interval. Relative bounds
        # constrain its lower and upper ends; crossing them selects nothing.
        if not relative and p1 is not None and p2 is not None and p2 < p1:
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
            values = update_fields.pop("values", None)
            out = out.update_limits(**update_fields)
            if values is not None:
                out = get_coord(data=values, units=units)
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
        # Two coordinates describing the same grids hold the same values;
        # this avoids materializing and comparing the value arrays.
        mine, theirs = getattr(self, "runs", None), getattr(other, "runs", None)
        grids = all(isinstance(x, Grid) for x in (mine or ()) + (theirs or ()))
        if mine is not None and theirs is not None and grids and mine == theirs:
            return True
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
                result = func(coord_data)
            new_coord = self.update(values=result)
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
    _rich_style = dascore_styles["coord_non"]
    _partial = True

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
        bounds = {"min": min, "max": max, "step": step}
        passed = {i: v for i, v in bounds.items() if v is not None}
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
# Nanoseconds per second: the tick of every exact time grid.
_NS_PER_S = 10**9
# A float spacing this close to a whole number of steps is on the grid;
# a float grid such as 0.1 cannot be held exactly, an off-grid label can.
_GRID_RTOL = 1e-6
# The dense-array guard. Stored arrays often carry sub-step jitter
# (GPS-stamped DAS time), so run detection would give roughly one run per
# sample; at or past this many samples, an array whose runs would
# outnumber this fraction of them keeps its labels as one stored run.
_MIN_RUN_GUARD_SIZE = 1_000
_MAX_RUN_FRACTION = 0.1
# A coordinate's summary carries its runs only up to this many, since each
# becomes an index row; past it the summary is an envelope.
_MAX_SUMMARY_RUNS = 256


def _fraction_step(step) -> Fraction | None:
    """A step given as a Fraction or (numerator, denominator) tuple, else None."""
    if isinstance(step, tuple):
        return Fraction(*step)
    return step if isinstance(step, Fraction) else None


def _is_null(value) -> bool:
    """Return True for None or a null scalar (NaN, NaT)."""
    if value is None:
        return True
    if _fraction_step(value) is not None:
        return False
    return bool(pd.isnull(_maybe_unpack(value)))


def _is_int(value) -> bool:
    """Return True for a python or numpy integer (not a bool)."""
    value = _maybe_unpack(value)
    return isinstance(value, int | np.integer) and not isinstance(
        value, bool | np.bool_
    )


def _declared_step(step, dtype):
    """
    A step declared on stored labels, as the scalar the labels are measured in.

    A fraction may only be whole (labels on a fractional grid are floors of
    ideal positions, so they do not state it) and becomes seconds for time
    and an integer otherwise; a zero or non-finite step is no grid.
    """
    if (fraction := _fraction_step(step)) is not None:
        if fraction.denominator != 1:
            msg = (
                f"A fractional step ({fraction}) cannot be declared on stored "
                "labels; build the range with start, step, and shape instead."
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


def _to_tick(value) -> int:
    """Return a time value as nanoseconds, or an integer value as itself."""
    value = _maybe_unpack(value)
    if is_datetime64(value) or is_timedelta64(value):
        return int(to_int(value))
    if isinstance(value, float | np.floating):
        if not float(value).is_integer():
            msg = f"An integer coordinate cannot hold the non-integer value {value}."
            raise CoordError(msg)
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        msg = f"{value!r} is not an integer or time value."
        raise CoordError(msg) from e


def _negate_for_search(values):
    """
    Negate values so descending labels can use ascending searchsorted.

    Exactness matters: converting ns-precision datetimes (or large ints) to
    float collapses nearby values, so time-like values negate on their int
    ns representation and signed numerics negate natively. Only unsigned
    ints (which would wrap) fall back to float.
    """
    array = np.atleast_1d(np.asarray(values))
    if dtype_time_like(array.dtype):
        return -to_int(array)
    if array.dtype.kind == "u":
        return -to_float(array)
    return -array


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


@dataclass(frozen=True, slots=True)
class Grid:
    """
    One evenly sampled run of labels.

    An exact run (``step_den`` above zero) works in ticks -- nanoseconds for
    time, the integers themselves otherwise -- and labels sample ``k`` as
    ``origin + (phase + (k0 + k) * step_num) // step_den``. The ideal grid
    therefore has a spacing of ``step_num / step_den`` ticks and an origin
    ``phase / step_den`` ticks after ``origin``, and each label is the floor
    of its ideal position, so a 1024 Hz grid never drifts.

    A float run (``step_den`` of zero) keeps ``step_num`` as a scalar step
    and spaces its labels linearly, as numpy does. A slice keeps the original
    ``parent_count``, origin and step, and maps samples through the integer
    offset ``k0`` and ``stride``. Evaluating the original expression keeps
    every label bit-for-bit, including numpy's endpoint rounding.

    An exact run folds ``k0`` into its origin and phase. A newly constructed
    float run also folds an explicit offset into its origin; only slices
    with ``parent_count`` retain their original grid.
    """

    origin: Any
    step_num: Any
    step_den: int
    count: int
    k0: int = 0
    phase: int = 0
    parent_count: int | None = None
    stride: int = 1

    def __post_init__(self):
        """Normalize the run to the one form its labels have."""
        # Counts come from numpy offsets as often as from python, and a
        # numpy integer here would leak into every id and every dump.
        for name in ("step_den", "count", "k0", "phase", "stride"):
            if type(value := getattr(self, name)) is not int:
                object.__setattr__(self, name, int(value))
        if self.count < 0:
            msg = f"A run cannot hold {self.count} samples."
            raise CoordError(msg)
        if not self.exact:
            if self.parent_count is None:
                object.__setattr__(
                    self, "origin", self.origin + self.k0 * self.step_num
                )
                object.__setattr__(self, "k0", 0)
            else:
                object.__setattr__(self, "parent_count", int(self.parent_count))
                if self.parent_count < 1 or not self.stride:
                    raise CoordError(
                        "A float slice needs a positive length and nonzero stride."
                    )
                if (self.k0, self.stride, self.count) == (0, 1, self.parent_count):
                    object.__setattr__(self, "parent_count", None)
            return
        num = int(self.step_num)
        whole, phase = divmod(self.phase + self.k0 * num, self.step_den)
        # Reduced by the common divisor of all three terms, never of the
        # step alone: from (3, 2, 1) the stride-two grid (6, 2, 1) must
        # stay as it is, since (3, 1, 0) keeps the labels but moves the
        # origin by half a tick.
        common = math.gcd(num, self.step_den, phase)
        object.__setattr__(self, "origin", int(self.origin) + whole)
        object.__setattr__(self, "step_num", num // common)
        object.__setattr__(self, "step_den", self.step_den // common)
        object.__setattr__(self, "phase", phase // common)
        object.__setattr__(self, "k0", 0)

    def __len__(self) -> int:
        return self.count

    @property
    def exact(self) -> bool:
        """Whether the labels come from an integer grid."""
        return self.step_den > 0

    @property
    def ideal_origin(self) -> Fraction:
        """The ideal position of the first sample, in ticks."""
        return Fraction(self.origin * self.step_den + self.phase, self.step_den)

    def labels(self, indices, dtype) -> np.ndarray:
        """The labels at these indices, which may lie outside the run."""
        indices = np.asarray(indices)
        if self.exact:
            offset = self.phase
            ticks = (offset + indices.astype(np.int64) * self.step_num) // self.step_den
            return np.asarray(self.origin + ticks).astype(dtype)
        indices = self.k0 + indices * self.stride
        start, step = self.origin, self.step_num
        num = self.count if self.parent_count is None else self.parent_count
        if num == 1 or np.dtype(dtype).kind in "mMO":
            return np.asarray(start + indices * step, dtype=dtype)
        # Match linspace's inferred floating dtype, rounding, and exact endpoint.
        last = (start + step * num) - step
        wide = np.result_type(start, last, 0.0)
        delta = np.subtract(last, start, dtype=wide)
        spacing = delta / (num - 1)
        scaled = indices.astype(wide)
        scaled = scaled / (num - 1) * delta if spacing == 0 else scaled * spacing
        return np.where(indices == num - 1, last, scaled + start).astype(dtype)

    def sliced(self, first: int, stride: int, count: int) -> Grid:
        """The run of the samples first, first + stride, ... (count of them)."""
        if not self.exact:
            if np.asarray(self.origin).dtype.kind in "mMO":
                step = self.step_num
                return Grid(self.origin + first * step, step * stride, 0, count)
            if self._float_arithmetic_exact():
                start = self.origin + (self.k0 + first * self.stride) * self.step_num
                flat = Grid(start, self.step_num * self.stride * stride, 0, count)
                if flat._float_arithmetic_exact():
                    return flat
            return replace(
                self,
                count=count,
                k0=self.k0 + first * self.stride,
                stride=self.stride * stride,
                parent_count=self.count
                if self.parent_count is None
                else self.parent_count,
            )
        if stride == 1:
            return Grid(
                self.origin,
                self.step_num,
                self.step_den,
                count,
                self.k0 + first,
                self.phase,
            )
        whole, phase = divmod(
            self.phase + (self.k0 + first) * self.step_num, self.step_den
        )
        return Grid(
            self.origin + whole,
            self.step_num * stride,
            self.step_den,
            count,
            phase=phase,
        )

    def _float_arithmetic_exact(self) -> bool:
        """Whether the float expression needs no rounding on its binary grid."""
        # All denominators are powers of two. Bound every intermediate in
        # units of the finest one, at the narrowest operand's precision.
        start, start_den = self.origin.as_integer_ratio()
        step, step_den = self.step_num.as_integer_ratio()
        den = max(start_den, step_den)
        start, step = start * (den // start_den), step * (den // step_den)
        count = self.count if self.parent_count is None else self.parent_count
        bits = min(
            np.finfo(np.result_type(value, 0.0)).nmant + 1
            for value in (self.origin, self.step_num)
        )
        return max(abs(start), abs(step * count), abs(start + step * count)) <= 2**bits

    def resized(self, count: int, dtype) -> Grid:
        """Restate a grid's extent, as an explicit resize or gap fill requests."""
        if self.exact:
            return self.sliced(0, 1, count)
        return Grid(self.labels(0, dtype)[()], self.step(dtype), 0, count)

    def index_of(self, ticks, forward: bool):
        """
        The sample index each label tick maps to.

        Forward: the first index whose label is at or past the tick in the
        run's direction. Otherwise the last index whose label is at or
        before it. Exact, since a label is the floor of its ideal position
        and a tick is an integer; Python integers, so no overflow.
        """
        num, den = self.step_num, self.step_den
        rel = (np.asarray(ticks, dtype=object) - self.origin) * den - self.phase
        if forward == (num > 0):  # label >= tick  <=>  ideal >= tick
            return -((-rel) // num) if num > 0 else rel // num
        # label <= tick  <=>  ideal < tick + 1
        return -((-(rel + den)) // num) - 1 if num > 0 else (rel + den) // num + 1

    def step(self, dtype):
        """The whole-tick (or float) spacing between neighbouring labels."""
        if not self.exact:
            return self.step_num * self.stride
        tick = round(Fraction(self.step_num, self.step_den))
        return np.timedelta64(tick, "ns") if np.dtype(dtype).kind in "mM" else tick

    def step_exact(self, dtype) -> Fraction:
        """The exact spacing in coordinate units, seconds for time."""
        assert self.exact, "only an exact grid states its spacing exactly"
        scale = _NS_PER_S if dtype_time_like(dtype) else 1
        return Fraction(self.step_num, self.step_den * scale)

    def canonical(self) -> tuple:
        """The terms which name the run's labels: origin, step, and phase."""
        if not self.exact:
            terms = (self.origin, self.step_num, 0, self.count)
            if self.parent_count is not None:
                terms += (self.parent_count, self.k0, self.stride)
            return terms
        return (self.origin, self.step_num, self.step_den, self.phase)


@dataclass(frozen=True, slots=True)
class Labels:
    """
    One window into a stored array of labels.

    The array itself lives in its coordinate's ``sources``, under the id it
    was hashed with on the way in. A slice, a stride, a reversal or a move
    to another coordinate only moves the window, so however far the labels
    travel they are copied once and hashed once.
    """

    id: str
    count: int
    offset: int = 0
    # negative where the window reads the stored array backwards
    stride: int = 1

    def __len__(self) -> int:
        return self.count

    def window(self, first: int, stride: int, count: int) -> Labels:
        """The window of the samples first, first + stride, ... within this one."""
        offset = self.offset + first * self.stride
        return Labels(self.id, count, offset, self.stride * stride)


def _store(sources: dict, values) -> Labels:
    """Add an array of labels to a store; the window names all of it."""
    # Always copied: a read-only view of a writable array would otherwise
    # change under the id hashed from it.
    values = np.array(values, copy=True)
    values.flags.writeable = False
    # An object array has no hash, so it is named by identity: two of them
    # share neither a store entry nor an id, however alike they look.
    key = uuid4().hex if values.dtype.hasobject else hash_array(values)
    sources.setdefault(key, values)
    return Labels(key, values.shape[0] if values.ndim else 1)


def _union_sources(stores) -> dict:
    """
    Merge label stores, refusing an id two unlike arrays answer to.

    Sources are trusted by their id, so a tampered dump is only ever
    visible here, where it meets the labels it claims to be.
    """
    out: dict = {}
    for store in stores:
        for key, values in store.items():
            held = out.setdefault(key, values)
            alike = held.shape == values.shape and held.dtype == values.dtype
            # An id names an array bit for bit, so its NaNs match too.
            equal_nan = alike and held.dtype.kind in "fc"
            same = alike and np.array_equal(held, values, equal_nan=equal_nan)
            if held is not values and not same:
                msg = f"Two different arrays of labels are named {key!r}."
                raise CoordError(msg)
    return out


def _check_window(run: Labels, values) -> None:
    """Ensure a window reads only samples its stored array holds."""
    length = values.shape[0] if values.ndim else 1
    last = run.offset + (run.count - 1) * run.stride
    inside = 0 <= min(run.offset, last) and max(run.offset, last) < length
    if run.count < 0 or run.stride == 0 or (run.count and not inside):
        msg = (
            f"A window of {run.count} samples from {run.offset} by {run.stride} "
            f"reads outside its {length} stored labels."
        )
        raise CoordError(msg)


def _seal_sources(sources) -> None:
    """Mark stored arrays read-only; an id names labels which cannot change."""
    for values in sources.values():
        values.flags.writeable = False


def _source(sources, run: Labels, dtype=None) -> np.ndarray:
    """The labels a window holds; the one place a stored array is read."""
    out = sources[run.id]
    if out.ndim:  # a lone label is held as the scalar array it came as
        stop = run.offset + run.count * run.stride
        out = out[run.offset : stop if stop >= 0 else None : run.stride]
    return out if dtype is None else out.astype(dtype, copy=False)


def _check_grid(grid: Grid, dtype) -> None:
    """Ensure an exact grid's labels fit its dtype, in value and in arithmetic."""
    dtype = np.dtype(dtype)
    info = np.iinfo(cast("Any", dtype) if dtype.kind in "iu" else np.int64)
    num, den, count = grid.step_num, grid.step_den, grid.count
    offset = grid.phase
    first = grid.origin + offset // den
    last = grid.origin + (offset + max(count - 1, 0) * num) // den
    wraps = min(first, last) < info.min or max(first, last) > info.max
    if abs(count * num) + den >= 2**63 or wraps:
        msg = (
            f"A grid of {count} samples with step {num}/{den} ticks from "
            f"{grid.origin} exceeds the {dtype} range."
        )
        raise CoordError(msg)


def _exact_run(values, dtype: np.dtype) -> Grid:
    """Resolve the exact grid a set of range inputs describes."""
    start, stop, step = (
        None if _is_null(v := values.get(x)) else _maybe_unpack(v)
        for x in ("start", "stop", "step")
    )
    shape = values.get("shape")
    start_tick = None if start is None else _to_tick(start)
    stop_tick = None if stop is None else _to_tick(stop)
    # get_coord screens out a shape which is not one nonzero length
    count = None if shape is None else int(next(iter(iterate(shape))))
    # The grid: a fraction step wins, then explicit grid fields, then a
    # scalar step, then the span divided by the count.
    phase = int(values.get("origin_offset") or 0)
    if (frac := _fraction_step(step)) is not None:
        if dtype.kind in "mM":
            frac = frac * _NS_PER_S
        num, den = frac.numerator, frac.denominator
    elif values.get("step_numerator") is not None:
        num = int(values["step_numerator"])
        den = int(values.get("step_denominator") or 1)
    elif step is not None:
        num, den, phase = _to_tick(step), 1, 0
    else:
        # get_coord screens out anything which states fewer than three
        assert start_tick is not None and stop_tick is not None and count is not None
        frac = Fraction(stop_tick - start_tick, count)
        num, den, phase = frac.numerator, frac.denominator, 0
    if den < 1:
        msg = "step_denominator must be positive."
        raise CoordError(msg)
    if 0 < abs(num) < den:
        msg = "A step smaller than one tick would repeat labels."
        raise CoordError(msg)
    if not 0 <= phase < den:
        msg = f"origin_offset must satisfy 0 <= offset < {den}, got {phase}."
        raise CoordError(msg)
    if count is None:
        assert start_tick is not None and stop_tick is not None  # they bound it
        if num == 0 or start_tick == stop_tick:
            count = 1
        else:
            span = stop_tick - start_tick
            if (span > 0) != (num > 0):
                msg = "Sign of step must match sign of stop - start"
                raise CoordError(msg)
            # Rounded to a tenth of a sample as a float, as the float range
            # always has, so a stop a hair past a sample does not add one.
            ratio = float(Fraction(span * den - phase, num))
            count = max(1, math.ceil(round(ratio, 1)))
    if start_tick is None:
        assert stop_tick is not None, "start or stop is present"
        # The ideal origin is count steps before stop; start is its floor.
        start_tick, phase = divmod(stop_tick * den - count * num, den)
    return Grid(start_tick, num, den, count, phase=phase)


def _float_run(values) -> tuple[Grid, np.dtype]:
    """Resolve a float (or coarse-time) range from a set of range inputs."""

    def _round_ratio(numerator, denominator, digits):
        """Round numerator/denominator, cheaply for scalars."""
        # Rounding python floats is ~10x faster than numpy scalars.
        ratio = _maybe_unpack(numerator / denominator)
        return round(float(ratio), digits)

    start, stop, step, shape = (
        values.get(x, None) for x in ("start", "stop", "step", "shape")
    )
    if not pd.isnull(shape):
        length = int(next(iter(iterate(shape))))
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
        stop = start + step * int(_maybe_unpack(np.ceil(span)))
    start_equal_stop = _maybe_unpack(start == stop)
    length = 1 if start_equal_stop else int(_round_ratio(stop - start, step, 0))
    # step should have the same sign as stop-start, see #321. Compare signs
    # via direct comparisons (rather than np.sign) since np.sign(datetime64)
    # returns a datetime64 which includes precision, so even if the sign is
    # the same, differing precision fails.
    diff = stop - start
    try:
        same_sign = ((step > zero) == (diff > zero)) & ((step < zero) == (diff < zero))
    except TypeError:  # mixed types (e.g. datetime.timedelta vs int zero)
        same_sign = np.sign(to_float(step)) == np.sign(to_float(diff))
    if not same_sign:
        msg = "Sign of step must match sign of stop - start"
        raise CoordError(msg)
    return Grid(start, step, 0, length), np.asarray(start + step).dtype


def _range_run(values) -> tuple[Grid, np.dtype]:
    """The single run, and its dtype, that a set of range inputs describes."""
    get = values.get
    dtype = _exact_dtype(get("start"), get("stop"), get("step"), get("shape"))
    if dtype is None:
        return _float_run(values)
    return _exact_run(values, dtype), dtype


def _coerce_run(run, sources: dict) -> Grid | Labels:
    """Coerce a run input (a run, an array of labels, or a dump) to a run."""
    if isinstance(run, np.ndarray):
        return _store(sources, run)
    if isinstance(run, Mapping):
        run = Labels(**run) if "id" in run else Grid(**run)
    elif not isinstance(run, Grid | Labels):
        msg = f"Runs must be a Grid or Labels, got {type(run)}."
        raise CoordError(msg)
    if isinstance(run, Labels):
        if run.id not in sources:
            msg = f"A run names labels no source holds: {run.id!r}."
            raise CoordError(msg)
        _check_window(run, sources[run.id])
    return run


def _promoted(values, dtype) -> Grid | None:
    """The grid these labels are exactly, or None where no grid is."""
    if values.ndim != 1 or len(values) < 2:
        return None
    diffs = _diffs(values)
    # A zero step would make one label answer for every sample, which
    # repeated labels do not; they stay as they are.
    if len(np.unique(diffs)) != 1 or diffs[0] == diffs[0] * 0:
        return None
    grid, _ = _range_run(dict(start=values[0], step=diffs[0], shape=(len(values),)))
    return grid if _grid_holds(grid, values, dtype) else None


def _grid_holds(grid: Grid, values, dtype) -> bool:
    """
    Whether a grid restates these labels exactly.

    Unlike get_coord's inference a run may never change a label or drop
    one, so a grid stands in for stored labels only where it holds all of
    them.
    """
    try:
        labels = grid.labels(np.arange(len(values)), dtype)
        if grid.exact:
            _check_grid(grid, dtype)
    except (CoordError, OverflowError):
        # Labels at the edge of their dtype can name a grid which cannot be
        # built, or whose arithmetic cannot reach them.
        return False
    return len(grid) == len(values) and bool(np.array_equal(labels, values))


def _fuse_runs(runs: tuple[Grid | Labels, ...]) -> tuple[Grid | Labels, ...]:
    """
    Fuse each run which is exactly the next samples of its predecessor.

    Two grids fuse where the second continues the first's arithmetic; two
    windows only where the second reads on through the same stored array,
    which costs no copy and keeps the id.
    """
    out = [runs[0]]
    for run in runs[1:]:
        prev = out[-1]
        if isinstance(prev, Grid) and isinstance(run, Grid) and _continues(prev, run):
            count = prev.count + run.count
            if (
                not prev.exact
                and prev.parent_count is None
                and run.parent_count is None
            ):
                out[-1] = replace(prev, count=count)
            else:
                out[-1] = prev.sliced(0, 1, count)
        elif isinstance(prev, Labels) and isinstance(run, Labels) and _abuts(prev, run):
            out[-1] = Labels(prev.id, prev.count + run.count, prev.offset, prev.stride)
        else:
            out.append(run)
    return tuple(out)


def _abuts(prev: Labels, run: Labels) -> bool:
    """Whether a window reads on from where its predecessor stopped."""
    return (prev.id, prev.stride) == (run.id, run.stride) and (
        run.offset == prev.offset + prev.count * prev.stride
    )


def _continues(prev: Grid, run: Grid) -> bool:
    """Whether a grid holds exactly the samples after its predecessor."""
    # A float run has a denominator of zero, so this also tells the two
    # kinds of grid apart.
    if (prev.step_num, prev.step_den) != (run.step_num, run.step_den):
        return False
    if not prev.exact:
        if prev.parent_count is not None or run.parent_count is not None:
            first_count = prev.count if prev.parent_count is None else prev.parent_count
            next_count = run.count if run.parent_count is None else run.parent_count
            return (prev.origin, first_count, prev.stride) == (
                run.origin,
                next_count,
                run.stride,
            ) and run.k0 == prev.k0 + prev.count * prev.stride
        return bool(prev.origin + prev.count * prev.step_num == run.origin)
    step = Fraction(prev.step_num, prev.step_den)
    return run.ideal_origin == prev.ideal_origin + prev.count * step


class NumericCoord(BaseCoord):
    """
    A coordinate whose labels are held as an ordered sequence of runs.

    A run is either a [`Grid`](`dascore.core.coords.Grid`), which states an
    evenly sampled stretch in a handful of numbers, or
    [`Labels`](`dascore.core.coords.Labels`), which stores them. One grid is
    the evenly sampled case and answers every question by arithmetic; one
    stored run is an arbitrary (or N-dimensional) array; several runs are a
    coordinate with holes in it, where each boundary records a break in
    sampling without changing any label.

    Coordinates are normally built with
    [`get_coord`](`dascore.core.coords.get_coord`) rather than directly.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.core.coords import concat_coords, get_coord
    >>>
    >>> # One evenly sampled run.
    >>> coord = get_coord(start=0.0, stop=10.0, step=1.0)
    >>> assert coord.runs_count == 1 and coord.evenly_sampled
    >>>
    >>> # Two runs separated by a gap.
    >>> other = get_coord(start=15.0, stop=25.0, step=1.0)
    >>> gappy = concat_coords(coord, other)
    >>> assert gappy.runs_count == 2 and gappy.missing().count == 5
    >>>
    >>> # Runs which continue exactly fuse back into one.
    >>> assert concat_coords(coord, get_coord(start=10.0, stop=20.0, step=1.0)) == \
get_coord(start=0.0, stop=20.0, step=1.0)
    """

    runs: tuple[Grid | Labels, ...]
    # The arrays the stored runs window, each under the id it was hashed
    # with; a window is only ever a view of one of them. An id is never
    # recomputed, so supplied sources are trusted to be the labels they
    # are keyed by; a dump is a contract, not input to validate.
    sources: FrozenDictType[str, ArrayLike] = FrozenDict()
    # A default no dtype equals: ``np.dtype(None)`` is float64, so the
    # inherited None makes exclude_defaults drop a float coord's dtype.
    dtype: Any = ""

    def __eq__(self, other) -> bool:
        """Whether two coordinates hold the same labels and say the same of them."""
        first, second = (
            x._label_fields() if isinstance(x, NumericCoord) else x
            for x in (self, other)
        )
        return sensible_model_equals(first, second)

    # Defined together, as the base class defines them: a class which states
    # __eq__ and not __hash__ is unhashable by a rule of python's, which
    # would replace the error the base class raises with a duller one.
    __hash__ = BaseCoord.__hash__

    def __deepcopy__(self, memo=None) -> Self:
        """Copy the coordinate, keeping its sources sealed."""
        out = super().__deepcopy__(memo)
        # A copied cache holds writable arrays detached from these sources.
        (out.__pydantic_private__ or {}).pop("_cache", None)
        _seal_sources(out.sources)
        return out

    def __setstate__(self, state) -> None:
        """Restore a pickled coordinate, keeping its sources sealed."""
        super().__setstate__(state)
        _seal_sources(self.sources)

    def _label_fields(self) -> dict:
        """
        The fields equality compares, stored runs spelled out as labels.

        An id names an array exactly, and equality has always been tolerant
        of a float's last bits, so the labels stand in for the ids.
        """
        out = self.model_dump(exclude={"runs", "sources"})
        out["runs"] = tuple(
            x if isinstance(x, Grid) else self._run_labels(x) for x in self.runs
        )
        return out

    @property
    def _rich_style(self) -> str:
        """Colour the shape the coordinate has, not the class it is."""
        if self.runs_count > 1:
            return dascore_styles["coord_segmented"]
        if self.evenly_sampled:
            return dascore_styles["coord_range"]
        return (
            dascore_styles["coord_monotonic"]
            if self._direction()
            else dascore_styles["coord_array"]
        )

    # --- construction

    @model_validator(mode="before")
    @classmethod
    def _validate_runs(cls, data: Any) -> Any:
        """Store any bare labels, coerce the runs, and derive what they imply."""
        if not isinstance(data, Mapping):
            return data
        data = dict(data)
        sources = dict(data.get("sources") or {})
        runs = tuple(_coerce_run(x, sources) for x in iterate(data.get("runs")))
        if not runs:
            msg = "A numeric coordinate needs at least one run."
            raise CoordError(msg)
        if len(runs) > 1:  # an empty run states nothing, not even a seam
            runs = tuple(x for x in runs if len(x)) or runs[:1]
        runs = _fuse_runs(runs)
        stored = [x for x in runs if isinstance(x, Labels)]
        # An entry no window reads is let go of, so the store never
        # outgrows the coordinate carrying it.
        data["sources"] = {x.id: sources[x.id] for x in stored}
        dtype = data.get("dtype")
        if _is_null(dtype) or dtype == "":
            if not stored:
                msg = "A coordinate built only from grids must state its dtype."
                raise CoordError(msg)
            dtype = np.result_type(*[sources[x.id].dtype for x in stored])
        dtype = np.dtype(dtype)
        for run in runs:
            if isinstance(run, Grid) and run.exact:
                _check_grid(run, dtype)
        data["runs"] = runs
        data["dtype"] = dtype
        if len(runs) == 1 and isinstance(runs[0], Labels):
            data["shape"] = _source(sources, runs[0]).shape
        else:
            if any(sources[x.id].ndim > 1 for x in stored):
                msg = "An N-dimensional coordinate holds exactly one run."
                raise CoordError(msg)
            data["shape"] = (sum(len(x) for x in runs),)
        data["step"] = _runs_step(runs, data.get("step"), dtype, sources)
        return data

    @classmethod
    def from_labels(cls, values, step=None, units=None) -> Self:
        """
        Build a coordinate holding these labels exactly, as one stored run.

        Parameters
        ----------
        values
            The labels, of any shape.
        step
            The grid the labels sit on; every spacing is then a whole
            number of steps.
        units
            Units for the coordinate.
        """
        values = np.asarray(values)
        return cls(runs=(values,), step=step, units=units, dtype=values.dtype)

    # --- runs

    @property
    def runs_count(self) -> int:
        """How many runs the coordinate holds."""
        return len(self.runs)

    @property
    def segments(self) -> tuple[NumericCoord, ...]:
        """Each run as a coordinate of its own, in order."""
        return tuple(self._with_runs((x,)) for x in self.runs)

    @property
    def _grid(self) -> Grid | None:
        """The one grid the coordinate is, or None."""
        runs = self.runs
        if len(runs) == 1 and isinstance(runs[0], Grid):
            return runs[0]
        return None

    @cached_method
    def _run_offsets(self) -> np.ndarray:
        """The first sample index of each run."""
        lens = [len(x) for x in self.runs]
        return array(np.cumsum([0, *lens[:-1]]))

    def _run_labels(self, run, indices=None) -> np.ndarray:
        """The labels of one run, at these indices or all of them."""
        if isinstance(run, Labels):
            values = _source(self.sources, run, self.dtype)
            return values if indices is None else values[indices]
        if indices is None:
            indices = np.arange(run.count)
        return run.labels(indices, self.dtype)

    def _with_runs(self, runs, dtype=None, step: Any = ...) -> Self:
        """A coordinate holding these runs, with this one's metadata."""
        return self.__class__(
            runs=tuple(runs),
            units=self.units,
            dtype=self.dtype if dtype is None else dtype,
            # a step passed here is a caller saying what the runs now sit on
            step=self.step if step is ... else step,
            sources=self.sources,
        )

    # --- evaluation

    @property
    @cached_method
    def values(self) -> ArrayLike:
        """Return the labels of the coordinate as an array."""
        runs = self.runs
        if len(runs) == 1:
            return array(self._run_labels(runs[0]))
        out = np.concatenate([self._run_labels(x) for x in runs])
        return array(out.astype(self.dtype, copy=False))

    def _get_index_values(self, indices):
        """Evaluate only the requested samples, each in its own run."""
        indices = np.asarray(indices)
        if (grid := self._grid) is not None:
            return grid.labels(
                np.where(indices < 0, indices + len(self), indices), self.dtype
            )
        if len(self.runs) == 1:
            return np.asarray(self.values)[indices]
        indices = np.where(indices < 0, indices + len(self), indices)
        offsets = self._run_offsets()
        which = np.searchsorted(offsets, indices, side="right") - 1
        out = np.empty(indices.shape, dtype=self.dtype)
        for num in np.unique(which):
            mask = which == num
            out[mask] = self._run_labels(self.runs[num], indices[mask] - offsets[num])
        return out

    @cached_method
    def _bounds(self) -> np.ndarray:
        """The extreme labels of every run, which min and max come from."""
        out = []
        for run in self.runs:
            if isinstance(run, Grid):
                out.append(run.labels([0, run.count - 1], self.dtype))
            elif (values := self._run_labels(run)).size:
                out.append(np.asarray([np.nanmin(values), np.nanmax(values)]))
        return np.concatenate(out) if out else np.empty(0, dtype=self.dtype)

    def _min(self):
        """Return min value."""
        bounds = self._bounds()
        return np.nanmin(bounds) if bounds.size else _get_nullish(self.dtype)

    def _max(self):
        """Return max value."""
        bounds = self._bounds()
        return np.nanmax(bounds) if bounds.size else _get_nullish(self.dtype)

    @cached_method
    def _direction(self) -> int:
        """1 for ascending labels, -1 for descending, 0 for neither."""
        runs = self.runs
        if len(runs) == 1:
            return self._run_direction(runs[0])
        # Runs are never assumed to chain: each must run one way and every
        # seam must continue it.
        edges = np.concatenate([_run_edges(x, self.dtype, self.sources) for x in runs])
        ascending = edges[0] < edges[-1]
        ends, starts = edges[1::2][:-1], edges[::2][1:]
        chained = np.all(starts > ends) if ascending else np.all(starts < ends)
        inner = {self._run_direction(x) for x in runs if len(x) > 1}
        if not chained or inner - {1 if ascending else -1}:
            return 0
        return 1 if ascending else -1

    def _run_direction(self, run) -> int:
        """1 if one run ascends, -1 if it descends, 0 if it does neither."""
        if isinstance(run, Grid):
            step = run.step(self.dtype)
            zero = _TD64_ZERO if is_timedelta64(step) else 0
            return -1 if step < zero else 1
        values = self._run_labels(run)
        if values.ndim != 1 or not values.size:
            return 0
        if len(values) == 1:
            return 1
        if not is_strictly_monotonic(values):
            return 0
        return 1 if values[0] < values[-1] else -1

    @property
    def sorted(self) -> bool:
        """Returns True if the labels ascend."""
        return self._direction() > 0

    @property
    def reverse_sorted(self) -> bool:
        """Returns True if the labels descend."""
        return self._direction() < 0

    @property
    def evenly_sampled(self) -> bool:
        """Returns True if the coord is one evenly sampled run."""
        return self._grid is not None

    @property
    def step_exact(self) -> Fraction | None:
        """The exact spacing in coordinate units (seconds for time), or None."""
        if (grid := self._grid) is not None and grid.exact:
            return grid.step_exact(self.dtype)
        return super().step_exact

    # --- indexing and selection

    def __getitem__(self, item):
        if isinstance(item, int | np.integer) and self.ndim == 1:
            if item >= len(self) or item < -len(self):
                raise IndexError(f"{item} exceeds coord length of {self}")
            return self._get_index_values(item)[()]
        if isinstance(item, slice) and self.ndim == 1:
            start = None if item.start is ... else item.start
            stop = None if item.stop is ... else item.stop
            indices = range(len(self))[slice(start, stop, item.step)]
            if not len(indices):
                return get_coord(data=np.empty(0, dtype=self.dtype), units=self.units)
            return self._slice_runs(indices)
        out = self.values[item]
        if not np.ndim(out):
            return out
        if self._grid is not None:
            # A grid states no label the arithmetic cannot restate, so the
            # result may be read back as one.
            return get_coord(data=out, units=self.units)
        # Stored labels are held exactly, and a stride or a reorder can
        # take the result off the declared grid, which then states nothing.
        step = self.step
        if step is not None and np.ndim(out) == 1 and is_strictly_monotonic(out):
            with suppress(CoordError, ValidationError):
                return get_coord(data=out, units=self.units, step=step, snap=False)
        return get_coord(data=out, units=self.units, snap=False)

    def _slice_runs(self, indices: range) -> BaseCoord:
        """The coordinate holding the samples a range of positions names."""
        stride = indices.step
        out = []
        for run, offset in zip(self.runs, self._run_offsets()):
            first, count = _run_span(indices, int(offset), len(run))
            if not count:
                continue
            if (first, stride, count) == (0, 1, len(run)):
                # Only a trimmed run may have become evenly sampled; an
                # untouched one still states the break it was held apart by.
                out.append(run)
            elif isinstance(run, Grid):
                out.append(run.sliced(first, stride, count))
            else:
                out.append(self._promoted_labels(run.window(first, stride, count)))
        # a backwards range reads the runs from the last one
        runs = out if stride > 0 else out[::-1]
        # A stride multiplies every spacing, so a step declared over the
        # samples it skips is kept only where the ones it keeps still sit
        # on it; a widened seam says they do not.
        with suppress(CoordError, ValidationError):
            return self._with_runs(runs)
        return self._with_runs(runs, step=None)

    def _promoted_labels(self, window: Labels) -> Grid | Labels:
        """A trimmed window as the grid it is, unless it contradicts the step."""
        run = _promoted(self._run_labels(window), self.dtype)
        if run is None:
            return window
        if not _is_null(step := self.step):
            # a grid on another spacing would restate the step and swallow
            # the positions the declared one says are missing
            spacing = abs(_maybe_unpack(run.step(self.dtype)))
            if spacing != abs(_maybe_unpack(step)):
                return window
        return run

    def index(self, indexer, axis: int | None = None) -> BaseCoord:
        """Index the coordinate; a slice keeps the runs, as ``coord[slice]`` does."""
        if isinstance(indexer, slice) and not axis:
            return self[indexer]
        return super().index(indexer, axis=axis)

    def select(
        self, args, relative=False, samples=False
    ) -> tuple[BaseCoord, slice | ArrayLike]:
        """Apply select, return selected coords and index for selecting data."""
        if is_array(args):
            return self._select_by_array(args, relative=relative, samples=samples)
        if samples:
            return self._select_by_samples(args)
        if not self._direction():
            return self._select_by_mask(args, relative)
        value_1, value_2 = self._get_slice_tuple(args, relative=relative)
        if value_1 is not None and value_2 is not None and value_1 > value_2:
            # crossed relative bounds name no interval at all
            return self.empty(), slice(0, 0)
        if self._grid is not None:
            start = self._get_index(value_1, forward=self.sorted)
            stop = self._get_index(value_2, forward=self.reverse_sorted)
            if self.reverse_sorted:
                start, stop = stop, start
            # we add 1 to stop in slice since its upper limit is exclusive
            start = None if start == 0 else start
            out = slice(start, (stop + 1) if stop is not None else stop)
        elif self.runs_count > 1:
            out = self._run_window(value_1, value_2)
        else:
            out = self._window(value_1, value_2)
        if self._slice_degenerate(out):
            return self.empty(), slice(0, 0)
        return self[out], out

    def _window(self, value_1, value_2) -> slice:
        """The samples of a monotonic coordinate inside [value_1, value_2]."""
        values = self.values
        if self.reverse_sorted:
            value_1, value_2 = value_2, value_1
            values = _negate_for_search(values)
            value_1 = None if value_1 is None else _negate_for_search(value_1)[0]
            value_2 = None if value_2 is None else _negate_for_search(value_2)[0]
        low = 0 if value_1 is None else int(np.searchsorted(values, value_1, "left"))
        high = (
            len(values)
            if value_2 is None
            else int(np.searchsorted(values, value_2, "right"))
        )
        return slice(low or None, high if high < len(values) else None)

    def _run_window(self, value_1, value_2) -> slice:
        """
        The samples inside [value_1, value_2], asked of each run in turn.

        A run answers from its own bounds, so placing a window never
        concatenates the coordinate's labels.
        """
        low, high = None, None
        for run, offset in zip(self.runs, self._run_offsets()):
            edges = _run_edges(run, self.dtype, self.sources)
            near, far = (edges[0], edges[-1]) if self.sorted else (edges[-1], edges[0])
            outside = (value_2 is not None and near > value_2) or (
                value_1 is not None and far < value_1
            )
            if outside:
                continue
            whole = (value_1 is None or value_1 <= near) and (
                value_2 is None or value_2 >= far
            )
            if whole:
                first, stop = 0, len(run)
            else:  # a boundary run; let it make the exact trim
                _, indexer = self._with_runs((run,)).select((value_1, value_2))
                assert isinstance(indexer, slice)  # a value window is contiguous
                first, stop, _ = indexer.indices(len(run))
                if stop <= first:
                    continue
            if low is None:
                low = int(offset) + first
            high = int(offset) + stop
        if low is None or high is None:
            return slice(0, 0)
        return slice(low or None, high if high < len(self) else None)

    def _select_by_mask(self, args, relative):
        """Select unordered labels by comparing every one of them."""
        args = self._get_slice_tuple(args, relative=relative)
        values = self.values
        out = np.ones_like(values, dtype=np.bool_)
        if (value_1 := self._get_compatible_value(args[0])) is not None:
            out = out & (values >= value_1)
        if (value_2 := self._get_compatible_value(args[1])) is not None:
            out = out & (values <= value_2)
        if not np.any(out):
            return self.empty(), out
        if np.all(out):
            return self, slice(None, None)
        # Convert boolean to int indexes; some consumers (eg lazy file
        # readers) index with these where booleans are not supported.
        if len(self.shape) == 1:
            out = np.arange(len(out))[out]
        return self[out], out

    def _get_index(self, value, forward=True):
        """Get the index corresponding to a value."""
        if (value := self._get_compatible_value(value)) is None:
            return value
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value[()]
        if (grid := self._grid) is None:
            return self._searched_index(value, forward)
        if not grid.exact:
            return self._get_float_index(grid, value, forward)
        if grid.step_num == 0:
            return self._get_zero_step_index(value, forward)
        if isinstance(value, Sized):
            return grid.index_of(self._bound_ticks(value, forward), forward).astype(
                np.int64
            )
        if isinstance(value, float | np.floating) and not math.isfinite(value):
            # Past one end of the coordinate; the checks below open that side.
            out = len(self) if (value > 0) == self.sorted else -1
        else:
            ticks = self._bound_ticks(value, forward)
            out = int(grid.index_of(ticks, forward)[0])
        if (forward and out < 0) or (not forward and out >= len(self)):
            return None
        return out

    def _searched_index(self, value, forward=True):
        """The index of a value in stored labels, found by search."""
        values = np.atleast_1d(self.values)
        new_value = np.atleast_1d(value)
        # since searchsorted only works on ascending arrays we negate
        # descending ones to get the same effect.
        if self.reverse_sorted:
            values = _negate_for_search(values)
            new_value = _negate_for_search(new_value)
        right = np.searchsorted(values, new_value, side="right")
        left = np.searchsorted(values, new_value, side="left")
        left_ok = (left < len(self)) & (left > 0)
        eq = left_ok & (values.take(left, mode="clip") == new_value)
        out = right if forward else left
        # where equal it should also be left values, so this behaves the
        # same way the grid path does.
        if not self.reverse_sorted:
            out[eq] = left[eq]
        return out if is_array(value) else int(out[0])

    def _get_zero_step_index(self, value, forward):
        """
        Get the index of a value for a coord with a step of 0.

        Every sample equals the first label, so the index is the first
        sample or one just outside the coord.
        """
        start = self.min()
        if forward:  # index of the first sample >= value
            return 0 if value <= start else len(self)
        return 0 if value >= start else -1

    def _bound_ticks(self, values, forward: bool) -> np.ndarray:
        """
        The integer ticks an array of query bounds is equivalent to.

        A bound between ticks asks for the labels past it, which are the
        labels past the next tick in that direction.
        """
        values = np.atleast_1d(values)
        if values.dtype.kind in "mM":  # already nanoseconds, the tick
            return values.astype("int64")
        if values.dtype.kind == "f":
            values = (np.ceil if forward == self.sorted else np.floor)(values)
        # A bound past the int64 range lies past the coordinate either way.
        limits = np.iinfo(np.int64)
        return np.clip(values, limits.min, limits.max).astype(np.int64)

    def _get_float_index(self, grid: Grid, value, forward=True):
        """Get the index corresponding to a value of a float range."""
        if grid.parent_count is not None:
            # The indexing helpers import coordinates, so load this here.
            from dascore.utils.indexing import _range_searchsorted  # noqa: PLC0415

            side = "left" if forward == self.sorted else "right"
            positions = _range_searchsorted(
                self, np.atleast_1d(value), side, estimate=False
            )
            out = positions if side == "left" else positions - 1
            if self.reverse_sorted:
                out = len(self) - 1 - out
            return out if isinstance(value, Sized) else int(out[0])
        start, step = grid.origin, grid.step_num
        if isinstance(value, Sized):
            func = np.ceil if forward else np.floor
            # Due to float weirdness we need a little bit of a fudge factor here.
            fraction = func(
                np.round((np.atleast_1d(value) - start) / step, decimals=10)
            )
            return fraction.astype(np.int64)
        # Scalar fast path. A zero step is caught after the division: python
        # scalars raise, numpy scalars give inf/nan, and testing a numpy step
        # for truthiness up front costs ~10x more on this hot path.
        try:
            fraction = round(float((value - start) / step), 10)
        except ZeroDivisionError:
            return self._get_zero_step_index(value, forward)
        if not math.isfinite(fraction):
            if not step:
                return self._get_zero_step_index(value, forward)
            # A bound past one end; its sign says which.
            out = len(self) if fraction > 0 else -1
        else:
            out = math.ceil(fraction) if forward else math.floor(fraction)
        if (forward and out < 0) or (not forward and out >= len(self)):
            return None
        return out

    def _samples_in(self, duration):
        """How many steps a duration spans, unrounded."""
        grid = self._grid
        if grid is None or not grid.exact:
            return super()._samples_in(duration)
        if dtype_time_like(self.dtype):
            ticks = Fraction(int(to_int(dc.to_timedelta64(duration))))
        else:
            ticks = Fraction(_maybe_unpack(np.asarray(duration)).item())
        return ticks / abs(Fraction(grid.step_num, grid.step_den))

    def _out_of_bounds_indices(self, array) -> np.ndarray:
        """Positions past either end, from the grid."""
        grid = self._grid
        if grid is None or not grid.exact:
            return super()._out_of_bounds_indices(array)
        # Past the last label its floor position, before the first its
        # ceiling, as the truncating division did.
        last = grid.labels(len(self) - 1, self.dtype)
        beyond = (array > last) if self.sorted else (array < last)
        return np.where(
            beyond, self._get_index(array, forward=False), self._get_index(array)
        ).astype(np.int64)

    def sort(self, reverse=False) -> tuple[BaseCoord, slice | ArrayLike]:
        """Sort the contents of the coord. Return new coord and slice for sorting."""
        direction = self._direction()
        if direction:
            if (direction < 0) == reverse:
                return self, slice(None)
            return self[::-1], slice(None, None, -1)
        argsort: ArrayLike = np.argsort(self.values)[:: -1 if reverse else 1]
        new = get_coord(data=self.values[argsort], units=self.units)
        return new, argsort

    # --- updates

    @compose_docstring(doc=get_docstring(BaseCoord.change_length))
    def change_length(self, length: int) -> Self:
        """
        {doc}
        """
        if (grid := self._grid) is None:
            return super().change_length(length)
        length = _validate_new_length(length)
        if len(self) == length:
            return self
        return self._with_runs((grid.resized(length, self.dtype),))

    def coord_range(self, extend: bool = True):
        """The span of the coordinate; extended, to its exclusive end."""
        grid = self._grid
        if not extend or grid is None or not grid.exact:
            return super().coord_range(extend=extend)
        first, end = grid.labels([0, len(self)], np.int64)
        span = np.abs(end - first)
        return np.timedelta64(span, "ns") if dtype_time_like(self.dtype) else span

    def snap(self) -> BaseCoord:
        """
        Snap the coordinates to evenly sampled grid points.

        The min and max remain unchanged; every interior value may move
        without bound.
        """
        if self.evenly_sampled:
            return self
        min_v, max_v = self.min(), self.max()
        if len(self) == 1:
            # time deltas need to be generated for dt case, hence the subtract
            zero = self._get_compatible_value(0)
            step = self._get_compatible_value(1) - zero
        else:
            dur = max_v - min_v
            if is_timedelta64(dur):  # hack to handle dts int division.
                raw = float(dur.astype(np.int64)) / (len(self) - 1)
                step = np.timedelta64(int(np.round(raw)), "ns")
            else:
                step = dur / (len(self) - 1)
            assert step > (_TD64_ZERO if is_timedelta64(step) else 0)
        if self.reverse_sorted:
            step = -step
            start, stop = max_v, min_v + step
        else:
            start, stop = min_v, max_v + step
        out = get_coord(start=start, stop=stop, step=step, units=self.units)
        return out.change_length(len(self))

    @compose_docstring(doc=get_docstring(BaseCoord.fuse))
    def fuse(self, tolerance=None, keep_step: bool = False) -> BaseCoord:
        """
        {doc}
        """
        if self.evenly_sampled or self.ndim != 1 or not self._direction():
            return self
        tol = self._fit_tolerance(tolerance)
        result, run = [], [self.runs[0]]
        fit = self._fit_run(run, tol, keep_step)
        for nxt in self.runs[1:]:
            trial = [*run, nxt]
            if (new_fit := self._fit_run(trial, tol, keep_step)) is not None:
                run, fit = trial, new_fit
            else:
                result.append(fit or (run[0], self.dtype))
                run, fit = [nxt], self._fit_run([nxt], tol, keep_step)
        result.append(fit or (run[0], self.dtype))
        runs, dtypes = zip(*result)
        # A re-fit spaces its labels evenly, which integers and coarse
        # times cannot always hold; the fit says what dtype they need.
        return self._with_runs(runs, dtype=np.result_type(*dtypes))

    def _fit_tolerance(self, tolerance):
        """
        Coerce the tolerance to the dtype expected for value deviations.

        None is no bound at all, which is what an infinite count asks for.
        """
        if isinstance(tolerance, GapTolerance) and tolerance.count is not None:
            if not np.isfinite(tolerance.count):
                return None
            # a count of steps is measured against the runs' own step
            steps = [abs(x.step(self.dtype)) for x in self.runs if isinstance(x, Grid)]
            tolerance = tolerance.count * get_middle_value(steps) if steps else 0
        return self._gap_tolerance(tolerance).excess

    def _fit_run(self, runs, tol, keep_step: bool = False) -> tuple | None:
        """Fit a stretch of runs to one grid within tol, with its dtype."""
        if len(runs) == 1 and isinstance(runs[0], Grid):
            return runs[0], self.dtype
        count = sum(len(x) for x in runs)
        if count < 2:
            return None
        ascending = self.sorted
        labels = [self._run_labels(x, [0, len(x) - 1]) for x in runs]
        first = labels[0][0]
        last = labels[-1][-1]
        span = last - first
        if is_timedelta64(span) or is_datetime64(first):
            span_ns = dc.to_timedelta64(span).astype(np.int64)
            step = np.timedelta64(int(np.round(span_ns / (count - 1))), "ns")
            zero = dc.to_timedelta64(0)
        else:
            step = span / (count - 1)
            zero = 0
        # Strictly monotonic runs guarantee a nonzero step matching the
        # sort direction.
        assert step != zero and (step > zero) == ascending
        candidate = get_coord(
            start=first, stop=last + step, step=step, units=self.units
        ).change_length(count)
        assert isinstance(candidate, NumericCoord)
        actual = np.concatenate([self._run_labels(x) for x in runs])
        deviation = np.max(np.abs(candidate.values - actual))
        if tol is not None and deviation > tol:
            return None
        if keep_step and not self._keeps_step(runs, labels, ascending):
            return None
        fit = candidate.runs[0]
        assert isinstance(fit, Grid)  # a re-fit is evenly sampled by construction
        return fit, candidate.dtype

    def _keeps_step(self, runs, labels, ascending: bool) -> bool:
        """
        Whether any seam between these runs skips a position of their grid.

        A seam a whole number of steps wider than one is a hole, which
        re-fitting would spread over the run; any other seam is
        misalignment, and stays the tolerance's business.
        """
        steps = [x.step(self.dtype) if isinstance(x, Grid) else self.step for x in runs]
        for num in range(1, len(runs)):
            step = steps[num - 1] if not _is_null(steps[num - 1]) else steps[num]
            if _is_null(step):
                continue  # no grid stated, so no position to have skipped
            span = abs(labels[num][0] - labels[num - 1][-1]) / abs(step)
            whole = np.round(span)
            if whole > 1 and abs(span - whole) <= _GRID_RTOL * whole:
                return False
        return True

    @compose_docstring(doc=get_docstring(BaseCoord.update_limits))
    def update_limits(self, min=None, max=None, step=None, **kwargs) -> BaseCoord:
        """
        {doc}
        """
        if (grid := self._grid) is not None:
            return self._range_limits(grid, min, max, step, **kwargs)
        if self.runs_count > 1 and min is not None and max is not None:
            msg = "Cannot specify both min and max in update_limits."
            raise ParameterError(msg)
        if sum(x is not None for x in [min, max, step]) > 1:
            msg = "At most one parameter can be specified in update_limits."
            raise ValueError(msg)
        out = self
        if not pd.isnull(step) and len(self):
            if self.runs_count > 1:
                msg = (
                    "A coordinate with holes has no single step; use fuse or "
                    "snap to get an evenly sampled coordinate first."
                )
                raise ParameterError(msg)
            out = self.snap().update_limits(step=step)
        elif min is not None:
            out = self._shifted(get_compatible_values(min, self.dtype) - self.min())
        elif max is not None:
            out = self._shifted(get_compatible_values(max, self.dtype) - self.max())
        return out.new(**kwargs) if kwargs else out

    def _range_limits(self, grid: Grid, min, max, step, **kwargs) -> BaseCoord:
        """Update the limits of a single evenly sampled run."""
        if all(x is not None for x in [min, max, step]):
            msg = "At most two parameters can be specified in update_limits."
            raise ValueError(msg)
        out: BaseCoord = self
        if min is not None and max is not None:
            # min is the new start, max the new exclusive stop, and the
            # count is kept; a spacing below one tick becomes floats.
            min = get_compatible_values(min, self.dtype)
            max = get_compatible_values(max, self.dtype)
            span = _to_tick(max) - _to_tick(min) if grid.exact else 0
            if abs(frac := Fraction(span, len(self))) >= 1:
                num, den = frac.as_integer_ratio()
                new = Grid(_to_tick(min), num, den, len(self))
                out = self._with_runs((new,))
            else:
                new_step = (max - min) / len(self)
                out = get_coord(start=min, stop=max, step=new_step, units=self.units)
            return out.new(**kwargs) if kwargs else out
        if step is not None:
            out = self._with_step(grid, step)
        assert isinstance(out, NumericCoord)
        if min is not None:
            out = out._shifted(get_compatible_values(min, self.dtype) - out.min())
        if max is not None:
            out = out._shifted(get_compatible_values(max, self.dtype) - out.max())
        return out.new(**kwargs) if kwargs else out

    def _with_step(self, grid: Grid, step) -> NumericCoord:
        """The same start and count on a new cadence."""
        if _fraction_step(step) is None:
            step = get_compatible_values(step, type(self.step))
        # Re-resolve: the inputs pick the representation the step needs,
        # and the start comes from the grid rather than from every label.
        out = get_coord(
            start=grid.labels(0, self.dtype)[()],
            step=step,
            shape=self.shape,
            units=self.units,
        )
        assert isinstance(out, NumericCoord)
        return out

    def _shifted(self, delta) -> Self:
        """Every label moved by delta; the runs move with them."""
        runs = []
        for run in self.runs:
            if isinstance(run, Labels):
                runs.append(self._run_labels(run) + delta)
            elif run.exact:
                runs.append(
                    Grid(
                        run.origin + _to_tick(delta),
                        run.step_num,
                        run.step_den,
                        run.count,
                        run.k0,
                        run.phase,
                    )
                )
            else:
                start = run.labels(0, self.dtype)[()] + delta
                runs.append(Grid(start, run.step(self.dtype), 0, run.count))
        # A shift off the coordinate's own dtype -- an integer moved half a
        # step -- states the labels it lands on, not the ones it left. A
        # tick grid has already refused a delta it cannot hold.
        exact = any(isinstance(x, Grid) and x.exact for x in self.runs)
        dtype = self.dtype if exact else np.asarray(self.min() + delta).dtype
        return self._with_runs(runs, dtype=dtype)

    def new(self, **kwargs):
        """Update coordinate; the runs are kept unless the labels change."""
        if "data" in kwargs or "values" in kwargs:
            # new labels state their own runs; the old step is not a claim
            # about them
            data = kwargs.get("data", kwargs.get("values"))
            units = kwargs.get("units", self.units)
            return get_coord(data=data, units=units, step=kwargs.get("step"))
        if "segments" in kwargs:
            units = kwargs.get("units", self.units)
            return get_coord(segments=kwargs["segments"], units=units)
        if not kwargs:
            return self
        if set(kwargs) <= {"units"}:
            return self.set_units(kwargs["units"])
        info: dict[str, Any] = dict(units=self.units)
        if (grid := self._grid) is None:
            info.update(data=self.values, step=self.step)
            return get_coord(**{**info, **kwargs})
        info["start"] = grid.labels(0, self.dtype)[()]
        info["stop"] = grid.labels(len(self), self.dtype)[()]
        # the step is always stated, since without one a new length would
        # re-derive the spacing from the span and move every label
        info["step"] = self.step
        if grid.exact and "step" not in kwargs:
            # the rounded step would move every label of a fractional grid
            info.update(
                step_numerator=grid.step_num,
                step_denominator=grid.step_den,
                origin_offset=grid.phase,
            )
        if "stop" not in kwargs and "step" not in kwargs:
            # a new stop or step re-derives the count, as a range always has
            info["shape"] = self.shape
        return get_coord(**{**info, **kwargs})

    # --- units

    def set_units(self, units) -> Self:
        """Set new units on the coordinate."""
        if units_match(self.units, units):
            return self
        return self.__class__(
            runs=self.runs,
            units=units,
            dtype=self.dtype,
            step=self.step,
            sources=self.sources,
        )

    def _convert_units(self, units) -> Self:
        """Convert units, or set units if none exist."""
        if dtype_time_like(self.dtype):  # time units are fixed
            return self
        if self.units is None:
            return self.set_units(units)
        # Each run converts in the form it is held in: re-reading the
        # converted labels would snap within a tolerance and erase a seam.
        pieces = [self._converted_run(x, units) for x in self.runs]
        dtype = np.result_type(*[x for _, x in pieces])
        step = self.step
        if step is not None:
            # a step is a difference, so an affine unit's offset cancels
            anchor = convert_units(step * 0, units, self.units)
            step = convert_units(step, units, self.units) - anchor
        runs = tuple(x for x, _ in pieces)
        return self.__class__(runs=runs, units=units, dtype=dtype, step=step)

    def _converted_run(self, run, units) -> tuple[Grid | Labels, np.dtype]:
        """One run's labels in new units, and the dtype they take."""
        if isinstance(run, Labels):
            # a genuinely new array: it enters a store of its own, hashed once
            values = convert_units(self._run_labels(run), units, self.units)
            return values, values.dtype
        if run.exact and run.step_den != 1:
            msg = (
                "Cannot convert the units of an integer coordinate with a "
                f"fractional step ({self.step_exact}); the result is not an "
                "integer grid."
            )
            raise CoordError(msg)
        # Never the endpoint past the last label: it wraps at a dtype limit.
        step = run.step(self.dtype)
        anchor = convert_units(step * 0, units, self.units)  # an affine offset
        start = convert_units(run.labels(0, self.dtype)[()], units, self.units)
        step = convert_units(step, units, self.units) - anchor
        return _range_run(dict(start=start, step=step, shape=(run.count,)))

    # --- discontinuities

    def _expected_spacing(self):
        """The spacing stored labels are held to: the step, else the median."""
        if not _is_null(self.step):
            return self.step
        diffs = _diffs(self.values)
        # the median magnitude, signed with the labels' direction, so
        # either orientation judges the same spacings
        median = np.median(np.abs(diffs))
        return median if self.sorted else -median

    def _seams(self) -> list[tuple]:
        """The ``(index, before, after, expected)`` rows of every discontinuity."""
        if not self._direction() or self.ndim != 1:
            return []
        if self.runs_count == 1:
            if isinstance(self.runs[0], Grid) or len(self) < 2:
                return []
            values = self.values
            diffs = _diffs(values)
            expected = self._expected_spacing()
            return [
                (int(i) + 1, values[i], values[i + 1], expected)
                for i in np.flatnonzero(diffs != expected)
            ]
        rows, offsets = [], self._run_offsets()
        for num in range(1, self.runs_count):
            prev, nxt = self.runs[num - 1], self.runs[num]
            before = self._run_labels(prev, [len(prev) - 1])[0]
            after = self._run_labels(nxt, [0])[0]
            rows.append((int(offsets[num]), before, after, self._run_step(prev)))
        return rows

    def _run_step(self, run):
        """The spacing a run expects after itself, or None."""
        if isinstance(run, Grid):
            return run.step(self.dtype)
        if not _is_null(self.step):
            return self.step
        if len(run) > 1:
            values = self._run_labels(run, [-2, -1])
            return values[-1] - values[0]
        return None

    def _holes(self) -> list[tuple]:
        """Each hole as ``(first missing label, last missing label, count)``."""
        step = self.step  # signed with the runs' direction
        rows = list(self._run_holes(self.runs[0], step))
        for (_, before, after, _), run in zip(self._seams(), self.runs[1:]):
            count = int(_on_grid(np.asarray([after - before]), step)[0]) - 1
            if count:
                rows.append(_hole(before, step, count))
            rows.extend(self._run_holes(run, step))
        return rows

    def _run_holes(self, run, step) -> list[tuple]:
        """The grid positions a stored run skips between its neighbours."""
        if isinstance(run, Grid) or len(run) < 2:
            return []
        values = self._run_labels(run)
        counts = _on_grid(_diffs(values), step)
        return [
            _hole(values[i], step, int(counts[i]) - 1)
            for i in np.flatnonzero(counts > 1)
        ]

    # --- identity, summary and display

    def _run_identity(self, run) -> tuple:
        """The payload naming one run's labels."""
        if isinstance(run, Labels):
            return ("labels", run.id, run.count, run.offset, run.stride)
        origin, num, den, extra, *window = run.canonical()
        if not run.exact:
            return (
                "float",
                self._hash_scalar(origin, "start"),
                self._hash_scalar(num, "step"),
                int(extra),
                *window,
            )
        return ("grid", origin, num, den, extra, int(run.count))

    def _id_components(self) -> tuple[Any, ...]:
        """The runs, the grid they declare, and the dtype they are read as."""
        components: tuple[Any, ...] = (
            tuple(int(x) for x in self.shape),
            str(np.dtype(self.dtype)),
            tuple(self._run_identity(x) for x in self.runs),
        )
        if not self.evenly_sampled and not _is_null(self.step):
            components += (("step", self._hash_scalar(self.step, "step")),)
        return components

    def to_summary(self, dims=()) -> CoordSummary:
        """Get the summary info about the coord, exact grid and runs included."""
        summary = super().to_summary(dims=dims)
        if (grid := self._grid) is not None:
            if not grid.exact:
                return summary
            terms = grid.canonical()[1:]
            return summary.model_copy(
                update=dict(zip(_EXACT_GRID_FIELDS, terms, strict=True))
            )
        if self.runs_count < 2 or self.runs_count > _MAX_SUMMARY_RUNS:
            return summary
        runs = tuple(x.to_summary(dims=dims) for x in self.segments)
        return summary.model_copy(update={"runs": runs})

    def _repr_fields(self) -> tuple[tuple[str, Text, bool], ...]:
        fields = super()._repr_fields()
        grid = self._grid
        if grid is None or not grid.exact or grid.step_den == 1:
            return fields
        # The rounded step misstates a fractional grid, so it is said exactly.
        exact = cast("Fraction", self.step_exact)
        text = Text(f"{exact}")
        if dtype_time_like(self.dtype):
            rate = 1 / abs(exact)
            rate_str = str(rate) if rate.denominator == 1 else f"{float(rate):g}"
            text += Text(" s") + Text(f" ({rate_str} Hz)", dascore_styles["units"])
        elif self.unit_str:
            text += Text(f" {self.unit_str}", dascore_styles["units"])
        return tuple(
            ("step", text, True) if name == "step" else (name, value, labelled)
            for name, value, labelled in fields
        )


def _runs_step(runs, declared, dtype, sources):
    """
    The grid every label of a coordinate sits on, or None.

    One grid states its own step. Otherwise a declared step is checked
    against every run and every seam, and raises where it does not hold; a
    step the runs merely share is dropped rather than raising.
    """
    if len(runs) == 1 and isinstance(runs[0], Grid):
        return runs[0].step(dtype)
    strict = not _is_null(declared)
    if strict:
        step = _declared_step(declared, dtype)
    else:
        steps = [x.step(dtype) for x in runs if isinstance(x, Grid)]
        if len(steps) != len(runs) or len({_maybe_unpack(x) for x in steps}) != 1:
            return None
        step = steps[0]
    stored = [(x, _source(sources, x, dtype)) for x in runs if isinstance(x, Labels)]
    if strict and any(x.ndim != 1 for _, x in stored):
        msg = "A declared step needs one-dimensional, monotonic values."
        raise CoordError(msg)
    # a declared step is the grid the labels sit on, not their spacing; its
    # sign follows the labels, as a grid's does
    magnitude = np.abs(np.asarray(step))[()]
    edges = np.concatenate(
        [_run_edges(x, dtype, sources) for x in runs if len(x)] or [np.empty(0)]
    )
    ascending = len(edges) < 2 or edges[-1] > edges[0]
    step = magnitude if ascending else -magnitude
    try:
        for run, values in stored:
            if len(run) > 1:
                if not is_strictly_monotonic(values):
                    msg = "A declared step needs one-dimensional, monotonic values."
                    raise CoordError(msg)
                _on_grid(_diffs(values), step)
        for num in range(1, len(runs)):
            seam = edges[2 * num] - edges[2 * num - 1]
            _on_grid(np.asarray([seam]), step)
    except CoordError:
        if strict:
            raise
        return None
    return step


def _run_edges(run, dtype, sources) -> np.ndarray:
    """The first and last label of a run."""
    if isinstance(run, Labels):
        values = _source(sources, run, dtype)
        flat = values if values.ndim == 1 else values.reshape(-1)
        return np.asarray([flat[0], flat[-1]])
    return run.labels([0, run.count - 1], dtype)


def _run_span(indices: range, offset: int, length: int) -> tuple[int, int]:
    """The first sample and the count a range of positions takes from one run."""
    step = indices.step
    low, high = offset - indices.start, offset + length - indices.start
    if step > 0:
        first, stop = -((-low) // step), -((-high) // step)
    else:  # a backwards range reaches the run's far edge first
        first, stop = high // step + 1, low // step + 1
    first, stop = max(first, 0), min(stop, len(indices))
    return indices.start + first * step - offset, max(stop - first, 0)


def concat_coords(*coords, units=None) -> BaseCoord:
    """
    Concatenate monotonic coordinates into a single coordinate.

    This operation is truth-preserving: no value is ever altered, and every
    boundary between inputs that does not continue exactly is kept as a run
    boundary. Use [`fuse`](`dascore.core.coords.NumericCoord.fuse`) on the
    result for tolerance-bounded gap absorption.

    Parameters
    ----------
    *coords
        Coordinates to concatenate. Each must be numeric and monotonic.
        Inputs are ordered by their envelopes, or, where every input holds
        one sample, by the order they are given in; they must share dtype
        kind, units, and sort direction, and must not overlap.
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
    flat: list[NumericCoord] = []
    for coord in coords:
        if isinstance(coord, Mapping):  # model_dump round-trip payloads
            coord = get_coord(**coord)
        if not isinstance(coord, BaseCoord):
            msg = f"concat_coords requires coordinates, got {type(coord)}."
            raise CoordError(msg)
        if coord.degenerate:
            continue
        if not isinstance(coord, NumericCoord):
            msg = (
                "concat_coords only supports numeric, monotonic coordinates, "
                f"got {type(coord)}."
            )
            raise CoordError(msg)
        flat.append(coord.set_units(units) if units is not None else coord)
    if not flat:
        msg = "concat_coords requires at least one non-empty coordinate."
        raise CoordError(msg)
    _check_concat(flat)
    multi = [x for x in flat if len(x) > 1]
    # Single-sample inputs state no direction of their own, so the order
    # they were given in states it.
    bounds = [x.min() for x in flat]
    ascending = multi[0].sorted if multi else len(flat) < 2 or bounds[-1] > bounds[0]
    step = flat[0].step
    dtype = np.result_type(*[x.dtype for x in flat])
    steps = {_maybe_unpack(x.step) for x in flat}
    sources = _union_sources(x.sources for x in flat)
    # Runs, not whole coordinates, are what may not overlap: an input can
    # hold samples which belong inside another's gap.
    # Sort on native values; float conversion would collapse ns datetimes.
    parts = sorted(
        (seg for x in flat for seg in x.segments),
        key=lambda x: x.min(),
        reverse=not ascending,
    )
    _check_chain(parts, ascending)
    runs = tuple(itertools.chain.from_iterable(x.runs for x in parts))
    out = NumericCoord(runs=runs, sources=sources, dtype=dtype, units=flat[0].units)
    if out.step is None and len(steps) == 1 and not _is_null(step):
        # stored runs carry no step of their own, so the one they shared is
        # restated here; it is refused if the seams do not meet it
        with suppress(CoordError, ValidationError):
            out = NumericCoord(
                runs=runs, sources=sources, dtype=dtype, units=out.units, step=step
            )
    return out


def _check_concat(coords) -> None:
    """Validate that coordinate dtypes and units can be concatenated."""
    for coord in coords:
        if coord.ndim != 1 or not coord._direction():
            msg = "Concatenated coordinates must be 1D and monotonic."
            raise CoordError(msg)
    # Width promotion within one dtype kind is lossless (i4+i8, f4+f8,
    # M8[s]+M8[ns]); mixing kinds (e.g. int64 + float64) can silently alter
    # values (ints above 2**53), so it is rejected outright. A grid counts
    # in nanosecond ticks, so a finer time unit beside one is refused too.
    kinds = {np.dtype(x.dtype).kind for x in coords}
    dtypes = {np.dtype(x.dtype) for x in coords}
    nano = {x for x in dtypes if x.str.endswith("[ns]")}
    if len(kinds) > 1 or (nano and nano != {np.result_type(*dtypes)}):
        msg = f"Concatenated coordinates must share compatible dtypes, got {dtypes}."
        raise CoordError(msg)
    if len({get_quantity(x.units) for x in coords}) > 1:
        msg = "All concatenated coordinates must have the same units."
        raise CoordError(msg)


def _check_chain(coords, ascending: bool) -> None:
    """Validate direction consistency and strict non-overlap."""
    for coord in coords:
        if len(coord) > 1 and (coord.sorted != ascending):
            msg = "All concatenated coordinates must share a sort direction."
            raise CoordError(msg)
    for prev, nxt in itertools.pairwise(coords):
        good = nxt.min() > prev.max() if ascending else nxt.max() < prev.min()
        if not good:
            msg = (
                "Concatenated coordinates must be monotonic and non-overlapping; "
                f"({nxt.min()}, {nxt.max()}) overlaps or precedes "
                f"({prev.min()}, {prev.max()})."
            )
            raise CoordError(msg)


def _labels_to_runs(values, step) -> tuple[tuple, np.dtype]:
    """
    The runs a monotonic array of labels states, read exactly.

    Each maximal evenly sampled stretch becomes a grid and each sampling
    break a run boundary. A dense array whose runs would outnumber a tenth
    of its samples keeps its labels as one stored run, which is faster to
    build, smaller, and no less exact.
    """
    if len(values) < (3 if _is_null(step) else 2):
        return (values,), values.dtype
    diffs = _diffs(values)
    signed = step
    if _is_null(step):
        # A diff belongs to a uniform run when it matches a neighbouring
        # diff; isolated diffs are seams (gaps or sampling changes).
        eq_next = diffs[:-1] == diffs[1:]
        in_run = np.zeros(len(diffs), dtype=bool)
        in_run[1:] |= eq_next
        in_run[:-1] |= eq_next
        splits = np.flatnonzero(~in_run) + 1
        if not np.any(in_run):
            return (values,), values.dtype
    else:
        # every spacing is a whole number of steps; more than one step
        # between neighbours is a seam with positions missing
        magnitude = np.abs(np.asarray(step))[()]
        signed = magnitude if values[-1] > values[0] else -magnitude
        splits = np.flatnonzero(_on_grid(diffs, signed) != 1) + 1
    dense = len(values) >= _MIN_RUN_GUARD_SIZE
    if dense and len(splits) + 1 > _MAX_RUN_FRACTION * len(values):
        return (values,), values.dtype
    runs: list[Any] = []
    dtypes = [values.dtype]
    for block in np.split(values, splits):
        if _is_null(step):
            # A block's spacings may still differ, so it becomes a grid
            # only where one reproduces every label exactly.
            runs.append(_promoted(block, values.dtype) or block)
            continue
        spec = dict(start=block[0], step=signed, shape=(len(block),))
        grid, dtype = _range_run(spec)
        # A declared step says which grid the labels sit on, not that one
        # restates them; where it does not, they are kept as they are.
        if not _grid_holds(grid, block, dtype):
            runs.append(block)
            continue
        runs.append(grid)
        dtypes.append(dtype)
    return tuple(runs), np.result_type(*dtypes)


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
    if not isinstance(coord, NumericCoord) or coord.evenly_sampled or len(coord) < 2:
        return None
    pieces = _fill_pieces(coord)
    first = pieces[0][1]
    for _, piece in pieces[1:]:
        if not _fill_same_step(coord, first, piece):
            msg = (
                f"Runs are sampled at different steps ({first.step(coord.dtype)} "
                f"and {piece.step(coord.dtype)}); resample them to one step "
                "before filling gaps."
            )
            raise CoordError(msg)
    max_missing = _fill_limit(coord, first, limit, samples)
    # each group: its anchor run, its filled length, and its runs' blocks
    groups: list[list] = []
    for source, piece in pieces:
        stop = source + len(piece)
        if groups:
            anchor, length, blocks = groups[-1]
            position = _fill_position(coord, anchor, piece)
            missing = position - length
            if missing < 0:
                label = coord._run_labels(piece, [0])[0]
                msg = (
                    f"Samples near {label} land on the same grid position "
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
    runs, out, offset = [], [], 0
    for anchor, length, blocks in groups:
        if len(blocks) == 1:  # untouched: keep the labels as they are
            kept = coord._slice_runs(range(blocks[0][0], blocks[0][1]))
            assert isinstance(kept, NumericCoord)
            runs.extend(kept.runs)
        else:
            runs.append(anchor.resized(length, coord.dtype))
        out.extend((start, stop, offset + pos) for start, stop, pos in blocks)
        offset += length
    return coord._with_runs(runs), tuple(out)


def _fill_pieces(coord: NumericCoord) -> list[tuple[int, Grid]]:
    """The runs of consecutive grid positions, each with its source offset."""
    pieces: list[tuple[int, Grid]] = []
    grids = [x.step(coord.dtype) for x in coord.runs if isinstance(x, Grid)]
    for run, offset in zip(coord.runs, coord._run_offsets()):
        if isinstance(run, Grid):
            pieces.append((int(offset), run))
            continue
        step = coord.step
        if _is_null(step) and len(run) == 1 and grids:
            step = grids[0]  # a lone sample takes the other runs' step
        if _is_null(step):
            msg = (
                "Filling gaps needs a coordinate with a declared step; this one "
                "has none. Use snap_coords or resample to put it on a grid first."
            )
            raise CoordError(msg)
        values = coord._run_labels(run)
        counts = _on_grid(_diffs(values), step)
        edges = np.flatnonzero(counts != 1) + 1
        for first, stop in itertools.pairwise([0, *edges.tolist(), len(values)]):
            grid, _ = _range_run(
                dict(start=values[first], step=step, shape=(stop - first,))
            )
            # floats pass _on_grid per spacing; the run must not drift
            labels = grid.labels(np.arange(stop - first), coord.dtype)
            drift = np.abs(labels - values[first:stop])
            if values.dtype.kind == "f" and np.max(drift) > abs(step) / 2:
                msg = (
                    f"Values drift more than half a step from the grid of "
                    f"step {step}; use snap_coords before filling gaps."
                )
                raise CoordError(msg)
            pieces.append((int(offset) + first, grid))
    return pieces


def _fill_same_step(coord: NumericCoord, first: Grid, other: Grid) -> bool:
    """
    Whether a run shares the first run's step.

    Exactly for ticks; for floats, closely enough that the run drifts from
    the first run's grid by a negligible fraction of a step.
    """
    if first.exact and other.exact:
        return first.step_exact(coord.dtype) == other.step_exact(coord.dtype)
    ratio = float(other.step(coord.dtype)) / float(first.step(coord.dtype))
    return bool(abs(ratio - 1) * max(len(other) - 1, 1) <= _GRID_RTOL)


def _fill_position(coord: NumericCoord, anchor: Grid, piece: Grid) -> int:
    """The position on the anchor's grid nearest the piece's first label."""
    label = coord._run_labels(piece, [0])[0]
    if not anchor.exact:
        start = anchor.labels(0, coord.dtype)[()]
        return int(np.round((label - start) / anchor.step(coord.dtype)))
    tick = _to_tick(label)
    after = int(anchor.index_of([tick], forward=True)[0])
    # the labels either side as the integer ticks the grid casts to dtype,
    # so the comparison stays in Python integers and cannot wrap
    ticks = [_to_tick(anchor.labels(pos, coord.dtype)) for pos in (after - 1, after)]
    return after - 1 if abs(ticks[0] - tick) <= abs(ticks[1] - tick) else after


def _fill_limit(coord: NumericCoord, step: Grid, limit, samples: bool):
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
    if step.exact:
        exact = step.step_exact(coord.dtype)
        if is_timedelta64(excess):
            # the limit was rounded to whole nanoseconds; allow that rounding
            excess = Fraction(2 * int(to_int(excess)) + 1, 2 * _NS_PER_S)
        return int(Fraction(excess) // abs(exact))
    scalar = step.step(coord.dtype)
    return math.floor(float(excess) / abs(float(scalar)) * (1 + _GRID_RTOL))


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
    _rich_style = dascore_styles["coord_array"]

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
        # Deliberately match BaseCoord/CoordRange parameter names for API parity.
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
    runs: tuple[Grid | Labels, ...] | None = None,
    sources: Mapping[str, np.ndarray] | None = None,
    snap: bool = True,
    step_numerator: int | None = None,
    step_denominator: int | None = None,
    origin_offset: int | None = None,
) -> BaseCoord:
    """
    Return a coordinate from provided inputs.

    Parameters
    ----------
    data
        An array indicating the values or an integer to specify the length
        of a partial coordinate.
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
        The runs (see [`NumericCoord`](`dascore.core.coords.NumericCoord`))
        the coordinate holds, normally from a dumped coordinate.
    sources
        The arrays the stored runs window, keyed by id; they accompany
        ``runs`` in a dumped coordinate. An id is taken on trust rather
        than recomputed, so a dump is a contract: pass back the arrays it
        named, not arrays of your own under its keys.
    snap
        If True (default), nearly evenly sampled data is read as one grid.
        If False, the data is read exactly: each evenly sampled stretch
        becomes a run of its own and no label is ever moved.
    step_numerator, step_denominator, origin_offset
        The exact grid of an integer or time range in ticks (see
        [`Grid`](`dascore.core.coords.Grid`)). Normally these come from a
        dumped coordinate; pass ``step`` as a `Fraction` or
        ``(numerator, denominator)`` tuple to state a fractional step.

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
            # a stated grid numerator is the step, in ticks
            stated = step if step is not None else step_numerator
            if any([start is None, stop is None, stated is None]):
                msg = "When data is not defined, start, stop, and step must be."
                raise CoordError(msg)

    def _new_max(data, min, step):
        """Get the new length to use."""
        # for int based data types we need to modify the end time
        # otherwise this will just go nuts
        dtype = getattr(min, "dtype", None)
        if dtype_time_like(dtype) or np.issubdtype(dtype, np.integer):
            max = min + (len(data) - 1) * step
        else:
            max = data[-1]
        return max

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

    def _maybe_get_start_stop_step(data):
        """Get start, stop, step, is_monotonic."""
        data = np.asarray(data)
        # special case for ndim arrays.
        if data.ndim > 1:
            return None, None, None, False
        view2 = data[1:]
        view1 = data[:-1]
        is_monotonic = is_strictly_monotonic(data)
        # the array cannot be evenly sampled if it isn't monotonic
        if is_monotonic:
            try:
                diffs = view2 - view1
            except TypeError:
                return None, None, None, False
            # sort once and derive the unique values from the sorted array
            # (np.unique would sort a second copy).
            sorted_diffs = np.sort(diffs)
            if sorted_diffs[0] == sorted_diffs[-1]:  # all diffs equal
                unique_diff = sorted_diffs[:1]
            else:
                mask = np.empty(len(sorted_diffs), dtype=np.bool_)
                mask[0] = True
                np.not_equal(sorted_diffs[1:], sorted_diffs[:-1], out=mask[1:])
                unique_diff = sorted_diffs[mask]
            if len(unique_diff) == 1 or all_diffs_close_enough(unique_diff):
                _min = data[0]
                # this is a poor man's median that preserves dtype
                _step = sorted_diffs[len(sorted_diffs) // 2]
                _max = _new_max(data, _min, _step)
                return _min, _max + _step, _step, is_monotonic
        return None, None, None, is_monotonic

    if segments is not None:
        # shape/dtype/step are derived fields, so they legitimately appear
        # alongside segments when round-tripping a model_dump (e.g. through
        # CoordManager); ignore them here.
        others = (data, values, start, min, stop, max)
        if any(x is not None for x in others):
            msg = "segments cannot be combined with other coordinate value inputs."
            raise CoordError(msg)
        out = concat_coords(*segments, units=units)
        if not _is_null(step) and not _is_null(out.step) and step != out.step:
            msg = f"step {step} contradicts the segments' step {out.step}."
            raise CoordError(msg)
        return out
    if runs is not None:
        return NumericCoord(
            runs=runs, sources=sources or {}, units=units, dtype=dtype, step=step
        )

    data = _get_array(data, values)
    shape = _get_shape(shape)
    spec = dict(
        start=start,
        stop=stop,
        step=step,
        shape=shape,
        step_numerator=step_numerator,
        step_denominator=step_denominator,
        origin_offset=origin_offset,
    )
    if data is None and shape is not None:
        attrs = dict(
            shape=shape, start=start, stop=stop, step=step, units=units, dtype=dtype
        )
        # A 1D shape with two of start/stop/step states a range; anything
        # less (or more dimensions) is a partial coord. A range that then
        # fails to validate is an error, not a partial.
        stated = sum(not _is_null(x) for x in (start, stop, step))
        stated += step_numerator is not None
        if len(shape) != 1 or shape[0] == 0 or stated < 2:
            return CoordPartial(**attrs)
        try:
            return _range_coord(spec, units)
        except (ValidationError, CoordError):
            # A float range that cannot be built is still a partial, as it
            # always was; an exact grid raises only for what is wrong.
            if _exact_dtype(start, stop, step, shape) is not None:
                raise
            return CoordPartial(**attrs)

    # maybe convert min/max to start stop.
    if start is None and min is not None:
        spec["start"] = start = min
    if stop is None and max is not None:
        spec["stop"] = stop = max
    _check_data_compatibility(data, start, stop, step)
    if data is None:
        return _range_coord(spec, units)
    # a data array was passed; read the runs it states
    if isinstance(data, dc.units.Quantity):  # handle attached units
        data, maybe_units = data.magnitude, data.units
        units = units if units is not None else maybe_units
    if isinstance(data, (int | np.integer)):
        return CoordPartial(
            shape=_get_shape(data),
            start=start,
            stop=stop,
            step=step,
            units=units,
            dtype=dtype,
        )
    if isinstance(data, BaseCoord):  # just return coordinate
        return data
    # An exact read takes a squeezed sample as one label, as readers did
    if not isinstance(data, np.ndarray) or not (snap or data.ndim):
        data = np.atleast_1d(data)
    kind = _get_coord_kind(data)
    if kind == "string":
        if units not in (None, ""):
            _raise_string_coord_error("unit conversion")
        if step not in (None, "") and not pd.isnull(step):
            _raise_string_coord_error("range operations")
        return CoordString(values=data)
    if kind == "empty":
        return CoordPartial(
            shape=data.shape, units=units, step=step, dtype=dtype or data.dtype
        )
    if kind == "single":
        # a lone sample with a step is the first sample of a grid
        if not _is_null(step):
            return _range_coord(dict(start=data[0], step=step, shape=(1,)), units)
        return NumericCoord.from_labels(data, units=units)
    if not _is_null(step):
        # a declared step is a claim about the grid, so the labels are read
        # against it exactly rather than fitted
        return _exact_coord(data, step=step, units=units)
    if snap:
        start, stop, step, monotonic = _maybe_get_start_stop_step(data)
        if start is not None:
            out = _range_coord(dict(start=start, stop=stop, step=step), units)
            # The change_length call helps with float off by one issues.
            return out.change_length(len(data))
    else:
        monotonic = data.ndim == 1 and is_strictly_monotonic(data)
        if monotonic and not pd.isnull(data).any():
            return _exact_coord(data, units=units)
    if not monotonic and np.all(pd.isnull(data)):
        # The values say nothing, but their type still does: an array of
        # NaT came from datetimes and should stay datetimes, as the empty
        # case above also keeps. Only a kind whose null the values can
        # actually hold is recorded; an object array of Nones has no null
        # but NaN, so claiming "object" would state a dtype which `values`
        # then contradicts.
        if dtype is None and data.dtype.kind in "fmM":
            dtype = data.dtype
        return CoordPartial(shape=data.shape, units=units, dtype=dtype)
    return NumericCoord.from_labels(data, units=units)


def _range_coord(spec: dict, units) -> BaseCoord:
    """The evenly sampled coordinate a set of range inputs describes."""
    grid, dtype = _range_run(spec)
    return NumericCoord(runs=(grid,), units=units, dtype=dtype)


def _exact_coord(values, step=None, units=None) -> BaseCoord:
    """
    The coordinate holding these labels exactly, runs and all.

    Monotonic labels keep their runs; anything else keeps them as one
    stored run.
    """
    values = np.asarray(values)
    if not _is_null(step):
        step = _declared_step(step, values.dtype)
        if values.ndim != 1 or not is_strictly_monotonic(values):
            msg = "A declared step needs one-dimensional, monotonic values."
            raise CoordError(msg)
    runs, dtype = _labels_to_runs(values, step)
    return NumericCoord(runs=runs, units=units, dtype=dtype, step=step)
