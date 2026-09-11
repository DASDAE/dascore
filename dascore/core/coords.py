"""Machinery for coordinates.

See ['Coordinate Internals'](`docs/notes/coordinate_internals.qmd`) for the
current coord-family and string-coordinate design notes.
"""

from __future__ import annotations

import abc
import hashlib
import itertools
import json
import math
import re
from collections.abc import Mapping, Sequence, Sized
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from operator import gt, lt
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Self, cast, overload

import numpy as np
import pandas as pd
from pydantic import (
    Field,
    ValidationError,
    field_serializer,
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
from dascore.utils.gaps import GapTolerance
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
    fingerprint: str | None = None
    # The exact grid of a CoordRange, in ticks; None for other coords.
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
        """Return True when the summary can reconstruct a CoordRange."""
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

    def to_coord(self) -> CoordRange:
        """Convert to coord range, if possible."""
        if not self.is_range_like:
            msg = "Cannot convert summary which is not evenly sampled to coord."
            raise CoordError(msg)
        step = self.step
        assert step is not None  # is_range_like above rules out a null step
        if (num := self.step_numerator) is not None:
            if self.len is None:
                msg = "An exact grid summary needs its length to rebuild a coord."
                raise CoordError(msg)
            return CoordRange(
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
        return CoordRange(start=start, stop=stop, step=step, units=self.units)


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
    def check_time_units(cls, data: Any) -> Any:
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
        data1, data2 = self.data, other.data
        intersection = np.intersect1d(data1, data2)
        coord1, slice1 = self.order(intersection)
        coord2, slice2 = other.order(intersection)
        return coord1, coord2, slice1, slice2

    @overload
    def __getitem__(self, item: int | np.integer) -> Any: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray) -> Self: ...

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
        return self.data

    def __hash__(self):
        """Disable Python hash semantics in favor of explicit fingerprints."""
        msg = "Coordinates are not hashable; use `fingerprint()` for stable IDs."
        raise TypeError(msg)

    def _get_fingerprintable_coord(self) -> Self:
        """Return a coordinate normalized for stable fingerprinting."""
        if self.units is None or dtype_time_like(self.dtype):
            return self
        # The unguarded conversion, deliberately. A coord already in base
        # units matches the guard and would come back with whatever dtype
        # it happens to have, while one which is not converts to floats --
        # so an integer range in metres and the same range in centimetres
        # would fingerprint differently. Converting both is what puts them
        # in one numeric form.
        _, units = get_factor_and_unit(self.units, simplify=True)
        return self._convert_units(units)

    def _hash_scalar(self, value, name: str = "start") -> tuple[str, str | None]:
        """
        Return a dtype-aware scalar hash token.

        The value is first conformed to the coordinate's own dtype, since
        a fingerprint identifies *values*, not how they were spelled: a
        range whose start was given as `0` holds the same coordinate as
        one given `0.0`, and a step of four milliseconds is the step of
        four million nanoseconds. Without this they would be stored under
        different identities and never deduplicate.
        """
        if value is None:
            return ("none", None)
        dtype = np.dtype(self.dtype) if self.dtype else None
        if dtype is not None:
            value = _conformed(value, _scalar_dtype(dtype, name))
        return ("scalar", hash_array(np.asarray([value])))

    @staticmethod
    def _coord_identity(coord: BaseCoord) -> str:
        """Return a stable identifier for one coordinate class."""
        cls = coord.__class__
        return f"{cls.__module__}.{cls.__qualname__}"

    @abc.abstractmethod
    def _fingerprint_components(self) -> tuple[Any, ...]:
        """Return subclass-specific fingerprint components."""

    @cached_method
    def fingerprint(self) -> str:
        """
        Return a stable fingerprint whose matches imply coord equality.

        Notes
        -----
        Fingerprints are designed for stable identifiers, not tolerant
        comparison. As a result, coordinates that are approximately equal can
        still have different fingerprints.
        """
        coord = self._get_fingerprintable_coord()
        payload = (
            self._coord_identity(coord),
            coord.unit_str,
            *coord._fingerprint_components(),
        )
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

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
    def limits(self) -> tuple[Any, Any]:
        """Returns a numpy datatype."""
        return self.min(), self.max()

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

    def simplify(self, tolerance=None) -> BaseCoord:
        """
        Return the simplest coordinate representing the same values.

        Unlike [`snap`](`dascore.core.coords.BaseCoord.snap`), which forces a
        uniform coordinate with unbounded interior error, simplify never moves
        any value by more than `tolerance`.

        Parameters
        ----------
        tolerance
            The maximum amount any coordinate value may change. For time-like
            coordinates this is a timedelta (numeric values interpreted as
            seconds). None or 0 permit only exact (lossless) simplifications.

        Notes
        -----
        Most coordinates are already in their simplest form and return
        themselves. [`CoordSegmented`](`dascore.core.coords.CoordSegmented`)
        re-fits its segments as evenly sampled ranges wherever the fit error
        stays within tolerance, possibly collapsing to a single
        [`CoordRange`](`dascore.core.coords.CoordRange`).
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
        time), as [`simplify`](`dascore.core.coords.BaseCoord.simplify`)
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
        For CoordRange stop will be max + step.
        """

    def update_data(
        self,
        data: ArrayLike | np.ndarray | None = None,
        values: ArrayLike | np.ndarray | None = None,
        **kwargs,
    ) -> BaseCoord:
        """
        Update the data of the coordinate.

        Parameters
        ----------
        data
            A new array to use.
        values
            Alias for data.
        """
        if data is None and values is None:
            return self
        data = values if data is None else data
        units = kwargs.get("units")
        return get_coord(data=data, units=units)

    def new(self, **kwargs):
        """Update coordinate."""
        info = self.model_dump(exclude_unset=True, exclude_defaults=True)
        if "data" in kwargs:
            kwargs["values"] = kwargs.pop("data")
        if "values" in kwargs:
            info.pop("shape", None)

        info.update(kwargs)
        return get_coord(**info)

    @property
    def data(self):
        """Return the internal data. Same as values attribute."""
        return self.values

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

    def get_slice_tuple(
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
        array = self.data[indexer]
        return get_coord(data=array, units=self.units)

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
            fingerprint=self.fingerprint(),
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
            out = out.update_limits(**update_fields).update_data(**update_fields)
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
        from `fingerprint()`, which is stricter and intended for stable IDs.

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
        # Ranges (the evenly sampled coords) with identical start/stop/step
        # have identical values; this avoids materializing and comparing
        # the value arrays.
        if isinstance(self, CoordRange) and isinstance(other, CoordRange):
            same = (
                self.start == other.start
                and self.stop == other.stop
                and self.step == other.step
            )
            if same:
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
            coord_data = self.data
            if dtype_time_like(coord_data):
                result = _reduce_time_like(func, coord_data)
            else:
                result = func(self.data)
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
        # We init a temporary array just to get numpy to do the
        # indexing. There is probably a faster way but this is robust.
        dummy = np.empty(self.shape)[item]
        return self.__class__(shape=dummy.shape)

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
                args = self.get_slice_tuple(args, relative=False)
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
            fingerprint=self.fingerprint(),
        )

    def _fingerprint_components(self) -> tuple[Any, ...]:
        """Return the scalar payload needed to fingerprint partial coords."""
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


# A float spacing this close to a whole number of steps is on the grid;
# a float grid such as 0.1 cannot be held exactly, an off-grid label can.
_GRID_RTOL = 1e-6
# The dense-array guard. Stored arrays often carry sub-step jitter
# (GPS-stamped DAS time), so run detection would give roughly one run per
# sample; at or past this many samples, an array whose runs would
# outnumber this fraction of them keeps its values as one array (with
# its declared step), which is faster to build, smaller, and no less
# exact.
_MIN_SEGMENT_GUARD_SIZE = 1_000
# A segmented coordinate's summary carries its runs only up to this many,
# since each becomes an index row; past it the summary is an envelope.
_MAX_SUMMARY_RUNS = 256
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

    The grid is reduced by the common divisor of all three terms, never of
    the step alone: from (num 3, den 2, offset 1) the stride-two grid (6, 2,
    1) must stay as it is, since (3, 1, 0) keeps the labels but moves the
    origin by half a tick. The labels must fit the dtype: a grid that would
    wrap an int8 or overflow int64 arithmetic is refused.
    """
    g = math.gcd(num, den, offset)
    num, den, offset = num // g, den // g, offset // g
    dtype = np.dtype(dtype)
    info = np.iinfo(np.int64 if dtype.kind in "mM" else cast("Any", dtype))
    stop_tick = start_tick + (offset + count * num) // den
    low, high = min(start_tick, stop_tick), max(start_tick, stop_tick)
    if abs(count * num) + den >= 2**63 or low < info.min or high > info.max:
        msg = (
            f"A grid of {count} samples with step {num}/{den} ticks from "
            f"{start_tick} exceeds the {dtype} range."
        )
        raise CoordError(msg)
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
        if len(shape) != 1:
            msg = "Coord range only works for 1D coords."
            raise CoordError(msg)
        count = int(shape[0])
        if count < 1:
            msg = "A range coordinate needs at least one sample."
            raise CoordError(msg)
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
        if start_tick is None or stop_tick is None or count is None:
            msg = (
                "Three of ('start', 'stop', 'step', 'shape') are required "
                f"to create CoordRange. You passed {values}"
            )
            raise CoordError(msg)
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
        if start_tick is None or stop_tick is None:
            msg = "start, stop, and step, or a shape, are needed."
            raise CoordError(msg)
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
    valid_count = sum(not pd.isnull(x) for x in _attrs)
    if valid_count < 3:
        msg = (
            f"Three of {req_values} are required to create CoordRange. "
            f"You passed {values}"
        )
        raise CoordError(msg)
    # Now get start, stop, step from length, if provided.
    start, stop, step, shape = _attrs
    if not pd.isnull(shape):
        shape = tuple(iterate(shape))
        if len(shape) != 1:
            msg = "Coord range only works for 1D coords."
            raise CoordError(msg)
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
    try:
        same_sign = ((step > zero) == (diff > zero)) & ((step < zero) == (diff < zero))
    except TypeError:  # mixed types (e.g. datetime.timedelta vs int zero)
        same_sign = np.sign(to_float(step)) == np.sign(to_float(diff))
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


class CoordRange(BaseCoord):
    """
    A coordinate representing a range of evenly sampled data.

    Nanosecond time and integer ranges hold their sampling exactly: labels
    are integer ticks (nanoseconds, or the integers themselves), the ideal
    grid has a spacing of ``step_numerator / step_denominator`` ticks and an
    origin ``origin_offset / step_denominator`` ticks after ``start``, and
    each label is the floor of its ideal position. Slices, strides,
    reversals, and selections therefore reproduce the labels of the original
    exactly, and a 1024 Hz grid never drifts the way a step rounded to
    976562 ns would. ``step`` reports the nearest whole-tick spacing;
    ``step_exact`` the fraction, in seconds for time.

    Float ranges, and time ranges in units other than nanoseconds, keep a
    scalar step and linearly spaced labels; their grid fields are None.

    Parameters
    ----------
    start
        The starting value.
    stop
        The ending value, exclusive.
    step
        The step between values; for exact grids also a `Fraction` or a
        ``(numerator, denominator)`` tuple in coordinate units.
    shape
        The sample count.
    step_numerator, step_denominator, origin_offset
        The exact grid in ticks, normally supplied only when rebuilding a
        dumped coordinate.
    """

    start: Any = None
    stop: Any = None
    step: Any = None
    step_numerator: int | None = None
    step_denominator: int | None = None
    origin_offset: int | None = None
    _evenly_sampled = True
    _rich_style = dascore_styles["coord_range"]

    @model_validator(mode="before")
    @classmethod
    def validate_start_stop_step_len(cls, values):
        """Resolve the range from any three of start, stop, step, and shape."""
        get = values.get
        dtype = _exact_dtype(get("start"), get("stop"), get("step"), get("shape"))
        fields = (
            _float_fields(values) if dtype is None else _exact_fields(values, dtype)
        )
        values.update(fields)
        return values

    # --- the grid

    @property
    def _exact(self) -> bool:
        """Whether the labels come from an integer grid."""
        return self.step_numerator is not None

    @property
    def _grid_terms(self) -> tuple[int, int, int]:
        """The numerator, denominator, and origin offset of an exact grid."""
        assert self._exact, "only an exact grid has terms"
        terms = (self.step_numerator, self.step_denominator, self.origin_offset)
        return cast("tuple[int, int, int]", terms)

    @property
    @cached_method
    def _start_tick(self) -> int:
        return _to_tick(self.start)

    @property
    def _ideal_origin(self) -> Fraction:
        """The ideal origin of an exact grid, in ticks."""
        _, den, offset = self._grid_terms
        return Fraction(self._start_tick * den + offset, den)

    def _labels(self, indices) -> np.ndarray:
        """The labels at these indices, which may lie outside the coordinate."""
        indices = np.asarray(indices)
        if self._exact:
            num, den, offset = self._grid_terms
            ticks = (offset + indices.astype(np.int64) * num) // den
            return np.asarray(self._start_tick + ticks).astype(self.dtype)
        if len(self) == 1 or np.dtype(self.dtype).kind in "mMO":
            return np.asarray(self.start + indices * self.step, dtype=self.dtype)
        # Match linspace's inferred floating dtype, rounding, and exact endpoint.
        last = self.stop - self.step
        dtype = np.result_type(self.start, last, 0.0)
        delta = np.subtract(last, self.start, dtype=dtype)
        step = delta / (len(self) - 1)
        scaled = indices.astype(dtype)
        scaled = scaled / (len(self) - 1) * delta if step == 0 else scaled * step
        values = np.where(indices == len(self) - 1, last, scaled + self.start)
        return values.astype(self.dtype)

    def _sliced(self, first: int, stride: int, count: int) -> Self:
        """The coordinate of the samples first, first + stride, ... (count)."""
        if self._exact:
            num, den, offset = self._grid_terms
            q, offset = divmod(offset + first * num, den)
            fields = _grid_fields(
                self._start_tick + q, stride * num, den, offset, count, self.dtype
            )
        else:
            start, step = self.start + first * self.step, self.step * stride
            fields = dict(
                start=start, stop=start + step * count, step=step, shape=(count,)
            )
        return self._construct(fields)

    def _construct(self, fields: dict) -> Self:
        """Build from fields already on a known grid, skipping re-validation."""
        # check_time_units forces time-like coords to seconds, testing start
        # for truthiness; a coord starting at exactly zero is left alone.
        start, dtype = fields["start"], fields.get("dtype", self.dtype)
        units = self.units
        if start and (is_timedelta64(start) or is_datetime64(start)):
            units = _second_quantity()
        return self.model_construct(
            _fields_set=set(self.model_fields_set) | set(fields),
            **{"units": units, "dtype": dtype, **fields},
        )

    # --- identity and display

    @property
    def step_exact(self) -> Fraction | None:
        """The exact spacing in coordinate units (seconds for time), or None."""
        if not self._exact:
            return super().step_exact
        num, den, _ = self._grid_terms
        return Fraction(num, den) / (_NS_PER_S if dtype_time_like(self.dtype) else 1)

    def _fingerprint_components(self) -> tuple[Any, ...]:
        """The scalar payload of a range, plus the grid when it is not whole ticks."""
        components = (
            self.shape,
            self._hash_scalar(self.start, "start"),
            self._hash_scalar(self.stop, "stop"),
            self._hash_scalar(self.step, "step"),
        )
        if self._exact and self.step_denominator != 1:
            return (*components, self._grid_terms)
        return components

    def _get_fingerprintable_coord(self) -> Self:
        if self._exact and self.step_denominator != 1:
            return self  # units cannot convert; see _convert_units
        return super()._get_fingerprintable_coord()

    def _repr_fields(self) -> tuple[tuple[str, Text, bool], ...]:
        fields = super()._repr_fields()
        if not self._exact or self.step_denominator == 1:
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

    def to_summary(self, dims=()) -> CoordSummary:
        """Get the summary info about the coord, exact grid included."""
        summary = super().to_summary(dims=dims)
        if not self._exact:
            return summary
        return summary.model_copy(
            update={name: getattr(self, name) for name in _EXACT_GRID_FIELDS}
        )

    # --- evaluation

    @property
    @cached_method
    def values(self) -> ArrayLike:
        """Return the values of the coordinate as an array."""
        return array(self._labels(np.arange(len(self))))

    def _get_index_values(self, indices):
        """Evaluate only requested samples using the same grid as ``values``."""
        indices = np.asarray(indices)
        return self._labels(np.where(indices < 0, indices + len(self), indices))

    @cached_method
    def __len__(self):
        return self.shape[0]

    def _min(self):
        """Return min value."""
        return np.min(self._labels([0, len(self) - 1]))

    def _max(self):
        """Return max value in range."""
        return np.max(self._labels([0, len(self) - 1]))

    @property
    def sorted(self) -> bool:
        """Returns true if sorted in ascending order."""
        zero = _TD64_ZERO if is_timedelta64(self.step) else 0
        return self.step >= zero

    @property
    def reverse_sorted(self) -> bool:
        """Returns true if sorted in descending order."""
        return not self.sorted

    # --- indexing and selection

    def __getitem__(self, item):
        if isinstance(item, int | np.integer):
            if item >= len(self) or item < -len(self):
                raise IndexError(f"{item} exceeds coord length of {self}")
            return self._get_index_values(item)[()]
        if isinstance(item, slice):
            start = None if item.start is ... else item.start
            end = None if item.stop is ... else item.stop
            # A (strided) slice of a range is still a range; see #567.
            indices = range(len(self))[slice(start, end, item.step)]
            if len(indices):
                return self._sliced(indices.start, indices.step, len(indices))
            return get_coord(data=np.empty(0, dtype=self.dtype), units=self.units)
        return get_coord(data=self.values[item], units=self.units)

    def index(self, indexer, axis: int | None = None) -> BaseCoord:
        """Index the coordinate; a slice keeps the grid, as ``coord[slice]`` does."""
        if isinstance(indexer, slice) and not axis:
            return self[indexer]
        return super().index(indexer, axis=axis)

    def coord_range(self, extend: bool = True):
        """The span of the coordinate; extended, to its exclusive end."""
        if not extend or not self._exact:
            return super().coord_range(extend=extend)
        first, end = self._labels([0, len(self)])
        return np.abs(end - first)

    @compose_docstring(doc=get_docstring(BaseCoord.change_length))
    def change_length(self, length: int) -> Self:
        """
        {doc}
        """
        length = _validate_new_length(length)
        return self if len(self) == length else self._sliced(0, 1, length)

    def select(
        self, args, relative=False, samples=False
    ) -> tuple[BaseCoord, slice | ArrayLike]:
        """
        Apply select, return selected coords and index to apply to array.

        Can return a CoordDegenerate if selection is outside of range.
        """
        if is_array(args):
            return self._select_by_array(args, relative=relative, samples=samples)
        elif samples:
            return self._select_by_samples(args)
        args = self.get_slice_tuple(args, relative=relative)
        start = self._get_index(args[0], forward=self.sorted)
        stop = self._get_index(args[1], forward=self.reverse_sorted)
        if self.reverse_sorted:
            start, stop = stop, start
        # we add 1 to stop in slice since its upper limit is exclusive
        start = None if start == 0 else start
        data = slice(start, (stop + 1) if stop is not None else stop)
        if self._slice_degenerate(data):
            return self.empty(), slice(0, 0)
        return self[data], data

    def sort(self, reverse=False) -> tuple[BaseCoord, slice | ArrayLike]:
        """Sort the contents of the coord. Return new coord and slice for sorting."""
        if reverse == self.reverse_sorted:
            return self, slice(None)
        return self[::-1], slice(None, None, -1)

    def _get_zero_step_index(self, value, forward):
        """
        Get the index of a value for a coord with a step of 0.

        Every sample equals start, so the index is the first sample or one
        just outside the coord, which makes the selection degenerate.
        """
        if forward:  # index of the first sample >= value
            return 0 if value <= self.start else len(self)
        return 0 if value >= self.start else -1

    def _index_of(self, ticks, forward: bool):
        """
        The sample index each label tick of an exact grid maps to.

        Forward: the first index whose label is at or past the tick in the
        coordinate's direction. Otherwise the last index whose label is at
        or before it. Exact, since a label is the floor of its ideal
        position and a tick is an integer; Python integers, so no overflow.
        """
        num, den, offset = self._grid_terms
        rel = (np.asarray(ticks, dtype=object) - self._start_tick) * den - offset
        if forward == (num > 0):  # label >= tick  <=>  ideal >= tick
            return -((-rel) // num) if num > 0 else rel // num
        # label <= tick  <=>  ideal < tick + 1
        return -((-(rel + den)) // num) - 1 if num > 0 else (rel + den) // num + 1

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

    def _get_index(self, value, forward=True):
        """Get the index corresponding to a value."""
        if (value := self._get_compatible_value(value)) is None:
            return value
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value[()]
        if not self._exact:
            return self._get_float_index(value, forward)
        if self.step_numerator == 0:
            return self._get_zero_step_index(value, forward)
        if isinstance(value, Sized):
            ticks = self._bound_ticks(value, forward)
            return self._index_of(ticks, forward).astype(np.int64)
        if isinstance(value, float | np.floating) and not math.isfinite(value):
            # Past one end of the coordinate; the checks below open that side.
            out = len(self) if (value > 0) == self.sorted else -1
        else:
            out = int(self._index_of(self._bound_ticks(value, forward), forward)[0])
        if (forward and out < 0) or (not forward and out >= len(self)):
            return None
        return out

    def _get_float_index(self, value, forward=True):
        """Get the index corresponding to a value of a float range."""
        start, step = self.start, self.step
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
        if not self._exact:
            return super()._samples_in(duration)
        if dtype_time_like(self.dtype):
            ticks = Fraction(int(to_int(dc.to_timedelta64(duration))))
        else:
            ticks = Fraction(_maybe_unpack(np.asarray(duration)).item())
        num, den, _ = self._grid_terms
        return ticks / abs(Fraction(num, den))

    def _out_of_bounds_indices(self, array) -> np.ndarray:
        """Positions past either end, from the grid."""
        if not self._exact:
            return super()._out_of_bounds_indices(array)
        # Past the last label its floor position, before the first its
        # ceiling, as the truncating division did.
        last = self._labels(len(self) - 1)
        beyond = (array > last) if self.sorted else (array < last)
        return np.where(
            beyond, self._get_index(array, forward=False), self._get_index(array)
        ).astype(np.int64)

    # --- updates

    def _translated(self, delta) -> Self:
        """Shift every label; the grid moves with them."""
        if self._exact:
            num, den, offset = self._grid_terms
            start_tick = self._start_tick + _to_tick(delta)
            fields = _grid_fields(start_tick, num, den, offset, len(self), self.dtype)
        else:
            start, stop = self.start + delta, self.stop + delta
            fields = dict(start=start, stop=stop, step=self.step, shape=self.shape)
        return self._construct(fields)

    def _with_step(self, step) -> CoordRange:
        """The same start and count on a new cadence."""
        if (frac := _fraction_step(step)) is None:
            step = get_compatible_values(step, type(self.step))
        whole_tick = frac is None and (_is_int(step) or is_timedelta64(step))
        if not self._exact or not (frac is not None or whole_tick):
            # Re-validate: the class picks the representation the step needs.
            return CoordRange(
                start=self.start, step=step, shape=self.shape, units=self.units
            )
        if frac is not None:
            num, den = (
                frac * (_NS_PER_S if dtype_time_like(self.dtype) else 1)
            ).as_integer_ratio()
        else:
            num, den = _to_tick(step), 1
        if 0 < abs(num) < den:
            msg = "A step smaller than one tick would repeat labels."
            raise CoordError(msg)
        return self._construct(
            _grid_fields(self._start_tick, num, den, 0, len(self), self.dtype)
        )

    @compose_docstring(doc=get_docstring(BaseCoord.update_limits))
    def update_limits(self, min=None, max=None, step=None, **kwargs) -> BaseCoord:
        """{doc}."""
        if all(x is not None for x in [min, max, step]):
            msg = "At most two parameters can be specified in update_limits."
            raise ValueError(msg)
        out = self
        if min is not None and max is not None:
            # min is the new start, max the new exclusive stop, and the count
            # is kept. An exact grid keeps an exact spacing when it is at
            # least a tick; below that the labels become floats.
            min = get_compatible_values(min, self.dtype)
            max = get_compatible_values(max, self.dtype)
            span = _to_tick(max) - _to_tick(min) if self._exact else 0
            if abs(frac := Fraction(span, len(self))) >= 1:
                num, den = frac.as_integer_ratio()
                fields = _grid_fields(_to_tick(min), num, den, 0, len(self), self.dtype)
                out = self._construct(fields)
            else:
                new_step = (max - min) / len(self)
                out = get_coord(start=min, stop=max, step=new_step, units=self.units)
            return out.new(**kwargs) if kwargs else out
        if step is not None:
            out = out._with_step(step)
        if min is not None:
            min = get_compatible_values(min, self.dtype)
            out = out._translated(min - out.min())
        if max is not None:
            max = get_compatible_values(max, self.dtype)
            out = out._translated(max - out.max())
        return out.new(**kwargs) if kwargs else out

    def new(self, **kwargs):
        """Update coordinate; an exact grid is kept unless the step changes."""
        if "data" in kwargs or "values" in kwargs:
            # new values state their own grid; the range's step is not a
            # claim about them
            data = kwargs.get("data", kwargs.get("values"))
            units = kwargs.get("units", self.units)
            return get_coord(data=data, units=units, step=kwargs.get("step"))
        info = self.model_dump(exclude_unset=True, exclude_defaults=True)
        # A new stop or step re-derives the count, as a range always has.
        if "stop" in kwargs or "step" in kwargs:
            info.pop("shape", None)
        if "step" in kwargs:
            for name in _EXACT_GRID_FIELDS:
                info.pop(name, None)
        return get_coord(**{**info, **kwargs})

    def _convert_units(self, units) -> Self:
        """Convert units, or set units if none exist."""
        if dtype_time_like(self.dtype):  # time units are fixed
            return self
        if self._exact and self.step_denominator != 1:
            msg = (
                "Cannot convert the units of an integer coordinate with a "
                f"fractional step ({self.step_exact}); the result is not an "
                "integer grid."
            )
            raise CoordError(msg)
        start = convert_units(self.start, to_units=units, from_units=self.units)
        stop = convert_units(self.stop, to_units=units, from_units=self.units)
        step = (stop - start) / len(self)
        return self.__class__(start=start, stop=stop, step=step, units=units)


class CoordArray(BaseCoord):
    """
    A coordinate with arbitrary values in an array.

    Can handle any number of dimensions.
    """

    values: ArrayLike
    _rich_style = dascore_styles["coord_array"]

    @model_validator(mode="before")
    @classmethod
    def validate_start_stop_step_len(cls, values):
        """Coerce the needed values from the inputs; check a declared step."""
        data = values["values"]
        values["dtype"] = data.dtype
        values["shape"] = data.shape
        step = values.get("step")
        if _is_null(step):
            values["step"] = None
        else:
            # a declared step is the grid the values sit on, not their
            # spacing; its sign follows the values, as a range's does
            if data.ndim != 1 or not is_strictly_monotonic(data):
                msg = "A declared step needs one-dimensional, monotonic values."
                raise CoordError(msg)
            magnitude = np.abs(np.asarray(_declared_step(step, data.dtype)))[()]
            step = magnitude if len(data) < 2 or data[-1] > data[0] else -magnitude
            _on_grid(_diffs(data), step)
            values["step"] = step
        return values

    def _convert_units(self, units) -> Self:
        """Convert units, or set units if none exist."""
        is_time = np.issubdtype(self.dtype, np.datetime64)
        is_time_delta = np.issubdtype(self.dtype, np.timedelta64)
        if self.units is None or is_time or is_time_delta:
            return self.set_units(units)
        values = convert_units(self.values, units, self.units)
        step = self.step
        if step is not None:
            # a step is a difference, so an affine unit's offset cancels
            anchor = convert_units(step * 0, units, self.units)
            step = convert_units(step, units, self.units) - anchor
        return self.new(units=units, values=values, step=step)

    def select(
        self, args, relative=False, samples=False
    ) -> tuple[BaseCoord, slice | ArrayLike]:
        """Apply select, return selected coords and index for selecting data."""
        if is_array(args):
            return self._select_by_array(args, relative=relative, samples=samples)
        elif samples:
            return self._select_by_samples(args)

        args = self.get_slice_tuple(args, relative=relative)
        values = self.values
        out = np.ones_like(values, dtype=np.bool_)
        val1 = self._get_compatible_value(args[0])
        val2 = self._get_compatible_value(args[1])
        if val1 is not None:
            out = out & (values >= val1)
        if val2 is not None:
            out = out & (values <= val2)
        if not np.any(out):
            return self.empty(), out
        if np.all(out):
            return self, slice(None, None)
        # Convert boolean to int indexes; some consumers (eg lazy file
        # readers) index with these where booleans are not supported.
        if len(self.shape) == 1:
            out = np.arange(len(out))[out]
        return self.new(values=values[out]), out

    def sort(self, reverse=False) -> tuple[BaseCoord, slice | ArrayLike]:
        """Sort the coord to be monotonic (maybe range)."""
        argsort: ArrayLike = np.argsort(self.values)[:: -1 if reverse else 1]
        arg_dict = self.model_dump()
        arg_dict["values"] = self.values[argsort]
        new = get_coord(**arg_dict)
        return new, argsort

    def snap(self):
        """
        Snap the coordinates to evenly sampled grid points.

        This will cause some loss of precision but often makes the coordinate
        much easier to work with. The min/max of the coordinate will remain
        unchanged.
        """
        values = self.values
        min_v, max_v = np.min(values), np.max(values)
        if len(self) == 1:
            # time deltas need to be generated for dt case, hence the subtract
            _zero = self._get_compatible_value(0)
            step = self._get_compatible_value(1) - _zero
            # we just use a step of 1 in case of len 1 coord.
        else:
            dur = max_v - min_v
            is_dt = is_timedelta64(dur)
            # hack to handle dts int division.
            if is_dt:
                _step = float(dur.astype(np.int64)) / (len(self) - 1)
                step = np.timedelta64(int(np.round(_step)), "ns")
            else:
                step = dur / (len(self) - 1)
            zero = dc.to_timedelta64(0) if is_timedelta64(step) else 0
            assert step > zero
        if self.reverse_sorted:
            step = -step
            start, stop = max_v, min_v + step
        else:
            start, stop = min_v, max_v + step
        # Get potential output, ensure it is the same length as original.
        out = get_coord(start=start, stop=stop, step=step, units=self.units)
        return out.change_length(len(self))

    @compose_docstring(doc=get_docstring(BaseCoord.update_limits))
    def update_limits(self, min=None, max=None, step=None, **kwargs) -> BaseCoord:
        """{doc}."""
        if sum(x is not None for x in [min, max, step]) > 1:
            msg = "At most one parameter can be specified in update_limits."
            raise ValueError(msg)
        out = self
        if not pd.isnull(step) and len(self):
            out = self.snap().update_limits(step=step)
        elif min is not None:
            diff = min - self.min()
            vals = self.values + diff
            out = get_coord(data=vals, units=self.units)
        elif max is not None:
            diff = max - self.max()
            vals = self.values + diff
            out = get_coord(data=vals, units=self.units)
        return out.new(**kwargs)

    def __getitem__(self, item) -> Self:
        out = self.values[item]
        if not np.ndim(out):
            return out
        # a declared step survives only an order it can be held against
        step = self.step if out.ndim == 1 and is_strictly_monotonic(out) else None
        return self.__class__(values=out, units=self.units, step=step)

    def _min(self):
        """Return min value."""
        return np.nanmin(self.values) if self.size else _get_nullish(self.dtype)

    def _max(self):
        """Return max value in range."""
        return np.nanmax(self.values) if self.size else _get_nullish(self.dtype)

    def _fingerprint_components(self) -> tuple[Any, ...]:
        """The array payload, and the grid it declares when it declares one."""
        components: tuple[Any, ...] = (("array", hash_array(self.values)),)
        if not _is_null(self.step):
            components += (("step", self._hash_scalar(self.step, "step")),)
        return components


def _negate_for_search(values):
    """
    Negate values so descending arrays can use ascending searchsorted.

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


class CoordMonotonicArray(CoordArray):
    """A coordinate with strictly increasing or decreasing values."""

    values: ArrayLike
    _rich_style = dascore_styles["coord_monotonic"]
    _sorted = True

    def _expected_spacing(self):
        """The spacing the values are held to: the declared step, else the median."""
        if not _is_null(self.step):
            return self.step
        diffs = _diffs(self.values)
        # the median magnitude, signed with the values' direction, so
        # either orientation judges the same spacings
        median = np.median(np.abs(diffs))
        return median if self.sorted else -median

    def _seams(self) -> list[tuple]:
        """Every neighbour spacing which is not the expected one."""
        values = self.values
        if len(values) < 2:
            return []
        diffs = _diffs(values)
        expected = self._expected_spacing()
        seams = np.flatnonzero(diffs != expected)
        return [(int(i) + 1, values[i], values[i + 1], expected) for i in seams]

    def _holes(self) -> list[tuple]:
        """The grid positions skipped between neighbours."""
        values = self.values
        counts = _on_grid(_diffs(values), self.step)
        return [
            _hole(values[i], self.step, int(counts[i]) - 1)
            for i in np.flatnonzero(counts > 1)
        ]

    def select(
        self, args, relative=False, samples=False
    ) -> tuple[BaseCoord, slice | ArrayLike]:
        """Apply select, return selected coords and index for selecting data."""
        if is_array(args):
            return self._select_by_array(args, relative=relative, samples=samples)
        elif samples:
            return self._select_by_samples(args)

        v1, v2 = self.get_slice_tuple(args, relative=relative)
        # reverse order if reverse monotonic. This is done so when we mult
        # by -1 in _get_index the inverted range is used.
        if self.reverse_sorted:
            v1, v2 = v2, v1
        start = self._get_index(v1, forward=False)
        new_start = start if start is not None and start > 0 else None
        stop = self._get_index(v2, forward=True)
        new_stop = stop if stop is not None and stop < len(self) else None
        # We need to add 1 to end so 1 sample get selected if start == stop
        if new_stop is not None:
            if self.values[new_stop] == v2:
                new_stop = new_stop + 1
        out = slice(new_start, new_stop)
        if self._slice_degenerate(out):
            return self.empty(), slice(0, 0)
        return self.new(values=self.values[out]), out

    def _get_index(self, value, forward=True):
        """
        Get the index corresponding to a value.

        Forward indicates if this is the max (left) value.
        """
        if (new_value := self._get_compatible_value(value)) is None:
            return new_value
        values = np.atleast_1d(self.values)
        # since search sorted only works on ascending monotonic arrays we
        # negative descending arrays to get the same effect.
        if self.reverse_sorted:
            values = _negate_for_search(values)
            new_value = _negate_for_search(new_value)
        # side = "right" if forward else "left"
        # out = np.atleast_1d(np.searchsorted(values, new_value, side=side))
        # Search values. Ensure the returned index is in bounds (eg values GT
        # coord max should still have a range in coords.
        new_value = np.atleast_1d(new_value)
        right = np.searchsorted(values, new_value, side="right")
        # right_ok = (right < len(self)) & (right < 0)
        left = np.searchsorted(values, new_value, side="left")
        left_ok = (left < len(self)) & (left > 0)
        eq = left_ok & (values.take(left, mode="clip") == new_value)
        out = right if forward else left
        # where equal it should also be left values. This makes the function
        # behavior consistent with BaseCoord._get_index.
        if not self.reverse_sorted:
            out[eq] = left[eq]
        return out if is_array(value) else int(out[0])

    def _step_meets_requirement(self, op):
        """Return True is any data increment meets the comp. requirement."""
        vals = self.values
        # we must iterate because first two elements might be equal.
        # but this wont iterate the whole array; just until sort order is found
        for ind in range(1, len(self)):
            if op(vals[ind], vals[ind - 1]):
                return True
        # we consider single valued arrays sorted, but not reverse sorted.
        if len(vals) == 1 and op is gt:
            return True
        return False

    @property
    @cached_method
    def sorted(self):
        """Determine is coord array is sorted in ascending order."""
        return self._step_meets_requirement(gt)

    @property
    @cached_method
    def reverse_sorted(self):
        """Determine is coord array is sorted in descending order."""
        return self._step_meets_requirement(lt)


def _coerce_segment(seg) -> BaseCoord:
    """Coerce a segment input (coord or dumped dict) to a coordinate."""
    if isinstance(seg, BaseCoord):
        return seg
    if isinstance(seg, dict):
        # Round-trip support: rebuild segments from model_dump payloads.
        if seg.get("values") is not None:
            return CoordMonotonicArray(**seg)
        return CoordRange(**seg)
    msg = f"Segments must be coordinates, got {type(seg)}."
    raise CoordError(msg)


def _maybe_promote_segment(seg: BaseCoord) -> BaseCoord:
    """Promote an exactly evenly sampled array segment to a CoordRange."""
    if not isinstance(seg, CoordMonotonicArray) or len(seg) < 2:
        return seg
    values = seg.values
    diffs = np.diff(values)
    if len(np.unique(diffs)) != 1:
        return seg
    step = diffs[0]
    if not _is_null(seg.step) and step != seg.step:
        return seg  # evenly spaced, but not at the grid it declares
    candidate = get_coord(
        start=values[0], stop=values[-1] + step, step=step, units=seg.units
    )
    # Only promote when the range reproduces the values bit-exactly; unlike
    # get_coord inference, segments must never change any value.
    if len(candidate) == len(seg) and np.array_equal(candidate.values, values):
        return candidate
    return seg


def _range_continues(prev: BaseCoord, seg: BaseCoord) -> bool:
    """Return True if seg is the next samples of prev's grid."""
    if not (isinstance(prev, CoordRange) and isinstance(seg, CoordRange)):
        return False
    if prev.step_exact != seg.step_exact:
        return False
    if prev._exact and seg._exact:
        # The ideal origins must line up, not just the rounded labels.
        num, den, _ = prev._grid_terms
        return seg._ideal_origin == prev._ideal_origin + len(prev) * Fraction(num, den)
    return bool(prev.step == seg.step and prev.stop == seg.start)


def _fuse_segments(segments: tuple[BaseCoord, ...]) -> tuple[BaseCoord, ...]:
    """Fuse adjacent segments that continue exactly (normal form)."""
    out = [segments[0]]
    for seg in segments[1:]:
        prev = out[-1]
        if _range_continues(prev, seg):
            out[-1] = prev.change_length(len(prev) + len(seg))
            continue
        both_arrays = isinstance(prev, CoordMonotonicArray) and isinstance(
            seg, CoordMonotonicArray
        )
        if both_arrays and _arrays_continue(prev, seg):
            # Adjacent arrays with no sampling expectation, or one they
            # both meet at the seam, share no boundary; fuse for canonical
            # form.
            values = np.concatenate([prev.values, seg.values])
            out[-1] = CoordMonotonicArray(
                values=values, units=prev.units, step=prev.step
            )
            continue
        out.append(seg)
    return tuple(out)


def _arrays_continue(prev: CoordMonotonicArray, seg: CoordMonotonicArray) -> bool:
    """Whether two array segments may fuse: no declared step, or one met at the seam."""
    if _is_null(prev.step) and _is_null(seg.step):
        return True
    if _is_null(prev.step) or _is_null(seg.step) or prev.step != seg.step:
        return False
    try:
        seam = _diffs(np.concatenate([prev.values[-1:], seg.values[:1]]))
        return bool(_on_grid(seam, prev.step) == 1)
    except CoordError:
        return False  # the same step on offset grids: a seam, not a continuation


def _one_grid(segments, step) -> bool:
    """Whether every seam between runs of ``step`` is a whole number of steps."""
    ascending = segments[0].min() < segments[-1].min() if len(segments) > 1 else True
    for prev, nxt in itertools.pairwise(segments):
        before = prev.max() if ascending else prev.min()
        after = nxt.min() if ascending else nxt.max()
        try:
            _on_grid(np.asarray([after - before]), step)
        except CoordError:
            return False
    return True


def _validate_segment_compat(segments: tuple[BaseCoord, ...]) -> None:
    """Validate segment types, dtypes, and units are compatible."""
    for seg in segments:
        if not isinstance(seg, CoordRange | CoordMonotonicArray):
            msg = (
                f"Segments must be CoordRange or CoordMonotonicArray, got {type(seg)}."
            )
            raise CoordError(msg)
        if not len(seg):
            msg = "Segments must not be empty."
            raise CoordError(msg)
    # Width promotion within one dtype kind is lossless (i4+i8, f4+f8,
    # M8[s]+M8[ns]); mixing kinds (e.g. int64 + float64) can silently alter
    # values (ints above 2**53), so it is rejected outright.
    kinds = {np.dtype(s.dtype).kind for s in segments}
    if len(kinds) > 1:
        dtypes = {np.dtype(s.dtype) for s in segments}
        msg = f"Segments must share compatible dtypes, got {dtypes}."
        raise CoordError(msg)
    units = {get_quantity(s.units) for s in segments}
    if len(units) > 1:
        msg = "All segments must have the same units."
        raise CoordError(msg)


def _validate_segment_chain(segments: tuple[BaseCoord, ...]) -> None:
    """Validate direction consistency and strict non-overlap of segments."""
    multi = [s for s in segments if len(s) > 1]
    ascending = multi[0].sorted if multi else segments[0].min() < segments[-1].min()
    for seg in multi:
        ok = seg.sorted if ascending else seg.reverse_sorted
        if not ok:
            msg = "All segments must be sorted in a consistent direction."
            raise CoordError(msg)
    for prev, nxt in itertools.pairwise(segments):
        if ascending:
            good = nxt.min() > prev.max()
        else:
            good = nxt.max() < prev.min()
        if not good:
            msg = (
                "Segments must be monotonic and non-overlapping; segment "
                f"({nxt.min()}, {nxt.max()}) overlaps or precedes "
                f"({prev.min()}, {prev.max()})."
            )
            raise CoordError(msg)


class CoordSegmented(BaseCoord):
    """
    A coordinate composed of an ordered sequence of monotonic segments.

    Values are the concatenation of the constituent monotonic coordinates. Segment
    boundaries preserve discontinuities such as data gaps without changing values.

    Notes
    -----
    - Direct construction requires at least two segments after normalization;
      use [`concat_coords`](`dascore.core.coords.concat_coords`) (or
      `get_coord(segments=...)`) which returns a plain coordinate when the
      inputs fuse into one segment.
    - Normalization promotes exactly evenly sampled array segments to ranges
      and fuses segments that continue exactly, so equal-valued segmented
      coordinates compare and fingerprint equal regardless of how they
      were assembled.
    - `step` is always None; use
      [`simplify`](`dascore.core.coords.BaseCoord.simplify`) to obtain an
      evenly sampled coordinate with bounded error, or
      [`snap`](`dascore.core.coords.BaseCoord.snap`) to force one.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.core.coords import concat_coords, get_coord
    >>>
    >>> # Two evenly sampled blocks separated by a gap.
    >>> c1 = get_coord(start=0.0, stop=10.0, step=1.0)
    >>> c2 = get_coord(start=15.0, stop=25.0, step=1.0)
    >>> coord = concat_coords(c1, c2)
    >>> assert coord.segment_count == 2
    >>> assert coord.min() == 0.0 and coord.max() == 24.0
    >>>
    >>> # Exactly contiguous blocks fuse back to a single range.
    >>> c3 = get_coord(start=10.0, stop=20.0, step=1.0)
    >>> fused = concat_coords(c1, c3)
    >>> assert fused == get_coord(start=0.0, stop=20.0, step=1.0)
    """

    # Note: typed as BaseCoord (not a union) because pydantic union dispatch
    # runs member before-validators on foreign instances; the model validator
    # below enforces the concrete segment types.
    segments: tuple[BaseCoord, ...]
    _rich_style = dascore_styles["coord_segmented"]

    @model_validator(mode="before")
    @classmethod
    def _validate_segments(cls, data: Any) -> Any:
        """Coerce, normalize, and validate segments; derive model fields."""
        if not isinstance(data, dict):
            return data
        segments = data.get("segments")
        if isinstance(segments, BaseCoord):
            segments = (segments,)
        segments = tuple(_coerce_segment(x) for x in iterate(segments))
        if not segments:
            msg = "CoordSegmented requires at least one segment."
            raise CoordError(msg)
        _validate_segment_compat(segments)
        _validate_segment_chain(segments)
        segments = _fuse_segments(tuple(_maybe_promote_segment(x) for x in segments))
        if len(segments) < 2:
            msg = (
                "Segments fuse into a single coordinate; use concat_coords "
                "or get_coord(segments=...) which return it directly."
            )
            raise CoordError(msg)
        seg_units = segments[0].units
        if (given := data.get("units")) is not None:
            if get_quantity(given) != get_quantity(seg_units):
                msg = (
                    f"units {given} do not match segment units {seg_units}. "
                    "Use set_units or convert_units instead."
                )
                raise CoordError(msg)
        data["segments"] = segments
        data["units"] = seg_units
        data["shape"] = (sum(len(x) for x in segments),)
        data["dtype"] = np.result_type(*[s.dtype for s in segments])
        # the step every run declares, else none: the labels follow one
        # grid, with positions missing between the runs
        steps = [x.step for x in segments]
        declared = all(not _is_null(x) for x in steps) and len(set(steps)) == 1
        data["step"] = steps[0] if declared and _one_grid(segments, steps[0]) else None
        return data

    @field_serializer("segments")
    def _serialize_segments(self, segments, _info):
        """Serialize each segment with its own (subclass) schema."""
        return [x.model_dump() for x in segments]

    def __eq__(self, other) -> bool:
        """Compare segment-wise (nested arrays break generic dump equality)."""
        if not isinstance(other, CoordSegmented):
            return False
        if len(self.segments) != len(other.segments):
            return False
        pairs = zip(self.segments, other.segments)
        return all(s1 == s2 for s1, s2 in pairs)

    __hash__ = BaseCoord.__hash__

    @property
    def segment_count(self) -> int:
        """Return the number of segments."""
        return len(self.segments)

    def to_summary(self, dims=()) -> CoordSummary:
        """Get the summary info about the coord, with a summary per run."""
        summary = super().to_summary(dims=dims)
        if self.segment_count > _MAX_SUMMARY_RUNS:
            return summary
        runs = tuple(x.to_summary(dims=dims) for x in self.segments)
        return summary.model_copy(update={"runs": runs})

    @cached_method
    def _segment_offsets(self) -> np.ndarray:
        """Return the starting sample index of each segment."""
        lens = [len(x) for x in self.segments]
        # Cached and shared by callers, so hand back a read-only array.
        return array(np.cumsum([0, *lens[:-1]]))

    @property
    @cached_method
    def values(self) -> ArrayLike:
        """Return the values of the coordinate as an array."""
        out = np.concatenate([x.values for x in self.segments])
        return array(out.astype(self.dtype, copy=False))

    def _get_index_values(self, indices):
        """Evaluate only the requested samples, each in its own segment."""
        indices = np.asarray(indices)
        indices = np.where(indices < 0, indices + len(self), indices)
        offsets = self._segment_offsets()
        which = np.searchsorted(offsets, indices, side="right") - 1
        out = np.empty(indices.shape, dtype=self.dtype)
        for num in np.unique(which):
            mask = which == num
            local = indices[mask] - offsets[num]
            out[mask] = self.segments[num]._get_index_values(local)
        return out

    def _as_monotonic(self) -> CoordMonotonicArray:
        """
        Return an equivalent (materialized) monotonic array coord.

        Deliberately not cached: transient materialization for rare
        operations (snap, get_next_index) must not permanently defeat the
        O(segments) memory model.
        """
        return CoordMonotonicArray(values=self.values, units=self.units)

    def _min(self):
        """Return min value (exact, no materialization)."""
        first, last = self.segments[0], self.segments[-1]
        return first.min() if self.sorted else last.min()

    def _max(self):
        """Return max value (exact, no materialization)."""
        first, last = self.segments[0], self.segments[-1]
        return last.max() if self.sorted else first.max()

    @property
    @cached_method
    def sorted(self) -> bool:
        """Return True if sorted in ascending order."""
        return bool(self.segments[0].min() < self.segments[-1].min())

    @property
    @cached_method
    def reverse_sorted(self) -> bool:
        """Return True if sorted in descending order."""
        return not self.sorted

    def _fingerprint_components(self) -> tuple[Any, ...]:
        """Return the payload needed to fingerprint segmented coords."""
        return (("segments", tuple(x.fingerprint() for x in self.segments)),)

    def new(self, **kwargs):
        """Update coordinate."""
        if "data" in kwargs or "values" in kwargs:
            data = kwargs.get("data", kwargs.get("values"))
            return get_coord(data=data, units=kwargs.get("units", self.units))
        segments = kwargs.pop("segments", self.segments)
        units = kwargs.pop("units", self.units)
        return self.__class__(segments=segments, units=units)

    def _rebuild_segments(self, segments) -> Self:
        """Return a coord holding these segments, or self if none moved."""
        if all(new is old for new, old in zip(segments, self.segments)):
            return self
        return self.__class__(segments=segments)

    def set_units(self, units) -> Self:
        """Set new units on the coordinate and all segments."""
        return self._rebuild_segments(tuple(x.set_units(units) for x in self.segments))

    def convert_units(self, units) -> Self:
        """
        Convert units, or set units if none exist.

        The guard is per segment rather than on `self.units`, which speaks
        only for the first: segments are admitted when their units are
        merely equal, so a coord in metres can hold a segment in `100 cm`,
        and that one still has work to do.
        """
        return self._convert_units(units)

    def _convert_units(self, units) -> Self:
        """Convert each segment, keeping self when none of them moved."""
        if dtype_time_like(self.dtype):
            return self
        return self._rebuild_segments(
            tuple(x.convert_units(units) for x in self.segments)
        )

    def _rebuild(self, segments) -> BaseCoord:
        """Build the simplest coordinate from a (non-empty) list of segments."""
        segments = tuple(segments)
        if len(segments) == 1:
            return segments[0]
        try:
            return self.__class__(segments=segments)
        except (ValidationError, CoordError):
            fused = _fuse_segments(tuple(_maybe_promote_segment(x) for x in segments))
            if len(fused) == 1:  # Segments fused to a single coordinate.
                return fused[0]
            raise

    def _slice_segments(self, start: int, stop: int) -> BaseCoord:
        """Return the coordinate for a contiguous sample range."""
        if stop <= start:
            return self.empty()
        out = []
        for seg, off in zip(self.segments, self._segment_offsets()):
            lo, hi = max(start - off, 0), min(stop - off, len(seg))
            if hi <= lo:
                continue
            sub = seg if (lo == 0 and hi == len(seg)) else seg[slice(lo, hi)]
            out.append(sub)
        return self._rebuild(out)

    def __getitem__(self, item):
        if isinstance(item, int | np.integer):
            length = len(self)
            index = int(item) + length if item < 0 else int(item)
            if not 0 <= index < length:
                msg = f"{item} exceeds coord length of {self}"
                raise IndexError(msg)
            offsets = self._segment_offsets()
            seg_ind = int(np.searchsorted(offsets, index, side="right")) - 1
            return self.segments[seg_ind][index - int(offsets[seg_ind])]
        if isinstance(item, slice):
            start = None if item.start is ... else item.start
            stop = None if item.stop is ... else item.stop
            item = slice(start, stop, item.step)
            start_i, stop_i, step_i = item.indices(len(self))
            if step_i == 1:
                return self._slice_segments(start_i, stop_i)
        out = self.values[item]
        if not np.ndim(out):
            return out
        # a declared grid survives only an order it can be held against
        keep = not _is_null(self.step) and is_strictly_monotonic(out)
        return get_coord(data=out, units=self.units, step=self.step if keep else None)

    def select(
        self, args, relative=False, samples=False
    ) -> tuple[BaseCoord, slice | ArrayLike]:
        """Apply select, return selected coords and index for selecting data."""
        if is_array(args):
            return self._select_by_array(args, relative=relative, samples=samples)
        if samples:
            return self._select_by_samples(args)
        # Delegate to each segment and compose the global slice from segment
        # offsets. The window over a monotonic coordinate keeps a contiguous
        # run of samples, and range segments answer in O(1), so selection
        # stays O(segments) and never materializes the concatenated values.
        v1, v2 = self.get_slice_tuple(args, relative=relative)
        kept, lo, hi = [], None, None
        for seg, off in zip(self.segments, self._segment_offsets()):
            seg_min, seg_max = seg.min(), seg.max()
            if (v2 is not None and seg_min > v2) or (v1 is not None and seg_max < v1):
                continue  # entirely outside the window
            inside_lo = v1 is None or v1 <= seg_min
            inside_hi = v2 is None or v2 >= seg_max
            if inside_lo and inside_hi:  # entirely inside; keep untouched
                sub, seg_lo, seg_hi = seg, 0, len(seg)
            else:  # boundary segment; delegate the exact trim
                sub, indexer = seg.select((v1, v2))
                assert isinstance(indexer, slice)  # a value window is contiguous
                seg_lo, seg_hi, _ = indexer.indices(len(seg))
                if seg_hi <= seg_lo:
                    continue
            if lo is None:
                lo = int(off) + seg_lo
            hi = int(off) + seg_hi
            kept.append(sub)
        if not kept:
            return self.empty(), slice(0, 0)
        assert hi is not None  # kept is non-empty, so the loop set hi
        new = self._rebuild(kept)
        start = None if lo == 0 else lo
        stop = None if hi >= len(self) else hi
        return new, slice(start, stop)

    def _get_index(self, value, forward=True):
        """Get the index corresponding to a value."""
        return self._as_monotonic()._get_index(value, forward=forward)

    def sort(self, reverse=False) -> tuple[BaseCoord, slice | ArrayLike]:
        """Sort the contents of the coord. Return new coord and slice for sorting."""
        forward_forward = not reverse and self.sorted
        reverse_reverse = reverse and self.reverse_sorted
        if forward_forward or reverse_reverse:
            return self, slice(None)
        segments = tuple(
            seg.sort(reverse=reverse)[0] for seg in reversed(self.segments)
        )
        return self.new(segments=segments), slice(None, None, -1)

    @compose_docstring(doc=get_docstring(BaseCoord.update_limits))
    def update_limits(self, min=None, max=None, step=None, **kwargs) -> BaseCoord:
        """{doc}."""
        if step is not None:
            msg = (
                "Segmented coordinates have no single step; use simplify or "
                "snap to get an evenly sampled coordinate first."
            )
            raise ParameterError(msg)
        if min is not None and max is not None:
            msg = "Cannot specify both min and max in update_limits."
            raise ParameterError(msg)
        out = self
        if min is not None:
            delta = get_compatible_values(min, self.dtype) - self.min()
            out = out._shift(delta)
        elif max is not None:
            delta = get_compatible_values(max, self.dtype) - self.max()
            out = out._shift(delta)
        return out.new(**kwargs) if kwargs else out

    def _shift(self, delta) -> Self:
        """Return a copy of the coordinate with all values shifted by delta."""
        segments = []
        for seg in self.segments:
            if isinstance(seg, CoordRange):
                new = seg.new(start=seg.start + delta, stop=seg.stop + delta)
            else:
                new = seg.new(values=seg.values + delta)
            segments.append(new)
        return self.new(segments=tuple(segments))

    def snap(self) -> CoordRange:
        """
        Snap the coordinates to evenly sampled grid points.

        The min/max of the coordinate remain unchanged; every interior value
        may move without bound. Use
        [`simplify`](`dascore.core.coords.BaseCoord.simplify`) for a
        tolerance-bounded alternative.
        """
        return self._as_monotonic().snap()

    def simplify(self, tolerance=None) -> BaseCoord:
        """
        Return the simplest coordinate representing the same values.

        Segments are greedily re-fit as evenly sampled ranges; a fit is
        accepted only when no value moves by more than `tolerance`. With a
        sufficient tolerance a fully contiguous segmented coordinate collapses
        to a single [`CoordRange`](`dascore.core.coords.CoordRange`).

        Parameters
        ----------
        tolerance
            The maximum amount any coordinate value may change. For time-like
            coordinates this is a timedelta (numeric values interpreted as
            seconds). None or 0 permit only exact simplifications.
        """
        tol = self._get_tolerance(tolerance)
        result = []
        run = [self.segments[0]]
        run_fit = self._fit_run(run, tol)
        for seg in self.segments[1:]:
            trial = [*run, seg]
            fit = self._fit_run(trial, tol)
            if fit is not None:
                run, run_fit = trial, fit
            else:
                result.append(run_fit if run_fit is not None else run[0])
                run, run_fit = [seg], self._fit_run([seg], tol)
        result.append(run_fit if run_fit is not None else run[0])
        return self._rebuild(result)

    def _get_tolerance(self, tolerance):
        """Coerce the tolerance to the dtype expected for value deviations."""
        if isinstance(tolerance, GapTolerance) and tolerance.count is not None:
            # a count of steps is measured against the runs' own step
            steps = [abs(x.step) for x in self.segments if not _is_null(x.step)]
            tolerance = tolerance.count * get_middle_value(steps) if steps else 0
        return self._gap_tolerance(tolerance).excess

    def _fit_run(self, run, tol) -> CoordRange | None:
        """Fit a run of segments to a single range within tol, or None."""
        if len(run) == 1 and isinstance(run[0], CoordRange):
            return run[0]
        n = sum(len(x) for x in run)
        if n < 2:
            return None
        ascending = self.sorted
        first = run[0].min() if ascending else run[0].max()
        last = run[-1].max() if ascending else run[-1].min()
        span = last - first
        if is_timedelta64(span) or is_datetime64(first):
            span_ns = dc.to_timedelta64(span).astype(np.int64)
            step = np.timedelta64(int(np.round(span_ns / (n - 1))), "ns")
            zero = dc.to_timedelta64(0)
        else:
            step = span / (n - 1)
            zero = 0
        # Strictly monotonic segments guarantee a nonzero step matching the
        # sort direction.
        assert step != zero and (step > zero) == ascending
        candidate = CoordRange(
            start=first, stop=last + step, step=step, units=self.units
        ).change_length(n)
        actual = np.concatenate([x.values for x in run])
        deviation = np.max(np.abs(candidate.values - actual))
        if deviation > tol:
            return None
        return candidate

    def _seams(self) -> list[tuple]:
        """One row per seam between runs, expecting the run's own step after it."""
        offsets = self._segment_offsets()
        ascending = self.sorted
        rows = []
        for num in range(1, len(self.segments)):
            prev, nxt = self.segments[num - 1], self.segments[num]
            before = prev.max() if ascending else prev.min()
            after = nxt.min() if ascending else nxt.max()
            rows.append((int(offsets[num]), before, after, self._expected_step(prev)))
        return rows

    def _holes(self) -> list[tuple]:
        """The grid positions skipped inside each run and between runs."""
        step = self.step  # signed with the runs' direction
        rows = list(self.segments[0]._holes())
        for (_, before, after, _), seg in zip(self._seams(), self.segments[1:]):
            count = int(_on_grid(np.asarray([after - before]), step)[0]) - 1
            if count:
                rows.append(_hole(before, step, count))
            rows.extend(seg._holes())
        return rows

    @staticmethod
    def _expected_step(seg) -> Any:
        """Return the expected next-sample spacing after a segment, or None."""
        if not _is_null(seg.step):
            return seg.step
        if len(seg) > 1:
            values = seg.values
            return values[-1] - values[-2]
        return None

    @classmethod
    def from_array(cls, array, step=None, tolerance=None, units=None) -> BaseCoord:
        """
        Build a coordinate from a monotonic array, detecting uniform runs.

        Values are preserved exactly; each maximal evenly sampled run becomes
        an evenly sampled segment and each internal sampling break becomes a
        segment boundary, so gaps inside the array are queryable via
        [`get_discontinuities`](`dascore.core.coords.BaseCoord.get_discontinuities`).
        Fully uniform arrays come back as a plain
        [`CoordRange`](`dascore.core.coords.CoordRange`) and arrays with no
        detectable runs as a plain monotonic coordinate.

        Parameters
        ----------
        array
            A strictly monotonic 1D array (numeric, datetime64, or
            timedelta64) with no missing values.
        step
            The grid the values sit on. Every spacing must then be a whole
            number of steps (a value off the grid raises), each run of
            consecutive positions becomes a range of this step, singletons
            included, and the coordinate reports its
            [`missing`](`dascore.core.coords.BaseCoord.missing`) positions.
            Without it runs are read from equal spacings alone.
        tolerance
            If not None, apply
            [`simplify`](`dascore.core.coords.BaseCoord.simplify`) with this
            tolerance to the result, re-fitting jittery runs and absorbing
            small gaps with bounded error.
        units
            Units for the coordinate.

        Notes
        -----
        A dense array whose runs would outnumber a tenth of its samples
        (past a thousand samples) keeps its values as one monotonic
        coordinate rather than as thousands of tiny runs; a declared step
        is kept on it, so `missing` answers the same on both sides.

        Examples
        --------
        >>> import numpy as np
        >>> from dascore.core.coords import CoordSegmented
        >>>
        >>> values = np.array([0.0, 1, 2, 3, 10, 11, 12, 13])
        >>> coord = CoordSegmented.from_array(values)
        >>> assert coord.segment_count == 2
        >>> assert len(coord.get_discontinuities("gaps")) == 1
        >>>
        >>> # A declared step turns isolated values into runs of one.
        >>> coord = CoordSegmented.from_array([1, 3, 4, 10, 11, 12], step=1)
        >>> assert coord.segment_count == 3 and coord.step == 1
        >>> assert coord.missing().count == 6
        """
        values = np.asarray(array)
        if values.ndim != 1:
            msg = "from_array requires a 1D array."
            raise CoordError(msg)
        if pd.isnull(values).any():
            msg = "from_array does not support missing values."
            raise CoordError(msg)
        if not _is_null(step):
            step = _declared_step(step, values.dtype)
        if len(values) < (3 if _is_null(step) else 2):
            out = get_coord(data=values, units=units, step=step)
        else:
            if not is_strictly_monotonic(values):
                msg = "from_array requires strictly monotonic values."
                raise CoordError(msg)
            diffs = _diffs(values)
            if _is_null(step):
                # A diff belongs to a uniform run when it matches a
                # neighboring diff; isolated diffs are seams (gaps or
                # sampling changes).
                eq_next = diffs[:-1] == diffs[1:]
                in_run = np.zeros(len(diffs), dtype=bool)
                in_run[1:] |= eq_next
                in_run[:-1] |= eq_next
                splits = np.flatnonzero(~in_run) + 1
            else:
                # every spacing is a whole number of steps; more than one
                # step between neighbours is a seam with positions missing
                magnitude = np.abs(np.asarray(step))[()]
                signed = magnitude if values[-1] > values[0] else -magnitude
                splits = np.flatnonzero(_on_grid(diffs, signed) != 1) + 1
            dense = len(values) >= _MIN_SEGMENT_GUARD_SIZE
            if dense and len(splits) + 1 > _MAX_SEGMENT_FRACTION * len(values):
                out = CoordMonotonicArray(values=values, units=units, step=step)
            elif _is_null(step):
                if not np.any(in_run):
                    # Singleton segments carry no direction and would be
                    # sorted ascending by concat_coords. Keep the recorded
                    # order.
                    out = CoordMonotonicArray(values=values, units=units)
                else:
                    blocks = np.split(values, splits)
                    segments = [
                        CoordMonotonicArray(values=x, units=units) for x in blocks
                    ]
                    out = concat_coords(*segments)
            else:
                # built in the values' own order: a coordinate of singleton
                # runs states no direction concat_coords could read
                runs = [
                    CoordRange(start=x[0], step=signed, shape=(len(x),), units=units)
                    for x in np.split(values, splits)
                ]
                out = runs[0] if len(runs) == 1 else CoordSegmented(segments=runs)
        if tolerance is not None:
            out = out.simplify(tolerance)
        return out


def concat_coords(*coords, units=None) -> BaseCoord:
    """
    Concatenate monotonic coordinates into a single coordinate.

    This operation is truth-preserving: no value is ever altered, and every
    boundary between inputs that does not continue exactly is recorded as a
    segment boundary. The result is a
    [`CoordSegmented`](`dascore.core.coords.CoordSegmented`) unless the inputs
    fuse into a single segment, in which case that coordinate is returned
    directly. Use [`simplify`](`dascore.core.coords.BaseCoord.simplify`) on
    the result for tolerance-bounded gap absorption.

    Parameters
    ----------
    *coords
        Coordinates to concatenate. Each must be evenly sampled
        ([`CoordRange`](`dascore.core.coords.CoordRange`)), monotonic
        ([`CoordMonotonicArray`](`dascore.core.coords.CoordMonotonicArray`)),
        or already segmented. Inputs are ordered by their envelopes; they
        must share dtype kind, units, and sort direction, and must not
        overlap.
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
    flat = []
    for coord in coords:
        if isinstance(coord, dict):  # model_dump round-trip payloads
            coord = _coerce_segment(coord)
        if isinstance(coord, CoordSegmented):
            flat.extend(coord.segments)
        elif isinstance(coord, CoordRange | CoordMonotonicArray):
            if len(coord):
                flat.append(coord)
        elif isinstance(coord, BaseCoord):
            if coord.degenerate:
                continue
            msg = (
                "concat_coords only supports evenly sampled, monotonic, or "
                f"segmented coordinates, got {type(coord)}."
            )
            raise CoordError(msg)
        else:
            msg = f"concat_coords requires coordinates, got {type(coord)}."
            raise CoordError(msg)
    if units is not None:
        flat = [x.set_units(units) for x in flat]
    if not flat:
        msg = "concat_coords requires at least one non-empty coordinate."
        raise CoordError(msg)
    _validate_segment_compat(tuple(flat))
    multi = [x for x in flat if len(x) > 1]
    ascending = multi[0].sorted if multi else True
    # Sort on native values; float conversion would collapse ns datetimes.
    flat.sort(key=lambda x: x.min(), reverse=not ascending)
    if len(flat) == 1:
        return _maybe_promote_segment(flat[0])
    _validate_segment_chain(tuple(flat))
    segments = _fuse_segments(tuple(_maybe_promote_segment(x) for x in flat))
    if len(segments) == 1:
        return segments[0]
    return CoordSegmented(segments=segments)


def _grid_pieces(coord: BaseCoord) -> list[tuple[int, CoordRange]]:
    """
    The runs of consecutive grid positions, each with its source offset.

    A range is one run; an array declaring a step splits at its holes.
    """
    segments = coord.segments if isinstance(coord, CoordSegmented) else (coord,)
    pieces, offset = [], 0
    for seg in segments:
        if isinstance(seg, CoordRange):
            pieces.append((offset, seg))
        elif isinstance(seg, CoordMonotonicArray) and not _is_null(seg.step):
            values = seg.values
            counts = _on_grid(_diffs(values), seg.step)
            edges = [0, *(np.flatnonzero(counts != 1) + 1).tolist(), len(values)]
            for start, stop in itertools.pairwise(edges):
                piece = get_coord(
                    start=values[start],
                    step=seg.step,
                    shape=(stop - start,),
                    units=seg.units,
                )
                pieces.append((offset + start, piece))
        else:
            msg = (
                "Filling gaps needs a coordinate with a declared step; this one "
                "has none. Use snap_coords or resample to put it on a grid first."
            )
            raise CoordError(msg)
        offset += len(seg)
    return pieces


def _same_step(first: CoordRange, other: CoordRange) -> bool:
    """Whether two runs share a step: exactly for ticks, closely for floats."""
    exact = first.step_exact, other.step_exact
    if None not in exact:
        return exact[0] == exact[1]
    ratio = float(other.step) / float(first.step)
    return bool(abs(ratio - 1) <= _GRID_RTOL)


def _grid_position(anchor: CoordRange, label) -> int:
    """The position on the anchor's grid nearest a label."""
    if not anchor._exact:
        return int(np.round((label - anchor.start) / anchor.step))
    num, den, offset = anchor._grid_terms
    ticks = (_to_tick(label) - anchor._start_tick) * den - offset
    guess = round(Fraction(ticks, num))
    # labels floor the ideal grid, so the nearest one may be a neighbour
    near = np.arange(guess - 1, guess + 2)
    return int(near[np.argmin(np.abs(anchor._labels(near) - label))])


def _max_missing(step: CoordRange, coord: BaseCoord, limit, samples: bool):
    """The most missing positions a filled hole may have, or None for any."""
    if limit is None:
        return None
    if samples:
        if not _is_int(limit) or limit < 0:
            msg = f"A sample limit must be a non-negative integer, got {limit!r}."
            raise ParameterError(msg)
        return int(limit)
    excess = coord._gap_tolerance(limit).excess
    if (exact := step.step_exact) is not None:
        if is_timedelta64(excess):
            excess = Fraction(int(to_int(excess)), _NS_PER_S)
        return int(Fraction(excess) // abs(exact))
    return math.floor(float(excess) / abs(float(step.step)) * (1 + _GRID_RTOL))


def _fill_layout(
    coord: BaseCoord, limit=None, samples: bool = False
) -> tuple[BaseCoord, tuple[tuple[int, int, int], ...]] | None:
    """
    Place every run of a coordinate on one grid, filling the holes between.

    Returns the filled coordinate and, per run, its ``(source start, source
    stop, target start)``, or None when there is nothing to fill.

    Parameters
    ----------
    coord
        A range, a segmented coordinate, or an array declaring a step.
    limit
        The widest hole to fill, in coordinate units (seconds for time), or
        missing samples when `samples` is True. Wider holes stay as seams
        between separate runs. None fills every hole.
    samples
        If True, `limit` counts missing samples.

    Notes
    -----
    Runs must share one step. A run starting off the grid of the runs
    before it is placed at the nearest position, moving its labels by at
    most half a step; two runs landing on the same position raise.
    """
    if isinstance(coord, CoordRange) or len(coord) < 2:
        return None
    pieces = _grid_pieces(coord)
    first = pieces[0][1]
    for _, piece in pieces[1:]:
        if not _same_step(first, piece):
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
    new = coords[0] if len(coords) == 1 else concat_coords(*coords)
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

    See ['Coordinate Internals'](`docs/notes/coordinate_internals.qmd`) for the
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
            fingerprint=self.fingerprint(),
        )

    def _fingerprint_components(self) -> tuple[Any, ...]:
        """Return the array payload needed to fingerprint string coords."""
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
    step_numerator, step_denominator, origin_offset
        The exact grid of an integer or time range in ticks (see
        [`CoordRange`](`dascore.core.coords.CoordRange`)). Normally
        these come from a dumped coordinate; pass ``step`` as a `Fraction`
        or ``(numerator, denominator)`` tuple to state a fractional step.

    Notes
    -----
    See ['Coordinate Internals'](`docs/notes/coordinate_internals.qmd`) for
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

    def _get_new_max(data, min, step):
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
                _max = _get_new_max(data, _min, _step)
                return _min, _max + _step, _step, is_monotonic
        return None, None, None, is_monotonic

    if segments is not None:
        # shape/dtype/step are derived fields on CoordSegmented, so they
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
        # A 1D shape with two of start/stop/step states a range; anything
        # less (or more dimensions) is a partial coord. A range that then
        # fails to validate is an error, not a partial.
        stated = sum(not _is_null(x) for x in (start, stop, step))
        if len(shape) != 1 or shape[0] == 0 or stated < 2:
            return CoordPartial(**attrs)
        try:
            return CoordRange(
                shape=shape, start=start, stop=stop, step=step, units=units, **grid
            )
        except (ValidationError, CoordError):
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
                val = data[0]
                return CoordRange(start=val, shape=(1,), step=step, units=units)
            return CoordMonotonicArray(values=data, units=units)
        if not _is_null(step):
            # a declared step is a claim about the grid, so the values are
            # read against it exactly rather than fitted
            return CoordSegmented.from_array(data, step=step, units=units)
        start, stop, step, monotonic = _maybe_get_start_stop_step(data)
        if start is not None:
            out = CoordRange(start=start, stop=stop, step=step, units=units)
            # The change_length call helps with float off by one issues.
            return out.change_length(len(data))
        elif monotonic:
            return CoordMonotonicArray(values=data, units=units)
        elif np.all(pd.isnull(data)):
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
        return CoordArray(values=data, units=units)
    else:
        return CoordRange(start=start, stop=stop, step=step, units=units, **grid)
