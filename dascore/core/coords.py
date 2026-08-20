"""Machinery for coordinates.

See ['Coordinate Internals'](`docs/notes/coordinate_internals.qmd`) for the
current coord-family and string-coordinate design notes.
"""

from __future__ import annotations

import abc
import fnmatch
import hashlib
import itertools
import json
import math
import re
from collections.abc import Mapping, Sequence, Sized
from contextlib import suppress
from functools import cache
from operator import gt, lt
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast, overload

import numpy as np
import pandas as pd
from pydantic import (
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)
from rich.text import Text
from typing_extensions import Self

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
from dascore.utils.display import get_nice_text
from dascore.utils.docs import compose_docstring, get_docstring
from dascore.utils.misc import (
    _get_nullish,
    _maybe_array_to_slice,
    _to_slice,
    _validate_sample_values,
    all_close,
    all_diffs_close_enough,
    cached_method,
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
        nfunc = np.datetime64 if is_datetime64(data) else np.timedelta64
        return np.atleast_1d(nfunc("NaT", "ns"))

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
        # this is a reverse coord
        if np.sign(step) == -1:
            start, stop = self.max, self.min + step
        else:
            start, stop = self.min, self.max + step
        return CoordRange(
            start=start,
            stop=stop,
            step=step,
            units=self.units,
        )


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


class BaseCoord(DascoreBaseModel, abc.ABC):
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

    @abc.abstractmethod
    def _convert_units(self, units) -> Self:
        """Perform the conversion; callers normally screen out no-op requests."""

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

    def __rich__(self):
        key_style = dascore_styles["keys"]
        base = Text("")
        base += Text(self.__class__.__name__, style=self._rich_style)
        base += Text("(")
        if not pd.isnull(self.min()):
            base += Text(" min: ", key_style)
            base += get_nice_text(self.min())
        if not pd.isnull(self.max()):
            base += Text(" max: ", key_style)
            base += get_nice_text(self.max())
        if not pd.isnull(self.step):
            base += Text(" step: ", key_style)
            base += get_nice_text(self.step)
        base += Text(" shape: ", key_style)
        base += get_nice_text(self.shape)
        base += Text(" dtype: ", key_style)
        base += get_nice_text(self.dtype)
        if self.units is not None:
            base += Text(" units: ", key_style)
            unit_str = get_quantity_str(self.units)
            base += get_nice_text(unit_str, style="units")
        base += Text(" )")
        return base

    def __str__(self):
        return str(self.__rich__())

    __repr__ = __str__

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

    @staticmethod
    def _hash_scalar(value) -> tuple[str, str | None]:
        """Return a dtype-aware scalar hash token."""
        if value is None:
            return ("none", None)
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

        Coordinates without internal segment structure return an empty
        dataframe.
        """
        if kind not in ("all", "gaps"):
            msg = f"kind must be 'all' or 'gaps', got {kind!r}"
            raise ParameterError(msg)
        columns = ["index", "before", "after", "delta", "excess"]
        return pd.DataFrame(columns=columns)

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
            step=self.step,
            dtype=self.dtype,
            units=self.units,
            dims=dims,
            len=len(self),
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
            ratio = duration / self.step
            if np.issubdtype(self.dtype, np.floating):
                nearest = np.round(ratio)
                # Adding a relative float value to coord.min() and subtracting
                # it back can introduce a small cancellation error. Snap only
                # those float-coordinate ratios that fall within that drift.
                tol = 10 * abs(np.spacing(self.min()) / self.step)
                if abs(ratio - nearest) <= tol:
                    ratio = nearest
            samples = int(np.ceil(ratio))
        if enforce_lt_coord and samples > len(self):
            msg = (
                f"value of {value} with samples={samples} results in a window "
                f"larger than coordinate length of {len(self)}."
            )
            raise ParameterError(msg)
        return samples

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
                # Calculate index: (value - min) / step
                indices = ((array - min_val) / self.step).astype(np.int64)
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
            self._hash_scalar(self.start),
            self._hash_scalar(self.stop),
            self._hash_scalar(self.step),
        )


class CoordRange(BaseCoord):
    """
    A coordinate represent a range of evenly sampled data.

    Parameters
    ----------
    start
        The starting value
    stop
        The ending value
    step
        The step between start and stop.

    Notes
    -----
    Like range and slice, CoordRange is exclusive of stop value.
    """

    start: Any = None
    stop: Any = None
    step: Any = None
    _evenly_sampled = True
    _rich_style = dascore_styles["coord_range"]

    def _new_grid(self, start, step, length: int) -> Self:
        """
        Return a CoordRange on an exactly known grid, skipping re-validation.

        `validate_start_stop_step_len` exists to *derive* shape and a
        normalized stop (always `start + step * length`) from loosely
        specified inputs. Callers below already know the sample count exactly
        -- it comes from index arithmetic on an existing, validated coord --
        so re-deriving it costs ~60us per call and can only reproduce what is
        passed in here.

        Only use this where start/step come from an already-validated
        CoordRange and length is computed from indices; anything taking user
        input must go through the validating constructor.
        """
        units = self.units
        # Mirror check_time_units, which forces time-like coords to seconds.
        # Note it tests `start` for truthiness, so a coord starting at exactly
        # zero is left alone; that quirk is reproduced here deliberately.
        if start and (is_timedelta64(start) or is_datetime64(start)):
            units = _second_quantity()
        return self.model_construct(
            # copy; model_construct stores the set by reference.
            _fields_set=set(self.model_fields_set),
            units=units,
            step=step,
            shape=(length,),
            # matches what the validator stores for dtype.
            dtype=np.asarray(start + step).dtype,
            start=start,
            stop=start + step * length,
        )

    @model_validator(mode="before")
    @classmethod
    def validate_start_stop_step_len(cls, values):
        """Coerce the needed values from the inputs."""

        def _maybe_unbox_scalar(value):
            """Extract scalar from a 1-element array-like value."""
            if isinstance(value, np.ndarray) and value.ndim > 0 and value.size == 1:
                return value.reshape(-1)[0]
            return value

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

        def _round_ratio(numerator, denominator, digits):
            """Round numerator/denominator, cheaply for scalars."""
            # Inputs are always scalar-like (multi-element arrays are rejected
            # by the pd.isnull check above) and rounding python floats is
            # ~10x faster than numpy scalars, hence the float conversion.
            ratio = _maybe_unbox_scalar(numerator / denominator)
            return round(float(ratio), digits)

        zero = _TD64_ZERO if is_timedelta64(step) else 0
        if step != zero:
            span = _round_ratio(stop - start, step, 1)
            int_val = int(_maybe_unbox_scalar(np.ceil(span)))
            stop = start + step * int_val
        start_equal_stop = _maybe_unbox_scalar(start == stop)
        length = 1 if start_equal_stop else int(_round_ratio(stop - start, step, 0))
        shape = (length,)
        values.update(dict(start=start, stop=stop, shape=shape, step=step))
        # step should have the same sign as stop-start, see #321.
        # Compare signs via direct comparisons (rather than np.sign) since
        # np.sign(datetime64) returns a datetime64 which includes precision,
        # so even if the sign is the same, differing precision fails; direct
        # comparisons are also much cheaper than to_float conversions.
        diff = stop - start
        step_ = values["step"]
        try:
            same_sign = ((step_ > zero) == (diff > zero)) & (
                (step_ < zero) == (diff < zero)
            )
        except TypeError:  # mixed types (e.g. datetime.timedelta vs int zero)
            same_sign = np.sign(to_float(step_)) == np.sign(to_float(diff))
        if not same_sign:
            msg = "Sign of step must match sign of stop - start"
            raise CoordError(msg)
        # Note: dtype was a property before but it messed up model
        # serialization.
        values["dtype"] = np.asarray(start + step).dtype
        return values

    def _fingerprint_components(self) -> tuple[Any, ...]:
        """Return the scalar payload needed to fingerprint range coords."""
        return (
            self.shape,
            self._hash_scalar(self.start),
            self._hash_scalar(self.stop),
            self._hash_scalar(self.step),
        )

    def __getitem__(self, item):
        if isinstance(item, (int | np.integer)):
            if item >= len(self):
                raise IndexError(f"{item} exceeds coord length of {self}")
            return self.values[item]
        # handle ... as None
        if isinstance(item, slice):
            start = None if item.start is ... else item.start
            end = None if item.stop is ... else item.stop
            item = slice(start, end, item.step)
            # A (possibly strided) slice of an evenly sampled range is still
            # evenly sampled, so preserve the step rather than collapsing to a
            # CoordMonotonicArray (which loses step for len==1). See #567.
            indices = range(len(self))[item]
            if len(indices):
                new_step = self.step * indices.step
                new_start = self.start + indices.start * self.step
                return self._new_grid(new_start, new_step, len(indices))
        out = self.values[item]
        return get_coord(data=out, units=self.units)

    @cached_method
    def __len__(self):
        return self.shape[0]

    def _convert_units(self, units) -> Self:
        """Convert units, or set units if none exist."""
        # cant convert time units
        if dtype_time_like(self.dtype):
            return self
        out = dict(units=units)
        start = convert_units(self.start, to_units=units, from_units=self.units)
        stop = convert_units(self.stop, to_units=units, from_units=self.units)
        step = (stop - start) / len(self)
        out["start"], out["stop"], out["step"] = start, stop, step
        return self.__class__(**out)

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
        # The sample count is known exactly from the indices, so build the
        # new grid directly rather than making the validator re-derive it.
        first = 0 if start is None else start
        last = (len(self) - 1) if stop is None else stop
        new_start = self[start] if start is not None else self.start
        new_coords = self._new_grid(new_start, self.step, last + 1 - first)
        return new_coords, data

    def sort(self, reverse=False) -> tuple[BaseCoord, slice | ArrayLike]:
        """Sort the contents of the coord. Return new coord and slice for sorting."""
        #
        forward_forward = not reverse and self.sorted
        reverse_reverse = reverse and self.reverse_sorted
        if forward_forward or reverse_reverse:
            return self, slice(None)
        new_step = -self.step
        if reverse:  # reversing a forward sorted CoordRange
            new_start = self.max()
        else:  # order a reverse sorted one
            new_start = self.min()
        # reversing preserves the sample count.
        out = self._new_grid(new_start, new_step, len(self))
        return out, slice(None, None, -1)

    def _get_zero_step_index(self, value, forward):
        """
        Get the index of a value for a coord with a step of 0.

        Every sample of such a coord equals start, so the index is either the
        first sample or one just outside the coord, which makes the
        selection degenerate.
        """
        start = self.start
        if forward:  # index of the first sample >= value
            return 0 if value <= start else len(self)
        return 0 if value >= start else -1

    def _get_index(self, value, forward=True):
        """Get the index corresponding to a value."""
        if (value := self._get_compatible_value(value)) is None:
            return value
        start, step = self.start, self.step
        if not isinstance(value, Sized):
            # Scalar fast path; avoids several small-array allocations.
            # Due to float weirdness we need a little bit of a fudge factor.
            # (float() first since rounding numpy scalars is ~10x slower)
            # A step of 0 is handled after the division (python scalars raise,
            # numpy scalars give inf/nan) since testing a numpy step for
            # truthiness up front costs ~10x more on this hot path.
            try:
                fraction = round(float((value - start) / step), 10)
            except ZeroDivisionError:
                return self._get_zero_step_index(value, forward)
            if not math.isfinite(fraction):
                # A zero step, whose samples all equal start, has a
                # degenerate but defined index. Ask it by truthiness: a
                # timedelta step compared to a bare 0 warns (and will
                # eventually raise) about the generic unit that implies.
                if not step:
                    return self._get_zero_step_index(value, forward)
                # Otherwise the fraction is infinite: a bound past one end of
                # the coord, either because the bound itself is infinite or
                # because it sits far enough from start to overflow the
                # division. Its sign says which end, and the range checks
                # below turn the end the samples are not on into an open
                # side. It cannot be NaN; a null bound, NaN and NaT alike,
                # is already None by the time it gets here.
                out = len(self) if fraction > 0 else -1
            else:
                out = math.ceil(fraction) if forward else math.floor(fraction)
            if forward and out < 0:
                return None
            if not forward and out >= len(self):
                return None
            return out
        # A 0d array is Sized but holds a single bound, so unbox it rather
        # than let the array path cast its infinity to the smallest int64.
        if getattr(value, "ndim", 1) == 0:
            return self._get_index(value[()], forward=forward)
        array = np.atleast_1d(value)
        func = np.ceil if forward else np.floor
        # Due to float weirdness we need a little bit of a fudge factor here.
        fraction = func(np.round((array - start) / step, decimals=10))
        return fraction.astype(np.int64)

    @compose_docstring(doc=get_docstring(BaseCoord.update_limits))
    def update_limits(self, min=None, max=None, step=None, **kwargs) -> BaseCoord:
        """{doc}."""
        if all(x is not None for x in [min, max, step]):
            msg = "At most two parameters can be specified in update_limits."
            raise ValueError(msg)
        # first case, we need to determine new dt.
        if min is not None and max is not None:
            new_step = (max - min) / len(self)
            return get_coord(start=min, stop=max, step=new_step, units=self.units)
        # For other combinations we just apply adjustments sequentially
        # after ensuring that the types are compatible.
        out = self
        if step is not None:
            step = get_compatible_values(step, type(self.step))
            new_stop = out.start + step * len(out)
            out = out.new(stop=new_stop, step=step)
        if min is not None:
            min = get_compatible_values(min, self.dtype)
            diff = min - out.start
            new_stop = out.stop + diff
            out = out.new(start=min, stop=new_stop)
        if max is not None:
            max = get_compatible_values(max, self.dtype)
            translation = (max + out.step) - out.stop
            new_start = self.start + translation
            # we add step so the new range is inclusive of stop.
            out = out.new(start=new_start, stop=max + out.step)
        return out.new(**kwargs)

    @property
    @cached_method
    def values(self) -> ArrayLike:
        """Return the values of the coordinate as an array."""
        if len(self) == 1:
            # Cached, so it must be read-only like the other branch.
            return array(np.asarray([self.start]))
        # note: linspace works better for floats that might have slightly
        # uneven spacing. It ensures the length of the output array is robust
        # to small deviations in spacing. However, this doesnt work for datetimes.
        if is_datetime64(self.start) or is_timedelta64(self.start):
            out = np.arange(self.start, self.stop, self.step)
        else:
            out = np.linspace(
                self.start, self.stop - self.step, num=len(self), dtype=self.dtype
            )
        # again, due to round-off error the array can one element longer than
        # anticipated. The slice here just ensures shape and len match.
        return array(out[: len(self)])

    def _min(self):
        """Return min value."""
        return np.min([self.start, self.stop - self.step])

    def _max(self):
        """Return max value in range."""
        # like range, coord range is exclusive of final value.
        # the min/max are needed for reverse sorted coord.
        return np.max([self.stop - self.step, self.start])

    @compose_docstring(doc=get_docstring(BaseCoord.change_length))
    def change_length(self, length: int) -> Self:
        """
        {doc}
        """
        # CoordRange is always 1D by construction; keep as an internal invariant.
        assert self.ndim == 1, "Can only change length for 1D coords."
        length = _validate_new_length(length)
        if len(self) == length:
            return self
        # Only the sample count changes; start/step are already valid.
        return self._new_grid(self.start, self.step, length)

    @property
    def sorted(self) -> bool:
        """Returns true if sorted in ascending order."""
        zero = _TD64_ZERO if is_timedelta64(self.step) else 0
        return self.step >= zero

    @property
    def reverse_sorted(self) -> bool:
        """Returns true if sorted in ascending order."""
        zero = _TD64_ZERO if is_timedelta64(self.step) else 0
        return self.step < zero


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
        """Coerce the needed values from the inputs."""
        values["dtype"] = values["values"].dtype
        values["shape"] = values["values"].shape
        return values

    def _convert_units(self, units) -> Self:
        """Convert units, or set units if none exist."""
        is_time = np.issubdtype(self.dtype, np.datetime64)
        is_time_delta = np.issubdtype(self.dtype, np.timedelta64)
        if self.units is None or is_time or is_time_delta:
            return self.set_units(units)
        values = convert_units(self.values, units, self.units)
        return self.new(units=units, values=values)

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
        out = CoordRange(start=start, stop=stop, step=step, units=self.units)
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
        return self.__class__(values=out, units=self.units)

    def _min(self):
        """Return min value."""
        return np.nanmin(self.values)

    def _max(self):
        """Return max value in range."""
        return np.nanmax(self.values)

    def _fingerprint_components(self) -> tuple[Any, ...]:
        """Return the array payload needed to fingerprint array coords."""
        return (("array", hash_array(self.values)),)


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
    candidate = CoordRange(
        start=values[0], stop=values[-1] + step, step=step, units=seg.units
    )
    # Only promote when the range reproduces the values bit-exactly; unlike
    # get_coord inference, segments must never change any value.
    if len(candidate) == len(seg) and np.array_equal(candidate.values, values):
        return candidate
    return seg


def _fuse_segments(segments: tuple[BaseCoord, ...]) -> tuple[BaseCoord, ...]:
    """Fuse adjacent segments that continue exactly (normal form)."""
    out = [segments[0]]
    for seg in segments[1:]:
        prev = out[-1]
        both_ranges = isinstance(prev, CoordRange) and isinstance(seg, CoordRange)
        if both_ranges and prev.step == seg.step and prev.stop == seg.start:
            out[-1] = CoordRange(
                start=prev.start, stop=seg.stop, step=prev.step, units=prev.units
            )
            continue
        both_arrays = isinstance(prev, CoordMonotonicArray) and isinstance(
            seg, CoordMonotonicArray
        )
        if both_arrays:
            # Adjacent irregular arrays carry no sampling expectation, so the
            # boundary between them has no meaning; fuse for canonical form.
            values = np.concatenate([prev.values, seg.values])
            out[-1] = CoordMonotonicArray(values=values, units=prev.units)
            continue
        out.append(seg)
    return tuple(out)


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

    Segments are normal coordinates ([`CoordRange`](`dascore.core.coords.CoordRange`)
    or [`CoordMonotonicArray`](`dascore.core.coords.CoordMonotonicArray`)); the
    values of the segmented coordinate are exactly the concatenation of the
    segment values. Segment boundaries record discontinuities (e.g. data gaps)
    without altering any value, which makes this the natural coordinate for
    data merged across nearly-contiguous blocks.

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
        data["step"] = None
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
        return get_coord(data=out, units=self.units)

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
        if tolerance is None:
            tolerance = 0
        if hasattr(tolerance, "units"):  # pint quantity tolerances
            target = "s" if dtype_time_like(self.dtype) else self.units
            tolerance = convert_units(tolerance.magnitude, target, tolerance.units)
        if dtype_time_like(self.dtype):
            tolerance = dc.to_timedelta64(tolerance)
            zero = dc.to_timedelta64(0)
        else:
            zero = 0
        if tolerance < zero:
            msg = "simplify tolerance must not be negative."
            raise ParameterError(msg)
        return tolerance

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

    @compose_docstring(doc=get_docstring(BaseCoord.get_discontinuities))
    def get_discontinuities(self, kind="all", tolerance=None) -> pd.DataFrame:
        """{doc}."""
        if kind not in ("all", "gaps"):
            msg = f"kind must be 'all' or 'gaps', got {kind!r}"
            raise ParameterError(msg)
        offsets = self._segment_offsets()
        ascending = self.sorted
        rows = []
        for num in range(1, len(self.segments)):
            prev, nxt = self.segments[num - 1], self.segments[num]
            before = prev.max() if ascending else prev.min()
            after = nxt.min() if ascending else nxt.max()
            delta = after - before
            expected = self._expected_step(prev)
            excess = np.abs(delta) - np.abs(expected) if expected is not None else None
            rows.append(
                dict(
                    index=int(offsets[num]),
                    before=before,
                    after=after,
                    delta=delta,
                    excess=excess,
                )
            )
        df = pd.DataFrame(rows, columns=["index", "before", "after", "delta", "excess"])
        if kind == "gaps":
            tol = self._get_tolerance(tolerance)
            excess = df["excess"]
            df = df[~pd.isnull(excess) & (excess > tol)]
        return df.reset_index(drop=True)

    @staticmethod
    def _expected_step(seg) -> Any:
        """Return the expected next-sample spacing after a segment, or None."""
        if isinstance(seg, CoordRange):
            return seg.step
        if len(seg) > 1:
            values = seg.values
            return values[-1] - values[-2]
        return None

    @classmethod
    def from_array(cls, array, tolerance=None, units=None) -> BaseCoord:
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
        tolerance
            If not None, apply
            [`simplify`](`dascore.core.coords.BaseCoord.simplify`) with this
            tolerance to the result, re-fitting jittery runs and absorbing
            small gaps with bounded error.
        units
            Units for the coordinate.

        Examples
        --------
        >>> import numpy as np
        >>> from dascore.core.coords import CoordSegmented
        >>>
        >>> values = np.array([0.0, 1, 2, 3, 10, 11, 12, 13])
        >>> coord = CoordSegmented.from_array(values)
        >>> assert coord.segment_count == 2
        >>> assert len(coord.get_discontinuities("gaps")) == 1
        """
        values = np.asarray(array)
        if values.ndim != 1:
            msg = "from_array requires a 1D array."
            raise CoordError(msg)
        if pd.isnull(values).any():
            msg = "from_array does not support missing values."
            raise CoordError(msg)
        if len(values) < 3:
            out = get_coord(data=values, units=units)
        else:
            diffs = np.diff(values)
            zero = diffs[0] - diffs[0]
            if not (np.all(diffs > zero) or np.all(diffs < zero)):
                msg = "from_array requires strictly monotonic values."
                raise CoordError(msg)
            # A diff belongs to a uniform run when it matches a neighboring
            # diff; isolated diffs are seams (gaps or sampling changes).
            eq_next = diffs[:-1] == diffs[1:]
            in_run = np.zeros(len(diffs), dtype=bool)
            in_run[1:] |= eq_next
            in_run[:-1] |= eq_next
            splits = np.flatnonzero(~in_run) + 1
            blocks = np.split(values, splits)
            segments = [CoordMonotonicArray(values=x, units=units) for x in blocks]
            out = concat_coords(*segments)
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
    coords. Plain string selectors use exact matching unless they contain `*`
    or `?`, in which case they are treated as unix-style wildcard patterns.
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
        if isinstance(args, str) and ("*" in args or "?" in args):
            pattern = re.compile(fnmatch.translate(args))
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
            len=len(self),
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
) -> BaseCoord:
    """
    Return a coordinate from provided inputs.

    This function figures out which kind of Coordinate should be returned
    for provided inputs.

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
        if any(x is not None for x in others) or not pd.isnull(step):
            msg = "segments cannot be combined with other coordinate value inputs."
            raise CoordError(msg)
        return concat_coords(*segments, units=units)

    data = _get_array(data, values)
    shape = _get_shape(shape)
    if data is None and shape is not None:
        attrs = dict(
            shape=shape, start=start, stop=stop, step=step, units=units, dtype=dtype
        )
        try:  # This could be a normal RangeCoord
            return CoordRange(**attrs)
        except (ValidationError, CoordError):  # If not it's a partial
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
            if not pd.isnull(step):
                val = data[0]
                return CoordRange(start=val, stop=val + step, step=step, units=units)
            return CoordMonotonicArray(values=data, units=units)
        start, stop, step, monotonic = _maybe_get_start_stop_step(data)
        if start is not None:
            out = CoordRange(start=start, stop=stop, step=step, units=units)
            # The change_length call helps with float off by one issues.
            return out.change_length(len(data))
        elif monotonic:
            return CoordMonotonicArray(values=data, units=units)
        elif np.all(pd.isnull(data)):
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
        return CoordRange(start=start, stop=stop, step=step, units=units)
