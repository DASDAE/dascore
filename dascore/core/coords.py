"""
Machinery for coordinates.
"""
import abc
from functools import cache
from operator import gt, lt
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import root_validator
from rich.text import Text
from typing_extensions import Self

import dascore as dc
from dascore.compat import array
from dascore.constants import dascore_styles
from dascore.exceptions import CoordError, ParameterError
from dascore.units import Quantity, Unit, get_conversion_factor, get_factor_and_unit
from dascore.utils.display import get_nice_text
from dascore.utils.misc import all_diffs_close_enough, iterate
from dascore.utils.models import ArrayLike, DascoreBaseModel, DTypeLike, UnitQuantity
from dascore.utils.time import is_datetime64, is_timedelta64


@cache
def _get_coord_filter_validators(dtype):
    """Get filter validators for a given input type."""

    def _is_sub_dtype(dtype1, dtype2):
        """helper function to get sub dtypes."""
        # uncomment these if validators that arent numpy types are needed.
        # with suppress(TypeError):
        #     if issubclass(dtype1, dtype2):
        #         return True
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
    for (cls, func) in validators:
        if _is_sub_dtype(dtype, cls):
            out.append(func)
    return tuple(out)


def _get_nullish_for_type(dtype):
    """Returns an appropriate null value for a given numpy type."""
    if np.issubdtype(dtype, np.datetime64):
        return np.datetime64("NaT")
    if np.issubdtype(dtype, np.timedelta64):
        return np.timedelta64("NaT")
    # everything else should be a NaN (which is a float). This is
    # a bit of a problem for ints, which have no null rep., but upcasting
    # to float will probably cause less damage then using None
    return np.NaN


class BaseCoord(DascoreBaseModel, abc.ABC):
    """
    Coordinate interface.

    Coordinates are used to manage labels and indexing along a single
    data dimension.
    """

    units: Optional[UnitQuantity] = None
    step: Any

    _rich_style = dascore_styles["default_coord"]
    _evenly_sampled = False
    _sorted = False
    _reverse_sorted = False

    @abc.abstractmethod
    def convert_units(self, unit) -> Self:
        """Convert from one unit to another. Set units if None are set."""

    @abc.abstractmethod
    def select(self, arg, relative=False) -> Tuple[Self, Union[slice, ArrayLike]]:
        """
        Returns an entity that can be used in a list for numpy indexing
        and selected coord.
        """

    @abc.abstractmethod
    def __getitem__(self, item) -> Self:
        """Should implement slicing and return new instance."""

    @cache
    def __len__(self):
        """Total number of elements."""
        return np.prod(self.shape)

    def __hash__(self):
        """
        Simply use default hash rather than smart one (object) id
        so that cache work.
        """
        return id(self)

    def __rich__(self):
        key_style = dascore_styles["keys"]
        base = Text("", "default")
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
        if not pd.isnull(self.units):
            base += Text(" units: ", key_style)
            base += get_nice_text(self.units, style="units")
        base += Text(" )")
        return base

    def __str__(self):
        return str(self.__rich__())

    @cache
    def min(self):
        """return min value"""
        return self._min()

    @cache
    def max(self):
        """return max value"""
        return self._max()

    @property
    def degenerate(self):
        """Returns true if coord is degenerate (empty)"""
        return not bool(len(self))

    @abc.abstractmethod
    def _min(self):
        """Returns (or generates) the array data."""

    @abc.abstractmethod
    def _max(self):
        """Returns (or generates) the array data."""

    @property
    @cache
    def limits(self) -> Tuple[Any, Any]:
        """Returns a numpy datatype"""
        return self.min(), self.max()

    @property
    @abc.abstractmethod
    def dtype(self) -> DTypeLike:
        """Returns a numpy datatype"""

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the coordinate data."""
        return self.data.shape

    @property
    def evenly_sampled(self) -> Tuple[int, ...]:
        """Returns True if the coord is evenly sampled."""
        return self._evenly_sampled

    @property
    def sorted(self) -> Tuple[int, ...]:
        """Returns True if the coord in sorted."""
        return self._sorted

    @property
    def reverse_sorted(self) -> Tuple[int, ...]:
        """Returns True if the coord in sorted in reverse order."""
        return self._reverse_sorted

    def set_units(self, units) -> Self:
        """Set new units on coordinates."""
        new = dict(self)
        new["units"] = units
        return self.__class__(**new)

    def simplify_units(self) -> Self:
        """Simplify the coordinate units."""
        _, unit = get_factor_and_unit(self.units, simplify=True)
        return self.convert_units(unit)

    @abc.abstractmethod
    def sort(self, reverse=False) -> Tuple["BaseCoord", Union[slice, ArrayLike]]:
        """
        Sort the contents of the coord. Return new coord and slice for sorting.
        """

    def snap(self) -> "CoordRange":
        """
        Snap the coordinates to evenly sampled grid points.

        This will cause some loss of precision but often makes the data much
        easier to work with.
        """
        return self

    @abc.abstractmethod
    def update_limits(self, start=None, stop=None, step=None) -> Self:
        """Update the limits or sampling of the coordinates."""

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
            unit = value.units
            uf = get_conversion_factor(unit, self.units)
            value = value.magnitude * uf
        # if null or ... just return None
        if pd.isnull(value) or value is Ellipsis:
            return None
        # special case for datetime and relative
        if relative:
            if np.issubdtype(self.dtype, np.datetime64):
                value = dc.to_timedelta64(value)
            value = self._get_relative_values(value)
        # apply validators. These can, eg, coerce to correct dtype.
        validators = _get_coord_filter_validators(self.dtype)
        out = value
        for func in validators:
            if out is not None:
                out = func(out)
        return out

    def _slice_degenerate(self, sliz):
        """
        Return bool indicating if the slice should yeild degenerate
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
        select: Union[slice, None, type(Ellipsis), Tuple[Any, Any]],
        relative=False,
    ) -> Tuple[Any, Any]:
        """
        Get a tuple with (start, stop) and perform basic checks.

        Parameters
        ----------
        select
            An object for determining select range.
        """

        def _validate_slice(select):
            """Validation for slices."""
            if not isinstance(select, slice):
                return select
            if select.step is not None:
                msg = (
                    "Step not supported in select/filtering. Use decimate for "
                    "proper down-sampling."
                )
                raise ParameterError(msg)
            return (select.start, select.stop)

        def _validate_none_or_ellipsis(select):
            """Ensure None, ... are converted to tuple."""
            if select is None or select is Ellipsis:
                select = (None, None)
            return select

        def _validate_len(select):
            """Ensure select is a tuple of proper length."""
            if len(select) != 2:
                msg = "Slice indices must be length 2 sequence."
                raise ParameterError(msg)
            return select

        # limits = self.limits
        # apply simple checks; ensure we have a len 2 tuple.
        for func in [_validate_slice, _validate_none_or_ellipsis, _validate_len]:
            select = func(select)
        p1, p2 = (self._get_compatible_value(x, relative=relative) for x in select)
        # reverse order if needed to ensure p1 < p2
        if p1 is not None and p2 is not None and p2 < p1:
            p1, p2 = p2, p1

        return p1, p2

    def _get_relative_values(self, value):
        """Get relative values based on start (pos) or stop (neg)"""
        pos = value >= 0
        return self.min() + value if pos else self.max() + value

    def empty(self, axes=None) -> Self:
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
            new_shape = np.array(self.shape)
            for ind in iterate(axes):
                new_shape[ind] = 0
        data = np.empty(tuple(new_shape), dtype=self.dtype)
        return get_coord(values=data)

    def index(self, indexer, axis: Optional[int] = None) -> Self:
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
        return get_coord(values=array, units=self.units)

    def get_attrs_dict(self, name):
        """Get attrs dict."""
        out = {f"{name}_min": self.min(), f"{name}_max": self.max()}
        if self.step:
            out[f"d_{name}"] = self.step
        if self.units:
            out[f"{name}_units"] = self.units
        return out


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

    start: Any
    stop: Any
    step: Any
    _evenly_sampled = True
    _rich_style = dascore_styles["coord_range"]

    @root_validator()
    def ensure_all_attrs_set(cls, values):
        """If any info is neglected the coord is invalid."""
        for name in ["start", "stop", "step"]:
            assert values[name] is not None
        return values

    @root_validator()
    def _set_stop(cls, values):
        """Set stop to integral value >= current stop."""
        dur = values["stop"] - values["start"]
        if values["step"] == 0:
            return values
        int_val = int(np.ceil(np.round(dur / values["step"], 1)))
        values["stop"] = values["start"] + values["step"] * int_val
        return values

    def __getitem__(self, item):
        if isinstance(item, int):
            if item >= len(self):
                raise IndexError(f"{item} exceeds coord length of {self}")
            return self.values[item]
        # Todo we can probably add more intelligent logic for slices.
        out = self.values[item]
        return get_coord(values=out, units=self.units)

    @cache
    def __len__(self):
        if self.start == self.stop:
            return 1
        out = abs((self.stop - self.start) / self.step)
        # due to floating point weirdness this can sometimes be very close
        # but not exactly an int, so we need to round.
        return int(np.round(out))

    def convert_units(self, units) -> Self:
        """
        Convert units, or set units if none exist.
        """
        is_time = np.issubdtype(self.dtype, np.datetime64)
        is_time_delta = np.issubdtype(self.dtype, np.timedelta64)
        if self.units is None or is_time or is_time_delta:
            return self.set_units(units)
        out = dict(units=units)
        factor = get_conversion_factor(self.units, units)
        for name in ["start", "stop", "step"]:
            out[name] = getattr(self, name) * factor
        return self.new(**out)

    def select(self, args, relative=False) -> Tuple[BaseCoord, Union[slice, ArrayLike]]:
        """
        Apply select, return selected coords and index to apply to array.

        Can return a CoordDegenerate if selection is outside of range.
        """
        args = self.get_slice_tuple(args, relative=relative)
        start = self._get_index(args[0], forward=self.sorted)
        stop = self._get_index(args[1], forward=self.reverse_sorted)
        if self.reverse_sorted:
            start, stop = stop, start
        # we add 1 to stop in slice since its upper limit is exclusive
        out = slice(start, (stop + 1) if stop is not None else stop)
        if self._slice_degenerate(out):
            return self.empty(), slice(0, 0)
        new_start = self[start] if start is not None else self.start
        new_end = self[stop] + self.step if stop is not None else self.stop
        new = self.new(start=new_start, stop=new_end)
        return new, out

    def sort(self, reverse=False) -> Tuple["BaseCoord", Union[slice, ArrayLike]]:
        """
        Sort the contents of the coord. Return new coord and slice for sorting.
        """
        #
        forward_forward = not reverse and self.sorted
        reverse_reverse = reverse and self.reverse_sorted
        if forward_forward or reverse_reverse:
            return self, slice(None)
        new_step = -self.step
        if reverse:  # reversing a forward sorted Coordrange
            new_start, new_stop = self.max(), self.min() + new_step
        else:  # order a reverse sorted one
            new_start, new_stop = self.min(), self.max() + new_step
        out = self.new(start=new_start, stop=new_stop, step=new_step)
        return out, slice(None, None, -1)

    def _get_index(self, value, forward=True):
        """Get the index corresponding to a value."""
        if (value := self._get_compatible_value(value)) is None:
            return value
        func = np.ceil if forward else np.floor
        start, _, step = self.start, self.stop, self.step
        # Due to float weirdness we need a little bit of a fudge factor here.
        fraction = func(np.round((value - start) / step, decimals=10))
        out = int(fraction)
        if (out <= 0 and forward) or (out >= len(self) and not forward):
            return None
        return out

    def update_limits(self, start=None, stop=None, step=None) -> Self:
        """
        Update the limits or sampling of the coordinates.

        If start and stop are defined a new step is determined and returned.
        Next, the step size is updated changing only the end. Then the start
        is updated changing the start/end. Then the end is updated changing
        the start/end.

        Notes
        -----
        For ease of use, stop is considered inclusive, meaning the real stop
        value in the output will be stop + step.
        """
        if all(x is not None for x in [start, stop, step]):
            msg = "At most two parameters can be specified in update_limits."
            raise ValueError(msg)
        # first case, we need to determine new dt.
        if start is not None and stop is not None:
            new_step = (stop - start) / len(self)
            return get_coord(start=start, stop=stop, step=new_step, units=self.units)
        # for other combinations we just apply adjustments sequentially.
        out = self
        if step is not None:
            new_stop = out.start + step * len(out)
            out = out.new(stop=new_stop, step=step)
        if start is not None:
            diff = start - out.start
            new_stop = out.stop + diff
            out = out.new(start=start, stop=new_stop)
        if stop is not None:
            translation = (stop + out.step) - out.stop
            new_start = self.start + translation
            # we add step so the new range is inclusive of stop.
            out = out.new(start=new_start, stop=stop + out.step)
        return out

    @property
    @cache
    def values(self) -> ArrayLike:
        """Return the values of the coordinate as an array."""
        if len(self) == 1:
            return np.array([self.start])
        # note: linspace works better for floats that might have slightly
        # uneven spacing. It ensures the length of the output array is robust
        # to small deviations in spacing. However, this doesnt work for datetimes.
        if is_datetime64(self.start) or is_timedelta64(self.start):
            out = np.arange(self.start, self.stop, self.step)
        else:
            out = np.linspace(self.start, self.stop - self.step, num=len(self))
        # again, due to roundoff error the array can one element longer than
        # anticipated. The slice here just ensures shape and len match.
        return array(out[: len(self)])

    def _min(self):
        """Return min value"""
        return np.min([self.start, self.stop - self.step])

    def _max(self):
        """Return max value in range."""
        # like range, coord range is exclusive of final value.
        # the min/max are needed for reverse sorted coord.
        return np.max([self.stop - self.step, self.start])

    @property
    def sorted(self) -> bool:
        """Returns true if sorted in ascending order."""
        return self.step >= 0

    @property
    def reverse_sorted(self) -> bool:
        """Returns true if sorted in ascending order."""
        return self.step < 0

    @property
    @cache
    def dtype(self):
        """Returns datatype."""
        # some types are weird so we create a small array here to let
        # numpy determine its dtype. It should only be 1 element long
        # so not expensive to do.
        return np.arange(self.start, self.start + self.step, self.step).dtype


class CoordArray(BaseCoord):
    """
    A coordinate with arbitrary values in an array.

    Can handle any number of dimensions.
    """

    values: ArrayLike
    _rich_style = dascore_styles["coord_array"]

    def convert_units(self, units) -> Self:
        """Convert units, or set units if none exist."""
        is_time = np.issubdtype(self.dtype, np.datetime64)
        is_time_delta = np.issubdtype(self.dtype, np.timedelta64)
        if self.units is None or is_time or is_time_delta:
            return self.set_units(units)
        factor = get_conversion_factor(self.units, units)
        return self.new(units=units, values=self.values * factor)

    def select(self, args, relative=False) -> Tuple[Self, Union[slice, ArrayLike]]:
        """Apply select, return selected coords and index for selecting data."""
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
        return self.new(values=values[out]), out

    def sort(self, reverse=False) -> Tuple[BaseCoord, Union[slice, ArrayLike]]:
        """Sort the coord to be monotonic (maybe range)."""
        argsort: ArrayLike = np.argsort(self.values)[:: -1 if reverse else 1]
        arg_dict = dict(self)
        arg_dict["values"] = self.values[argsort]
        new = get_coord(**arg_dict)
        return new, argsort

    def snap(self):
        """
        Snap the coordinates to evenly sampled grid points.

        This will cause some loss of precision but often makes the coordinate
        much easier to work with.
        """
        values = self.values
        min_v, max_v = np.min(values), np.max(values)
        if len(self) == 1:
            # time deltas need to be generated for dt case, hence the subtract
            _zero = self._get_compatible_value(0)
            step = _zero - _zero
        else:
            dur = max_v - min_v
            is_dt = is_timedelta64(dur)
            # hack to handle dts int division.
            if is_dt:
                _step = float(dur.astype(np.int64)) / (len(self) - 1)
                step = np.timedelta64(int(np.round(_step)), "ns")
            else:
                step = dur / (len(self) - 1)
            assert step > 0
        if self.reverse_sorted:
            step = -step
            start, stop = max_v, min_v + step
        else:
            start, stop = min_v, max_v + step
        return CoordRange(start=start, stop=stop, step=step, units=self.units)

    def update_limits(self, start=None, stop=None, step=None) -> Self:
        """
        Update the limits or sampling of the coordinates.

        This is more limited than with CoordRange since the data are not
        evenly sampled. In order to change the step, you must first call
        [snap](`dascore.core.coords
        """
        if sum(x is not None for x in [start, stop, step]) > 1:
            msg = "At most one parameter can be specified in update_limits."
            raise ValueError(msg)
        out = self
        if step is not None:
            out = self.snap().update_limits(step=step)
        elif start is not None:
            diff = start - self.min()
            vals = self.values + diff
            out = get_coord(values=vals, units=self.units)
        elif stop is not None:
            diff = stop - self.max()
            vals = self.values + diff
            out = get_coord(values=vals, units=self.units)
        return out

    def __getitem__(self, item) -> Self:
        out = self.values[item]
        if not np.ndim(out):
            return out
        return self.__class__(values=out, units=self.units)

    def __hash__(self):
        return hash(id(self))

    def _min(self):
        """Return min value"""
        try:
            return np.min(self.values)
        except ValueError:  # degenerate data case
            return _get_nullish_for_type(self.dtype)

    def _max(self):
        """Return max value in range."""
        try:
            return np.max(self.values)
        except ValueError:  # degenerate data case
            return _get_nullish_for_type(self.dtype)

    @property
    @cache
    def dtype(self):
        """Returns datatype."""
        return self.values.dtype

    @property
    @cache
    def shape(self):
        """Return the shape of the coordinate."""
        return np.shape(self.values)

    def __eq__(self, other):
        try:
            self_d, other_d = dict(self), dict(other)
        except (TypeError, ValueError):
            return False
        v1, v2 = self_d.pop("values", None), other_d.pop("values", None)
        shapes_same = v1.shape == v2.shape
        # Frustratingly, all cose doesn't work with datetime64 so we we need
        # this short-circuiting equality check.
        values_same = shapes_same and ((np.all(v1 == v2) or np.allclose(v1, v2)))
        if values_same and self_d == other_d and values_same:
            return True
        return False


class CoordMonotonicArray(CoordArray):
    """
    A coordinate with strictly increasing or decreasing values.
    """

    values: ArrayLike
    _rich_style = dascore_styles["coord_monotonic"]
    _sorted = True

    def select(self, args, relative=False) -> Tuple[Self, Union[slice, ArrayLike]]:
        """Apply select, return selected coords and index for selecting data."""
        v1, v2 = self.get_slice_tuple(args, relative=relative)
        # reverse order if reverse monotonic. This is done so when we mult
        # by -1 in _get_index the inverted range is used.
        if self.reverse_sorted:
            v1, v2 = v2, v1
        start = self._get_index(v1, left=True)
        new_start = start if start is not None and start > 0 else None
        stop = self._get_index(v2, left=False)
        new_stop = stop if stop is not None and stop < len(self) else None
        out = slice(new_start, new_stop)
        if self._slice_degenerate(out):
            return self.empty(), slice(0, 0)
        return self.new(values=self.values[out]), out

    def _get_index(self, value, left=True):
        """
        Get the index corresponding to a value.

        Left indicates if this is the min value.
        """
        if (value := self._get_compatible_value(value)) is None:
            return value
        values = self.values
        side_dict = {True: "left", False: "right"}
        # since search sorted only works on ascending monotonic arrays we
        # negative descending arrays to get the same effect.
        if self.reverse_sorted:
            values = values * -1
            value = value * -1
        ind = np.searchsorted(values, value, side=side_dict[left])
        return ind

    def _step_meets_requirement(self, op):
        """Return True is any data increment meets the comp. requirement."""
        vals = self.values
        # we must iterate because first two elements might be equal.
        for ind in range(1, len(self)):
            if op(vals[ind], vals[ind - 1]):
                return True
        return False

    @property
    @cache
    def sorted(self):
        """Determine is coord array is sorted in ascending order."""
        return self._step_meets_requirement(gt)

    @property
    @cache
    def reverse_sorted(self):
        """Determine is coord array is sorted in descending order."""
        return self._step_meets_requirement(lt)


class CoordDegenerate(CoordArray):
    """
    A coordinate with degenerate (empty on one axis) data.
    """

    values: ArrayLike
    step: Any = None
    _rich_style = dascore_styles["coord_degenerate"]

    def select(self, args, relative=False) -> Tuple[Self, Union[slice, ArrayLike]]:
        """Select for Degenerate coords does nothing."""
        return self, slice(None, None)

    def empty(self, axes=None) -> Self:
        """Empty simply returns self."""
        return self

    def snap(self, axes=None) -> Self:
        """Empty simply returns self."""
        return self

    @property
    def evenly_sampled(self):
        """If the degenerate was evenly sampled."""
        return self.step is not None


def get_coord(
    *,
    values: Union[ArrayLike, None, np.ndarray] = None,
    start=None,
    stop=None,
    step=None,
    units: Union[None, Unit, Quantity, str] = None,
) -> BaseCoord:
    """
    Given multiple types of input, return a coordinate.

    Parameters
    ----------
    values
        An array of values.
    start
        The start value of the array, inclusive.
    stop
        The stopping value of an array, exclusive.
    step
        The sampling spacing of an array.
    units
        Indication of units.

    Notes
    -----
    The following combinations of input parameters are typical:
        (start, stop, step)
        (values)
        (values, step) - useful for length 1 arrays.
    """

    def _check_inputs(data, start, stop, step):
        """Ensure input combinations are valid."""
        if data is None:
            if any([start is None, stop is None, step is None]):
                msg = "When data is not defined, start, stop, and step must be."
                raise CoordError(msg)

    def _maybe_get_start_stop_step(data):
        """Get start, stop, step, is_monotonic"""
        data = np.array(data)
        view2 = data[1:]
        view1 = data[:-1]
        is_monotonic = np.all(view1 > view2) or np.all(view2 > view1)
        # the array cannot be evenly sampled if it isn't monotonic
        if is_monotonic:
            diffs = view2 - view1
            unique_diff = np.unique(diffs)
            # here we are dealing with a length 0 or 1 array.
            if len(view2) < 2:
                return None, None, None, False
            if len(unique_diff) == 1 or all_diffs_close_enough(unique_diff):
                _min = data[0]
                _max = data[-1]
                # this is a poor man's median that preserves dtype
                _step = np.sort(diffs)[len(diffs) // 2]
                return _min, _max + _step, _step, is_monotonic
        return None, None, None, is_monotonic

    _check_inputs(values, start, stop, step)
    # data array was passed; see if it is monotonic/evenly sampled
    if values is not None:
        if isinstance(values, BaseCoord):  # just return coordinate
            return values
        if np.size(values) == 0:
            return CoordDegenerate(values=values, units=units, step=step)
        # special case of len 1 array that specify step
        elif len(values) == 1 and not pd.isnull(step):
            val = values[0]
            return CoordRange(start=val, stop=val + step, step=step, units=units)
        start, stop, step, monotonic = _maybe_get_start_stop_step(values)
        if start is not None:
            out = CoordRange(start=start, stop=stop, step=step, units=units)
            return out
        elif monotonic:
            return CoordMonotonicArray(values=values, units=units)
        return CoordArray(values=values, units=units)
    else:
        return CoordRange(start=start, stop=stop, step=step, units=units)


def get_coord_from_attrs(attrs, name):
    """
    Try to get a coordinate from an attributes dict.

    For this to work, attrs must have {name}_min, {name}_max, d_{name}
    and optionally {name}_units. If the requirements are not met a
    CoordError is raised.
    """
    start, stop = attrs.get(f"{name}_min"), attrs.get(f"{name}_max")
    step = attrs.get(f"d_{name}")
    units = attrs.get(f"{name}_units")
    if all([x is not None for x in [start, stop, step]]):
        return get_coord(start=start, stop=stop + step, step=step, units=units)
    msg = f"Could not get coordinate from {attrs}"
    raise CoordError(msg)
