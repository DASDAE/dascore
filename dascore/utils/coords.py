"""
Machinery for coordinates.
"""
import abc
from contextlib import suppress
from functools import cache
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import root_validator
from rich.text import Text
from typing_extensions import Self

import dascore as dc
from dascore.compat import array
from dascore.constants import PatchType
from dascore.exceptions import CoordError, ParameterError, SelectRangeError
from dascore.utils.display import get_nice_string
from dascore.utils.misc import iterate
from dascore.utils.models import ArrayLike, DascoreBaseModel, DTypeLike, Unit
from dascore.utils.time import is_datetime64, is_timedelta64, to_number
from dascore.utils.units import get_conversion_factor


@cache
def _get_coord_filter_validators(dtype):
    """Get filter validators for a given input type."""

    def _is_sub_dtype(dtype1, dtype2):
        """helper function to get sub dtypes."""
        with suppress(TypeError):
            if issubclass(dtype1, dtype2):
                return True
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


def assign_coords(patch: PatchType, **kwargs) -> PatchType:
    """
    Add non-dimensional coordinates to a patch.

    Parameters
    ----------
    patch
        The patch to which coordinates will be added.
    **kwargs
        Used to specify the name, dimension, and values of the new
        coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> patch_1 = dc.get_example_patch()
    >>> coords = patch_1.coords
    >>> dist = coords['distance']
    >>> time = coords['time']
    >>> # Add a single coordinate associated with distance dimension
    >>> lat = np.arange(0, len(dist)) * .001 -109.857952
    >>> out_1 = patch_1.assign_coords(latitude=('distance', lat))
    >>> # Add multiple coordinates associated with distance dimension
    >>> lon = np.arange(0, len(dist)) *.001 + 41.544654
    >>> out_2 = patch_1.assign_coords(
    ...     latitude=('distance', lat),
    ...     longitude=('distance', lon),
    ... )
    >>> # Add multi-dimensional coordinates
    >>> quality = np.ones_like(patch_1.data)
    >>> out_3 = patch_1.assign_coords(
    ...     quality=(patch_1.dims, quality)
    ... )
    """
    coords = patch.coords.update_coords(**kwargs)
    return patch.new(coords=coords, dims=patch.dims)


class BaseCoord(DascoreBaseModel, abc.ABC):
    """
    Coordinate interface.

    Coordinates are used to manage labels and indexing along a single
    data dimension.
    """

    units: Optional[Unit]
    step: Any
    _even_sampling = False
    _rich_style = "bold white"

    @abc.abstractmethod
    def convert_units(self, unit) -> Self:
        """Convert from one unit to another. Set units if None are set."""

    @abc.abstractmethod
    def select(self, arg) -> Tuple[Self, Union[slice, ArrayLike]]:
        """
        Returns an entity that can be used in a list for numpy indexing
        and selected coord.
        """

    @abc.abstractmethod
    def __getitem__(self, item) -> Self:
        """Should implement slicing and return new instance."""

    @abc.abstractmethod
    def __len__(self):
        """Total number of elements."""

    def __hash__(self):
        """
        Simply use default hash rather than smart one (object) id
        so that cache work.
        """
        return id(self)

    def __rich__(self):
        base = Text(self.__class__.__name__, style=self._rich_style)
        t1 = Text("")
        if not pd.isnull(self.min()):
            t1 += Text(f" min={get_nice_string(self.min())}")
        if not pd.isnull(self.max()):
            t1 += Text(f" max={get_nice_string(self.max())}")
        if not pd.isnull(self.step):
            t1 += f" step={get_nice_string(self.step)}"
        t1 += f" shape={self.shape}"
        t1 += f" dtype={self.dtype}"
        if not pd.isnull(self.units):
            t1 += f" units={get_nice_string(self.units)}"
        out = Text.assemble(base, f"({t1[1:]})")
        return out

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

    def set_units(self, units) -> Self:
        """Set new units on coordinates."""
        new = dict(self)
        new["units"] = units
        return self.__class__(**new)

    def sort(self) -> Tuple["BaseCoord", Union[slice, ArrayLike]]:
        """
        Sort the contents of the coord. Return new coord and slice for sorting.
        """
        return self, slice(None)

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

    def _get_compatible_value(self, value, invert=False):
        """
        Return values that are compatible with dtype/units of coord.

        This is used, for example, to coerce values in select tuple
        so direct comparison with coord values is possible.
        """
        # strip units and v
        if hasattr(value, "units"):
            unit = value.units
            i_unit = (1 / unit).units
            if i_unit == self.units:
                value = 1 / value
                unit = i_unit
            uf = get_conversion_factor(unit, self.units)
            value = value.magnitude * uf
        # if null or ... just return None
        if pd.isnull(value) or value is Ellipsis:
            return None
        # apply validators. These can, eg, coerce to correct dtype.
        validators = _get_coord_filter_validators(self.dtype)
        out = value
        for func in validators:
            if out is not None:
                out = func(out)
        return out if not invert else 1 / out

    def _validate_slice(self, sliz, args):
        """Validate that the slice is not empty, else raise."""
        if sliz.start is not None and sliz.start == sliz.stop:
            msg = f"{args} selects a region between samples, no data to return."
            raise SelectRangeError(msg)

    def get_slice_tuple(
        self,
        select: Union[slice, None, type(Ellipsis), Tuple[Any, Any]],
        check_oder: bool = True,
        invert=False,
    ) -> Tuple[Any, Any]:
        """
        Get a tuple with (start, stop) and perform basic checks.

        Parameters
        ----------
        select
            An object for determining select range.
        check_oder
            Ensure select[0] <= select[1]
        invert
            If true, return the inverted range, meaning 1/end, 1/start.
            This is useful, for example, for getting frequencies (1/s) when
            coordinates are in s.
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

        limits = self.limits
        # apply simple checks; ensure we have a len 2 tuple.
        for func in [_validate_slice, _validate_none_or_ellipsis, _validate_len]:
            select = func(select)
        p1, p2 = (self._get_compatible_value(x, invert=invert) for x in select)
        # reverse order if needed.
        if p1 is not None and p2 is not None and p2 < p1:
            p1, p2 = p2, p1
        # Perform check that the requested select range isn't out of bounds
        if limits is not None:
            start, stop = p1, p2
            bad_start = start is not None and start > limits[1]
            bad_end = stop is not None and stop < limits[0]
            if bad_end or bad_start:
                msg = (
                    f"The select range ({start}, {stop}) is out of bounds for "
                    f"data with limits {limits}"
                )
                raise SelectRangeError(msg)
        return p1, p2

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
    _even_sampling = True
    _rich_style = "bold green"

    @root_validator()
    def ensure_all_attrs_set(cls, values):
        """If any info is neglected the coord is invalid."""
        for name in ["start", "stop", "step"]:
            assert values[name] is not None
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
        out = abs((self.stop - self.start) / self.step)
        # due to floating point weirdness this can sometimes be very close
        # but not exactly an int, so we need to round.
        return int(np.round(out))

    def convert_units(self, units) -> Self:
        """
        Convert units, or set units if none exist.
        """
        if self.units is None:
            return self.set_units(units)
        out = dict(units=units)
        factor = get_conversion_factor(self.units, units)
        for name in ["start", "stop", "step"]:
            out[name] = getattr(self, name) * factor
        return self.new(**out)

    def select(self, args) -> Tuple[Self, Union[slice, ArrayLike]]:
        """Apply select, return selected coords and index for selecting data."""
        args = self.get_slice_tuple(args)
        start = self._get_index(args[0])
        stop = self._get_index(args[1], forward=False)
        # we add 1 to stop in slice since its upper limit is exclusive
        out = slice(start, (stop + 1) if stop is not None else stop)
        self._validate_slice(out, args)
        new_start = self[start] if start is not None else self.start
        new_end = self[stop] + self.step if stop is not None else self.stop
        new = self.new(start=new_start, stop=new_end)
        return new, out

    def _get_index(self, value, forward=True):
        """Get the index corresponding to a value."""
        if (value := self._get_compatible_value(value)) is None:
            return value
        func = np.ceil if forward else np.floor
        start, _, step = self.start, self.stop, self.step
        # Due to float weirdness we need a little bit of a fudge factor here.
        fraction = func(np.round((value - start) / step, decimals=10))
        out = int(fraction)
        if (out <= 0 and forward) or out >= len(self):
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
        # note: linspace works better for floats that might have slightly
        # uneven spacing. It ensures the length of the output array is robust
        # to small deviations in spacing. However, this doesnt work for datetimes.
        if is_datetime64(self.start) or is_timedelta64(self.start):
            out = np.arange(self.start, self.stop, self.step)
        else:
            out = np.linspace(self.start, self.stop - self.step, num=len(self))
        return array(out)

    def _min(self):
        """Return min value"""
        return self.start

    def _max(self):
        """Return max value in range."""
        # like range, coord range is exclusive of final value.
        return self.stop - self.step

    @property
    @cache
    def dtype(self):
        """Returns datatype."""
        return np.arange(self.start, self.start + self.step, self.step).dtype


class CoordArray(BaseCoord):
    """
    A coordinate with arbitrary values in an array.

    Can handle any number of dimensions.
    """

    values: ArrayLike
    _rich_style = "bold #cd0000"

    def convert_units(self, units) -> Self:
        """Convert units, or set units if none exist."""
        if self.units is None:
            return self.set_units(units)
        factor = get_conversion_factor(self.units, units)
        return self.new(units=units, values=self.values * factor)

    def select(self, args) -> Tuple[Self, Union[slice, ArrayLike]]:
        """Apply select, return selected coords and index for selecting data."""
        args = self.get_slice_tuple(args)
        values = self.values
        out = np.ones_like(values, dtype=np.bool_)
        val1 = self._get_compatible_value(args[0])
        val2 = self._get_compatible_value(args[1])
        if val1 is not None:
            out = out & (values >= val1)
        if val2 is not None:
            out = out & (values <= val2)
        if not np.any(out):
            self._validate_slice(slice(0, 0), args)
        return self.new(values=values[out]), out

    def sort(self) -> Tuple[BaseCoord, Union[slice, ArrayLike]]:
        """Sort the coord to be monotonic (maybe range)."""
        argsort: ArrayLike = np.argsort(self.values)
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
        is_datetime = np.issubdtype(self.dtype, (np.datetime64, np.timedelta64))
        if is_datetime:
            values = to_number(self.values)
        max_v, min_v = np.max(values), np.min(values)
        step = (max_v - min_v) / (len(self) - 1)
        if is_datetime:
            max_v = np.datetime64(int(max_v), "ns")
            min_v = np.datetime64(int(min_v), "ns")
            step = np.timedelta64(int(np.round(step)), "ns")
        return CoordRange(start=min_v, stop=max_v, step=step, units=self.units)

    def update_limits(self, start=None, stop=None, step=None) -> Self:
        """
        Update the limits or sampling of the coordinates.

        This is more limited than with CoordRange since the data are not
        evenly sampled. In order to change the step, you must first call
        [snap](`dascore.utils.coords
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

    @cache
    def __len__(self):
        return len(self.values)

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
    _rich_style = "bold #d64806"

    def select(self, args) -> Tuple[Self, Union[slice, ArrayLike]]:
        """Apply select, return selected coords and index for selecting data."""
        v1, v2 = self.get_slice_tuple(args, check_oder=False)
        # reverse order if reverse monotonic. This is done so when we mult
        # by -1 in _get_index the inverted range is used.
        if self._is_reversed:
            v1, v2 = v2, v1
        # start_forward = False if self._is_reversed else True
        start = self._get_index(v1, left=True)
        new_start = start if start is not None and start > 0 else None
        stop = self._get_index(v2, left=False)
        new_stop = stop if stop is not None and stop < len(self) else None
        out = slice(new_start, new_stop)
        self._validate_slice(out, args)
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
        if self._is_reversed:
            values = values * -1
            value = value * -1
        ind = np.searchsorted(values, value, side=side_dict[left])
        return ind

    @property
    @cache
    def _is_reversed(self):
        vals = self.values
        return (vals[1] - vals[0]) < 0


class CoordDegenerate(CoordArray):
    """
    A coordinate with degenerate (empty on one axis) data.
    """

    values: ArrayLike
    _rich_style = "bold #d40000"

    def select(self, args) -> Tuple[Self, Union[slice, ArrayLike]]:
        """Select for Degenerate coords does nothing."""
        return self, slice(None, None)

    def empty(self, axes=None) -> Self:
        """Empty simply returns self."""
        return self


def get_coord(
    *,
    values: Union[ArrayLike, None, np.ndarray] = None,
    start=None,
    stop=None,
    step=None,
    units: Union[None, Unit, str] = None,
) -> BaseCoord:
    """
    Given multiple types of input, return a coordinate.
    """

    def _check_inputs(data, start, stop, step):
        """Ensure input combinations are valid."""
        if data is None:
            if any([start is None, stop is None, step is None]):
                msg = "When data is not defined, start, stop, and step must be."
                raise CoordError(msg)

    def _all_diffs_close(diffs):
        """Check if all the diffs are 'close' handling timedeltas."""
        if np.issubdtype(diffs.dtype, np.timedelta64):
            diffs = diffs.astype(np.int64)
        return np.allclose(diffs, diffs[0])

    def _maybe_get_start_stop_step(data):
        """Get start, stop, step, is_monotonic"""
        data = np.array(data)
        view2 = data[1:]
        view1 = data[:-1]
        is_monotonic = np.all(view1 > view2) or np.all(view2 > view1)
        # the array cannot be evenly sampled if it isn't monotonic
        if is_monotonic:
            unique_diff = np.unique(view2 - view1)
            # here we are dealing with a length 0 or 1 array.
            if not len(unique_diff):
                return None, None, None, False
            if len(unique_diff) == 1 or _all_diffs_close(unique_diff):
                _min = data[0]
                _max = data[-1]
                _step = unique_diff[0]
                return _min, _max + _step, _step, is_monotonic
        return None, None, None, is_monotonic

    _check_inputs(values, start, stop, step)
    # data array was passed; see if it is monotonic/evenly sampled
    if values is not None:
        if isinstance(values, BaseCoord):  # just return coordinate
            return values
        if np.size(values) == 0:
            return CoordDegenerate(values=values, units=units)
        # special case of len 1 array that specify step
        elif len(values) == 1 and step is not None:
            val = values[0]
            return CoordRange(start=val, stop=val, step=step, units=units)
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
