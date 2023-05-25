"""
Machinery for coordinates.
"""
import abc
from functools import cache
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import root_validator
from typing_extensions import Self

from dascore.exceptions import CoordError
from dascore.utils.misc import iterate
from dascore.utils.models import ArrayLike, DascoreBaseModel, DTypeLike, Unit
from dascore.utils.time import is_datetime64, to_datetime64, to_number
from dascore.utils.units import get_conversion_factor


class BaseCoord(DascoreBaseModel, abc.ABC):
    """Coordinate interface."""

    units: Optional[Unit]
    _even_sampling = False

    @abc.abstractmethod
    def convert_units(self, unit) -> Self:
        """Convert from one unit to another. Set units if None are set."""

    @abc.abstractmethod
    def filter(self, arg) -> Tuple[Self, Union[slice, ArrayLike]]:
        """
        Returns an entity that can be used in a list for numpy indexing
        and filtered coord.
        """

    @abc.abstractmethod
    def __getitem__(self, item) -> Self:
        """Should implement slicing and return new instance."""

    @abc.abstractmethod
    def __len__(self):
        """Total number of elements."""

    @property
    @abc.abstractmethod
    def min(self):
        """Returns (or generates) the array data."""

    @property
    @abc.abstractmethod
    def max(self):
        """Returns (or generates) the array data."""

    @property
    @abc.abstractmethod
    def dtype(self) -> DTypeLike:
        """Returns a numpy datatype"""

    def set_units(self, units) -> Self:
        """Set new units on coordinates."""
        new = dict(self)
        new["units"] = units
        return self.__class__(**new)

    def _cast_values(self, value):
        """
        Function to cast input values to current dtype.

        Used, for example, to allow strings to represent dates.
        """
        if np.issubdtype(self.dtype, np.datetime64):
            return to_datetime64(value)
        return value

    def _get_value_for_indexing(self, value, invert=False):
        """Function to get a value for indexing."""
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
        return self._cast_values(value if not invert else 1 / value)

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

    def get_query_range(
        self, start, end, invert=False
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Get a tuple of query ranges normalized to coordinate units.

        Parameters
        ----------
        start
            The starting value of the query.
        end
            The ending value of the query.
        invert
            If true, return the inverted range, meaning 1/end, 1/start.
            This is useful, for example, for getting frequencies (1/s) when
            coordinates are in s.
        """
        out = [
            self._get_value_for_indexing(start, invert=invert),
            self._get_value_for_indexing(end, invert=invert),
        ]
        if not any(pd.isnull(out)):
            return tuple(sorted(out))
        return tuple(out)

    @property
    def data(self):
        """Return the internal data. Same as values attribute."""
        return self.values


class CoordRange(BaseCoord):
    """A coordinate represent a range of evenly sampled data."""

    start: Any
    stop: Any
    step: Any
    _even_sampling = True

    @root_validator()
    def ensure_all_attrs_set(cls, values):
        """If any info is neglected the coord is invalid."""
        for name in ["start", "stop", "step"]:
            assert values[name] is not None
        return values

    def __getitem__(self, item):
        return self.values[item]

    @cache
    def __len__(self):
        out = abs((self.stop - self.start) / self.step)
        # due to floating point weirdness this can sometimes be very close
        # but not exactly an int, so we need to round.
        return int(np.round(out))

    def convert_units(self, units) -> Self:
        """Convert units, or set units if none exist."""
        if self.units is None:
            return self.set_units(units)
        out = dict(units=units)
        factor = get_conversion_factor(self.units, units)
        for name in ["start", "stop", "step"]:
            out[name] = getattr(self, name) * factor
        return self.update(**out)

    def filter(self, args) -> Tuple[Self, Union[slice, ArrayLike]]:
        """Apply filter, return filtered coords and index for filtering data."""
        if isinstance(args, Sequence):
            assert len(args) == 2, "Only length two sequence allowed for indexing."
            start = self._get_index(args[0])
            stop = self._get_index(args[1], forward=False)
            out = slice(start, stop)
            new_start = self[start] if start else self.start
            new_end = self[stop] if stop else self.stop
            new = self.update(start=new_start, stop=new_end)
            return new, out

    def _get_index(self, value, forward=True):
        """Get the index corresponding to a value."""
        if (value := self._get_value_for_indexing(value)) is None:
            return value
        func = np.ceil if forward else np.floor
        start, _, step = self.start, self.stop, self.step
        # Due to float weirdness we need a little bit of a fudge factor here.
        fraction = func(np.round((value - start) / step, decimals=10))
        out = int(fraction)
        if out <= 0 or out >= len(self):
            return None
        return out

    @property
    def values(self) -> ArrayLike:
        """Return the values of the coordinate as an array."""
        # note: linspace works better for floats that might have slightly
        # uneven spacing. It ensures the length of the output array is robust
        # to small deviations in spacing. However, this doesnt work for datetimes.
        if is_datetime64(self.start):
            return np.arange(self.start, self.stop, self.step)
        return np.linspace(self.start, self.stop - self.step, num=len(self))

    @property
    def min(self):
        """Return min value"""
        return self.start

    @property
    def max(self):
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
    """

    values: ArrayLike

    def convert_units(self, units) -> Self:
        """Convert units, or set units if none exist."""
        if self.units is None:
            return self.set_units(units)
        factor = get_conversion_factor(self.units, units)
        return self.update(units=units, values=self.values * factor)

    def filter(self, args) -> Tuple[Self, Union[slice, ArrayLike]]:
        """Apply filter, return filtered coords and index for filtering data."""
        if isinstance(args, Sequence):
            assert len(args) == 2, "Only length two sequence allowed for indexing."
            values = self.values
            out = np.ones_like(values, dtype=np.bool_)
            val1 = self._get_value_for_indexing(args[0])
            val2 = self._get_value_for_indexing(args[1])
            if val1 is not None:
                out = out & (values >= val1)
            if val2 is not None:
                out = out & (values <= val2)
            return self.update(values=values[out]), out

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

    def __getitem__(self, item):
        return self.values[item]

    def __hash__(self):
        return hash(id(self))

    @cache
    def __len__(self):
        return len(self.values)

    @property
    @cache
    def min(self):
        """Return min value"""
        return np.min(self.values)

    @property
    @cache
    def max(self):
        """Return max value in range."""
        return np.max(self.values)

    @property
    @cache
    def dtype(self):
        """Returns datatype."""
        return self.values.dtype

    def __eq__(self, other):
        try:
            self_d, other_d = dict(self), dict(other)
        except TypeError:
            return False
        v1, v2 = self_d.pop("values", None), other_d.pop("values", None)
        # Frustratingly, allcose doesn't work with datetime64 so we we need
        # this short-circuiting equality check.
        if self_d == other_d and (np.all(v1 == v2) or np.allclose(v1, v2)):
            return True
        return False


class CoordMonotonicArray(CoordArray):
    """
    A coordinate with strictly increasing or decreasing values.
    """

    values: ArrayLike

    def filter(self, args) -> Tuple[Self, Union[slice, ArrayLike]]:
        """Apply filter, return filtered coords and index for filtering data."""
        if isinstance(args, Sequence):
            assert len(args) == 2, "Only length two sequence allowed for indexing."
            start = self._get_index(args[0])
            new_start = start if start is not None and start > 0 else None
            stop = self._get_index(args[1], forward=False)
            new_stop = stop if stop is not None and stop < len(self) else None
            out = slice(new_start, new_stop)
            return self.update(values=self.values[out]), out

    def _get_index(self, value, forward=True):
        """Get the index corresponding to a value."""
        if (value := self._get_value_for_indexing(value)) is None:
            return value
        side_dict = {True: "left", False: "right"}
        values = self.values
        mod = 0 if forward else -1
        # since search sorted only works on ascending monotonic arrays we
        # can negative descending arrays to get the same effect.
        if values[0] > values[1]:
            values = values * -1
            value = value * -1
        out = np.searchsorted(values, value, side=side_dict[forward])
        return out + mod


def get_coord(
    *,
    values: Optional[ArrayLike] = None,
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
        start, stop, step, monotonic = _maybe_get_start_stop_step(values)
        if start is not None:
            out = CoordRange(start=start, stop=stop, step=step, units=units)
            assert len(out) == len(values), "failed!"
            return out
        elif monotonic:
            return CoordMonotonicArray(values=values, units=units)
        return CoordArray(values=values, units=units)
    else:
        return CoordRange(start=start, stop=stop, step=step, units=units)


class CoordManager(DascoreBaseModel):
    """
    Class for managing coordinates.

    Attributes
    ----------
    coord_map
        A mapping of {coord_name: Coord}
    dim_map
        A mapping of {coord_name: (dimensions, )
    """

    dims: Tuple[str, ...]
    coord_map: Dict[str, BaseCoord]
    dim_map: Dict[str, Tuple[str, ...]]

    def __getitem__(self, item):
        return self.coord_map[item]

    def update(self, **kwargs) -> Self:
        """
        Update the coordinates, return a new Coordinate Manager.

        Input values can be of the same form as initialization.
        To drop coordinates, simply pass {coord_name: None}

        """
        coords = dict(self.coord_map)
        for item, value in kwargs.items():
            if value is None:
                coords.pop(item, None)
            else:
                coords[item] = value
        out = dict(coord_map=coords, dim_map=self.dim_map, dims=self.dims)
        return self.__class__(**out)

    def drop_dim(self, dim: Union[str, Sequence[str]]) -> Self:
        """Drop one or more dimension."""
        coord_name_to_kill = []
        dims_to_kill = set(iterate(dim))
        for name, dims in self.dims.items():
            if dims_to_kill & set(dims):
                coord_name_to_kill.append(name)
        coord_map = {
            i: v for i, v in self.coord_map.items() if i not in coord_name_to_kill
        }
        dim_map = {i: v for i, v in self.dim_map.items() if i not in coord_name_to_kill}
        dims = tuple(x for x in self.dims if x not in dims_to_kill)
        return self.__class__(coord_map=coord_map, dim_map=dim_map, dims=dims)

    def __rich__(self) -> str:
        out = ["[bold] Coordinates [/bold]"]
        for name, coord in self.coord_map.items():
            dims = self.dim_map[name]
            new = f"    {name}: {dims}: {coord}"
            out.append(new)
        return "\n".join(out)


def get_coord_manager(coord_dict, dims) -> CoordManager:
    """
    Create a coordinate manager.
    """

    def _coord_from_simple(name, coord):
        """
        Get coordinates from {coord_name: coord} where coord_name is dim name.
        """
        if name not in dims:
            msg = (
                "Coordinates that are not named the same as dimensions"
                "must be passed as a tuple of the form: "
                "(dimension, coord) "
            )
            raise CoordError(msg)
        assert name in dims
        return get_coord(values=coord), (name,)

    def _coord_from_nested(coord):
        """
        Get coordinates from {coord_name: (dim_name, coord)} or
        {coord_name: ((dim_names...,), coord)}
        """
        if not len(coord) == 2:
            msg = (
                "Second input for coords must be length two of the form:"
                " (dimension, coord) or ((dimensions,...), coord)"
            )
            raise CoordError(msg)
        dim_names = iterate(coord[0])
        # all dims must be in the input dims.
        if not (d1 := set(dim_names)).issubset((d2 := set(dims))):
            bad_dims = d2 - d1
            msg = (
                f"Coordinate specified invalid dimension(s) {bad_dims}."
                f" Valid dimensions are {dims}"
            )
            raise CoordError(msg)
        coord_out = get_coord(values=coord[1])
        return coord_out, dim_names

    # each dimension must have a coordinate
    if not (cset := set(coord_dict)).issuperset((dset := set(dims))):
        missing = cset - dset
        msg = (
            f"All dimensions must have coordinates. The following "
            f"are missing {missing}"
        )
        raise CoordError(msg)

    coord_map, dim_map = {}, {}
    for name, coord in coord_dict.items():
        if isinstance(coord, (BaseCoord, ArrayLike, np.ndarray)):
            coord_map[name], dim_map[name] = _coord_from_simple(name, coord)
        else:
            coord_map[name], dim_map[name] = _coord_from_nested(coord)
    return CoordManager(coord_map=coord_map, dim_map=dim_map, dims=dims)
