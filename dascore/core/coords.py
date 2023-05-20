"""
Machinery for coordinates.
"""
import abc
from functools import cache
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import root_validator
from typing_extensions import Self

from dascore.exceptions import CoordError
from dascore.utils.models import ArrayLike, DascoreBaseModel, DTypeLike, Unit
from dascore.utils.time import is_datetime64, to_datetime64
from dascore.utils.units import get_conversion_factor


class BaseCoord(DascoreBaseModel, abc.ABC):
    """Coordinate interface."""

    units: Optional[Unit]

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

    def _get_value_for_indexing(self, value):
        """Function to get a value for indexing."""
        # strip units and v
        if hasattr(value, "units"):
            uf = get_conversion_factor(value.units, self.units)
            value = value.magnitude * uf
        # if null or ... just return None
        if pd.isnull(value) or value is Ellipsis:
            return None
        return self._cast_values(value)

    def sort(self) -> Tuple["BaseCoord", Union[slice, ArrayLike]]:
        """
        Sort the contents of the coord. Return new coord and slice for sorting.
        """
        return self, slice(None)


class CoordRange(BaseCoord):
    """A coordinate represent a range of evenly sampled data."""

    start: Any
    stop: Any
    step: Any

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
        return self.stop

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
        if self_d == other_d and np.allclose(v1, v2):
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
            if len(unique_diff) == 1 or _all_diffs_close(unique_diff):
                _min = data[0]
                _max = data[-1]
                _step = unique_diff[0]
                return _min, _max + _step, _step, is_monotonic
        return None, None, None, is_monotonic

    _check_inputs(values, start, stop, step)
    # data array was passed; see if it is monotonic/evenly sampled
    if values is not None:
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
