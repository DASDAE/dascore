"""
Machinery for coordinates.
"""
import abc
from functools import cache
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from typing_extensions import Self

from dascore.exceptions import CoordError
from dascore.utils.models import ArrayLike, DascoreBaseModel, DTypeLike, Unit
from dascore.utils.units import get_conversion_factor


class BaseCoord(DascoreBaseModel, abc.ABC):
    """Coordinate interface."""

    units: Optional[Unit]

    @abc.abstractmethod
    def convert_units(self, unit) -> Self:
        """Convert from one unit to another. Set units if None are set."""

    @abc.abstractmethod
    def select(self, arg) -> Union[slice, ArrayLike]:
        """Returns an entity that can be used in a list for numpy indexing."""

    @abc.abstractmethod
    def __getitem__(self, item) -> Self:
        """Should implement slicing and return new instance."""

    @abc.abstractmethod
    def __len__(self):
        """Total number of elements."""

    @property
    @abc.abstractmethod
    def values(self) -> ArrayLike:
        """Returns (or generates) the array data."""

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


class CoordRange(BaseCoord):
    """A coordinate represent a range of evenly sampled data."""

    start: Any
    stop: Any
    step: Any

    def __getitem__(self, item):
        pass

    def __len__(self):
        return int(abs(((self.stop - self.start) + self.step) / self.step))

    def convert_units(self, units) -> Self:
        """Convert units, or set units if none exist."""
        if self.units is None:
            return self.set_units(units)
        out = dict(self)
        out["units"] = units
        factor = get_conversion_factor(self.units, units)
        for name in ["start", "stop", "step"]:
            out[name] = getattr(self, name) * factor
        return self.__class__(**out)

    def select(self, args) -> Union[slice, ArrayLike]:
        """Return an object for indexing along a dimension."""
        if isinstance(args, Sequence):
            assert len(args) == 2, "Only length two sequence allowed for indexing."
            start = self._get_index(args[0])
            stop = self._get_index(args[1], forward=False)
            return slice(start, stop)

    def _get_index(self, value, forward=True):
        """Get the index corresponding to a value."""
        if pd.isnull(value):
            return None
        func = np.ceil if forward else np.floor
        out = int(func((value - self.start + self.step) / self.step))
        if out < 0 or out > self.stop:
            return None
        return out

    @property
    def values(self) -> ArrayLike:
        """Return the values of the coordinate as an array."""
        return np.arange(self.start, self.stop + self.step, self.step)

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
    """A coordinate with an array"""

    data: ArrayLike


def get_coord(
    *,
    data: Optional[ArrayLike] = None,
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

    def _maybe_get_start_stop_step(data):
        """Get start, stop, step if data is monotonic else return tuple of None"""
        view2 = data[1:]
        view1 = data[:-1]
        is_monotonic = np.all(view1 > view2) or np.all(view2 > view1)
        # the array cannot be evenly sampled if it isn't monotonic
        if is_monotonic:
            diff = view2 - view1
            if len(np.unique(diff)) == 1:
                _min = data[0]
                _max = data[-1]
                _step = diff[0]
                return _min, _max, _step
        return None, None, None

    _check_inputs(data, start, stop, step)
    # data array was passed; see if it is monotonic/evenly sampled
    if data is not None:
        start, stop, step = _maybe_get_start_stop_step(data)
        if start is not None:
            return CoordRange(start=start, stop=stop, step=step, units=units)
        return CoordArray(data=data, units=units)
    else:
        return CoordRange(start=start, stop=stop, step=step, units=units)
