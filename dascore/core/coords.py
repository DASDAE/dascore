"""Machinery for coordinates."""
from __future__ import annotations

import abc
from functools import cache
from operator import gt, lt
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from pydantic import (
    model_serializer,
    model_validator,
)
from rich.text import Text
from typing_extensions import Self

import dascore as dc
from dascore.compat import array
from dascore.constants import dascore_styles
from dascore.exceptions import CoordError, ParameterError
from dascore.units import (
    Quantity,
    Unit,
    convert_units,
    get_factor_and_unit,
    get_quantity,
    get_quantity_str,
)
from dascore.utils.display import get_nice_text
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import (
    all_close,
    all_diffs_close_enough,
    cached_method,
    iterate,
    sanitize_range_param,
)
from dascore.utils.models import (
    ArrayLike,
    DascoreBaseModel,
    DTypeLike,
    UnitQuantity,
)
from dascore.utils.time import dtype_time_like, is_datetime64, is_timedelta64, to_float

# Valid values for min/max
min_max_type = TypeVar("min_max_type")

step_type = TypeVar("step_type")


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
        value = float(value) if value is not None else np.NaN
    elif np.issubdtype(dtype, np.integer):
        value = int(value)
    return value


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

    dtype: str
    min: min_max_type
    max: min_max_type
    step: step_type | None = None
    units: UnitQuantity | None = None

    @model_serializer(when_used="json")
    def ser_model(self) -> dict[str, str]:
        """Serialize the model to json."""
        return {i: str(v) for i, v in self.model_dump().items()}

    @model_validator(mode="before")
    @classmethod
    def get_correct_dtype_cast_values(cls, data: Any) -> Any:
        """Ensure the correct dtype is provided and value conform to it."""
        if isinstance(data, dict):
            min_val = data["min"]
            dtype = _get_dtype(min_val, data.get("dtype"))
            data["dtype"] = str(dtype).split("[")[0]
            for name in ["min", "max", "step"]:
                val = data.get(name)
                data[name] = ensure_consistent_dtype(val, name, dtype)
        return data

    def to_coord(self) -> CoordRange:
        """Convert to coord range, if possible."""
        if pd.isnull(self.step):
            msg = "Cannot convert summary which is not evenly sampled to coord."
            raise CoordError(msg)
        step = self.step
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
        # uncomment these if validators that arent numpy types are needed.
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

    _rich_style = dascore_styles["default_coord"]
    _evenly_sampled = False
    _sorted = False
    _reverse_sorted = False

    @model_validator(mode="before")
    @classmethod
    def check_time_units(cls, data: Any) -> Any:
        """Ensure time units are s if dattype is time-like."""
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

    @abc.abstractmethod
    def convert_units(self, unit) -> Self:
        """Convert from one unit to another. Set units if None are set."""

    @abc.abstractmethod
    def select(self, arg, relative=False) -> tuple[Self, slice | ArrayLike]:
        """
        Returns an entity that can be used in a list for numpy indexing
        and selected coord.
        """

    @abc.abstractmethod
    def __getitem__(self, item) -> Self:
        """Should implement slicing and return new instance."""

    @cached_method
    def __len__(self):
        """Total number of elements."""
        return np.prod(self.shape)

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

    def __array__(self):
        """Numpy method for getting array data with `np.array(coord)`."""
        return self.data

    @cached_method
    def min(self):
        """Return min value."""
        return self._min()

    @cached_method
    def max(self):
        """Return max value."""
        return self._max()

    @property
    def unit_str(self) -> str:
        """Return a unit string."""
        return get_quantity_str(self.units)

    @property
    def degenerate(self):
        """Returns true if coord is degenerate (empty)."""
        return not bool(len(self))

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
    @abc.abstractmethod
    def dtype(self) -> DTypeLike:
        """Returns a numpy datatype."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the coordinate data."""
        return self.data.shape

    @property
    def evenly_sampled(self) -> tuple[int, ...]:
        """Returns True if the coord is evenly sampled."""
        return self._evenly_sampled

    @property
    def sorted(self) -> tuple[int, ...]:
        """Returns True if the coord in sorted."""
        return self._sorted

    @property
    def reverse_sorted(self) -> tuple[int, ...]:
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
    def sort(self, reverse=False) -> tuple[BaseCoord, slice | ArrayLike]:
        """Sort the contents of the coord. Return new coord and slice for sorting."""

    def snap(self) -> CoordRange:
        """
        Snap the coordinates to evenly sampled grid points.

        This will cause some loss of precision but often makes the data much
        easier to work with.
        """
        return self

    @abc.abstractmethod
    def update_limits(self, min=None, max=None, step=None, **kwargs) -> Self:
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
    ) -> Self:
        """
        Update the data of the coordinate.

        Parameters
        ----------
        data
            A new array to use.
        values
            Same as data, but deprecated. Here for compatibility reasons.
        """
        if data is None and values is None:
            return self
        data = values if data is None else data
        units = kwargs.get("units")
        return get_coord(data=data, units=units)

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
            value = convert_units(value.magnitude, self.units, value.units)
        # if null or ... just return None
        if pd.isnull(value) or value is Ellipsis:
            return None
        # special case for datetime and relative
        if relative:
            if np.issubdtype(self.dtype, np.datetime64):
                value = dc.to_timedelta64(value)
            value = self._get_relative_values(value)
        # apply validators. These can, eg, coerce to correct dtype.
        out = get_compatible_values(value, self.dtype)
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
        select: slice | None | type(Ellipsis) | tuple[Any, Any],
        relative=False,
    ) -> tuple[Any, Any]:
        """
        Get a tuple with (start, stop) and perform basic checks.

        Parameters
        ----------
        select
            An object for determining select range.
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
        return get_coord(data=data)

    def index(self, indexer, axis: int | None = None) -> Self:
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

    def get_attrs_dict(self, name):
        """Get attrs dict."""
        out = {f"{name}_min": self.min(), f"{name}_max": self.max()}
        if self.step:
            out[f"{name}_step"] = self.step
        if self.units:
            out[f"{name}_units"] = self.units
        return out

    def to_summary(self, dims=()) -> CoordSummary:
        """Get the summary info about the coord."""
        return CoordSummary(
            min=self.min(),
            max=self.max(),
            step=self.step,
            dtype=self.dtype,
            units=self.units,
        )

    def update(self, **kwargs):
        """Update parts of the coordinate."""
        info = self.model_dump()
        update_fields = {
            i: v for i, v in kwargs.items() if not all_close(v, info.get(i))
        }
        units = update_fields.pop("units", None)
        _ = update_fields.pop("dtype", None)
        out = self.update_limits(**update_fields).update_data(**update_fields)
        if units is not None:
            out = out.convert_units(units)
        return out

    def get_sample_count(self, value, samples=False) -> int:
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
        """
        if not self.evenly_sampled:
            msg = "Coordinate is not evenly sampled, cant get sample count."
            raise CoordError(msg)
        if samples:
            if not isinstance(value, int | np.integer):
                msg = "When samples==True values must be integers."
                raise ParameterError(msg)
            samples = int(value)
        else:
            compat_val = self._get_compatible_value(value, relative=True)
            duration = compat_val - self.min()
            samples = int(np.ceil(duration / self.step))
        if samples > len(self):
            msg = (
                f"value of {value} results in a window larger than coordinate "
                f"length of {len(self)}"
            )
            raise ParameterError(msg)
        return min(samples, len(self))

    def get_next_index(self, value, samples=False, allow_out_of_bounds=False) -> int:
        """
        Get the index a value would have in a coordinate.

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

        Examples
        --------
        >>> from dascore.core import get_coord
        >>> coord = get_coord(start=0, stop=10, step=1)
        >>> # Find the index for a value contained by the coordinate.
        >>> assert coord.get_next_index(1) == 1
        >>> # The next (not closest) index is return for value not in coord.
        >>> assert coord.get_next_index(2.000001) == 3
        """
        # handle samples
        if samples:
            min_val, max_val = 0, len(self) - 1
            value = int(np.round(value))
            # account for negative indexing
            value = value if value >= 0 else value + max_val + 1
        else:
            value = self._get_compatible_value(value)
            min_val, max_val = self.min(), self.max()
        # handle out of bounds cases
        if (is_gt := value > max_val) or (value < min_val):
            if not allow_out_of_bounds:
                msg = f"Value: {value} is out of bounds for {self}"
                raise ValueError(msg)
            return max_val if is_gt else min_val
        # samples should already have the answer, just return
        if samples:
            return value
        # otherwise get forward and backward inds
        for_index = self._get_index(value, forward=True)
        back_index = self._get_index(value, forward=False)
        ranges = [x for x in [for_index, back_index] if x is not None]
        assert len(ranges)
        return ranges[0]


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

    @model_validator(mode="before")
    @classmethod
    def ensure_all_attrs_set(cls, values):
        """If any info is neglected the coord is invalid."""
        for name in ["start", "stop", "step"]:
            assert values[name] is not None
        # step should have the same sign as stop-start, see #321.
        diff = values["stop"] - values["start"]
        # note: we need to the to_float since np.sign(datetime64) returns a
        # datetime64 which includes precision, so even if the sign is the same
        # if the precision is different this validation fails.
        if not np.sign(to_float(values["step"])) == np.sign(to_float(diff)):
            msg = "Sign of step must match sign of stop - start"
            raise CoordError(msg)
        return values

    @model_validator(mode="before")
    @classmethod
    def _set_stop(cls, values):
        """Set stop to integral value >= current stop."""
        start, stop = values.get("start"), values.get("stop")
        step = values.get("step")
        assert all(x is not None for x in [start, stop, step])
        dur = stop - start
        if step == 0:
            return values
        int_val = int(np.ceil(np.round(dur / step, 1)))
        values["stop"] = start + step * int_val
        return values

    def __getitem__(self, item):
        if isinstance(item, int):
            if item >= len(self):
                raise IndexError(f"{item} exceeds coord length of {self}")
            return self.values[item]
        # handle ... as None
        if isinstance(item, slice):
            start = None if item.start is ... else item.start
            end = None if item.stop is ... else item.stop
            # Todo we can probably add more intelligent logic for slices.
            item = slice(start, end, item.step)
        out = self.values[item]
        return get_coord(data=out, units=self.units)

    @cached_method
    def __len__(self):
        if self.start == self.stop:
            return 1
        out = abs((self.stop - self.start) / self.step)
        # due to floating point weirdness this can sometimes be very close
        # but not exactly an int, so we need to round.
        return int(np.round(out))

    def convert_units(self, units) -> Self:
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

    def select(self, args, relative=False) -> tuple[BaseCoord, slice | ArrayLike]:
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

    def sort(self, reverse=False) -> tuple[BaseCoord, slice | ArrayLike]:
        """Sort the contents of the coord. Return new coord and slice for sorting."""
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

    @compose_docstring(doc=BaseCoord.update_limits.__doc__)
    def update_limits(self, min=None, max=None, step=None, **kwargs) -> Self:
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
        """Return min value."""
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
    @cached_method
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
        values = convert_units(self.values, units, self.units)
        return self.new(units=units, values=values)

    def select(self, args, relative=False) -> tuple[Self, slice | ArrayLike]:
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
            assert step > 0
        if self.reverse_sorted:
            step = -step
            start, stop = max_v, min_v + step
        else:
            start, stop = min_v, max_v + step
        # get potential output, ensure it is the same length as original
        out = CoordRange(start=start, stop=stop, step=step, units=self.units)
        # A hack to deal with those pesky off-by-one errors.
        if out.shape != self.shape:
            diff = len(self) - len(out)
            new_stop = stop + diff * step  # increase or decrease coord length
            out = out.update(stop=new_stop)
        assert len(out) == len(self)
        return out

    @compose_docstring(doc=BaseCoord.update_limits.__doc__)
    def update_limits(self, min=None, max=None, step=None, **kwargs) -> Self:
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
    @cached_method
    def dtype(self):
        """Returns datatype."""
        return self.values.dtype

    @property
    @cached_method
    def shape(self):
        """Return the shape of the coordinate."""
        return np.shape(self.values)

    # def __eq__(self, other):
    #     # Frustratingly, all cose doesn't work with datetime64 so we we need
    #     # this short-circuiting equality check.
    #     if values_same and self_d == other_d and values_same:


class CoordMonotonicArray(CoordArray):
    """A coordinate with strictly increasing or decreasing values."""

    values: ArrayLike
    _rich_style = dascore_styles["coord_monotonic"]
    _sorted = True

    def select(self, args, relative=False) -> tuple[Self, slice | ArrayLike]:
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


class CoordDegenerate(CoordArray):
    """A coordinate with degenerate (empty on one axis) data."""

    values: ArrayLike
    step: Any = None
    _rich_style = dascore_styles["coord_degenerate"]

    def select(self, args, relative=False) -> tuple[Self, slice | ArrayLike]:
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
    data: ArrayLike | None | np.ndarray = None,
    values: ArrayLike | None | np.ndarray = None,
    start=None,
    min=None,
    stop=None,
    max=None,
    step=None,
    units: None | Unit | Quantity | str = None,
    dtype: str | np.dtype = None,
) -> BaseCoord:
    """
    Given multiple types of input, return a coordinate.

    This function automatically figures out which kind of Coordinate
    should be returned for a given type of input.

    Parameters
    ----------
    data
        An array indicating the values.
    values
        Deprecated, use data instead.
    start
        The start value of the array, inclusive.
    min
        The minimum value, same as start.
    stop
        The stopping value of an array, exclusive.
    step
        The sampling spacing of an array.
    units
        Indication of units.
    dtype
        Only used for compatibility with kwargs produced by other
        functions. Doesn't do anything as dtype is inferred from other
        arguments.

    Notes
    -----
    The following combinations of input parameters are typical:
        (start, stop, step)
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

    def _maybe_get_start_stop_step(data):
        """Get start, stop, step, is_monotonic."""
        data = np.array(data)
        view2 = data[1:]
        view1 = data[:-1]
        is_monotonic = np.all(view1 > view2) or np.all(view2 > view1)
        # the array cannot be evenly sampled if it isn't monotonic
        if is_monotonic:
            diffs = view2 - view1
            unique_diff = np.unique(diffs)
            if len(unique_diff) == 1 or all_diffs_close_enough(unique_diff):
                _min = data[0]
                # this is a poor man's median that preserves dtype
                sorted_diffs = np.sort(diffs)
                _step = sorted_diffs[len(sorted_diffs) // 2]
                _max = _get_new_max(data, _min, _step)
                return _min, _max + _step, _step, is_monotonic
        return None, None, None, is_monotonic

    # ensure data and values are not used
    if data is not None and values is not None:
        msg = "Cannot specify both data and values. Use only data."
        raise CoordError(msg)
    elif values is not None:
        data = values
    # maybe convert min/max to start stop.
    if start is None and min is not None:
        start = min
    if stop is None and max is not None:
        stop = max
    _check_data_compatibility(data, start, stop, step)
    # data array was passed; see if it is monotonic/evenly sampled
    if data is not None:
        if not isinstance(data, np.ndarray | BaseCoord):
            data = np.array(data)
        # values = np.array(values)  # ensure we have a numpy array
        if isinstance(data, BaseCoord):  # just return coordinate
            return data
        if np.size(data) == 0:
            return CoordDegenerate(values=data, units=units, step=step)
        # special case of len 1 array either get range, if step specified
        # or sorted monotonic array if not.
        elif len(data) == 1:
            if not pd.isnull(step):
                val = data[0]
                return CoordRange(start=val, stop=val + step, step=step, units=units)
            return CoordMonotonicArray(values=data, units=units)
        start, stop, step, monotonic = _maybe_get_start_stop_step(data)
        if start is not None:
            out = CoordRange(start=start, stop=stop, step=step, units=units)
            return out
        elif monotonic:
            return CoordMonotonicArray(values=data, units=units)
        return CoordArray(values=data, units=units)
    else:
        return CoordRange(start=start, stop=stop, step=step, units=units)
