"""Module for handling units."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import TypeVar

import numpy as np
import pandas as pd
import pint
from pint import DimensionalityError, Quantity, UndefinedUnitError, Unit

import dascore as dc
from dascore.compat import is_array
from dascore.exceptions import UnitError
from dascore.utils.misc import unbyte
from dascore.utils.time import dtype_time_like, is_datetime64, is_timedelta64, to_float

str_or_none = TypeVar("str_or_none", None, str)
numeric = TypeVar("numeric", np.ndarray, int, float)


@cache
def get_registry():
    """Get the pint unit registry."""
    ureg = pint.UnitRegistry(cache_folder=":auto:")
    # a few custom defs, we may need our own unit registry if this
    # gets too long.
    ureg.define("PI=pi")
    ureg.define("RADIANS=radians")
    ureg.define("Radians=radians")
    ureg.define("Radian=radians")
    # define strain
    ureg.define("strain=[]=ϵ")
    # allow multiplication with offset units.
    ureg.autoconvert_offset_to_baseunit = True
    # set the shortest display for units.
    # .formatter was added in new versions of pint; this makes it work with both
    formatter = getattr(ureg, "formatter", ureg)
    formatter.default_format = "~"
    pint.set_application_registry(ureg)
    return ureg


@cache
def get_unit(value) -> Unit:
    """
    Convert a value to a pint unit.

    Usually quantities, generated with
    [`get_quantity`](`dascore.units.get_quantity`), are easy to work
    with.

    Examples
    --------
    >>> import dascore as dc
    >>>
    >>> # Create unit from string
    >>> unit = dc.get_unit('m/s')
    >>> assert str(unit) == 'm / s'
    >>>
    >>> # Create unit from existing quantity
    >>> quantity = dc.get_quantity('10 Hz')
    >>> unit = dc.get_unit(quantity.units)
    >>> assert str(unit) == 'Hz'
    """
    if isinstance(value, Quantity):
        assert value.magnitude == 1.0
        value = value.units
    return get_registry().Unit(value)


@cache
def _str_to_quant(qunat_str):
    """Get quantity from a string; cache output."""
    if isinstance(qunat_str, Unit):
        qunat_str = str(qunat_str)  # ensure unit is converted to quantity
    ureg = get_registry()
    return ureg.Quantity(qunat_str)


def get_quantity(value: str_or_none) -> Quantity | None:
    """
    Convert a value to a pint quantity.

    Parameters
    ----------
    value
        The value to convert to a quantity.

    Examples
    --------
    >>> import dascore as dc
    >>> meters = dc.get_quantity("m")
    >>> accel = dc.get_quantity("m/s^2")
    >>>
    >>> # This can also convert date times.
    >>> many_seconds = dc.get_quantity(dc.to_timedelta64(200))
    """
    value = unbyte(value)
    if value is None or value is ... or value == "":
        return None
    if isinstance(value, Quantity):
        return value
    elif is_datetime64(value) | is_timedelta64(value):
        return to_float(value) * dc.get_unit("s")
    return _str_to_quant(value)


def get_factor_and_unit(
    value: str_or_none, simplify: bool = False
) -> tuple[float, str_or_none]:
    """Convert a mixed unit/scaling factor to scale_factor and unit str."""
    quant = get_quantity(value)
    if quant is None:
        return 1.0, None
    if simplify:
        quant = quant.to_base_units()
    return quant.magnitude, get_quantity_str(quant.units)


@cache
def _get_conversion_factors(from_quant, to_quant) -> tuple[float, float, float]:
    """Get multiplicative and additive conversion factors."""
    add_mag = (0 * from_quant).to(0 * to_quant).magnitude
    # need to convert from and to units to deltas for proper conversion.
    from_delta = (1 * from_quant.units) - (from_quant.units * 0)
    to_delta = (1 * to_quant.units) - (to_quant.units * 0)
    mult_mag1 = from_delta.to(to_delta).magnitude
    return mult_mag1 * from_quant.magnitude, add_mag, 1 / to_quant.magnitude


def convert_units(
    data: numeric,
    to_units: None | str | Quantity,
    from_units: None | str | Quantity = None,
) -> numeric:
    """
    Convert units in array from one type of units to another.

    Parameters
    ----------
    data
        The data to convert.
    to_units
        The desired units after the conversion
    from_units
        The current units of the data. If None, simply set the units.

    Raises
    ------
    [UnitError](`dascore.exceptions.UnitError`) if conversion is not possible
    or if the datatype is not compatible (e.g., datetime must always be
    [time])
    """
    if isinstance(data, Quantity):  # an existing quantity
        from_units, data = data.units, data.magnitude
    to_units, from_units = get_quantity(to_units), get_quantity(from_units)
    if from_units is None:
        return data
    elif to_units is None:
        msg = "Cannot convert units to_units are not specified"
        raise UnitError(msg)
    try:
        mult1, add, mult2 = _get_conversion_factors(from_units, to_units)
    except DimensionalityError as e:
        raise UnitError(str(e))
    return (data * mult1 + add) * mult2


def assert_dtype_compatible_with_units(dtype, quantity) -> Quantity:
    """
    Return quantity if it is compatible with dtype.

    If not raise [UnitError](`dascore.exceptions.UnitError`).
    """
    if not dtype_time_like(dtype):
        return get_quantity(quantity)
    if (quant := get_quantity(quantity)) != get_quantity("s"):
        msg = (
            "For arrays with dtypes of datetime64 and timedelta64 the "
            "only allowable units are s."
        )
        raise UnitError(msg)
    return quant


def invert_quantity(unit: pint.Unit | str) -> pint.Unit | None:
    """Invert a unit."""
    # just get magnitude for isnull test to avoid warning of casting
    # quantity to array.
    unit_test = unit.magnitude if hasattr(unit, "magnitude") else unit
    if pd.isnull(unit_test):
        return None
    quant = get_quantity(unit)
    return 1 / quant


def get_quantity_str(quant_value: str | Quantity | None) -> str | None:
    """
    Ensure a unit/quantity is valid and return its string representation.

    If it is not valid raise a [UnitError](`dascore.exceptions.UnitError`).

    Parameters
    ----------
    quant_value
        A input specifying a quantity.
    """
    quant_value = unbyte(quant_value)
    if quant_value is None or quant_value == "":
        return None
    try:
        quant = get_quantity(quant_value)
    except UndefinedUnitError:
        msg = f"DASCore failed to parse the following unit/quantity: {quant_value}"
        raise UnitError(msg)
    if isinstance(quant_value, Quantity):
        if quant.magnitude == 1.0:
            quant_value = str(quant.units)
        else:
            quant_value = str(quant)
    return str(quant_value)


def get_inverted_quant(quant, data_units):
    """Convert to inverted units."""
    if quant is None:
        return quant, True
    if quant.units == get_unit("dimensionless"):
        msg = (
            "Both inputs must be quantities to get filter parameters. "
            f"You passed ({quant}, {data_units})"
        )
        raise UnitError(msg)
    data_units = get_unit(data_units)
    inverted_units = (1 / data_units).units
    units_inversed = True
    if data_units.dimensionality == quant.units.dimensionality:
        quant, units_inversed = 1 / quant, False
    # try to get invert units, otherwise raise.
    try:
        mag = quant.to(inverted_units).magnitude
    except DimensionalityError as e:
        raise UnitError(str(e))
    return mag, units_inversed


def get_filter_units(
    arg1: Quantity | float,
    arg2: Quantity | float,
    to_unit: str | Quantity,
    dim: None | str = None,
) -> tuple[float, float]:
    """
    Get a tuple for applying filter based on dimension coordinates.

    Parameters
    ----------
    arg1
        The lower bound of the filter params
    arg2
        The upper bound of the filter params.
    to_unit
        The units to which the filter should be applied. The returned
        units will be 1/to_units.
    dim
        The dimension name the opration is applied on. Only used for
        raising a more helpful error message.

    Examples
    --------
    >>> from dascore.units import get_filter_units, Hz, s
    >>>
    >>> # Passing a tuple in Hz leaves the output in Hz
    >>> assert get_filter_units(1 * Hz, 10 * Hz, s) == (1., 10.)
    >>> assert get_filter_units(None, 10 * Hz, s) == (None, 10.)
    >>> assert get_filter_units(1 * Hz, 10 * Hz, s) == (1., 10.)
    >>>
    >>> # Passing a tuple in seconds will convert to Hz and switch order, if needed.
    >>> assert get_filter_units(1 * s, 10 * s, s) == (0.1, 1.)
    >>> assert get_filter_units(None, 10 * s, s) == (None, 0.1)
    >>> assert get_filter_units(10 * s, None, s) == (0.1, None)
    """

    def _ensure_same_units(quant1, quant2):
        """Ensure the arguments ar ok."""
        not_none = quant1 is not None and quant2 is not None
        if not_none and quant1.units != quant2.units:
            msg = f"Units must match, {quant1} and {quant2} were provided."
            raise UnitError(msg)

    def _check_to_units(to_unit, dim):
        """Ensure to units are valid."""
        if to_unit is None:
            dim_str = "" if dim is None else dim
            msg = (
                f"Cannot use units on dimension {dim_str} because it has " f"no units."
            )
            raise UnitError(msg)

    # fast-path for non-unit, non-quantity inputs.
    unitable = (Quantity, Unit)
    arg1 = None if arg1 is ... else arg1
    arg2 = None if arg2 is ... else arg2
    if not (isinstance(arg1, unitable) or isinstance(arg2, unitable)):
        return arg1, arg2
    # if we are here it means at least on unit is specified. Check to_unit.
    _check_to_units(to_unit, dim)
    # get inverse of desired output units and ensure units are pure.
    to_quant = get_quantity(to_unit)
    assert to_quant.magnitude == 1.0
    to_units = get_quantity(to_unit).units
    quant1, quant2 = get_quantity(arg1), get_quantity(arg2)
    _ensure_same_units(quant1, quant2)
    out1, inverted1 = get_inverted_quant(quant1, to_units)
    out2, inverted2 = get_inverted_quant(quant2, to_units)
    # if inverted units weren't passed 1 and 2 must be swapped
    if not (inverted1 or inverted2):
        out1, out2 = out2, out1
    return out1, out2


def quant_sequence_to_quant_array(sequence: Sequence[Quantity]) -> Quantity:
    """
    Convert a sequence of Quantities (eg list) to a Quantity array.

    Will simplify all quantities. Raises an error if not all elements have
    the same units.

    Parameters
    ----------
    sequence
        A sequence of Quantities.

    Notes
    -----
    This is probably not efficient for large lists.
    """
    if is_array(sequence):
        # This is a numpy array, just return multiplied by quantity.
        return sequence * get_quantity("dimensionless")
    # iterate the sequence and manually convert to base units.
    try:
        base_unit_sequence = [x.to_base_units() for x in sequence]
    except AttributeError:
        msg = "Not all values in sequence are quantities."
        raise UnitError(msg)
    if not len(base_unit_sequence):
        return np.array([]) * get_quantity("dimensionless")
    units = {x.units for x in base_unit_sequence}
    if len(units) != 1:
        msg = "Not all values in sequence have compatible units."
        raise UnitError(msg)
    array = np.array([x.magnitude for x in base_unit_sequence])
    return array * next(iter(units))


def __getattr__(name):
    """
    Allows arbitrary units (quantities) to be imported from this module.

    For example:
    from dascore.units import m

    is the same as
    from dascore.units import get_quantity
    m = get_quantity("m")
    """
    return get_quantity(name)
