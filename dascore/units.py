"""
Module for handling units.
"""
from functools import cache
from typing import Optional, Tuple, TypeVar, Union

import pandas as pd
import pint
from pint import DimensionalityError, Quantity, UndefinedUnitError, Unit  # noqa

from dascore.exceptions import UnitError
from dascore.utils.misc import unbyte

str_or_none = TypeVar("str_or_none", None, str)


@cache
def get_registry():
    """Get the pint unit registry."""
    ureg = pint.UnitRegistry(cache_folder=":auto:")
    # a few custom defs, we may need our own unit registry if this
    # gets too long.
    ureg.define("PI=pi")
    pint.set_application_registry(ureg)
    return ureg


@cache
def get_unit(value) -> Unit:
    """Convert a value to a pint unit."""
    return get_registry().Unit(value)


@cache
def _str_to_quant(qunat_str):
    """Get quantity from a string; cache output."""
    if isinstance(qunat_str, Unit):
        qunat_str = str(qunat_str)  # ensure unit is converted to quantity
    ureg = get_registry()
    return ureg.Quantity(qunat_str)


def get_quantity(value: str_or_none) -> Optional[Quantity]:
    """Convert a value to a pint quantity."""
    if value is None or value is ...:
        return None
    if isinstance(value, Quantity):
        return value
    return _str_to_quant(value)


def get_factor_and_unit(
    value: str_or_none, simplify: bool = False
) -> Tuple[float, str_or_none]:
    """Convert a mixed unit/scaling factor to scale_factor and unit str"""
    quant = get_quantity(value)
    if quant is None:
        return 1.0, None
    if simplify:
        quant = quant.to_base_units()
    return quant.magnitude, str(quant.units)


@cache
def get_conversion_factor(from_quant, to_quant) -> float:
    """Get a conversion factor for converting from one unit to another."""
    from_quant, to_quant = get_quantity(from_quant), get_quantity(to_quant)
    if from_quant is None or to_quant is None:
        return 1
    mag1, mag2 = from_quant.magnitude, to_quant.magnitude
    mag_ratio = mag1 / mag2
    try:
        unit_ratio = to_quant.units.from_(from_quant.units).magnitude
    except DimensionalityError as e:
        raise UnitError(str(e))

    return mag_ratio * unit_ratio


def invert_quantity(unit: Union[pint.Unit, str]) -> pint.Unit:
    """Invert a unit"""
    if pd.isnull(unit):
        return None
    quant = get_quantity(unit)
    return 1 / quant


def validate_quantity(quant_str) -> Optional[str]:
    """
    Ensure a unit string is valid and return it.

    If it is not valid raise a [UnitError](`dascore.execptions.UnitError`).

    Parameters
    ----------
    quant_str
        A string input specifying a quantity (unit + scaling factors).
    """
    quant_str = unbyte(quant_str)
    if quant_str is None or quant_str == "":
        return None
    try:
        get_quantity(quant_str)
    except UndefinedUnitError:
        msg = f"DASCore failed to parse the following unit/quantity: {quant_str}"
        raise UnitError(msg)
    return str(quant_str)


def get_filter_units(
    arg1: Union[Quantity, float], arg2: Union[Quantity, float], to_unit: str
) -> Tuple[float, float]:
    """
    Get a tuple for applying filter based on dimension coordinates.

    Parameters
    ----------
    arg1
        The lower bound of the filter params
    arg2
        The upper bound of the filter params.
    data_unit
        The units of the axis which will be filtered.

    Examples
    --------
    >>> from dascore.units import get_filter_units, Hz, s
    >>> # Passing a tuple in Hz leaves the output in Hz
    >>> assert get_filter_units(1 * Hz, 10 * Hz, s) == (1., 10.)
    >>> assert get_filter_units(None, 10 * Hz, s) == (None, 10.)
    >>> assert get_filter_units(1 * Hz, 10 * Hz, s) == (1., 10.)
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

    def get_inverted_quant(quant, data_units):
        """Convert to inverted units."""
        if quant is None:
            return quant, True
        if quant.units == get_unit("dimensionless"):
            msg = (
                "Both inputs must be quantities to get filter parameters. "
                f"You passed ({arg1}, {arg2})"
            )
            raise UnitError(msg)
        data_units = get_unit(data_units)
        inverted_units = (1 / data_units).units
        inversed_units = True
        if data_units.dimensionality == quant.units.dimensionality:
            quant, inversed_units = 1 / quant, False
        mag = get_conversion_factor(quant, inverted_units)
        return mag, inversed_units

    # fast-path for non-unit, non-quantity inputs.
    unitable = (Quantity, Unit)
    arg1 = None if arg1 is ... else arg1
    arg2 = None if arg2 is ... else arg2
    if not (isinstance(arg1, unitable) or isinstance(arg2, unitable)):
        return arg1, arg2
    to_units = get_quantity(to_unit).units
    quant1, quant2 = get_quantity(arg1), get_quantity(arg2)
    _ensure_same_units(quant1, quant2)
    out1, inverted1 = get_inverted_quant(quant1, to_units)
    out2, inverted2 = get_inverted_quant(quant2, to_units)
    # if inverted units weren't passed 1 and 2 must be swapped
    if not (inverted1 or inverted2):
        out1, out2 = out2, out1
    return out1, out2


def __getattr__(name):
    """
    This is a bit of magic; it allows arbitrary units to be imported from
    this module. For example:

    from dascore.units import m

    is the same as
    from dascore.units import get_unit
    m = get_unit("m")
    """
    return get_unit(name)
