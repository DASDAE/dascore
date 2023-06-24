"""
Module for handling units.
"""
from functools import cache
from typing import Optional, Tuple, Union

import pint
from pint import Quantity, UndefinedUnitError, Unit  # noqa

from dascore.exceptions import UnitError
from dascore.utils.misc import unbyte


@cache
def get_registry():
    """Get the pint unit registry."""
    ureg = pint.UnitRegistry()
    # a few custom defs, we may need our own unit registry if this
    # gets too long.
    ureg.define("PI=pi")
    return ureg


def get_unit(value):
    """Convert a value to a pint unit."""
    return get_registry().Unit(value)


def get_quantity(value):
    """Convert a value to a pint quantity."""
    if isinstance(value, Unit):
        value = str(value)  # ensure unit is converted to quantity
    ureg = get_registry()
    return ureg.Quantity(value)


def get_unit_and_factor(value: str) -> Tuple[float, str]:
    """Convert a mixed unit/scaling factor to scale_factor and unit str"""
    quant = get_quantity(value)
    return quant.magnitude, str(quant.units)


def get_conversion_factor(from_unit, to_unit) -> float:
    """Get a conversion factor for converting from one unit to another."""
    if from_unit is None or to_unit is None:
        return 1
    from_quant, to_quant = get_quantity(from_unit), get_quantity(to_unit)
    mag1, mag2 = from_quant.magnitude, to_quant.magnitude
    mag_ratio = mag1 / mag2
    unit_ratio = to_quant.units.from_(from_quant.units).magnitude
    return mag_ratio * unit_ratio


def invert_unit(unit: Union[pint.Unit, str]) -> pint.Unit:
    """Invert a unit"""
    out = 1 / pint.Unit(unit)
    return out.units


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
