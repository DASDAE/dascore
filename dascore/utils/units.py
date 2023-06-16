"""
Module for handling units.
"""
from functools import cache
from typing import Union

import pint
from pint import UndefinedUnitError

from dascore.exceptions import UnitError


@cache
def get_registry():
    """Get the pint unit registery."""
    return pint.UnitRegistry()


def unit(value):
    """Convert a value to a unit."""
    return get_registry()(value)


def get_conversion_factor(from_unit, to_unit) -> float:
    """Get a conversion factor for converting from one unit to another."""
    if from_unit is None or to_unit is None:
        return 1
    unit1, unit2 = pint.Unit(from_unit), pint.Unit(to_unit)
    return unit2.from_(unit1).magnitude


def invert_unit(unit: Union[pint.Unit, str]) -> pint.Unit:
    """Invert a unit"""
    out = 1 / pint.Unit(unit)
    return out.units


def validate_units(unit_str) -> str:
    """
    Ensure a unit string is valid and return it.

    If it is not valid raise a [UnitError](`dascore.execptions.UnitError`).

    Parameters
    ----------
    unit_str
        A string input specifying units.
    """
    try:
        unit(unit_str)
    except UndefinedUnitError:
        msg = f"{unit_str} is not a unit supported by DASCore"
        raise UnitError(msg)
    return unit_str
