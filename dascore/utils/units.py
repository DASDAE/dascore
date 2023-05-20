"""
Module for handling units.
"""
from functools import cache
from typing import Union

import pint


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
