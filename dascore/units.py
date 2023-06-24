"""
Module for handling units.
"""
from functools import cache
from typing import Optional, Tuple, TypeVar, Union

import pint
from pint import Quantity, UndefinedUnitError, Unit  # noqa

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
def get_quantity(value: str_or_none) -> Optional[Quantity]:
    """Convert a value to a pint quantity."""
    if value is None:
        return value
    if isinstance(value, Unit):
        value = str(value)  # ensure unit is converted to quantity
    ureg = get_registry()
    return ureg.Quantity(value)


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


def get_filter_units(
    arg1: Union[Quantity, float], arg2: Union[Quantity, float], to_unit: str
) -> Tuple[float, float]:
    """
    Get a tuple for applying filter based on dimension coordinates.

    Parameters
    ----------
    input_value
        An array of
    data_unit

    Returns
    -------
    """

    def _check_args(arg1_compat, arg2_compat):
        """Ensure the arguments ar ok."""
        if not arg1_compat and arg2_compat:
            msg = (
                "Both inputs must be quantities to get filter parameters. "
                f"You passed ({arg1}, {arg2})"
            )
            raise UnitError(msg)

    def get_inverted_quant(quant, data_units):
        """Convert to inverted units."""
        quant = get_quantity(quant)
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
        fact, unit = get_factor_and_unit(quant, inverted_units)
        return quant.magnitude * fact, inversed_units

    # fast-path for non-unit, non-quantity inputs.
    unitable = (Quantity, Unit)
    if not (isinstance(arg1, unitable) and isinstance(arg2, unitable)):
        return arg1, arg2
    to_units = get_unit(to_unit)
    out1, inverted = get_inverted_quant(arg1, to_units)
    out2, inverted = get_inverted_quant(arg2, to_units)
    return out2, out1, inverted
    # _check_args(arg1, arg2)
    #
    # out_unit = get_quantity(to_unit)


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
