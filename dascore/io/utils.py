"""Shared utilities for IO implementations."""

from dascore.core.coords import BaseCoord, CoordSegmented, get_coord
from dascore.exceptions import CoordError


def get_exact_coord(values, units=None) -> BaseCoord:
    """Return an exact coordinate, including for non-monotonic values."""
    try:
        return CoordSegmented.from_array(values, tolerance=0, units=units)
    except CoordError:
        return get_coord(data=values, units=units)
