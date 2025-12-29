"""
Compatibility module for DASCore.

All components/functions that may be exchanged for other numpy/scipy
compatible libraries should go in this model.
"""

from __future__ import annotations

from contextlib import suppress

import numpy as np
from numpy import floor, interp, ndarray  # NOQA
from numpy.random import RandomState
from rich.progress import Progress  # NOQA
from scipy.interpolate import interp1d  # NOQA
from scipy.ndimage import zoom  # NOQA
from scipy.signal import decimate, resample, resample_poly  # NOQA

random_state = RandomState(42)


class DataArray:
    """A dummy class for when xarray isn't installed."""


with suppress(ImportError):
    from xarray import DataArray  # NOQA


def array(array):
    """Wrapper function for creating 'immutable' arrays."""
    xp = get_array_namespace(array)
    out = xp.asarray(array)
    # Setting the write flag to false makes the array immutable unless
    # the flag is switched back. TODO: check on pytorch.
    if hasattr(out, "setflags"):
        out.setflags(write=False)
    return out


def get_array_namespace(*arrays):
    """Return Array API namespace for the first array with __array_namespace__."""
    for arr in arrays:
        if hasattr(arr, "__array_namespace__"):
            return arr.__array_namespace__()
    return np


def is_array(maybe_array):
    """
    Determine if an object is array like (compatible with array api).
    """
    array_ns = getattr(maybe_array, "__array_namespace__", None)
    as_array = getattr(array_ns, "asarray", None)
    return as_array is not None


def array_at_least(array, nd: int):
    """
    Expand an array to be a certain dimensionality if it isn't already.

    Parameters
    ----------
    array
        A numeric array.
    nd
        The number of dimensions the array out to have.
    """
    xp = get_array_namespace(array)
    array = xp.asarray(array)
    while array.ndim < nd:
        array = xp.expand_dims(array, 0)
    if array.ndim < nd:
        msg = f"Failed to promote {array} to dimension {nd}."
        raise ValueError(msg)
    return array
