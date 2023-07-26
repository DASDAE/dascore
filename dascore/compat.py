"""
Compatibility module for DASCore.

All components/functions that may be exchanged for other numpy/scipy
compatible libraries should go in this model.
"""
from __future__ import annotations

from contextlib import suppress

from numpy import floor, interp  # NOQA
from scipy.interpolate import interp1d  # NOQA
from scipy.ndimage import zoom  # NOQA
from scipy.signal import decimate, resample, resample_poly  # NOQA


class DataArray:
    """A dummy class for when xarray isn't installed."""


with suppress(ImportError):
    from xarray import DataArray  # NOQA


def array(array):
    """Wrapper function for creating 'immutable' arrays."""
    array.setflags(write=False)
    return array
