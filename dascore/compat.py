"""
Compatibility module for DASCore.

All components/functions that may be exchanged for other numpy/scipy
compatible libraries should go in this model.
"""

from numpy import array, floor, interp  # NOQA
from scipy.interpolate import interp1d  # NOQA
from typing import NewType
from scipy.ndimage import zoom  # NOQA
from scipy.signal import decimate, resample, resample_poly  # NOQA


# Try to import data array. If Xarray isn't installed just create dummy type.
try:
    from xarray import DataArray
except ImportError:
    DataArray = NewType("DataArray", None)
