"""
Compatibility module for DASCore.

All components/functions that may be exchanged for other numpy/scipy
compatible libraries should go in this model.
"""

from numpy import array, floor, interp  # NOQA
from scipy.interpolate import interp1d  # NOQA
from scipy.ndimage import zoom  # NOQA
from scipy.signal import decimate, resample, resample_poly  # NOQA
