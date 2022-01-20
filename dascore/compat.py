"""
Compatibility module for DASCore.

All components/functions that may be exchanged for other numpy/scipy
compatible libraries should go in this model.
"""

from scipy.signal import decimate, resample, resample_poly  # NOQA
from scipy.interpolate import interp1d  # NOQA
from numpy import floor, array, interp # NOQA
from scipy.ndimage import zoom  # NOQA
