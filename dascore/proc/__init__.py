"""
Module containing patch processing routines.
"""
from __future__ import annotations

import dascore.proc.aggregate as agg
from .basic import *  # noqa
from .coords import *  # noqa
from .correlate import correlate, correlate_shift
from .detrend import detrend
from .filter import median_filter, pass_filter, sobel_filter, savgol_filter, gaussian_filter, slope_filter, notch_filter
from .resample import decimate, interpolate, resample
from .rolling import rolling
from .taper import taper, taper_range
from .units import convert_units, set_units, simplify_units
from .whiten import whiten
from .hampel import hampel_filter
from .wiener import wiener_filter
