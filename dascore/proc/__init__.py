"""
Module containing patch processing routines.
"""
from __future__ import annotations
from .aggregate import aggregate
from .basic import *  # noqa
from .coords import *  # noqa
from .correlate import correlate
from .detrend import detrend
from .filter import median_filter, pass_filter, sobel_filter, savgol_filter, gaussian_filter
from .resample import decimate, interpolate, resample
from .rolling import rolling
from .select import select
from .taper import taper
from .units import convert_units, set_units, simplify_units
from .whiten import whiten
