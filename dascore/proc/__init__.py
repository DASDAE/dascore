"""
Module containing patch processing routines.
"""
from __future__ import annotations
from .aggregate import aggregate
from .basic import *  # noqa
from .coords import *  # noqa
from .detrend import detrend
from .filter import median_filter, pass_filter, sobel_filter
from .resample import decimate, interpolate, iresample, resample
from .rolling import rolling
from .select import iselect, select
from .taper import taper
from .units import convert_units, set_units, simplify_units
