"""
Module containing patch processing routines.
"""
from .aggregate import aggregate
from .basic import *  # noqa
from .coords import snap_coords, sort_coords
from .detrend import detrend
from .filter import median_filter, pass_filter, sobel_filter
from .resample import decimate, interpolate, iresample, resample
from .select import iselect, select
from .taper import taper
from .units import convert_units, set_units, simplify_units
