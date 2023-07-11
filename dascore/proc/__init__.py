"""
Module to Patch Processing.
"""
from .aggregate import aggregate
from .basic import (
    abs,
    apply_operator,
    normalize,
    rename_coords,
    squeeze,
    standardize,
    transpose,
    update_coords,
)
from .coords import snap_coords, sort_cords
from .detrend import detrend
from .filter import median_filter, pass_filter, sobel_filter
from .resample import decimate, interpolate, iresample, resample
from .select import select
from .taper import taper
from .units import convert_units, set_units, simplify_units
