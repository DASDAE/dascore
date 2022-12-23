"""
Module to Patch Processing.
"""
from .aggregate import aggregate
from .basic import abs, normalize, rename, squeeze, standardize, transpose
from .detrend import detrend
from .filter import median_filter, pass_filter, sobel_filter
from .resample import decimate, interpolate, iresample, resample
from .select import select
