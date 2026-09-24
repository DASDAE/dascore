"""
Module containing patch processing routines.
"""
from __future__ import annotations

import dascore.proc.aggregate as agg
from .basic import *  # noqa
from .coords import *  # noqa
from .correlate import correlate, correlate_shift
from .detrend import Detrend
from .filter import median_filter, pass_filter, SobelFilter, savgol_filter, gaussian_filter, slope_filter, notch_filter
from .resample import decimate, interpolate, resample
from .rolling import rolling
from .taper import taper, taper_range
from .mute import line_mute, slope_mute
from .units import ConvertUnits, SetUnits, SimplifyUnits
from .whiten import whiten
from .hampel import hampel_filter
from .wiener import wiener_filter
from .align import align_to_coord
from .tile_apply import reassemble
from .inventory import enrich
