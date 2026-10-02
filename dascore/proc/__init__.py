"""
Module containing patch processing routines.
"""
from __future__ import annotations

import dascore.proc.aggregate as agg
from .basic import *  # noqa
from .coords import *  # noqa
from .correlate import correlate, correlate_shift
from .detrend import Detrend
from .filter import MedianFilter, PassFilter, SobelFilter, SavgolFilter, GaussianFilter, slope_filter, NotchFilter
from .resample import decimate, interpolate, resample
from .rolling import rolling
from .taper import taper, taper_range
from .mute import line_mute, slope_mute
from .units import ConvertUnits, SetUnits, SimplifyUnits
from .whiten import whiten
from .hampel import HampelFilter
from .wiener import WienerFilter
from .align import align_to_coord
from .tile_apply import reassemble
from .inventory import enrich
