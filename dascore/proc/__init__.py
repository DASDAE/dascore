"""
Module containing patch processing routines.
"""
from __future__ import annotations

import dascore.proc.aggregate as agg
from .basic import *  # noqa
from .coords import *  # noqa
from .correlate import correlate, CorrelateShift
from .detrend import Detrend
from .filter import MedianFilter, PassFilter, SobelFilter, SavgolFilter, GaussianFilter, slope_filter, NotchFilter
from .resample import Decimate, Interpolate, Resample
from .rolling import rolling
from .taper import Taper, TaperRange
from .mute import LineMute, SlopeMute
from .units import ConvertUnits, SetUnits, SimplifyUnits
from .whiten import whiten
from .hampel import HampelFilter
from .wiener import WienerFilter
from .align import AlignToCoord
from .tile_apply import Reassemble
from .inventory import enrich
