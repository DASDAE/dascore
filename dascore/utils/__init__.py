"""Utilities for dascore."""
from __future__ import annotations
from .time import to_datetime64, to_timedelta64
from .moving import (
    move_max,
    move_mean,
    move_median,
    move_min,
    move_std,
    move_sum,
    moving_window,
)
