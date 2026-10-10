"""
Read and write data from Terra15 DAS interrogators.

Terra15's website can be found [here](https://terra15.com.au/)

Notes
-----
Terra15 files contain ``GPS_time`` and ``posix_time`` arrays; DASCore uses GPS
time. GPS timestamps can jitter, occasionally decrease between samples, and differ
from the ``dT`` attribute, so by default (``snap=("time",)``) time is put on the
grid spanning the first and last written GPS times, checked against the
quartile times; where those sit more than ``dascore.config.max_shift`` steps
off it, the times are kept as stored with a warning.

Unwritten trailing rows with zero-filled timestamps are excluded whether snapping is enabled or disabled. Passing ``snap=False`` preserves the written samples' raw timestamps (``snap_dims``, the released spelling, is deprecated).
"""
from __future__ import annotations
from .core import Terra15FormatterV4, Terra15FormatterV5, Terra15FormatterV6
