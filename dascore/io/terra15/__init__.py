"""
Read and write data from Terra15 DAS interrogators.

Terra15's website can be found [here](https://terra15.com.au/)

Notes
-----
Terra15 files contain ``GPS_time`` and ``posix_time`` arrays; DASCore uses GPS
time. GPS timestamps can jitter, occasionally decrease between samples, and differ
from the ``dT`` attribute, so DASCore regularizes them as follows:

1. Compute ``dt = (max(gps_time) - min(gps_time)) / (len(gps_time) - 1)``.

2. Build ``min(gps_time) + dt * np.arange(len(gps_time))`` and convert it with
[to_datetime64](dascore.utils.time.to_datetime64).

3. Convert ``gps_time[0]`` and ``gps_time[-1]`` to ``datetime64`` scan bounds.

Unwritten trailing rows with zero-filled timestamps are excluded whether snapping is enabled or disabled. Passing ``snap=False`` (or ``snap_dims=False`` to `read`) preserves the written samples' raw timestamps.

The scan bounds must exactly match the loaded patch's ``time_min`` and ``time_max``.
"""
from __future__ import annotations
from .core import Terra15FormatterV4, Terra15FormatterV5, Terra15FormatterV6
