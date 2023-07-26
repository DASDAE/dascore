"""
Module for reading and writing data recorded by Terra15 DAS interrogators.

Terra15's website can be found [here](https://terra15.com.au/)

Notes
-----
There are a few implementation details to note. The tricky part of the
implementation is how time is handled. The Terra15 files contain two arrays
corresponding to GPS_time and posix_time. We elected to simply use GPS time.

However, there are a few issues. First, there can be significant jitter in the
GPS time vector, and occasionally, sample n+1 has a smaller timestamp than n.
This reeks havoc on the pandas indexes used by xarray. Second, the spacing is
often different from the dT attributes. Our implementation does the following:

1. dt = (max(gps_time) - min(gps_time)) / (len(gps_time) - 1). This ensures
min(gps_time) + len(gps_time) * dt ≈ max(gps_time).

2. The time array returned by the parser is calculated by min(gps_time) +
dt * np.arange(len(gps_time)) which insures it is monotonically increasing.
The time is then cast to datetime64 with
[to_datetime64](dascore.utils.time.to_datettime64).

3. The start/end time returned by the scan function are gps_time[0] and
gps_time[-1], cast to datetime64 objects.

It is very important that the scan method returns exactly the same time_min
and time_max as contained in the patch when loaded into memory.
"""
from __future__ import annotations
from .core import Terra15FormatterV4, Terra15FormatterV5, Terra15FormatterV6
