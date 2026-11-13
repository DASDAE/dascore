"""
Support for PUNIU HiFi-DAS HDF5 files.

HiFi-DAS is a distributed acoustic sensing interrogator made by PUNIU which
uses TGD-OFDR sensing (https://puniutech.com/en/products/hifi-das-v2-0/).

Version 1 files store a two-dimensional differential signal in a ``/default``
dataset whose attributes declare the axis order, channel range, and clock.
Patches are always returned with dims ``(time, distance)``, whichever axis
order the file stores.

Distance is absolute: channel ``n`` sits at ``n * dx0`` metres, where ``dx0``
is ``spatial_sampling_rate`` rounded half-up to 0.1 m, as the acquisition
software's MATLAB reader does. Sampled channels are ``step`` channels apart,
so files that start on different channels line up along the fiber. The
original channel numbers are kept in a ``channel`` coordinate on the
distance axis.

The ``spatial_resolution`` attribute becomes ``gauge_length``, matching the
acquisition reader. Time comes from ``epoch`` (seconds) plus ``ns``, with no
timezone adjustment. Signal values are returned as stored, without
calibration or units.

Examples
--------
>>> import dascore as dc
>>> # Read 100 m to 200 m along the fiber from a file:
>>> # patch = dc.read("puniu_file.h5", distance=(100, 200))[0]
"""

from .core import PuniuV1
