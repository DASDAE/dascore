"""
Read HDF5 files exported by Uptech Sensing AS1000 interrogators.

Examples
--------
Read a registered test file:

>>> import dascore as dc
>>> spool = dc.read("examples://uptech_as1000_1.hdf5")

The reader expects ``Acquisition/StrainRate`` and ``Acquisition/Time``
datasets. Format metadata is read from attributes on the strain-rate dataset;
in particular, ``sampling_interval`` is the spatial channel pitch.

``Acquisition/Time`` holds float64 epoch seconds, which only resolve to a few
hundred nanoseconds at a modern epoch, so consecutive samples are never exactly
evenly spaced. The nominal ``acquisition_frequency`` is used to sanity check
the mean step but is not stored on the patch; the time coordinate built from
the timestamps is authoritative.
"""
from .core import UptechH5V1

__all__ = ["UptechH5V1"]
