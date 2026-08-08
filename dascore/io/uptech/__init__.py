"""
Read HDF5 files exported by Uptech Sensing AS1000 interrogators.

Examples
--------
Read a registered test file:

>>> import dascore as dc
>>> from dascore.utils.downloader import fetch
>>> path = fetch("uptech_as1000_1.hdf5")
>>> spool = dc.read(path)

The reader expects ``Acquisition/StrainRate`` and ``Acquisition/Time``
datasets. Format metadata is read from attributes on the strain-rate dataset;
in particular, ``sampling_interval`` is the spatial channel pitch.
"""
from .core import UptechH5V1

__all__ = ["UptechH5V1"]
