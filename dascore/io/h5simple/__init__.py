"""
Support for generic H5 files.

This attempts to use some common names to read in some very simple h5 files.

The FORESEE data in PubDAS motivated this module.

A root ``data_units`` attr that is not units (eg "same as input DAS data") is
kept as ``data_units_description``, with a warning. A root ``time_units`` attr
of the form "<unit> relative to ..." makes time a timedelta coordinate of
offsets from that unstated origin (and is not kept as an attr); add the origin
with
``patch.update_coords(time=patch.get_coord("time") + origin)``.
"""
from __future__ import annotations

from .core import H5Simple
