"""
Read Febus DAS interrogator formats.

See https://www.febus-optics.com/en/.

Interrogator identity
---------------------
A1 files derive ``interrogator.name`` from the source ``Hostname`` (for example,
``"fa1-24090193"``). T1 files derive it from ``device_name`` and derive
``interrogator.instrument_type`` from ``device``; the format supplies their
manufacturer and model. G1 BSL and MTX HDF5 files contain no interrogator metadata.

Distance sampling
-----------------
Reads return an evenly sampled ``distance`` coordinate because the interrogator fixes
the spatial sampling. A1 files state the spacing in their headers. G1 BSL/MTX HDF5
and T1 files store each sample's distance, which can contain sub-step jitter. Known
G1 files store an even grid as float32; its quantization exceeds ``get_coord``'s
tolerance, so DASCore restores the stated grid.

The correction is well under a millimeter in known files but is not bounded. A
genuinely discontinuous distance axis, such as one spanning several acquisition
zones, would be mapped onto one grid. ``scan(..., snap=False)`` reports stored
``distance`` and ``time`` values exactly.
"""

from __future__ import annotations

from .core import Febus1 as Febus1
from .core import Febus2 as Febus2
from .core import FebusBSLH5V1 as FebusBSLH5V1
from .core import FebusMTXH5V1 as FebusMTXH5V1
from .core import FebusT1V1 as FebusT1V1
