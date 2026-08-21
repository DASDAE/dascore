"""
Support for Febus format.

This is used by the Febus DAS interrogator.

More info about febus can be found here: https://www.febus-optics.com/en/

Interrogator identity
---------------------
A1 files report ``interrogator.name`` from the Source's ``Hostname`` (eg
"fa1-24090193"). T1 files report it from ``device_name``, plus
``interrogator.instrument_type`` from ``device``; their
``interrogator.manufacturer`` and ``interrogator.model`` are asserted by the
format, not read from the header. The G1 BSL and MTX HDF5 files carry no
interrogator metadata, so they report none.
"""

from __future__ import annotations

from .core import Febus1 as Febus1
from .core import Febus2 as Febus2
from .core import FebusBSLH5V1 as FebusBSLH5V1
from .core import FebusMTXH5V1 as FebusMTXH5V1
from .core import FebusT1V1 as FebusT1V1
