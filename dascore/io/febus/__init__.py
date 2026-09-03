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

Distance sampling
-----------------
The interrogator fixes the spatial sampling, so a read returns an evenly
sampled ``distance`` coordinate. A1 files state the spacing in the header and
the coordinate is built from it. The G1 BSL/MTX HDF5 and T1 files instead
store a distance per sample, and those arrays can carry sub-step jitter: the
G1 files seen so far hold an even grid restated in float32, whose
quantization exceeds the tolerance ``get_coord`` uses to recognize an even
coordinate and leaves a monotonic coord with no step. Those are put back on
the grid they restate, which depends only on the first and last stored values
and the sample count, so files spanning the same fiber agree sample for
sample and still merge.

The correction is well under a millimeter on the files seen so far, but it is
not bounded in principle: a file whose distance axis were genuinely
discontinuous, such as one covering several acquisition zones, would be
smeared onto a single grid. ``scan(..., snap=False)`` reports the stored
values exactly, for ``distance`` as for ``time``.
"""

from __future__ import annotations

from .core import Febus1 as Febus1
from .core import Febus2 as Febus2
from .core import FebusBSLH5V1 as FebusBSLH5V1
from .core import FebusMTXH5V1 as FebusMTXH5V1
from .core import FebusT1V1 as FebusT1V1
