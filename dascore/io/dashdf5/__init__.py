"""
Support for DAS-HDF5 format.

This format is used by the
[PoroTomo](https://github.com/openEDI/documentation/tree/main/PoroTomo) project
 as one of the two modes of storing DAS data which were recorded by a Silixa iDAS
 interrogator.
"""
from __future__ import annotations

from .core import DASHDF5V1
