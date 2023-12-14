"""
Basic support for CF (climate and forcasting) style DAS data.

This was created mainly for reading PoroTomo data from Brady Hotsprings,
so it may be quite lacking for other CF data files.

More info on PoroTomo here:
https://github.com/openEDI/documentation/tree/main/PoroTomo
"""
from __future__ import annotations

from .core import CF1_7
