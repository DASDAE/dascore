"""
Deprecated alias for :mod:`dascore.io.sintela`.

The Sintela binary reader moved to :mod:`dascore.io.sintela` when protobuf
support was added, so both Sintela formats share one module. This shim keeps
``from dascore.io.sintela_binary import SintelaBinaryV3`` working for code
written against v0.1.18 and earlier.
"""

from __future__ import annotations

from dascore.io.sintela import SintelaBinaryV3

__all__ = ["SintelaBinaryV3"]
