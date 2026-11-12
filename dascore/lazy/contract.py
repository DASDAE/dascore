"""
Chunk contracts: what one chunk of an operation's output needs of its input.

An operation which can run on a lazy array says so by returning a contract
from `PatchProcessor.chunk_contract`, for the input's dtype and the plan
`get_metadata` worked out. The contract is what lets the work be split
into chunks later without changing the result: each output chunk is
computed from the input it names, and nothing else. Returning `None` says
the operation cannot run on a lazy array.

This module imports nothing from the rest of DASCore, because the
processors in `dascore.proc` and `dascore.transform` import it.

Examples
--------
>>> import numpy as np
>>> import dascore as dc
>>> from dascore.lazy.contract import Pointwise, Unchanged
>>>
>>> class Double(dc.PatchProcessor):
...     '''Double every sample.'''
...
...     name = None  # so this example claims no name of its own
...
...     def chunk_contract(self, dtype, plan):
...         return Pointwise()
...
...     def kernel(self, data):
...         return data * 2
>>>
>>> assert isinstance(Double().chunk_contract(np.dtype("f8"), {}), Pointwise)
>>> assert dc.PatchProcessor().chunk_contract(np.dtype("f8"), {}) is None
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkContract:
    """What one chunk of an operation's output needs of its input."""


@dataclass(frozen=True)
class Pointwise(ChunkContract):
    """
    An output chunk needs only the input chunk at the same place.

    The output has the input's shape, and each sample is computed from the
    input sample at its own position alone.
    """


@dataclass(frozen=True)
class Unchanged(ChunkContract):
    """
    The kernel hands its input back, for this dtype and plan.

    The step does nothing, so it adds nothing to the chain; the patch is
    handed back as it is, with its own id, as the eager no-op does. Only
    for a no-op decided by the dtype or the plan, never by the values.
    """
