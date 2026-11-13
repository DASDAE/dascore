"""
The kernel every operation runs on a lazy array, unless it registers its own.

It computes nothing. It asks the processor for its
[chunk contract](`dascore.lazy.contract`) for the input's dtype and the
plan -- a contract written above the class which owns the kernel describes
some other kernel, so it does not count -- then:

- no contract, or a `reconcile` of the processor's own, which would be
  handed the lazy array: refuses, before anything is read;
- [`Unchanged`](`dascore.lazy.contract.Unchanged`): hands the data back, so
  the operation is the same no-op it is on data in memory;
- any other contract: appends the operation to the chain, pending until
  the operation settles it with the shape, dtype and `data_id` of what it
  made.

[`PatchProcessor.kernel_for`](`dascore.core.processor.PatchProcessor.kernel_for`)
answers with it for the lazy backend.

Examples
--------
>>> import dascore as dc
>>> from dascore.core.source import ArraySource
>>> from dascore.lazy.contract import Pointwise
>>> from dascore.lazy.recipe import Recipe
>>>
>>> class Negate(dc.PatchProcessor):
...     '''Negate every sample.'''
...
...     name = None  # so this example claims no name of its own
...
...     def chunk_contract(self, dtype, plan):
...         return Pointwise()
...
...     def kernel(self, data):
...         return -data
>>>
>>> recipe = Recipe.from_source(ArraySource.full((3, 2), -1.0))
>>> patch = dc.Patch(data=recipe, coords={"x": range(3), "y": range(2)},
...                  dims=("x", "y"))
>>> out = Negate()(patch)
>>> assert out.backend == "lazy" and len(out.data.chain) == 1
>>> assert out.data.chain.last.data_id == out.attrs.data_id
>>> assert (out.load().data == 1).all()
"""

from __future__ import annotations

from typing import Any

from dascore.core.processor import PatchProcessor, _computes
from dascore.lazy.array import LazyArray
from dascore.lazy.chain import Node
from dascore.lazy.contract import ChunkContract, Unchanged
from dascore.utils.array_api import lazy_refusal


def lazy_kernel(processor, data: LazyArray, **plan: Any) -> LazyArray:
    """
    Answer for an operation on a lazy array by its chunk contract.

    Parameters
    ----------
    processor
        The [`PatchProcessor`](`dascore.core.processor.PatchProcessor`)
        being run.
    data
        The lazy array it runs on.
    **plan
        The arguments `get_metadata` worked out for its kernel.

    Raises
    ------
    LazyOperationError
        If the operation states no contract this backend can keep.
    """
    contract = _contract(processor, data.dtype, plan)
    if isinstance(contract, Unchanged):
        return data
    if isinstance(contract, ChunkContract):
        return data._appended(Node(processor, plan, contract))
    # No contract: refused, and nothing read.
    raise lazy_refusal(processor.name or type(processor).__name__)


def _contract(processor, dtype, plan: dict[str, Any]):
    """
    Return the processor's contract, if it describes the kernel that runs.

    A subclass which writes its own kernel means it, as `kernel_for` says;
    a contract it inherits from above that kernel was written for another.
    """
    mro = type(processor).__mro__
    # A reconcile is handed the kernel's result, and may read it.
    if type(processor).reconcile is not PatchProcessor.reconcile:
        return None
    owner = next((x for x in mro if _computes(x)), mro[0])
    stated = next(x for x in mro if "chunk_contract" in x.__dict__)
    if not issubclass(stated, owner):
        return None
    return processor.chunk_contract(dtype, plan)
