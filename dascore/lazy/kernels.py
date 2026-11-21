"""
The kernel every operation runs on a lazy array, unless it registers its own.

It computes nothing. It asks the processor for its
[chunk contract](`dascore.lazy.contract`) for the input's dtype and the
plan -- a contract written above the class which owns the kernel describes
some other kernel, so it does not count -- then:

- no contract, or a `reconcile` of the processor's own, which would be
  handed the lazy array: refuses, before anything is read, unless the
  `numpy_fallback` setting allows the data to be read into memory for it;
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
>>> assert len(out.data.chain) == 1
>>> assert out.data.chain.last.data_id == out.attrs.data_id
>>> assert (out.load().data == 1).all()
"""

from __future__ import annotations

from typing import Any

from dascore.core.processor import PatchProcessor, _computes
from dascore.lazy.array import LazyArray
from dascore.lazy.chain import Node
from dascore.lazy.contract import ChunkContract, Unchanged
from dascore.lazy.execute import _converted
from dascore.utils.array_api import decide_fallback, lazy_refusal


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
        If the operation states no contract this backend can keep, and the
        `numpy_fallback` setting does not allow its data to be read.
    """
    name = processor.name or type(processor).__name__
    contract = _contract(processor, data.dtype, plan)
    if isinstance(contract, Unchanged):
        return data
    library = data._backend.name
    kernel = type(processor).kernel_for(library)
    if isinstance(contract, ChunkContract):
        if (converted := _converted(processor, library)) is not None:
            # It runs a chunk at a time still, each chunk through NumPy.
            decide_fallback(
                f"{converted} has no {library} implementation",
                "each chunk is converted to numpy to run it",
            )
        return data._appended(Node(processor, plan, contract))
    # No contract: the data are read whole, if the setting allows it.
    decide_fallback(
        f"{name} cannot run on a lazy patch a chunk at a time",
        "the data were read into memory to run it",
        refusal=lazy_refusal(name),
    )
    # Read on the library its chunks compute on. An operation with no
    # If the operation has no kernel for that library either, it says so
    # itself, as it does for data in memory.
    loaded = data.load()
    out = kernel(processor, loaded, **plan)
    # Handed back unchanged is "nothing to do", as it is in memory: the
    # patch is left as it was, lazy and under its own id.
    return data if out is loaded else out


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
