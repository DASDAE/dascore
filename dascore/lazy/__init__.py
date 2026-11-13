"""
Arrays which are read only when asked for, and the operations pending on them.

- `dascore.lazy.recipe`: a recipe holds no data; each of its members says
  which window of which stored source fills which box of the output.
- `dascore.lazy.array`: a `LazyArray` is a recipe and a chain of pending
  operations, which is what a patch holds when its data are not read yet.
- `dascore.lazy.chain`: the steps of that chain.
- `dascore.lazy.contract`: chunk contracts, what an operation's output
  chunk needs of its input, which let it run on a lazy array.
- `dascore.lazy.kernels`: the kernel every operation runs on a lazy array.
- `dascore.lazy.execute`: running the chain, a chunk at a time.

Import names from the module which defines them; this package imports
none of its modules, so `dascore.lazy.contract` can be imported while
DASCore itself is still being imported.
"""

from __future__ import annotations
