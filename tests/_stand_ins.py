"""
Another package's objects, for tests: plain Python classes registered with
dascore's interfaces.
"""

from __future__ import annotations

import numpy as np

from dascore.core.interfaces import (
    BaseCoordManager,
    BaseLazyArray,
    BaseNumericCoord,
    BaseSpool,
    BaseStringCoord,
)


class _StandIn:
    """Another package's object: plain Python, answering as one of dascore's."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __len__(self):
        return len(self._inner)

    def __iter__(self):
        return iter(self._inner)

    def __getitem__(self, item):
        return self._inner[item]

    def __contains__(self, item):
        return item in self._inner

    def __add__(self, other):
        return self._inner + getattr(other, "_inner", other)

    def __eq__(self, other):
        return self._inner == getattr(other, "_inner", other)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._inner, dtype=dtype)


@BaseNumericCoord.register
class _NumericCoord(_StandIn):
    """Another package's numeric coordinate."""


@BaseStringCoord.register
class _StringCoord(_StandIn):
    """Another package's text coordinate."""


@BaseCoordManager.register
class _Manager(_StandIn):
    """Another package's coordinate manager."""


@BaseSpool.register
class _Spool(_StandIn):
    """Another package's spool."""


@BaseLazyArray.register
class _LazyArray:
    """Another package's lazy array, counting the times it is read."""

    def __init__(self, array):
        self._array = np.asarray(array)
        self.reads = 0

    shape = property(lambda self: self._array.shape)
    dtype = property(lambda self: self._array.dtype)
    ndim = property(lambda self: self._array.ndim)

    def __getitem__(self, item):
        return _LazyArray(self._array[item])

    def __array__(self, dtype=None, copy=None):
        self.reads += 1
        return self._array
