"""
Compatibility module for DASCore.

All components/functions that may be exchanged for other numpy/scipy
compatible libraries should go in this model.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Protocol, runtime_checkable

import numpy as np
from numpy import floor, interp, ndarray  # NOQA
from numpy.random import RandomState
from rich.progress import Progress  # NOQA
from scipy.interpolate import interp1d  # NOQA
from scipy.ndimage import zoom  # NOQA
from scipy.signal import decimate, resample, resample_poly  # NOQA
from upath import UPath  # NOQA

random_state = RandomState(42)


class DataArray:
    """A dummy class for when xarray isn't installed."""


with suppress(ImportError):
    from xarray import DataArray  # NOQA


def array(array):
    """Wrapper function for creating 'immutable' arrays."""
    out = np.asarray(array)
    # Setting the write flag to false makes the array immutable unless
    # the flag is switched back.
    out.setflags(write=False)
    return out


def is_array(maybe_array):
    """Determine if an object is array like."""
    # This is here so that we can support other array types in the future.
    return isinstance(maybe_array, np.ndarray)


@runtime_checkable
class BufferedReaderLike(Protocol):
    """Protocol for buffered reader objects."""

    def read(self, n: int = -1) -> bytes: ...  # NOQA
    def seek(self, offset: int, whence: int = 0) -> int: ...  # NOQA
    def tell(self) -> int: ...  # NOQA
    def close(self) -> None: ...  # NOQA


@runtime_checkable
class BufferedWriterLike(Protocol):
    """Protocol for buffered writer objects."""

    def write(self, data: bytes) -> int: ...  # NOQA
    def flush(self) -> None: ...  # NOQA
    def seek(self, offset: int, whence: int = 0) -> int: ...  # NOQA
    def tell(self) -> int: ...  # NOQA
    def close(self) -> None: ...  # NOQA
