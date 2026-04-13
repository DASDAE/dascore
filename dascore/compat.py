"""
Compatibility module for DASCore.

All components/functions that may be exchanged for other numpy/scipy
compatible libraries should go in this model.
"""

from __future__ import annotations

from contextlib import suppress

import numpy as np
from h5py import Dataset as H5Dataset
from numpy import floor, interp, ndarray  # NOQA
from numpy.random import RandomState
from rich.progress import Progress  # NOQA
from scipy.interpolate import interp1d  # NOQA
from scipy.ndimage import zoom  # NOQA
from scipy.signal import decimate, resample, resample_poly  # NOQA

random_state = RandomState(42)


class DataArray:
    """A dummy class for when xarray isn't installed."""


with suppress(ImportError):
    from xarray import DataArray  # NOQA

MATERIALIZE_NOW = (H5Dataset,)


def is_array_like(maybe_array):
    """Determine if an object looks like an array without materializing it."""
    # Some array-like types depend on external resources and must be converted
    # immediately rather than preserved by reference.
    if MATERIALIZE_NOW and isinstance(maybe_array, MATERIALIZE_NOW):
        return False
    attrs = ("shape", "dtype")
    protocols = (
        "__array__",
        "__array_ufunc__",
        "__array_function__",
        "__dlpack__",
        "__array_namespace__",
    )
    return all(hasattr(maybe_array, attr) for attr in attrs) and any(
        hasattr(maybe_array, protocol) for protocol in protocols
    )


def _make_immutable(maybe_array):
    """Set writable flag to false if the array exposes one."""
    flags = getattr(maybe_array, "flags", None)
    if flags is None or not hasattr(flags, "writeable"):
        return maybe_array
    with suppress(AttributeError, ValueError):
        maybe_array.flags.writeable = False
    return maybe_array


def array(array):
    """
    Preserve array-like inputs for patch creation and reconstruction.

    NumPy arrays and array-likes with a writable flag are marked read-only when
    possible. Other preserved array-likes are treated as logically immutable,
    but DASCore does not guarantee physical immutability for them.
    """
    out = array if is_array_like(array) else np.asarray(array)
    return _make_immutable(out)


def is_array(maybe_array):
    """
    Determine if an object is a numpy array.
    """
    # This is here so that we can support other array types in the future.
    return isinstance(maybe_array, np.ndarray)
