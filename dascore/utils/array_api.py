"""
Utilities for working with the Python array API standard.

DASCore patch data can be backed by any array library which implements the
[array API standard](https://data-apis.org/array-api/latest/). This module
holds the small set of helpers needed to write backend-agnostic code and to
cross the boundary back to numpy when a numpy-only implementation is used.
"""

from __future__ import annotations

from typing import Any, TypeGuard

import array_api_compat.numpy as np_namespace
import numpy as np
from array_api_compat import array_namespace as _array_namespace
from array_api_compat import device, to_device

__all__ = [
    "array_namespace",
    "asarray_like",
    "backend_name",
    "device",
    "is_numpy",
    "to_numpy",
]

# The key used by patch functions written against the array API standard.
ARRAY_API_BACKEND = "array_api"

# The key used by patch functions which require numpy arrays.
NUMPY_BACKEND = "numpy"


def array_namespace(*arrays: Any) -> Any:
    """
    Return the array API namespace shared by arrays.

    Array-likes which don't implement the standard, but which numpy can
    consume through ``__array__``, get the numpy namespace; numpy code is
    what has always handled them.

    Parameters
    ----------
    *arrays
        One or more arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.utils.array_api import array_namespace
    >>>
    >>> xp = array_namespace(np.array([1, 2]))
    >>> assert xp.sum(np.array([1, 2])) == 3
    """
    try:
        return _array_namespace(*arrays)
    except TypeError:
        return np_namespace


def is_numpy(array: Any) -> TypeGuard[np.ndarray]:
    """Return True if the array is a numpy array."""
    # Not array_api_compat.is_numpy_array; this is called by every patch
    # function and the isinstance check is several times faster.
    return isinstance(array, np.ndarray | np.generic)


def backend_name(array: Any) -> str:
    """
    Return the name of the array backend which owns array.

    The name is the top-level package of the array's namespace, so
    numpy arrays return "numpy", jax arrays "jax", dask arrays "dask", etc.

    Parameters
    ----------
    array
        Any array which implements the array API standard.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.utils.array_api import backend_name
    >>>
    >>> assert backend_name(np.array([1, 2])) == "numpy"
    """
    if is_numpy(array):
        return NUMPY_BACKEND
    name = array_namespace(array).__name__
    # array_namespace falls back to numpy for array-likes which don't
    # implement the standard; those are numpy's problem as well.
    # array_api_compat wraps incomplete backends in modules named after them
    # (eg array_api_compat.dask.array); native namespaces are named after
    # their own package (eg jax.numpy).
    name = name.removeprefix("array_api_compat.")
    return name.split(".")[0]


def to_numpy(array: Any) -> np.ndarray:
    """
    Convert an array from any backend to a numpy array.

    Parameters
    ----------
    array
        Any array which implements the array API standard.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.utils.array_api import to_numpy
    >>>
    >>> assert isinstance(to_numpy(np.array([1, 2])), np.ndarray)
    """
    # asarray returns numpy arrays unchanged and handles any backend which
    # implements __array__.
    try:
        return np.asarray(array)
    except (TypeError, ValueError, RuntimeError):
        # Arrays which live on another device (eg a gpu) refuse implicit
        # conversion, so they must be copied to the host first.
        return np.asarray(to_device(array, "cpu"))


def asarray_like(array: Any, like: Any) -> Any:
    """
    Return array converted to the backend and device of like.

    Parameters
    ----------
    array
        The array to convert.
    like
        An array whose backend and device the output should match.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.utils.array_api import asarray_like
    >>>
    >>> out = asarray_like(np.array([1, 2]), np.array([3.0]))
    >>> assert isinstance(out, np.ndarray)
    """
    if is_numpy(like):
        return np.asarray(array)
    xp = array_namespace(like)
    return xp.asarray(array, device=device(like))
