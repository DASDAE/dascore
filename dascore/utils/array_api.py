"""
Utilities for working with the Python array API standard.

DASCore patch data can be backed by any array library which implements the
[array API standard](https://data-apis.org/array-api/latest/). This module
holds the small set of helpers needed to write backend-agnostic code and to
cross the boundary back to numpy when a numpy-only implementation is used.
Which array-likes are accepted as patch data is a separate question, and
lives in [compat](`dascore.compat`).
"""

from __future__ import annotations

import warnings
from typing import Any, TypeGuard

import array_api_compat.numpy as np_namespace
import numpy as np
from array_api_compat import array_namespace as _array_namespace
from array_api_compat import device, is_array_api_obj, to_device

from dascore.compat import is_array
from dascore.warnings import NumpyFallbackWarning

__all__ = [
    "array_namespace",
    "asarray_like",
    "backend_name",
    "device",
    "is_foreign",
    "is_numpy",
    "namespace_name",
    "to_numpy",
    "warn_numpy_fallback",
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
    """Return True if the array is a numpy array or scalar."""
    # Numpy scalars are excluded by is_array but count here; they carry
    # __array_namespace__, so they must not be treated as foreign arrays.
    return is_array(array) or isinstance(array, np.generic)


def is_foreign(array: Any) -> bool:
    """
    Return True if the array belongs to a non-numpy array API backend.

    These are the arrays which have to cross the numpy boundary explicitly.

    Parameters
    ----------
    array
        Any object.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.utils.array_api import is_foreign
    >>>
    >>> assert not is_foreign(np.array([1, 2]))
    >>> assert not is_foreign(10)
    >>> assert not is_foreign(np.float32)  # a dtype, not an array
    """
    # Array classes (eg np.float32 used as a dtype) inherit the array
    # protocols from their instances, so exclude types explicitly.
    if is_numpy(array) or isinstance(array, type):
        return False
    # Everything else is only an array if it has a shape; checking here
    # keeps scalar operands off the slower classification below.
    if getattr(array, "shape", None) is None:
        return False
    # Not just __array_namespace__; backends which don't implement the
    # standard natively (eg torch, dask) are recognized by their type.
    return is_array_api_obj(array)


def warn_numpy_fallback(name: str, backend: str, stacklevel: int = 3) -> None:
    """
    Warn that name has no implementation for the given array backend.

    Parameters
    ----------
    name
        The name of the function which lacks an implementation.
    backend
        The name of the array backend of the input data.
    stacklevel
        The stack level, as understood by warnings.warn, of the function
        calling this one.
    """
    msg = (
        f"{name} has no {backend} implementation; the data were converted "
        "to numpy and the output converted back. Silence this with "
        "dascore.utils.misc.suppress_warnings(NumpyFallbackWarning)."
    )
    warnings.warn(msg, NumpyFallbackWarning, stacklevel=stacklevel + 1)


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
    # array_namespace falls back to numpy for array-likes which don't
    # implement the standard; those are numpy's problem as well.
    if is_numpy(array):
        return NUMPY_BACKEND
    return namespace_name(array_namespace(array))


def namespace_name(namespace: Any) -> str:
    """
    Return the backend name of an array API namespace.

    Parameters
    ----------
    namespace
        An array API namespace module.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.utils.array_api import namespace_name
    >>>
    >>> assert namespace_name(np) == "numpy"
    """
    # array_api_compat wraps incomplete backends in modules named after them
    # (eg array_api_compat.dask.array); native namespaces are named after
    # their own package (eg jax.numpy).
    name = namespace.__name__.removeprefix("array_api_compat.")
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
