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
    "can_nan_reduce",
    "device",
    "is_foreign",
    "is_numpy",
    "is_python_scalar",
    "namespace_name",
    "nan_reduce",
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


def is_python_scalar(value: Any) -> bool:
    """
    Return True if the value is a scalar every array API function accepts.

    Parameters
    ----------
    value
        Any object.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.utils.array_api import is_python_scalar
    >>>
    >>> assert is_python_scalar(1.0)
    >>> assert not is_python_scalar(np.float64(1.0))  # subclasses float
    """
    # Not isinstance; np.float64 subclasses float but is not a python scalar.
    return type(value) in (int, float, bool)


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


def _replace_nan(array: Any, value: float) -> Any:
    """Return a floating point array with its nans replaced by a value."""
    xp = array_namespace(array)
    fill = xp.asarray(value, dtype=array.dtype, device=device(array))
    return xp.where(xp.isnan(array), fill, array)


def _all_nan(array: Any, axis, keepdims: bool) -> Any:
    """Return a mask of the slices which hold nothing but nans."""
    xp = array_namespace(array)
    return xp.all(xp.isnan(array), axis=axis, keepdims=keepdims)


def _real_dtype(array):
    """Return the dtype a magnitude of the array has."""
    xp = array_namespace(array)
    if xp.isdtype(array.dtype, "complex floating"):
        return xp.finfo(array.dtype).dtype
    return array.dtype


def _nan_extremum(name, array, axis, keepdims):
    """Return the min or max of an array, ignoring nans."""
    xp = array_namespace(array)
    # Infinity rather than the largest finite value, which a slice holding
    # an infinity would otherwise beat.
    fill = float("inf") if name == "min" else float("-inf")
    out = getattr(xp, name)(_replace_nan(array, fill), axis=axis, keepdims=keepdims)
    # Numpy returns nan for a slice of nothing but nans; the fill value would
    # otherwise leak out here.
    nan = xp.asarray(float("nan"), dtype=array.dtype, device=device(array))
    return xp.where(_all_nan(array, axis, keepdims), nan, out)


def _nan_count(array, axis, keepdims):
    """Return the number of values which are not nan, as a float."""
    xp = array_namespace(array)
    # Counted as integers so a long trace cannot overflow a narrow float,
    # then widened so it can divide arrays of any floating dtype.
    counts = xp.sum(xp.astype(~xp.isnan(array), xp.int64), axis=axis, keepdims=keepdims)
    counts = xp.astype(counts, xp.float64)
    # Empty slices divide to nan rather than raising, as numpy does.
    nan = xp.asarray(float("nan"), device=device(array))
    return xp.where(counts == 0, nan, counts)


def _nan_reduce(name: str, array: Any, axis=None, keepdims: bool = False) -> Any:
    """Reduce an array with the array API standard, ignoring nans."""
    xp = array_namespace(array)
    # Only floating point data can hold nans; integers reduce plainly.
    if not xp.isdtype(array.dtype, ("real floating", "complex floating")):
        return getattr(xp, name)(array, axis=axis, keepdims=keepdims)
    if name in {"min", "max"}:
        return _nan_extremum(name, array, axis, keepdims)
    total = xp.sum(_replace_nan(array, 0), axis=axis, keepdims=keepdims)
    if name == "sum":
        return total
    counts = _nan_count(array, axis, keepdims)
    if name == "mean":
        return xp.astype(total / counts, array.dtype)
    # Only std is left; it needs the mean with the reduced axes still in
    # place. The magnitude keeps complex deviations from cancelling out.
    deviation = xp.abs(array - _nan_reduce("mean", array, axis, keepdims=True)) ** 2
    zero = xp.asarray(0, dtype=deviation.dtype, device=device(array))
    # Masked with the input's nans, not the deviation's; an infinity makes
    # an indeterminate deviation which numpy keeps rather than skips.
    center = xp.sum(
        xp.where(xp.isnan(array), zero, deviation),
        axis=axis,
        keepdims=keepdims,
    )
    return xp.astype(xp.sqrt(center / counts), _real_dtype(array))


# The dtype kinds each reduction accepts in the array API standard. Numpy
# is more permissive, eg it takes the min of a boolean array, so anything
# outside these is left to numpy.
NAN_REDUCE_DTYPES = {
    "max": ("integral", "real floating"),
    "mean": ("integral", "real floating", "complex floating"),
    "min": ("integral", "real floating"),
    "std": ("integral", "real floating", "complex floating"),
    "sum": ("integral", "real floating", "complex floating"),
}


def can_nan_reduce(name: str, array: Any) -> bool:
    """
    Return True if a nan-skipping reduction can be done in the array's own
    namespace, rather than by numpy.

    Parameters
    ----------
    name
        The name of the reduction; see
        [nan_reduce](`dascore.utils.array_api.nan_reduce`).
    array
        The array to reduce.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.utils.array_api import can_nan_reduce
    >>>
    >>> assert can_nan_reduce("mean", np.array([1.0, 2.0]))
    """
    xp = array_namespace(array)
    if getattr(xp, f"nan{name}", None) is not None:
        return True
    return xp.isdtype(array.dtype, NAN_REDUCE_DTYPES[name])


def nan_reduce(name: str, array: Any, axis=None, keepdims: bool = False) -> Any:
    """
    Reduce an array along an axis, ignoring nans.

    The backend's own implementation is used when it has one, since it is
    both faster and exactly what dascore did before it supported other
    array backends; its promotion rules then apply rather than numpy's.
    Reductions the standard cannot express, such as the minimum of a
    boolean array, are applied by numpy and converted back.

    Parameters
    ----------
    name
        The name of the reduction; one of min, max, mean, std, or sum.
    array
        The array to reduce.
    axis
        The axis, or axes, to reduce along. If None, reduce all of them.
    keepdims
        If True, leave the reduced axes in the output with length one.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.utils.array_api import nan_reduce
    >>>
    >>> array = np.array([1.0, np.nan, 3.0])
    >>> assert nan_reduce("mean", array) == 2.0
    """
    if name not in NAN_REDUCE_DTYPES:
        msg = f"{name} is not a reduction; use {sorted(NAN_REDUCE_DTYPES)}."
        raise ValueError(msg)
    xp = array_namespace(array)
    # The standard has no integer mean or std, and numpy promotes them.
    # This happens before the backend's own function is used so that they
    # all agree about integers.
    if name in {"mean", "std"} and xp.isdtype(array.dtype, "integral"):
        array = xp.astype(array, xp.float64)
    if (func := getattr(xp, f"nan{name}", None)) is not None:
        return func(array, axis=axis, keepdims=keepdims)
    if not xp.isdtype(array.dtype, NAN_REDUCE_DTYPES[name]):
        warn_numpy_fallback(f"nan{name}", backend_name(array))
        out = getattr(np, f"nan{name}")(to_numpy(array), axis=axis, keepdims=keepdims)
        return asarray_like(out, array)
    return _nan_reduce(name, array, axis=axis, keepdims=keepdims)
