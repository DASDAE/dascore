"""
Compatibility module for DASCore.

All components/functions that may be exchanged for other numpy/scipy
compatible libraries should go in this model.
"""

from __future__ import annotations

import warnings
from contextlib import suppress
from functools import wraps

import numpy as np
from numpy import floor, interp  # NOQA
from numpy.random import RandomState
from scipy.interpolate import interp1d  # NOQA
from scipy.ndimage import zoom  # NOQA
from scipy.signal import decimate, resample, resample_poly  # NOQA

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


def maybe_jit(compiler="numba", required=False, **compiler_kwargs):
    """
    A decorator for applying a Jit if the compiler module is installed.

    Parameters
    ----------
    compiler
        The name of the module that needs to be installed.
    required
        If True an ImportError is raised if the wrapped function is called
        and the compiler module is not installed. If False, issue a warning.
    **compiler_kwargs
        Keyword arguments passed to the compiler function.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.compat import maybe_jit
    >>>
    >>> # JIT function for fast numba compilation
    >>> @maybe_jit(nopython=True, nogil=True)
    ... def jit_func(array):
    ...     ...
    >>>
    >>> # Using objects from the numba module.
    >>> # This is a bit more tricky and requires a wrapper function. Cache
    >>> # Can make this easier.
    >>> from functools import cache
    >>> from dascore.utils.misc import optional_import
    >>>
    >>> @cache
    ... def jit_wrapper():
    ...     numba = optional_import("numba")
    ...
    ...     @maybe_jit
    ...     def jit_func(array):
    ...         for a in numba.prange(len(array)):
    ...             ...
    ...
    ...     return jit_func
    >>>
    >>> # An extra call is required to use the jit'ed function but cache
    >>> # makes it nearly free.
    >>> out = jit_wrapper()(np.array([1,2,3]))

    Notes
    -----
    By design, the warning/exception is only raised when calling the wrapped
    function. Users who don't use the function should be unaffected.
    """
    # This happens when the decorator is used without parens; use default values
    if callable(compiler):
        return maybe_jit()(compiler)

    if compiler != "numba":
        raise NotImplementedError("Only number supported for now.")

    def _wraper(func):
        try:
            import numba
        except ImportError:

            @wraps(func)
            def decorated(*args, **kwargs):
                if required:
                    msg = (
                        f"{func.__name__} requires python module "
                        f"{compiler} but it is not installed. "
                    )
                    raise ImportError(msg)
                else:
                    msg = (
                        f"{func.__name__} can be compiled to improve performance. "
                        f"Please install {compiler} to enable JIT."
                    )
                    warnings.warn(msg, UserWarning)
                return func(*args, **kwargs)

            return decorated
        return numba.jit(**compiler_kwargs)(func)

    return _wraper
