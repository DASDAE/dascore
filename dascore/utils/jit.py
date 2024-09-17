"""
Module for applying just in time compilation to speed up functions.
"""

from __future__ import annotations

import warnings
from functools import wraps


class _DummyNumba:
    """A simple class for acting like numba when numba is not installed."""

    # This isn't intended to make everything work, only basic stuff needed
    # for tests to pass when numba isn't installed.
    def prange(self, count):
        """A mock of numba's prange."""
        yield from range(count)


def maybe_numba_jit(required=False, _missing_numba=False, **compiler_kwargs):
    """
    Use numba to apply JIT compilation to the decorated function.

    Parameters
    ----------
    required
        If True an ImportError is raised if the wrapped function is called
        and the compiler module is not installed. If False, issue a warning.
    _missing_numba
        If true, simulate missing the numba package. Only used for testing.
    **compiler_kwargs
        Keyword arguments passed to the compiler function.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.utils.jit import maybe_numba_jit
    >>>
    >>> # JIT function for fast numba compilation
    >>> @maybe_numba_jit(nopython=True, nogil=True)
    ... def jit_func(array):
    ...     ...
    >>>
    >>> # You can use the numba module inside the function without importing
    >>> # it because numba is added to the functions globals by the decorator.
    >>> # To get linters to ignore this seemingly undefined variable use #noqa
    >>> @maybe_numba_jit
    ... def jit_func(array):
    ...     for a in numba.prange(len(array)):  # noqa
    ...         pass
    ...     return array
    >>>

    >>> out = jit_func(np.array([1,2,3]))

    Notes
    -----
    - By design, the warning/exception is only raised when calling the wrapped
      function. Users who don't use the function should be unaffected.
    - After the jit the original function can be accessed via the `func`
      attribute. This is useful to testing in python mode.
    """
    # This happens when the decorator is used without parens; use default values
    if callable(required):
        return maybe_numba_jit()(required)

    has_numba = True
    try:
        import numba

        if _missing_numba:
            raise ImportError("Simulating missing numba.")
    except ImportError:
        numba = _DummyNumba()
        has_numba = False

    def _wraper(func):
        # Add numba to the functions global namespace so that it can be used
        # within the function without having to import it.
        globs = getattr(func, "__globals__", {})
        globs["numba"] = numba

        if not has_numba:

            @wraps(func)
            def decorated(*args, **kwargs):
                if required:
                    msg = (
                        f"{func.__name__} requires python module "
                        f"numba but it is not installed. "
                    )
                    raise ImportError(msg)
                else:
                    msg = (
                        f"{func.__name__} can be compiled to improve performance. "
                        f"Please install numba to enable JIT."
                    )
                    warnings.warn(msg, UserWarning)
                return func(*args, **kwargs)

            out_func = decorated
        else:
            out_func = numba.jit(**compiler_kwargs)(func)
        out_func.func = func  # make original func accessible via .func
        return out_func

    return _wraper
