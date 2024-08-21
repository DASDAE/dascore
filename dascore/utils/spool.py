"""Utilities for working with the Patch class."""

from __future__ import annotations

import functools

from dascore.constants import SpoolType


def spool_function():
    """
    Decorator to mark a function as a spool method.
    """

    def _wrapper(func):
        @functools.wraps(func)
        def _func(spool, *args, **kwargs):
            out: SpoolType = func(spool, *args, **kwargs)
            return out

        _func.func = func  # attach original function
        _func.__wrapped__ = func

        return _func

    return _wrapper
