"""
Building another package's objects in dascore's place.

A `Patch` subclass holding another package's coordinates, managers and
lazy arrays sets `_provider` to a `Provider` of the functions that build
them. Its constructor and `update` build with it, as do dascore's
`get_coord` and `get_coord_manager` while an operation runs on one of its
patches. See [Interfaces](`dascore/docs/notes/interfaces.qmd`).

This module imports nothing else from dascore but its interfaces, so any
module may import it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

from dascore.core.interfaces import BaseCoord, BaseCoordManager, BaseLazyArray

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True)
class Provider:
    """
    The functions which build another package's objects in dascore's place.

    Each takes the arguments of dascore's function of the same name:
    [`get_coord`](`dascore.core.coords.get_coord`),
    [`get_coord_manager`](`dascore.core.coordmanager.get_coord_manager`)
    and [`lazy_array`](`dascore.lazy.array.lazy_array`), which builds the
    array a patch holds for a recipe. Each runs with no provider active, so
    one that falls back to dascore's function gets dascore's object.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.core.provider import Provider
    >>> from dascore.lazy.array import lazy_array
    >>>
    >>> built = []
    >>> def my_coord(**kwargs):
    ...     built.append(kwargs)  # a package builds its own coordinate here
    ...     return dc.get_coord(**kwargs)
    >>>
    >>> class MyPatch(dc.Patch):
    ...     _provider = Provider(my_coord, dc.get_coord_manager, lazy_array)
    >>>
    >>> out = MyPatch(dc.get_example_patch()).dft("time")
    >>> assert built  # the provider built the transform's new coordinate
    """

    get_coord: Callable[..., BaseCoord]
    get_coord_manager: Callable[..., BaseCoordManager]
    lazy_array: Callable[..., BaseLazyArray]


_ACTIVE: ContextVar[Provider | None] = ContextVar("provider", default=None)


class providing:  # noqa: N801
    """
    Build with a provider inside a block; None is dascore's.

    A class rather than a generator, as `operation_context` is: every patch
    built and every operation run enters one.
    """

    __slots__ = ("_provider", "_token")

    def __init__(self, provider: Provider | None):
        self._provider = provider
        self._token = None

    def __enter__(self):
        """Make the provider active, unless it already is."""
        if _ACTIVE.get() is not self._provider:
            self._token = _ACTIVE.set(self._provider)
        return self

    def __exit__(self, *exception):
        """Make active again what was before."""
        if self._token is not None:
            _ACTIVE.reset(self._token)
        return False


def provided(func: Callable[P, R]) -> Callable[P, R]:
    """Hand a call to the active provider's function of the same name."""
    # A Callable need not have a name; the functions this decorates do.
    name = getattr(func, "__name__")

    @functools.wraps(func)
    def build(*args: P.args, **kwargs: P.kwargs) -> R:
        if (provider := _ACTIVE.get()) is None:
            return func(*args, **kwargs)
        # None active, so the provider falling back to dascore's does not recurse.
        with providing(None):
            return getattr(provider, name)(*args, **kwargs)

    return build
