"""
Building another package's objects, and reading with its readers, in dascore's place.

A `Patch` subclass holding another package's coordinates, managers and
lazy arrays sets `_provider` to a `Provider` of the functions that build
them. Its constructor and `update` build with it, as do dascore's
`get_coord` and `get_coord_manager` while an operation runs on one of its
patches. A package reading some of dascore's formats itself states
`Readers`, which dascore's reads and scans ask first inside a
`reading_with` block. See [Interfaces](`dascore/docs/notes/interfaces.qmd`).

This module imports nothing else from dascore but its interfaces, so any
module may import it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ParamSpec, TypeVar

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


@dataclass(frozen=True)
class Readers:
    """
    Another package's readers of some of dascore's formats.

    While they are active ([`reading_with`](`dascore.core.provider.reading_with`)),
    dascore asks them before its own readers, and each function returns
    None to leave the question to dascore. A read or a scan is still
    dascore's: its walk, selection, validation, ids and errors.

    - `get_format(manager, file_format, file_version, fiber_io_hint)`
      returns the `(format, version)` of the resource a manager holds, as
      [`get_format`](`dascore.io.core.get_format`) finds it.
    - `prepare_read(fiber_io, manager, snap, keys)` returns what
      `FiberIO._prepare_read` does for the resource: its patches' metadata,
      and a loader of their arrays. A scan takes the metadata. The loader
      fills the arrays a read loads; a backend which reads for itself, as
      the lazy backend does, is handed each stored window instead and
      reads it with dascore's reader, so the readers answer only for files
      that reader reads too.
    - `reads(fiber_io, manager)` returns whether `prepare_read` answers for
      the resource, so that a scan hands it over unopened.

    A function which reads a resource it takes from the manager, and then
    leaves the file to dascore, rewinds it first. Errors are the package's:
    a scan skips a file `get_format` raises `UnknownFiberFormatError` for,
    as it skips one no reader of dascore's reads, and a file whose
    `prepare_read` raises, as it skips one whose reader fails.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.core.provider import Readers, reading_with
    >>>
    >>> asked = []
    >>> def get_format(manager, *args):
    ...     asked.append(manager.source)  # a package detects the format here
    ...     return None  # and leaves it to dascore when it can't
    >>>
    >>> # Readers which read nothing themselves.
    >>> readers = Readers(get_format, lambda *args: None, lambda *args: False)
    >>> with reading_with(readers):
    ...     spool = dc.read("examples://prodml_2.1.h5")
    >>> assert asked
    """

    get_format: Callable[..., tuple[str, str] | None]
    prepare_read: Callable[..., Any]
    reads: Callable[..., bool]


_READERS: ContextVar[Readers | None] = ContextVar("readers", default=None)


@contextmanager
def reading_with(readers: Readers | None) -> Iterator[None]:
    """Read with another package's readers inside the block; None is dascore's."""
    token = _READERS.set(readers)
    try:
        yield
    finally:
        _READERS.reset(token)


def active_readers() -> Readers | None:
    """The readers `reading_with` made active, or None."""
    return _READERS.get()
