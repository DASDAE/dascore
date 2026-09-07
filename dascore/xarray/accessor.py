"""
The ``.dc`` accessor: DASCore's patch methods on an xarray DataArray.

Importing this module registers the accessor. `patch_to_xarray` and
`spool_to_xarray` import it themselves, so anything DASCore hands back
already carries it; a DataArray from anywhere else needs
``import dascore.xarray.accessor`` first, since registering it means
importing xarray, which DASCore does not do on its own.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from importlib.util import find_spec
from typing import Any

import dascore as dc
from dascore.exceptions import PatchConversionError
from dascore.utils.array_api import is_foreign
from dascore.utils.misc import optional_import
from dascore.warnings import NumpyFallbackWarning
from dascore.xarray.patch import patch_to_xarray, xarray_to_patch

# What is left of memory before an operation which cannot stay lazy is
# refused rather than attempted. Materializing an array is not the only
# thing holding memory, so the whole of what is free is not available.
_MEMORY_HEADROOM = 0.8


def _available_memory() -> int | None:
    """Bytes a materialized array could use, or None if unknowable."""
    try:  # psutil is not a dependency; the guard is skipped without it
        psutil = optional_import("psutil")
    except Exception:
        return None
    return int(psutil.virtual_memory().available * _MEMORY_HEADROOM)


def _too_large_to_materialize(data) -> int | None:
    """The size of an array which would not fit in memory, else None."""
    if not is_foreign(data):
        return None  # already in memory, so materializing costs nothing
    size = getattr(data, "nbytes", None)
    available = _available_memory()
    if size is None or available is None:
        return None
    return int(size) if int(size) > available else None


def _largest_to_materialize(values) -> int | None:
    """
    The size of the largest array in a call which would not fit, else None.

    A conversion converts the arguments too, so an operation is refused
    for an argument which will not fit just as it is for the patch it
    is called on.
    """
    # a bare array is converted like a patch's data, so it weighs the same
    arrays = (x.data if isinstance(x, dc.Patch) else x for x in values)
    sizes = [_too_large_to_materialize(x) for x in arrays]
    found = [x for x in sizes if x is not None]
    return max(found) if found else None


def _as_patch(value):
    """A DataArray as the patch it describes; anything else unchanged."""
    xr = optional_import("xarray")
    return xarray_to_patch(value) if isinstance(value, xr.DataArray) else value


def _as_xarray(value):
    """A patch as the DataArray it describes; anything else unchanged."""
    return patch_to_xarray(value) if isinstance(value, dc.Patch) else value


def _guarded(call: Callable, name: str, size: int):
    """
    Run ``call``, refusing to materialize an array which will not fit.

    An operation which cannot work on the array as it stands says so by
    warning before it converts anything, so raising on that warning
    stops the conversion rather than reporting it afterwards. An
    operation which stays on the array's own backend never warns and is
    never stopped.

    Any such warning during the call is taken to be about this array,
    including one from a callback the caller passed in. Refusing an
    operation which would have fit is the safe way to be wrong here,
    and the message says which array was judged too large.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", NumpyFallbackWarning)
        try:
            return call()
        except NumpyFallbackWarning as error:
            msg = (
                f"'{name}' cannot work on this array without converting it "
                f"to NumPy, and its {size:,} bytes exceed what memory has "
                "free. Select a smaller part of it, or compute it yourself "
                "if the estimate is wrong."
            )
            raise PatchConversionError(msg) from error


class _PatchMethods:
    """
    DASCore's patch methods, on a DataArray.

    Every public name a `Patch` has is reachable here. A call converts
    the DataArray to a patch, runs the method, and converts a patch it
    returns back to a DataArray; a result which is not a patch -- an
    array, a spool, a figure -- comes back as it is. Arguments are
    converted the same way, so a method taking another patch takes
    another DataArray here.

    A dask-backed DataArray is passed through as it stands, so a method
    which works on the array's own backend leaves it lazy. One which
    cannot warns and converts, as it does on a patch, and where it
    announces that conversion and the array is larger than free memory
    the warning is raised instead, so the conversion never starts. A
    method which converts silently, without announcing it, is not
    caught: it materializes the array as it would on a patch, and fails
    on memory the same way.
    """

    def __init__(self, data_array):
        self._data_array = data_array

    def __getattr__(self, name: str) -> Any:
        """Forward one name to the patch this DataArray describes."""
        if name.startswith("_"):
            msg = f"The accessor forwards a patch's public names, not {name!r}."
            raise AttributeError(msg)
        patch = xarray_to_patch(self._data_array)
        try:
            # asked of the patch, not of its class, so a namespace the
            # patch resolves for itself is reachable and a name only its
            # class has is not
            value = getattr(patch, name)
        except AttributeError as error:
            msg = f"Neither this accessor nor a Patch has an attribute {name!r}."
            raise AttributeError(msg) from error
        if not callable(value):
            # a property states something about the patch, so there is no
            # call to convert arguments for; a patch it states is still a
            # patch, and comes back as the DataArray it describes
            return _as_xarray(value)
        return self._method(patch, value, name)

    def _method(self, patch, bound: Callable, name: str) -> Callable:
        """Wrap one patch method so it takes and returns DataArrays."""

        @functools.wraps(bound)
        def call(*args, **kwargs):
            args = tuple(_as_patch(x) for x in args)
            kwargs = {k: _as_patch(v) for k, v in kwargs.items()}
            run = functools.partial(bound, *args, **kwargs)
            # an argument can be the large one, so every patch in the
            # call is weighed, not only the one being called on
            patches = [patch, *(x for x in (*args, *kwargs.values()))]
            size = _largest_to_materialize(patches)
            out = run() if size is None else _guarded(run, name, size)
            return _as_xarray(out)

        # `add` and its kind are objects with methods of their own
        # (`reduce`, `accumulate`); wrapping the call must not hide them
        for extra in ("reduce", "accumulate", "outer", "at"):
            if (nested := getattr(bound, extra, None)) is not None:
                setattr(call, extra, self._method(patch, nested, extra))
        return call

    def __dir__(self) -> list[str]:
        """
        Offer the patch's own names, so completion finds them.

        A namespace is resolved when it is asked for rather than being
        an attribute, so `dir` does not report one; the patch says which
        it has, and they are as reachable here as any method.
        """
        patch = xarray_to_patch(self._data_array)
        names = set(dir(patch)) | set(patch.get_registered_namespaces())
        return sorted(x for x in names if not x.startswith("_"))

    def __repr__(self) -> str:
        """Say what this is and what it wraps."""
        shape = "x".join(str(x) for x in self._data_array.shape)
        return f"DASCore patch methods on a {shape} DataArray"

    def to_patch(self) -> dc.Patch:
        """Return the patch this DataArray describes."""
        return xarray_to_patch(self._data_array)


def register() -> None:
    """
    Register the ``.dc`` accessor on ``xarray.DataArray``.

    Registering what is already registered only warns about replacing
    it, so a second call is skipped; anything else holding the name is
    left alone to warn, since that is a collision worth hearing about.
    """
    xr = optional_import("xarray")
    # A patch is one array, so a DataArray is what it maps onto; a
    # Dataset holding several has no single patch to be.
    if getattr(xr.DataArray, "dc", None) is _PatchMethods:
        return
    xr.register_dataarray_accessor("dc")(_PatchMethods)


# Importing this module must not require xarray -- everything which
# walks the package imports it -- while registering plainly does.
if find_spec("xarray") is not None:
    register()

__all__ = ["register"]
