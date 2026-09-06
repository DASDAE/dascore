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
    cannot warns and converts, as it does on a patch, unless the array
    is too large for memory, in which case it raises instead of trying.
    """

    def __init__(self, data_array):
        self._data_array = data_array

    def __getattr__(self, name: str) -> Any:
        """Forward one name to the patch this DataArray describes."""
        if name.startswith("_") or not hasattr(dc.Patch, name):
            msg = f"Neither this accessor nor a Patch has an attribute {name!r}."
            raise AttributeError(msg)
        patch = xarray_to_patch(self._data_array)
        value = getattr(patch, name)
        if not callable(value):
            # a property states something about the patch; there is no
            # call to convert arguments for, and no patch to convert back
            return value
        return self._method(patch, value, name)

    def _method(self, patch, bound: Callable, name: str) -> Callable:
        """Wrap one patch method so it takes and returns DataArrays."""

        @functools.wraps(bound)
        def call(*args, **kwargs):
            args = tuple(_as_patch(x) for x in args)
            kwargs = {k: _as_patch(v) for k, v in kwargs.items()}
            run = functools.partial(bound, *args, **kwargs)
            size = _too_large_to_materialize(patch.data)
            out = run() if size is None else _guarded(run, name, size)
            return _as_xarray(out)

        return call

    def __dir__(self) -> list[str]:
        """Offer the patch's own names, so completion finds them."""
        return sorted(x for x in dir(dc.Patch) if not x.startswith("_"))

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


register()

__all__ = ["register"]
