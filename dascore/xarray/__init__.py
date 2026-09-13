"""
xarray integration: conversions between dascore and xarray objects.

Everything here treats xarray (and dask, for the spool tree) as an
optional dependency: the submodules import them lazily, except
`dascore.xarray.index`, whose classes subclass xarray classes and which
is therefore only imported behind an optional-import guard.
"""

from __future__ import annotations

from dascore.xarray.patch import patch_to_xarray, xarray_to_patch
from dascore.xarray.spool import spool_to_xarray

__all__ = ["patch_to_xarray", "spool_to_xarray", "xarray_to_patch"]
