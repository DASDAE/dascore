"""
Modules for reading and writing fiber data.
"""
from __future__ import annotations

import dascore.utils.pd
from dascore.core.accessor import register_patch_accessor
from dascore.io.core import FiberIO, read, scan, scan_to_df, write, PatchFileSummary
from dascore.utils.io import BinaryReader, BinaryWriter
from dascore.utils.hdf5 import (
    HDF5Writer,
    HDF5Reader,
    PyTablesWriter,
    PyTablesReader,
    H5Reader,
    H5Writer,
)
from dascore.utils.pd import dataframe_to_patch, patch_to_dataframe
from dascore.utils.io import (
    xarray_to_patch,
    patch_to_xarray,
    patch_to_obspy,
    obspy_to_patch,
)


@register_patch_accessor("io")
class PatchIO:
    """IO namespace for Patch."""

    def __init__(self, patch):
        self._patch = patch

    def write(self, *args, **kwargs):
        """See [`write`](`dascore.io.core.write`)."""
        return write(self._patch, *args, **kwargs)

    def to_dataframe(self, *args, **kwargs):
        """See [`patch_to_dataframe`](`dascore.utils.pd.patch_to_dataframe`)."""
        return patch_to_dataframe(self._patch, *args, **kwargs)

    def to_xarray(self, *args, **kwargs):
        """See [`patch_to_xarray`](`dascore.utils.io.patch_to_xarray`)."""
        return patch_to_xarray(self._patch, *args, **kwargs)

    def to_obspy(self, *args, **kwargs):
        """See [`patch_to_obspy`](`dascore.utils.io.patch_to_obspy`)."""
        return patch_to_obspy(self._patch, *args, **kwargs)
