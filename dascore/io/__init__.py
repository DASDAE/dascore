"""
Modules for reading and writing fiber data.
"""
from __future__ import annotations

import dascore.utils.pd
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
from dascore.utils.misc import MethodNameSpace
from dascore.utils.pd import dataframe_to_patch, patch_to_dataframe
from dascore.utils.io import (
    xarray_to_patch,
    patch_to_xarray,
    patch_to_obspy,
    obspy_to_patch,
)


class PatchIO(MethodNameSpace):
    write = write
    to_dataframe = patch_to_dataframe
    to_xarray = patch_to_xarray
    to_obspy = patch_to_obspy
