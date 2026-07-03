"""
Modules for reading and writing fiber data.
"""
from __future__ import annotations

import dascore.utils.pd
from dascore.io.codec import get_codec, get_codec_registry
from dascore.io.core import (
    BaseCodec,
    BaseStorage,
    FiberIO,
    PatchFileSummary,
    get_codecs,
    get_storage,
    read,
    scan,
    scan_to_df,
    write,
)
from dascore.utils.hdf5 import (
    H5Reader,
    H5Writer,
    HDF5Reader,
    HDF5Writer,
    PyTablesReader,
    PyTablesWriter,
)
from dascore.utils.io import (
    BinaryReader,
    BinaryWriter,
    obspy_to_patch,
    patch_to_obspy,
    patch_to_xarray,
    xarray_to_patch,
)
from dascore.utils.namespace import PatchNameSpace
from dascore.utils.pd import dataframe_to_patch, patch_to_dataframe


class PatchIO(PatchNameSpace):
    name = "io"

    write = write
    to_dataframe = patch_to_dataframe
    to_xarray = patch_to_xarray
    to_obspy = patch_to_obspy
