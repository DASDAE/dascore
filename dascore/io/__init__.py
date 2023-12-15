"""
Modules for reading and writing fiber data.
"""
from __future__ import annotations
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


class PatchIO(MethodNameSpace):
    write = write
