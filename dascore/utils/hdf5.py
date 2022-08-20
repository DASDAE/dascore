"""
Utilities for working with HDF5 files.

Pytables should only be imported in this module in case we need to switch
out the hdf5 backend in the future.
"""
from contextlib import contextmanager

import tables as tb

from dascore.utils.misc import suppress_warnings

HDF5ExtError = tb.HDF5ExtError
NoSuchNodeError = tb.NoSuchNodeError


@contextmanager
def open_hdf5_file(path, mode="r"):
    """Open a hdf5 file and return a reference to it."""
    # Note: We suppress DataTypeWarnings because pytables fails to read
    # 8 bit enum indicating true or false written by h5py. See:
    # https://github.com/PyTables/PyTables/issues/647
    with suppress_warnings(tb.DataTypeWarning):
        with tb.open_file(path, mode) as fi:
            yield fi
    return
