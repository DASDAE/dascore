"""
Utilities for working with HDF5 files.

Pytables should only be imported in this module in case we need to switch
out the hdf5 backend in the future.
"""
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Union

import tables

from dascore.exceptions import InvalidFileHandler
from dascore.utils.misc import suppress_warnings

HDF5ExtError = tables.HDF5ExtError
NoSuchNodeError = tables.NoSuchNodeError


@contextmanager
def open_hdf5_file(
    path_or_handler: Union[Path, str, tables.File],
    mode: Literal["r", "w", "a"] = "r",
) -> tables.File:
    """
    A helper function for getting a `tables.file.File` object.

    If a file reference (str or Path) is passed this context manager will
    close the file, if a File

    Parameters
    ----------
    path_or_handler
        The input
    mode
        The mode in which to open the file.

    Raises
    ------
    InvalidBuffer if a writable mode is requested from a read only handler.
    """

    def _validate_mode(current_mode, desired_mode):
        """Ensure modes are compatible else raise."""
        if desired_mode == "r":
            return
        # if a or w is desired the current mode should be w
        if not current_mode == "w":
            msg = (
                f"A HDF5 file handler with mode 'r' was provided but "
                f"mode: {desired_mode} was requested."
            )
            raise InvalidFileHandler(msg)

    if isinstance(path_or_handler, (str, Path)):
        # Note: We suppress DataTypeWarnings because pytables fails to read
        # 8 bit enum indicating true or false written by h5py. See:
        # https://github.com/PyTables/PyTables/issues/647
        with suppress_warnings(tables.DataTypeWarning):
            with tables.open_file(path_or_handler, mode) as fi:
                yield fi
    elif isinstance(path_or_handler, tables.File):
        _validate_mode(path_or_handler.mode, mode)
        yield path_or_handler
