"""
Utilities for working with HDF5 files.

Pytables should only be imported in this module in case we need to switch
out the hdf5 backend in the future.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from h5py import Empty  # noqa (we purposely re-import this other places)
from h5py import File as H5pyFile

from dascore.utils.misc import (
    _maybe_make_parent_directory,
    _maybe_unpack,
    unbyte,
)


class H5Reader(H5pyFile):
    """A thin wrapper around the h5py File object for reading."""

    mode = "r"
    constructor = H5pyFile

    @classmethod
    def get_handle(cls, resource):
        """Get the File object from various sources."""
        if isinstance(resource, cls | H5pyFile):
            return resource
        try:
            _maybe_make_parent_directory(resource)
            return cls.constructor(resource, mode=cls.mode)
        except TypeError:
            msg = f"Couldn't get handle from {resource} using {cls}"
            raise NotImplementedError(msg)


class H5Writer(H5Reader):
    """A thin wrapper around h5py for writing files."""

    mode = "a"


def unpack_scalar_h5_dataset(dataset):
    """
    Unpack a scalar H5Py dataset.
    """
    assert dataset.size == 1
    # This gets weird because datasets can be of shape () or (1,).
    value = dataset[()]
    if isinstance(value, np.ndarray):
        value = value[0]
    return value


def h5_matches_structure(h5file: H5pyFile, structure: Sequence[str]):
    """
    Check if an H5 file matches a spec given by a structure.

    Parameters
    ----------
    h5file
        A an open h5file as returned by h5py.File.
    structure
        A sequence of strings which indicates required groups/datasets/attrs.
        For example ("data", "data/raw", "data/raw.sampling") would require
        the 'data' group to exist, the data/raw group/dataset to exist and
        that raw has an attributed called 'sampling'.
    """
    for address in structure:
        split = address.split(".")
        assert len(split) in {1, 2}, "address can have at most one '.'"
        if len(split) == 2:
            base, attr = split
        else:
            base, attr = split[0], None
        try:
            obj = h5file[base]
        except KeyError:
            return False
        if attr is not None and attr not in set(obj.attrs):
            return False
    return True


def extract_h5_attrs(
    h5file: H5pyFile,
    name_map: dict[str, str],
    fill_values=None,
):
    """
    Extract attributes from h5 file based on structure.

    Parameters
    ----------
    h5file
        A an open h5file as returned by h5py.File.
    name_map
        A mapping from {old_name: new_name}. Old name must include one
        dot which separates the path from the attribute name.
        eg {"DasData.SamplingRate": "sampling_rate"}.

    Raises
    ------
    KeyError if any datasets/attributes are missing.
    """
    fill_values = fill_values or {}
    out = {}
    for address, out_name in name_map.items():
        split = address.split(".")
        assert len(split) == 2, "Struct must have exactly one '.'"
        base, attr = split
        obj = h5file[base]
        value = _maybe_unpack(unbyte(obj.attrs[attr]))
        out[out_name] = fill_values.get(value, value)
    return out
