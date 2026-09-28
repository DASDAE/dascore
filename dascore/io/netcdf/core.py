"""Core NetCDF IO implementation built on xarray."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

import dascore as dc
from dascore.constants import snap_type, windows_type
from dascore.exceptions import MissingOptionalDependencyError
from dascore.io import FiberIO
from dascore.utils.hdf5 import H5Reader, get_h5py_file
from dascore.utils.misc import optional_import

from .utils import (
    dataset_to_patch_meta,
    get_cf_version,
    is_netcdf4_file,
    parse_cf_version,
    read_dataset_array,
    spool_to_cf_dataset,
)

# netCDF engines that write the HDF5-based files the reader can open.
_HDF5_NETCDF_ENGINES = ("netCDF4", "h5netcdf")


def _require_hdf5_netcdf_backend() -> None:
    """
    Raise a helpful error when no HDF5-capable netCDF engine is installed.

    Without one, xarray's ``to_netcdf`` silently falls back to its scipy
    backend and writes NETCDF3 classic — a file this module's own
    HDF5-based reader and format detection cannot open. Refusing to write
    beats producing an archive DASCore cannot round trip.
    """
    if any(importlib.util.find_spec(name) for name in _HDF5_NETCDF_ENGINES):
        return
    msg = (
        "Writing netcdf_cf requires an HDF5-capable netCDF engine; install "
        "h5netcdf (or netCDF4), e.g. 'pip install h5netcdf'. Without one, "
        "xarray would write NETCDF3 classic, which DASCore cannot read back."
    )
    raise MissingOptionalDependencyError(msg)


def _open_xarray_dataset(resource: H5Reader):
    """
    Open one NetCDF-4 resource as an xarray dataset without downloading it.

    NetCDF-4 is HDF5, so DASCore hands the streaming ``h5py`` handle it already
    uses for format detection to xarray's ``h5netcdf`` engine. Remote resources
    are read over the network via ``h5py`` range requests (the same streaming
    and no-range fallback path as remote HDF5) rather than being materialized
    locally first. The ``netCDF4`` C engine is not used here because it requires
    a real filesystem path.

    The returned dataset owns only the ``h5netcdf`` wrapper; closing it does not
    close DASCore's underlying ``h5py`` handle, which stays owned by the
    ``IOResourceManager``.
    """
    xr = optional_import("xarray")
    # h5netcdf is the streaming-capable engine; import here for a clear error.
    optional_import("h5netcdf")
    h5_file = get_h5py_file(resource)
    return xr.open_dataset(h5_file, engine="h5netcdf")


def _int_for_bool(attrs: dict) -> dict:
    """Return attrs with booleans as the integer flags netCDF can store.

    Sequences and arrays of booleans hit the same netCDF limit as scalars,
    so they are converted whole rather than left to abort the write.
    """

    def convert(value):
        if isinstance(value, bool | np.bool_):
            return int(value)
        if isinstance(value, np.ndarray) and value.dtype == bool:
            return value.astype(int)
        if isinstance(value, list | tuple) and any(
            isinstance(x, bool | np.bool_) for x in value
        ):
            return type(value)(
                int(x) if isinstance(x, bool | np.bool_) else x for x in value
            )
        return value

    return {i: convert(v) for i, v in attrs.items()}


class NetCDFCFV18(FiberIO):
    """NetCDF-4 IO using xarray for read/write and CF markers for detection."""

    name = "NETCDF_CF"
    version = "1.8"
    preferred_extensions = ("nc", "nc4", "netcdf")

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        if not is_netcdf4_file(resource):
            return None
        cf_version = get_cf_version(resource)
        if not cf_version:
            return None
        try:
            if parse_cf_version(cf_version) >= (1, 6):
                return self.version
        except (TypeError, ValueError):
            pass
        return None

    def read_array(
        self, resource: H5Reader, windows: windows_type = (), key: str = ""
    ) -> np.ndarray:
        """
        Slice the payload variable through xarray.

        The selection goes through xarray rather than the stored dataset
        so CF decoding (scaling, offsets, fill values) applies exactly as
        it does in `read`.
        """
        with _open_xarray_dataset(resource) as dataset:
            where = str(getattr(resource, "filename", "the resource"))
            return read_dataset_array(dataset, windows, key, where)

    def write(self, spool, resource: Path, encoding: dict | None = None, **kwargs):
        """
        Write a Spool to NetCDF-4 through xarray.

        Parameters
        ----------
        encoding
            Passed to xarray's ``Dataset.to_netcdf``.
        """
        optional_import("xarray")  # raises a helpful error if xarray is absent
        _require_hdf5_netcdf_backend()
        dataset = spool_to_cf_dataset(spool)
        # netCDF has no boolean attribute type, so a bool attr aborts the
        # write. Inventory enrichment routinely sets one
        # (closed_fiber_loop), and CF's own convention for a flag is an
        # integer, so they are stored as 0/1 rather than dropped.
        dataset["data"].attrs = _int_for_bool(dataset["data"].attrs)
        dataset.to_netcdf(resource, encoding=encoding)

    def get_metadata(
        self, resource: H5Reader, *, snap: snap_type = True
    ) -> list[dc.PatchMeta]:
        """Scan NetCDF file metadata without loading the full payload array.

        Remote resources are streamed via the ``h5netcdf`` engine over the
        existing ``h5py`` handle, so only metadata bytes are fetched and the
        file is not downloaded.
        """
        with _open_xarray_dataset(resource) as dataset:
            return dataset_to_patch_meta(dataset, snap)
