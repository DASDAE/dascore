"""Core NetCDF IO implementation built on xarray."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

import dascore as dc
from dascore.constants import snap_type, windows_type
from dascore.exceptions import MissingOptionalDependencyError
from dascore.io import FiberIO
from dascore.io.utils import resolve_keyed_source
from dascore.io.xdas.utils import is_xdas_file
from dascore.utils.hdf5 import H5Reader, get_h5py_file
from dascore.utils.io import BinaryReader
from dascore.utils.misc import optional_import, suppress_warnings
from dascore.xarray.patch import _cf_units

from .utils import (
    dataset_to_patch_meta,
    get_cf_version,
    get_xarray_data_var_name,
    is_netcdf4_file,
    parse_cf_version,
    payload_datasets,
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


def _open_classic_dataset(resource: BinaryReader):
    """Open a netCDF-3 file as an xarray dataset; its arrays are read whole."""
    return optional_import("xarray").open_dataset(resource, engine="scipy")


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


def _take_file_units(dataset):
    """
    Copy the root ``units`` onto a payload with neither ``units`` nor
    ``data_units``, in place, when they read as CF units (``m s-1``).
    """
    name = get_xarray_data_var_name(dataset)
    attrs, units = dataset[name].attrs, dataset.attrs.get("units")
    if {"units", "data_units"} & set(attrs) or not isinstance(units, str):
        return
    with suppress_warnings(UserWarning):  # text which is not units stays out
        if _cf_units(units, name) is not None:
            attrs["units"] = units


def _states_cf(obj) -> bool:
    """Return True when the attrs of ``obj`` state CF 1.6 or later."""
    try:
        return parse_cf_version(get_cf_version(obj) or "") >= (1, 6)
    except (TypeError, ValueError):
        return False


def _takes_cf(resource: H5Reader) -> bool:
    """Return True for a non-XDAS netCDF-4 file stating CF 1.6 or later."""
    if not is_netcdf4_file(resource) or is_xdas_file(resource):
        return False
    return _states_cf(resource)


class _NetCDFReader:
    """Read netCDF through xarray, one patch per payload variable."""

    # Whether a payload with no units takes the file's root ``units`` when
    # they parse and the file does not state CF.
    _file_units = False
    _open = staticmethod(_open_xarray_dataset)

    def _read_array(self, resource, windows: windows_type, key: str) -> np.ndarray:
        """
        Slice the keyed payload variable through xarray.

        The selection goes through xarray rather than the stored dataset
        so CF decoding (scaling, offsets, fill values) applies exactly as
        it does in `read`.
        """
        with self._open(resource) as dataset:
            where = str(getattr(resource, "filename", "the resource"))
            payloads = payload_datasets(dataset)
            if len(payloads) > 1:
                dataset = resolve_keyed_source(payloads, key, where)
            else:
                (dataset,) = payloads.values()
            return read_dataset_array(dataset, windows, key, where)

    def _get_metadata(self, resource, snap: snap_type) -> list[dc.PatchMeta]:
        """Describe each payload of the file without loading its array."""
        out = []
        with self._open(resource) as dataset:
            for payload in payload_datasets(dataset).values():
                if self._file_units and not _states_cf(dataset):
                    _take_file_units(payload)
                out.extend(dataset_to_patch_meta(payload, snap))
        return out


class _HDF5NetCDFReader(_NetCDFReader):
    """Read netCDF-4 (HDF5) files."""

    def read_array(
        self, resource: H5Reader, windows: windows_type = (), key: str = ""
    ) -> np.ndarray:
        """Slice the keyed payload variable through xarray."""
        return self._read_array(resource, windows, key)

    def get_metadata(
        self, resource: H5Reader, *, snap: snap_type = ()
    ) -> list[dc.PatchMeta]:
        """Scan NetCDF file metadata, one per payload, without loading arrays.

        Remote resources are streamed via the ``h5netcdf`` engine over the
        existing ``h5py`` handle, so only metadata bytes are fetched and the
        file is not downloaded.
        """
        return self._get_metadata(resource, snap)


class NetCDFCFV18(_HDF5NetCDFReader, FiberIO):
    """NetCDF-4 IO using xarray for read/write and CF markers for detection."""

    name = "NETCDF_CF"
    version = "1.8"
    preferred_extensions = ("nc", "nc4", "netcdf")

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        return self.version if _takes_cf(resource) else None

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


class NetCDFNoCF(_HDF5NetCDFReader, FiberIO):
    """
    NetCDF-4 files version 1.8 does not take, read as NETCDF_CF version 0.

    These state no CF version, or one older than 1.6. Version 0 is a
    fallback, tried after every other format, and claims only a file
    written through netCDF (``_NCProperties``) with a dimensioned root
    variable. It reads like version 1.8, except that a payload with no
    units takes the file's root ``units`` when they parse; it does not
    write.
    """

    name = "NETCDF_CF"
    version = "0"
    fallback = True
    # Files outside CF may state their payloads' units once, on the file.
    _file_units = True

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return "0" for a netCDF-4 file that version 1.8 does not take."""
        if "_NCProperties" not in resource.attrs or _takes_cf(resource):
            return None
        has_variable = any("DIMENSION_LIST" in resource[x].attrs for x in resource)
        return self.version if has_variable else None


class NetCDF3V1(_NetCDFReader, FiberIO):
    """
    netCDF-3 classic and 64-bit offset files, read like NETCDF_CF.

    A file stating CF 1.6 or later reads as NETCDF_CF version 1.8 does;
    others as version 0, taking the file's units. Arrays are read whole.
    """

    name = "NETCDF3"
    version = "1"
    preferred_extensions = ("nc",)
    _file_units = True
    _open = staticmethod(_open_classic_dataset)

    def get_version(self, resource: BinaryReader, **kwargs) -> str | None:
        """Return "1" for a file starting with the netCDF-3 magic bytes."""
        return self.version if resource.read(4) in (b"CDF\x01", b"CDF\x02") else None

    def read_array(
        self, resource: BinaryReader, windows: windows_type = (), key: str = ""
    ) -> np.ndarray:
        """Slice the keyed payload variable."""
        return self._read_array(resource, windows, key)

    def get_metadata(
        self, resource: BinaryReader, *, snap: snap_type = ()
    ) -> list[dc.PatchMeta]:
        """Describe each payload variable without slicing it."""
        return self._get_metadata(resource, snap)
