"""Core NetCDF IO implementation built on xarray."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Literal

import numpy as np

import dascore as dc
from dascore.exceptions import MissingOptionalDependencyError
from dascore.io import FiberIO
from dascore.io.core import ScanPayload, make_scan_payload
from dascore.io.utils import (
    get_exact_coord,
    resolve_keyed_source,
    windows_to_slices,
)
from dascore.utils.hdf5 import H5Reader, get_h5py_file
from dascore.utils.io import patch_to_xarray, xarray_to_patch
from dascore.utils.misc import optional_import, raise_on_extra_kwargs

from .utils import (
    XDAS_PAYLOAD_VARIABLE,
    get_cf_version,
    get_coord_manager_for_coordless_data_var,
    get_xarray_data_var_name,
    is_netcdf4_file,
    parse_cf_version,
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

    def get_format(
        self,
        resource: H5Reader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """Return format tuple if file is a CF-convention NetCDF-4, else False."""
        if not is_netcdf4_file(resource):
            return False
        cf_version = get_cf_version(resource)
        if not cf_version:
            return False
        try:
            if parse_cf_version(cf_version) >= (1, 6):
                return self.name, self.version
        except (TypeError, ValueError):
            pass
        return False

    def read(self, resource: H5Reader, **kwargs) -> dc.Spool:
        """Read a NetCDF-4 file into a Spool, streaming remote resources."""
        with _open_xarray_dataset(resource) as dataset:
            data_var_name = get_xarray_data_var_name(dataset)
            data_array = dataset[data_var_name].load()
            patch = self._patch_from_dataset(dataset, data_var_name, data_array)
        patch = self._select_from_kwargs(patch, kwargs)
        if not patch.data.size:
            return dc.spool([])
        return dc.spool([patch])

    def read_array(
        self,
        resource: H5Reader,
        windows: dict[str, tuple[int, int]],
        source_patch_key="",
        **kwargs,
    ) -> np.ndarray:
        """
        Slice the payload variable through xarray.

        The selection goes through xarray rather than the stored dataset
        so CF decoding (scaling, offsets, fill values) applies exactly as
        it does in `read`.
        """
        raise_on_extra_kwargs(kwargs, "windows and source_patch_key")
        with _open_xarray_dataset(resource) as dataset:
            data_var_name = get_xarray_data_var_name(dataset)
            resolve_keyed_source(
                {self._get_source_patch_key(data_var_name): data_var_name},
                source_patch_key,
                where=str(getattr(resource, "filename", "the resource")),
            )
            data_array = dataset[data_var_name]
            slices = windows_to_slices(windows, data_array.dims, data_array.shape)
            return data_array[slices].to_numpy()

    def _get_write_encoding(self, **kwargs):
        """Translate explicit write options into xarray encoding hints."""
        compression = kwargs.get("compression")
        if compression not in ("gzip", None, False):
            msg = "xarray netcdf4 writing currently supports only gzip compression."
            raise ValueError(msg)
        chunks = kwargs.get("chunks")
        encoding: dict[str, object] = {}
        if chunks not in (None, False, True):
            encoding["chunksizes"] = tuple(chunks)
        if compression == "gzip":
            encoding["zlib"] = True
            encoding["complevel"] = kwargs.get("compression_opts", 4)
            encoding["shuffle"] = True
        return encoding

    def write(self, spool: dc.Patch | dc.Spool, resource: Path, **kwargs) -> None:
        """
        Write a Spool to NetCDF-4 through xarray.

        Parameters
        ----------
        kwargs
            compression: 'gzip', None, or False
            compression_opts: gzip level 1-9 (default 4)
            chunks: True to defer chunking to xarray/backend defaults, or an
                explicit tuple of chunk sizes
        """
        patch = self._validate_and_extract_patch(spool)
        optional_import("xarray")  # raises a helpful error if xarray is absent
        _require_hdf5_netcdf_backend()
        array = patch_to_xarray(patch).rename("data")
        # netCDF has no boolean attribute type, so a bool attr aborts the
        # write. Inventory enrichment routinely sets one
        # (closed_fiber_loop), and CF's own convention for a flag is an
        # integer, so they are stored as 0/1 rather than dropped. Every
        # patch attr lands on the data variable; the dataset carries only
        # what is set below.
        array.attrs = _int_for_bool(array.attrs)
        dataset = array.to_dataset()
        dataset.attrs["Conventions"] = f"CF-{self.version}"
        encoding = self._get_write_encoding(**kwargs)
        dataset.to_netcdf(
            resource,
            encoding={"data": encoding} if encoding else None,
        )

    def scan(
        self, resource: H5Reader, snap: bool = True, **kwargs
    ) -> list[ScanPayload]:
        """Scan NetCDF file metadata without loading the full payload array.

        Remote resources are streamed via the ``h5netcdf`` engine over the
        existing ``h5py`` handle, so only metadata bytes are fetched and the
        file is not downloaded.
        """
        with _open_xarray_dataset(resource) as dataset:
            data_var_name = get_xarray_data_var_name(dataset)
            # None is a valid xarray key for XDAS-style files whose primary
            # payload is stored under a None variable name.
            data_array = dataset[data_var_name]
            coords = {
                name: (coord.dims, self._get_scan_coord(coord, snap=snap))
                for name, coord in data_array.coords.items()
            }
            attrs = dict(data_array.attrs)
            dims = data_array.dims
            shape = data_array.shape
            dtype = str(data_array.dtype)
            source_patch_key = self._get_source_patch_key(data_var_name)
            coord_manager = self._coord_manager_from_data_array(
                dataset, data_array, coords, dims, shape
            )
        return [
            make_scan_payload(
                attrs=attrs | {"_source_patch_key": source_patch_key},
                coords=coord_manager,
                dims=dims,
                shape=shape,
                dtype=dtype,
                source_patch_key=source_patch_key,
            )
        ]

    @staticmethod
    def _get_scan_coord(coord, snap=True):
        """Return a coordinate for scanning; snap only controls exactness."""
        values = coord.values
        if np.ndim(values) != 1:
            return values
        units = coord.attrs.get("units")
        if snap:
            return dc.core.get_coord(data=values, units=units)
        return get_exact_coord(values, units=units)

    def _get_source_patch_key(self, data_var_name):
        """Normalize the selected xarray payload name to a patch id."""
        return XDAS_PAYLOAD_VARIABLE if data_var_name is None else data_var_name

    def _coord_manager_from_data_array(self, dataset, data_array, coords, dims, shape):
        """Return coords from xarray when present or reconstruct dim coords."""
        if coords:
            return dc.get_coord_manager(coords=coords, dims=dims)
        return get_coord_manager_for_coordless_data_var(dataset, dims=dims, shape=shape)

    def _patch_from_dataset(self, dataset, data_var_name, data_array):
        """Build one patch from an xarray dataset and selected data variable."""
        source_patch_key = self._get_source_patch_key(data_var_name)
        attrs = dict(data_array.attrs) | {"_source_patch_key": source_patch_key}
        if data_array.coords:
            return xarray_to_patch(data_array).update(attrs=attrs)
        coords = self._coord_manager_from_data_array(
            dataset,
            data_array,
            coords={},
            dims=data_array.dims,
            shape=data_array.shape,
        )
        return dc.Patch(
            data=data_array.data,
            coords=coords,
            dims=data_array.dims,
            attrs=attrs,
        )

    def _select_from_kwargs(self, patch: dc.Patch, kwargs: dict) -> dc.Patch:
        """Apply coordinate selection kwargs to one loaded patch."""
        coord_kwargs = {k: v for k, v in kwargs.items() if k in patch.coords.coord_map}
        return patch.select(**coord_kwargs) if coord_kwargs else patch

    def _validate_and_extract_patch(self, spool: dc.Patch | dc.Spool) -> dc.Patch:
        """Validate write input and return the single supported patch."""
        patches = [spool] if isinstance(spool, dc.Patch) else list(spool)
        if len(patches) == 0:
            msg = "Cannot write empty spool"
            raise ValueError(msg)
        if len(patches) > 1:
            msg = "Multi-patch spools not yet supported for NetCDF output"
            raise NotImplementedError(msg)
        return patches[0]
