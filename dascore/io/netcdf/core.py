"""Core NetCDF IO implementation built on xarray."""

from __future__ import annotations

from pathlib import Path

import dascore as dc
from dascore.constants import SpoolType
from dascore.io import FiberIO
from dascore.io.core import ScanPayload, _make_scan_payload
from dascore.utils.hdf5 import H5Reader
from dascore.utils.io import patch_to_xarray, xarray_to_patch
from dascore.utils.misc import optional_import

from .utils import (
    XDAS_PAYLOAD_VARIABLE,
    get_cf_version,
    get_coord_manager_for_coordless_data_var,
    get_xarray_data_var_name,
    is_netcdf4_file,
    parse_cf_version,
)


class NetCDFCFV18(FiberIO):
    """NetCDF-4 IO using xarray for read/write and CF markers for detection."""

    name = "NETCDF_CF"
    version = "1.8"
    preferred_extensions = ("nc", "nc4", "netcdf")

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
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

    def read(self, resource: Path, **kwargs) -> SpoolType:
        """Read a NetCDF-4 file into a Spool."""
        xr = optional_import("xarray")
        with xr.open_dataset(resource) as dataset:
            data_var_name = get_xarray_data_var_name(dataset)
            data_array = dataset[data_var_name].load()
            patch = self._patch_from_dataset(dataset, data_var_name, data_array)
        patch = self._select_from_kwargs(patch, kwargs)
        if not patch.data.size:
            return dc.spool([])
        return dc.spool([patch])

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

    def write(self, spool: SpoolType, resource: Path, **kwargs) -> None:
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
        dataset = patch_to_xarray(patch).rename("data").to_dataset()
        dataset.attrs["Conventions"] = f"CF-{self.version}"
        encoding = self._get_write_encoding(**kwargs)
        dataset.to_netcdf(
            resource,
            encoding={"data": encoding} if encoding else None,
        )

    def scan(self, resource: H5Reader, **kwargs) -> list[ScanPayload]:
        """Scan NetCDF file metadata without loading the full payload array."""
        xr = optional_import("xarray")
        dataset_path = resource.filename
        with xr.open_dataset(dataset_path) as dataset:
            data_var_name = get_xarray_data_var_name(dataset)
            # None is a valid xarray key for XDAS-style files whose primary
            # payload is stored under a None variable name.
            data_array = dataset[data_var_name]
            coords = {
                name: (coord.dims, coord.values)
                for name, coord in data_array.coords.items()
            }
            attrs = dict(data_array.attrs)
            dims = data_array.dims
            shape = data_array.shape
            dtype = str(data_array.dtype)
            source_patch_id = self._get_source_patch_id(data_var_name)
            coord_manager = self._coord_manager_from_data_array(
                dataset, data_array, coords, dims, shape
            )
        return [
            _make_scan_payload(
                attrs=attrs | {"_source_patch_id": source_patch_id},
                coords=coord_manager,
                dims=dims,
                shape=shape,
                dtype=dtype,
                source_patch_id=source_patch_id,
            )
        ]

    def _get_source_patch_id(self, data_var_name):
        """Normalize the selected xarray payload name to a patch id."""
        return XDAS_PAYLOAD_VARIABLE if data_var_name is None else data_var_name

    def _coord_manager_from_data_array(self, dataset, data_array, coords, dims, shape):
        """Return coords from xarray when present or reconstruct dim coords."""
        if coords:
            return dc.get_coord_manager(coords=coords, dims=dims)
        return get_coord_manager_for_coordless_data_var(dataset, dims=dims, shape=shape)

    def _patch_from_dataset(self, dataset, data_var_name, data_array):
        """Build one patch from an xarray dataset and selected data variable."""
        source_patch_id = self._get_source_patch_id(data_var_name)
        attrs = dict(data_array.attrs) | {"_source_patch_id": source_patch_id}
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

    def _validate_and_extract_patch(self, spool: SpoolType) -> dc.Patch:
        """Validate write input and return the single supported patch."""
        patches = [spool] if isinstance(spool, dc.Patch) else list(spool)
        if len(patches) == 0:
            msg = "Cannot write empty spool"
            raise ValueError(msg)
        if len(patches) > 1:
            msg = "Multi-patch spools not yet supported for NetCDF output"
            raise NotImplementedError(msg)
        return patches[0]
