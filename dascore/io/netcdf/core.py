"""Core NetCDF IO implementation with CF conventions."""

from __future__ import annotations

from pathlib import Path

import h5py

import dascore as dc
from dascore.constants import SpoolType
from dascore.io import FiberIO
from dascore.io.core import ScanPayload, _make_scan_payload
from dascore.utils.hdf5 import H5Reader
from dascore.utils.io import patch_to_xarray
from dascore.utils.misc import optional_import

from .utils import (
    extract_patch_attrs_from_netcdf,
    find_main_data_variable,
    get_xarray_data_var_name,
    get_xarray_engine,
    get_cf_data_attrs,
    get_cf_global_attrs,
    get_cf_version,
    is_netcdf4_file,
    iter_written_aux_coords,
    parse_cf_version,
    read_netcdf_coordinates,
    XDAS_PAYLOAD_VARIABLE,
    coord_attrs,
)


class NetCDFCFV18(FiberIO):
    """NetCDF-4 IO with CF-1.8 conventions, using xarray/netcdf4 for IO."""

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
        engine = get_xarray_engine()
        # Read structural metadata first so xarray only has to resolve the final
        # payload variable and dense coordinate values.
        data_var_name, attrs_dict, coords = self._read_metadata(resource)
        data_var_name, data_array, data = self._read_data_array(
            resource, xr, engine, data_var_name
        )
        patch = self._build_patch(
            data=data,
            data_array=data_array,
            coords=coords,
            attrs_dict=attrs_dict,
            data_var_name=data_var_name,
        )
        patch = self._apply_coordinate_filtering(patch, kwargs)
        if not patch.data.size:
            return dc.spool([])
        return dc.spool([patch])

    def _read_metadata(self, resource: Path):
        """Read NetCDF metadata needed before opening through xarray."""
        with h5py.File(resource, "r") as h5file:
            data_var_name = self._get_data_variable_name(h5file)
            attrs_dict = self._get_patch_attrs(h5file)
            coords = read_netcdf_coordinates(h5file, data_var_name)
        return data_var_name, attrs_dict, coords

    def _read_data_array(self, resource: Path, xr, engine: str, data_var_name: str):
        """Load the selected xarray data variable from disk."""
        # TODO: consider a lazy read path for large NetCDF arrays.
        with xr.open_dataset(resource, engine=engine) as dataset:
            data_array = dataset.get(data_var_name)
            if data_array is None:
                data_var_name = get_xarray_data_var_name(dataset)
                data_array = dataset[data_var_name]
            data = data_array.load().data
        return data_var_name, data_array, data

    def _build_patch(self, *, data, data_array, coords, attrs_dict, data_var_name: str):
        """Merge coordinate metadata and construct the output patch."""
        source_patch_id = (
            XDAS_PAYLOAD_VARIABLE if data_var_name is None else data_var_name
        )
        # Start from the HDF-derived coordinates, then add any extra xarray-only
        # coordinates that were materialized during decode.
        coords_dict = {
            name: (
                coord.values
                if name in coords.dims
                else (coords.dim_map[name], coord.values)
            )
            for name, coord in coords.coord_map.items()
        }
        for name, coord in data_array.coords.items():
            if name not in coords_dict:
                coords_dict[name] = (coord.dims, coord.values)
        coords = dc.get_coord_manager(coords=coords_dict, dims=coords.dims)
        return dc.Patch(
            data=data,
            coords=coords,
            dims=coords.dims,
            attrs=attrs_dict | {"_source_patch_id": source_patch_id},
        )

    def _build_data_array(self, patch: dc.Patch):
        """Convert a patch to an xarray DataArray with coordinate attrs."""
        data_array = patch_to_xarray(patch).rename("data")
        for name, coord in patch.coords.coord_map.items():
            if coord._partial:
                continue
            data_array.coords[name].attrs.update(coord_attrs(name, coord))
        return data_array

    def _build_dataset(self, patch: dc.Patch, data_array):
        """Create the xarray Dataset and attach CF metadata."""
        global_attrs = get_cf_global_attrs(patch.attrs, self.version)
        if patch.attrs.data_type:
            global_attrs["source_data_type"] = patch.attrs.data_type
        dataset = data_array.to_dataset()
        # NetCDF attrs cannot safely preserve DASCore's null-ish metadata, so
        # filter those out before handing the dataset to xarray.
        dataset.attrs.update(
            {
                key: value
                for key, value in global_attrs.items()
                if value not in (None, "")
            }
        )
        dataset["data"].attrs.update(
            get_cf_data_attrs(patch.attrs.data_type or "acoustic_signal")
        )
        aux_coord_names = tuple(iter_written_aux_coords(patch))
        if aux_coord_names:
            dataset["data"].attrs["coordinates"] = " ".join(aux_coord_names)
        return dataset

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
        Write a Spool to NetCDF-4 with CF-1.8 conventions.

        Parameters
        ----------
        kwargs
            compression: 'gzip', None, or False
            compression_opts: gzip level 1-9 (default 4)
            chunks: True to defer chunking to xarray/backend defaults, or an
                explicit tuple of chunk sizes
        """
        patch = self._validate_and_extract_patch(spool)
        optional_import("xarray")
        engine = get_xarray_engine()
        # Build the xarray object first, then attach CF metadata and optional
        # storage hints in separate steps to keep write policy isolated.
        data_array = self._build_data_array(patch)
        dataset = self._build_dataset(patch, data_array)
        encoding = self._get_write_encoding(**kwargs)
        dataset.to_netcdf(
            resource,
            engine=engine,
            encoding={"data": encoding} if encoding else None,
        )

    def scan(self, resource: H5Reader, **kwargs) -> list[ScanPayload]:
        """Scan NetCDF file to extract metadata without loading data."""
        data_var_name = self._get_data_variable_name(resource)
        data_var = resource[data_var_name]
        coords = read_netcdf_coordinates(resource, data_var_name)
        attrs_dict = self._get_patch_attrs(resource)
        return [
            _make_scan_payload(
                attrs=attrs_dict,
                coords=coords,
                dims=coords.dims,
                shape=data_var.shape,
                dtype=str(data_var.dtype),
                source_patch_id=data_var_name,
            )
        ]

    def _get_data_variable_name(self, resource: H5Reader) -> str:
        """Return the main NetCDF data variable name or raise."""
        data_var_name = find_main_data_variable(resource)
        if data_var_name is None:
            msg = "No suitable data variable found in NetCDF file"
            raise ValueError(msg)
        return data_var_name

    def _get_patch_attrs(self, resource: H5Reader) -> dict:
        """Extract patch attrs from a NetCDF resource."""
        return extract_patch_attrs_from_netcdf(resource)

    def _apply_coordinate_filtering(self, patch: dc.Patch, kwargs: dict) -> dc.Patch:
        """Apply coordinate selection kwargs to a loaded patch."""
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
