"""Core NetCDF IO implementation with CF conventions."""

from __future__ import annotations

import h5py

import dascore as dc
from dascore.constants import SpoolType
from dascore.io import FiberIO
from dascore.utils.hdf5 import H5Reader, H5Writer

from .utils import (
    create_dimension_scale,
    datetime64_to_cf_time,
    extract_patch_attrs_from_netcdf,
    find_main_data_variable,
    get_cf_data_attrs,
    get_cf_distance_attrs,
    get_cf_global_attrs,
    get_cf_time_attrs,
    get_cf_version,
    is_netcdf4_file,
    read_netcdf_coordinates,
)


class NetCDFCFV18(FiberIO):
    """
    NetCDF format support with CF-1.8 conventions.

    This implementation uses h5py as the backend to read/write NetCDF-4 files
    following the Climate and Forecast (CF) metadata conventions version 1.8.

    Attributes
    ----------
    name : str
        Format identifier ("NETCDF_CF")
    version : str
        CF convention version ("1.8")
    preferred_extensions : tuple
        File extensions for NetCDF files
    """

    name = "NETCDF_CF"
    version = "1.8"
    preferred_extensions = ("nc", "nc4", "netcdf")

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """
        Determine if file is NetCDF-4 format with CF conventions.

        Parameters
        ----------
        resource
            Open h5py.File object
        **kwargs
            Additional keyword arguments

        Returns
        -------
        tuple[str, str] | bool
            Tuple of (format_name, version) if valid, False otherwise
        """
        # Resource is already opened by FiberIO type casting
        if not is_netcdf4_file(resource):
            return False

        # Check for CF conventions
        cf_version = get_cf_version(resource)
        if cf_version:
            # Support older versions by using our implementation
            if cf_version in ("1.8", "1.7", "1.6"):
                return self.name, self.version
            # Try to read newer versions with our CF-1.8 support
            try:
                cf_ver_float = float(cf_version)
                if cf_ver_float > 1.8:
                    return self.name, self.version
            except ValueError:
                pass

        # Generic NetCDF-4 file without explicit CF version
        return self.name, self.version

    def read(self, resource: H5Reader, **kwargs) -> SpoolType:
        """
        Read NetCDF-4 file into a DASCore Spool.

        Parameters
        ----------
        resource
            Open h5py.File object
        **kwargs
            Additional keyword arguments for coordinate filtering
            (e.g., time=(start, end), distance=(min, max))

        Returns
        -------
        SpoolType
            Spool containing read patches

        Raises
        ------
        ValueError
            If no suitable data variable is found in the file
        """
        coords = read_netcdf_coordinates(resource)
        data_var_name = self._get_data_variable_name(resource)
        data_var = resource[data_var_name]
        attrs_dict = self._create_patch_attributes(resource, data_var_name)

        patch = dc.Patch(data=data_var[:], coords=coords, attrs=attrs_dict)
        patch = self._apply_coordinate_filtering(patch, kwargs)

        return dc.spool([patch])

    def write(self, spool: SpoolType, resource: H5Writer, **kwargs) -> None:
        """
        Write a Spool to NetCDF-4 format with CF-1.8 conventions.

        Parameters
        ----------
        spool
            Spool or single Patch to write
        resource
            Open h5py.File object for writing
        **kwargs
            Additional options:

            - compression: str, compression type ('gzip', 'lzf', 'szip')
            - compression_opts: int, compression level (1-9 for gzip)
            - chunks: bool or tuple, chunking strategy

        Raises
        ------
        ValueError
            If spool is empty
        NotImplementedError
            If spool contains multiple patches
        """
        patch = self._validate_and_extract_patch(spool)

        self._write_global_attributes(resource, patch.attrs)
        self._write_coordinates(resource, patch.coords)
        self._write_data_variable(resource, patch, **kwargs)

    def scan(self, resource: H5Reader, **kwargs) -> list[dc.PatchAttrs]:
        """
        Scan NetCDF file to extract metadata without loading data.

        Parameters
        ----------
        resource
            Open h5py.File object
        **kwargs
            Additional keyword arguments

        Returns
        -------
        list[dc.PatchAttrs]
            List of patch attributes with metadata
        """
        coords = read_netcdf_coordinates(resource)
        attrs_dict = self._build_scan_attributes(resource, coords)
        patch_attrs = dc.PatchAttrs(**attrs_dict)

        return [patch_attrs]

    def _get_data_variable_name(self, resource: H5Reader) -> str:
        """Get the main data variable name, raising error if not found."""
        data_var_name = find_main_data_variable(resource)
        if data_var_name is None:
            msg = "No suitable data variable found in NetCDF file"
            raise ValueError(msg)
        return data_var_name

    def _create_patch_attributes(self, resource: H5Reader, data_var_name: str) -> dict:
        """Create patch attributes dictionary."""
        attrs_dict = extract_patch_attrs_from_netcdf(resource)

        # Add data variable name if it's meaningful
        if data_var_name not in ("data", "acoustic_data", "das_data"):
            attrs_dict["tag"] = attrs_dict.get("tag", "") or data_var_name

        return attrs_dict

    def _apply_coordinate_filtering(self, patch: dc.Patch, kwargs: dict) -> dc.Patch:
        """Apply coordinate filtering to patch if kwargs provided."""
        if not kwargs:
            return patch

        coord_kwargs = {k: v for k, v in kwargs.items() if k in patch.coords.coord_map}
        return patch.select(**coord_kwargs) if coord_kwargs else patch

    def _validate_and_extract_patch(self, spool: SpoolType) -> dc.Patch:
        """Validate spool and extract single patch."""
        patches = [spool] if isinstance(spool, dc.Patch) else list(spool)

        if len(patches) == 0:
            msg = "Cannot write empty spool"
            raise ValueError(msg)

        if len(patches) > 1:
            msg = "Multi-patch spools not yet supported for NetCDF output"
            raise NotImplementedError(msg)

        return patches[0]

    def _build_scan_attributes(self, resource: H5Reader, coords) -> dict:
        """Build attributes dictionary for scan method."""
        attrs_dict = extract_patch_attrs_from_netcdf(resource)

        # Add data variable name if it's meaningful (same logic as read method)
        data_var_name = self._get_data_variable_name(resource)
        if data_var_name not in ("data", "acoustic_data", "das_data"):
            attrs_dict["tag"] = attrs_dict.get("tag", "") or data_var_name

        # Add file metadata
        attrs_dict["file_format"] = self.name
        attrs_dict["file_version"] = self.version  # Use handler version for consistency
        attrs_dict["path"] = getattr(resource, "filename", "")

        # Add coordinate information to attrs
        if coords.coord_map:
            coords_summary = coords.to_summary_dict()
            attrs_dict["coords"] = coords_summary

        return attrs_dict

    def _prepare_coordinate_data(self, name: str, coord) -> tuple:
        """Prepare coordinate data and attributes for writing."""
        coord_data = coord.values

        if name == "time":
            return self._prepare_time_coordinate(name, coord, coord_data)
        elif name == "distance":
            return self._prepare_distance_coordinate(name, coord, coord_data)
        else:
            return self._prepare_generic_coordinate(name, coord, coord_data)

    def _prepare_time_coordinate(self, name: str, coord, coord_data) -> tuple:
        """Prepare time coordinate data and CF attributes."""
        cf_time_data = datetime64_to_cf_time(coord_data)
        cf_attrs = get_cf_time_attrs(name)

        # Add bounds info if available
        if hasattr(coord, "step") and coord.step:
            cf_attrs["bounds"] = f"{name}_bounds"

        return cf_time_data, cf_attrs

    def _prepare_distance_coordinate(self, name: str, coord, coord_data) -> tuple:
        """Prepare distance coordinate data and CF attributes."""
        cf_attrs = get_cf_distance_attrs(name)

        # Add coordinate reference info if available
        if hasattr(coord, "units") and coord.units:
            cf_attrs["units"] = str(coord.units)

        return coord_data, cf_attrs

    def _prepare_generic_coordinate(self, name: str, coord, coord_data) -> tuple:
        """Prepare generic coordinate data and CF attributes."""
        cf_attrs = {
            "long_name": name.replace("_", " ").title(),
            "standard_name": name.lower(),
        }

        # Try to determine units
        if hasattr(coord, "units"):
            cf_attrs["units"] = str(coord.units)
        elif "depth" in name.lower():
            cf_attrs["units"] = "m"
            cf_attrs["positive"] = "down"
        else:
            cf_attrs["units"] = "1"

        return coord_data, cf_attrs

    def _write_global_attributes(self, h5file: h5py.File, attrs: dc.PatchAttrs) -> None:
        """Write CF-compliant global attributes."""
        global_attrs = get_cf_global_attrs(attrs, self.version)

        # Add NetCDF-specific properties
        global_attrs["_NCProperties"] = (
            f"version=2,netcdf=dascore-{dc.__version__},"
            f"hdf5={h5py.version.hdf5_version}"
        )

        # Add feature type for CF compliance
        global_attrs["featureType"] = "timeSeries"

        # Add data type information if available
        if attrs.data_type:
            global_attrs["source_data_type"] = attrs.data_type

        # Write all global attributes
        for key, value in global_attrs.items():
            if value is not None and value != "":
                h5file.attrs[key] = value

    def _write_coordinates(self, h5file: h5py.File, coords) -> None:
        """Write coordinate variables as CF-compliant dimension scales."""
        for name, coord in coords.coord_map.items():
            coord_data, cf_attrs = self._prepare_coordinate_data(name, coord)
            create_dimension_scale(h5file, name, coord_data, cf_attrs)

    def _write_data_variable(
        self, h5file: h5py.File, patch: dc.Patch, **kwargs
    ) -> None:
        """Write the main data variable with CF-compliant attributes."""
        compression_settings = self._get_compression_settings(**kwargs)
        data_var = self._create_data_dataset(h5file, patch, compression_settings)
        self._attach_dimension_scales(data_var, patch, h5file)
        self._add_data_variable_attributes(data_var, patch)

    def _get_compression_settings(self, **kwargs) -> dict:
        """Get compression settings with sensible defaults."""
        compression = kwargs.get("compression", "gzip")
        compression_opts = kwargs.get("compression_opts", 4)  # Balance speed/size
        chunks = kwargs.get("chunks", True)
        return {
            "compression": compression,
            "compression_opts": compression_opts,
            "chunks": chunks,
        }

    def _create_data_dataset(self, h5file: h5py.File, patch: dc.Patch, settings: dict):
        """Create the main data dataset with compression."""
        chunks = settings["chunks"]
        if chunks is True:
            # Auto-chunking with reasonable defaults for DAS data
            shape = patch.data.shape
            # Chunk by ~100 time samples and ~100 distance samples
            chunks = tuple(min(100, s) for s in shape)

        return h5file.create_dataset(
            "data",
            data=patch.data,
            chunks=chunks,
            compression=settings["compression"],
            compression_opts=settings["compression_opts"],
            shuffle=True,  # Enable shuffle filter for better compression
        )

    def _attach_dimension_scales(self, data_var, patch: dc.Patch, h5file: h5py.File):
        """Attach dimension scales to data variable."""
        for i, dim_name in enumerate(patch.dims):
            if dim_name in h5file:
                dim_scale = h5file[dim_name]
                data_var.dims[i].attach_scale(dim_scale)
                data_var.dims[i].label = dim_name

    def _add_data_variable_attributes(self, data_var, patch: dc.Patch):
        """Add CF-compliant attributes to data variable."""
        data_attrs = get_cf_data_attrs(patch.attrs.data_type or "acoustic_signal")
        data_attrs = self._enhance_data_attributes(data_attrs, patch)

        # Write all attributes
        for key, value in data_attrs.items():
            if value is not None:
                data_var.attrs[key] = value

        # Add dimension names as coordinates attribute for CF compliance
        coord_names = list(patch.coords.coord_map.keys())
        data_var.attrs["coordinates"] = " ".join(coord_names)

    def _enhance_data_attributes(self, data_attrs: dict, patch: dc.Patch) -> dict:
        """Enhance data attributes with additional CF-compliant metadata."""
        # Ensure required CF attributes
        if "long_name" not in data_attrs:
            data_attrs["long_name"] = "Distributed Acoustic Sensing data"

        # Add grid mapping if we have geographic coordinates
        if hasattr(patch.coords, "crs"):
            data_attrs["grid_mapping"] = "crs"

        # Add ancillary variables if quality info exists
        if hasattr(patch.attrs, "quality") and patch.attrs.quality:
            data_attrs["ancillary_variables"] = "quality_flag"

        return data_attrs
