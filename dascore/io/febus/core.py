"""
IO module for reading Febus data.
"""

from __future__ import annotations

import re
import warnings
from types import EllipsisType
from typing import Any, cast

import numpy as np

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO
from dascore.utils.hdf5 import H5Reader
from dascore.utils.io import TextReader
from dascore.utils.models import UTF8Str

from .a1utils import (
    _get_febus_version_str,
    _read_febus,
    _yield_attrs_coords,
)
from .g1utils import _get_g1_coords_and_attrs, _get_g1_patch, _is_g1_file

_float_select_type = tuple[float | None | EllipsisType, float | None | EllipsisType]
_time_select_type = tuple[
    opt_timeable_types | EllipsisType,
    opt_timeable_types | EllipsisType,
]


class FebusPatchAttrs(dc.PatchAttrs):
    """
    Patch attrs for febus.

    Attributes
    ----------
    source
        The source designation
    zone
        The zone designations
    """

    gauge_length: float = np.nan
    gauge_length_units: str = "m"
    pulse_width: float = np.nan
    pulse_width_units: str = "m"

    group: str = ""
    source: str = ""
    zone: str = ""

    folog_a1_software_version: UTF8Str = ""


class FebusBOTDRStrainAttrs(dc.PatchAttrs):
    """Attributes for BOTDR (DTSS) systems written in strain."""


class FebusMTXAttrs(dc.PatchAttrs):
    """Attributes for Febus Brillouin spectra files."""


_MTX_H5_DATASETS = frozenset(
    {"distances", "end_times", "mtx", "start_times", "temperatures"}
)
_MTX_DIMS = ("time", "distance", "frequency")
_MTX_ATTR_EXCLUDES = frozenset({"temperature"})


def _maybe_scalar(value):
    """Convert hdf5 attribute arrays into python scalar values when possible."""
    if isinstance(value, bytes | np.bytes_):
        return value.decode("utf-8", errors="replace")
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return array


def _to_snake_case(value):
    """Convert Febus attribute names to snake_case."""
    value = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", value)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", value).lower()


def _get_mtx_attrs(resource):
    """Return normalized Febus MTX HDF5 attributes."""
    attrs = {
        _to_snake_case(key): _maybe_scalar(value)
        for key, value in resource.attrs.items()
    }
    attrs = {
        key: value for key, value in attrs.items() if key not in _MTX_ATTR_EXCLUDES
    }
    data_type = attrs.pop("febus_data_kind", "brillouin_spectrum")
    attrs.update(
        {
            "data_category": "DSS",
            "data_type": data_type,
        }
    )
    return attrs


def _is_mtx_h5(resource):
    """Return True if the resource looks like a Febus MTX HDF5 file."""
    dataset_names = set(resource)
    has_datasets = _MTX_H5_DATASETS.issubset(dataset_names)
    if not has_datasets or "formatVersion" not in resource.attrs:
        return False
    try:
        format_version = int(_maybe_scalar(resource.attrs["formatVersion"]))
    except (TypeError, ValueError):
        return False
    return format_version == 1


def _get_mtx_frequency(resource):
    """Return the Brillouin frequency coordinate."""
    mtx = resource["mtx"]
    freq_len = mtx.shape[-1]
    freq_start = float(_maybe_scalar(resource.attrs["freq_offset_abs"]))
    freq_step = float(_maybe_scalar(resource.attrs["freq_step"]))
    return dc.get_coord(
        start=freq_start,
        step=freq_step,
        shape=freq_len,
        units="MHz",
    )


def _get_mtx_coords(resource, dims=_MTX_DIMS):
    """Return the coordinate manager for a Febus MTX HDF5 file."""
    mtx = resource["mtx"]
    time = dc.get_coord(data=dc.to_datetime64(resource["start_times"][...]))
    distance = dc.get_coord(data=resource["distances"][...], units="m")
    frequency = _get_mtx_frequency(resource)
    temperature = dc.get_coord(data=resource["temperatures"][...], units="degC")
    assert mtx.ndim == 3
    coords = {
        "frequency": frequency,
        "time": time,
        "distance": distance,
        "temperature": ("time", temperature),
    }
    return dc.get_coord_manager(cast(Any, coords), dims=dims)


def _get_mtx_patch(resource, attr_cls, select_kwargs=None):
    """Read a Febus MTX HDF5 file into a patch."""
    select_kwargs = {} if select_kwargs is None else select_kwargs
    data_node = resource["mtx"]
    assert data_node.ndim == 3
    coords = _get_mtx_coords(resource)
    coords, data = coords.select(array=data_node, **select_kwargs)
    if 0 in coords.shape:
        return None
    attrs = _get_mtx_attrs(resource)
    data = np.asarray(data)
    return dc.Patch(data=data, coords=coords, attrs=attr_cls(**attrs))


class Febus2(FiberIO):
    """Support for Febus V 2.

    This should cover all versions 2.* of the format (maybe).
    """

    name = "febus"
    preferred_extensions = ("hdf5", "h5")
    version = "2"

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """
        Return True if file contains febus version 8 data else False.

        Parameters
        ----------
        resource
            A path to the file which may contain terra15 data.
        """
        version_str = _get_febus_version_str(resource)
        if version_str:
            return self.name, version_str
        return False

    def scan(self, resource: H5Reader, **kwargs) -> list[dc.PatchAttrs]:
        """Scan a febus file, return summary information about the file's contents."""
        out = []
        file_version = _get_febus_version_str(resource)
        extras = {
            "path": resource.filename,
            "file_format": self.name,
            "file_version": str(file_version),
        }
        for attr, cm, _ in _yield_attrs_coords(resource):
            attr["coords"] = cm.to_summary_dict()
            attr.update(dict(extras))
            out.append(FebusPatchAttrs(**attr))
        return out

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read a febus spool of patches."""
        patches = _read_febus(
            resource, time=time, distance=distance, attr_cls=FebusPatchAttrs
        )
        return dc.spool(patches)


class Febus1(Febus2):
    """Support for Febus V 1.

    This is here to support legacy Febus (eg pubdas Valencia)
    """

    version = "1"


class FebusG1CSV1(FiberIO):
    """
    A CSV format used by Febus' G1 for storing DSTS files.
    """

    name = "febus_g1_csv"
    preferred_extensions = ("bsl", "mtx")
    version = "1"

    def get_format(self, resource: TextReader, **kwargs) -> tuple[str, str] | bool:
        """Get the name/version of a G1 file else return False."""
        is_g1_file = _is_g1_file(resource)
        resource.seek(0)  # proactively set resource back to position 0.
        return (self.name, self.version) if is_g1_file else False

    def scan(self, resource: TextReader, **kwargs) -> list[dc.PatchAttrs]:
        """Get the coords and attrs of a G1 file."""
        # Handle case of unsupported files (eg spectrum).
        try:
            coords, attrs = _get_g1_coords_and_attrs(resource)
        except NotImplementedError as f:
            warnings.warn(str(f), stacklevel=2)
            return []
        attrs.update(
            {
                "path": getattr(resource, "name", ""),
                "file_format": self.name,
                "file_version": str(self.version),
            }
        )
        attrs_no_private = {i: v for i, v in attrs.items() if not i.startswith("_")}
        attrs = FebusBOTDRStrainAttrs(coords=coords, **attrs_no_private)
        return [attrs]

    def read(self, resource: TextReader, **kwargs) -> dc.BaseSpool:
        """Read a G1 file, return a Patch object."""
        pa = _get_g1_patch(resource, attr_cls=FebusBOTDRStrainAttrs)
        return dc.spool([pa])


class FebusMTXH5V1(FiberIO):
    """HDF5 format used by Febus for storing Brillouin spectra."""

    name = "febus_mtx_h5"
    preferred_extensions = ("h5",)
    version = "1"

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """Get the name/version of an MTX HDF5 file else return False."""
        return (self.name, self.version) if _is_mtx_h5(resource) else False

    def scan(self, resource: H5Reader, **kwargs) -> list[dc.PatchAttrs]:
        """Scan a Febus MTX HDF5 file."""
        attrs = _get_mtx_attrs(resource)
        coords = _get_mtx_coords(resource)
        attrs.update(
            {
                "coords": coords.to_summary_dict(),
                "dims": coords.dims,
                "path": resource.filename,
                "file_format": self.name,
                "file_version": str(self.version),
            }
        )
        return [FebusMTXAttrs(**attrs)]

    def read(
        self,
        resource: H5Reader,
        frequency: _float_select_type | None = None,
        time: _time_select_type | None = None,
        distance: _float_select_type | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read a Febus MTX HDF5 file into a spool."""
        select_kwargs = {
            key: value
            for key, value in {
                "frequency": frequency,
                "time": time,
                "distance": distance,
            }.items()
            if value is not None
        }
        patch = _get_mtx_patch(
            resource,
            attr_cls=FebusMTXAttrs,
            select_kwargs=select_kwargs,
        )
        return dc.spool([] if patch is None else [patch])
