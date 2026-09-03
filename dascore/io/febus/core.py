"""
IO module for reading Febus data.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

import dascore as dc
from dascore.constants import (
    float_select_type,
    opt_timeable_types,
    time_select_type,
    timeable_types,
)
from dascore.io import FiberIO, ScanPayload
from dascore.io.core import make_scan_payload
from dascore.io.utils import resolve_keyed_source, slice_dataset
from dascore.models import OptionalFiniteFloat, UTF8Str
from dascore.utils.hdf5 import H5Reader
from dascore.utils.io import TextReader
from dascore.utils.misc import raise_on_extra_kwargs

from .a1utils import (
    _flatten_febus_info,
    _get_febus_version_str,
    _get_source_patch_key,
    _read_febus,
    _read_febus_array,
    _yield_attrs_coords,
)
from .g1utils import (
    _BSL_DIMS,
    _MTX_DIMS,
    _bsl_version,
    _get_bsl_attrs,
    _get_bsl_coords,
    _get_bsl_patch,
    _get_g1_coords_and_attrs,
    _get_g1_patch,
    _get_mtx_attrs,
    _get_mtx_coords,
    _get_mtx_patch,
    _is_g1_file,
    _mtx_version,
)
from .t1utils import _get_t1_patch, _is_t1_file, _scan_t1

# Kept as module-local names for the many signatures below; the shared
# definitions live in dascore.constants.
_float_select_type = float_select_type
_time_select_type = time_select_type


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

    gauge_length: OptionalFiniteFloat = None
    pulse_length: OptionalFiniteFloat = None

    group: str = ""
    source: str = ""
    zone: str = ""

    folog_a1_software_version: UTF8Str = ""


class FebusBOTDRStrainAttrs(dc.PatchAttrs):
    """Attributes for BOTDR (DTSS) systems written in strain."""


class FebusMTXAttrs(dc.PatchAttrs):
    """Attributes for Febus Brillouin spectra files."""


class Febus2(FiberIO):
    """Support for Febus V 2.

    This should cover all versions 2.* of the format (maybe).
    """

    name = "febus"
    preferred_extensions = ("hdf5", "h5")
    version = "2"

    def get_format(
        self,
        resource: H5Reader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """
        Return True if file contains febus version 8 data else False.

        Parameters
        ----------
        resource
            An open h5 file which may contain febus data.
        """
        version_str = _get_febus_version_str(resource)
        if version_str:
            return self.name, version_str
        return False

    def scan(self, resource: H5Reader, **kwargs) -> list[ScanPayload]:
        """Scan a febus file, return summary information about the file's contents."""
        out = []
        for attr, cm, feb in _yield_attrs_coords(resource):
            attrs = FebusPatchAttrs.from_dict(attr).update(
                _source_patch_key=_get_source_patch_key(feb)
            )
            out.append(
                make_scan_payload(
                    attrs=attrs,
                    coords=cm,
                    dtype=str(feb.zone[feb.data_name].dtype),
                    source_patch_key=attrs["_source_patch_key"],
                )
            )
        return out

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        source_patch_key=(),
        **kwargs,
    ) -> dc.Spool:
        """Read a febus spool of patches."""
        patches = _read_febus(
            resource,
            time=time,
            distance=distance,
            source_patch_key=source_patch_key,
            attr_cls=FebusPatchAttrs,
        )
        return dc.spool(patches)

    def read_array(
        self,
        resource: H5Reader,
        windows: dict[str, tuple[int, int]],
        source_patch_key="",
        **kwargs,
    ) -> np.ndarray:
        """
        Read one zone's window out of its block-structured data cube.

        ``source_patch_key`` is the ``group:source:zone`` name `scan`
        reports; a file holding several zones needs one.
        """
        raise_on_extra_kwargs(kwargs, "windows and source_patch_key")
        # pairs, not a mapping: two zones can generate one name, and the
        # default read refuses such a key rather than picking one
        zones = [
            (_get_source_patch_key(zone), zone)
            for zone in _flatten_febus_info(resource)
        ]
        where = str(getattr(resource, "filename", "the resource"))
        febus = resolve_keyed_source(zones, source_patch_key, where=where)
        return _read_febus_array(febus, windows)


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

    def get_format(
        self,
        resource: TextReader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """Get the name/version of a G1 file else return False."""
        is_g1_file = _is_g1_file(resource)
        resource.seek(0)  # proactively set resource back to position 0.
        return (self.name, self.version) if is_g1_file else False

    def scan(self, resource: TextReader, **kwargs) -> list[ScanPayload]:
        """Get the coords and attrs of a G1 file."""
        # Handle case of unsupported files (eg spectrum).
        try:
            coords, attrs = _get_g1_coords_and_attrs(resource)
        except NotImplementedError as f:
            warnings.warn(str(f), stacklevel=2)
            return []
        attrs_no_private = {i: v for i, v in attrs.items() if not i.startswith("_")}
        attrs = FebusBOTDRStrainAttrs(**attrs_no_private)
        return [make_scan_payload(attrs=attrs, coords=coords, dtype="float64")]

    def read(self, resource: TextReader, **kwargs) -> dc.Spool:
        """Read a G1 file, return a Patch object."""
        pa = _get_g1_patch(resource, attr_cls=FebusBOTDRStrainAttrs)
        return dc.spool([pa])


class FebusMTXH5V1(FiberIO):
    """
    HDF5 format used by Febus for storing Brillouin spectra.

    As with the BSL files, ``time`` holds the start of each acquisition
    window and the non-dimensional ``sample_span`` coord holds its length.
    """

    name = "febus_mtx_h5"
    preferred_extensions = ("h5", "hdf5")
    version = "1"

    def get_format(
        self,
        resource: H5Reader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """Get the name/version of an MTX HDF5 file else return False."""
        version = _mtx_version(resource)
        return (self.name, self.version) if version == self.version else False

    def scan(
        self, resource: H5Reader, snap: bool = True, **kwargs
    ) -> list[ScanPayload]:
        """Scan a Febus MTX HDF5 file."""
        attrs = _get_mtx_attrs(resource)
        coords = _get_mtx_coords(resource, snap=snap)
        return [
            make_scan_payload(
                attrs=FebusMTXAttrs(**attrs),
                coords=coords,
                dtype=str(resource["mtx"].dtype),
            )
        ]

    def read(
        self,
        resource: H5Reader,
        frequency: _float_select_type | None = None,
        time: _time_select_type | None = None,
        distance: _float_select_type | None = None,
        **kwargs,
    ) -> dc.Spool:
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
        attrs = _get_mtx_attrs(resource)
        patch = _get_mtx_patch(
            resource,
            attr_cls=FebusMTXAttrs,
            attrs=attrs,
            select_kwargs=select_kwargs,
        )
        return dc.spool([] if patch is None else [patch])

    def read_array(
        self,
        resource: H5Reader,
        windows: dict[str, tuple[int, int]],
        snap: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Slice the ``mtx`` dataset directly.
        """
        raise_on_extra_kwargs(kwargs, "windows and snap")
        return slice_dataset(resource["mtx"], _MTX_DIMS, windows)


class FebusBSLH5V1(FiberIO):
    """
    HDF5 format used by Febus G1 for storing BSL strain files.

    Samples are not instantaneous; each one covers an acquisition window.
    The ``time`` coord holds the start of that window and the non-dimensional
    ``sample_span`` coord, mapped to ``time``, holds how long it ran.
    """

    name = "febus_bsl_h5"
    preferred_extensions = ("h5", "hdf5")
    version = "1"

    def get_format(
        self,
        resource: H5Reader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """Get the name/version of a BSL HDF5 file else return False."""
        version = _bsl_version(resource)
        return (self.name, self.version) if version == self.version else False

    def scan(
        self, resource: H5Reader, snap: bool = True, **kwargs
    ) -> list[ScanPayload]:
        """Scan a Febus BSL HDF5 file."""
        attrs = _get_bsl_attrs(resource)
        coords = _get_bsl_coords(resource, snap=snap)
        return [
            make_scan_payload(
                attrs=FebusBOTDRStrainAttrs(**attrs),
                coords=coords,
                dtype=str(resource["bsl_data"].dtype),
            )
        ]

    def read(
        self,
        resource: H5Reader,
        time: _time_select_type | None = None,
        distance: _float_select_type | None = None,
        **kwargs,
    ) -> dc.Spool:
        """Read a Febus BSL HDF5 file into a spool."""
        select_kwargs = {
            key: value
            for key, value in {"time": time, "distance": distance}.items()
            if value is not None
        }
        attrs = _get_bsl_attrs(resource)
        patch = _get_bsl_patch(
            resource,
            attr_cls=FebusBOTDRStrainAttrs,
            attrs=attrs,
            select_kwargs=select_kwargs,
        )
        return dc.spool([] if patch is None else [patch])

    def read_array(
        self,
        resource: H5Reader,
        windows: dict[str, tuple[int, int]],
        snap: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Slice the ``bsl_data`` dataset directly.
        """
        raise_on_extra_kwargs(kwargs, "windows and snap")
        return slice_dataset(resource["bsl_data"], _BSL_DIMS, windows)


class FebusT1V1(FiberIO):
    """
    IO support for FEBUS T1 DTS HDF5 files.

    Each file typically covers one acquisition session; each row in
    Temperature / Time represents one measurement sweep.

    Only Temperature is exposed as the primary Patch data_type.
    Stokes / AntiStokes live in the same file but on a different distance
    grid (DistanceSignal, 4501 pts vs 1103 pts for Temperature), so they
    would need separate Patch objects — out of scope for this reader.

    Additionally, it's possible to have multiple fibers on a single
    interrogator and this doesn't account for that in any way.
    """

    name = "FEBUS_T1"
    version = "1"

    preferred_extensions = ("hdf5", "h5")

    def get_format(
        self, resource: H5Reader, **kwargs
    ) -> tuple[str, str] | Literal[False]:
        """Return (name, version) if this is a FEBUS T1 file, else False."""
        return (self.name, self.version) if _is_t1_file(resource) else False

    def scan(
        self, resource: H5Reader, snap: bool = True, **kwargs
    ) -> list[ScanPayload]:
        """Return a list with one PatchAttrs for the file's temperature data."""
        return [_scan_t1(resource, snap=snap)]

    def read(
        self,
        resource: H5Reader,
        time: tuple[timeable_types, timeable_types] | None = None,
        distance: tuple[float, float] | None = None,
        **kwargs,
    ) -> dc.Spool:
        """
        Read temperature data into a list containing one Patch.

        Parameters
        ----------
        resource
            Open h5py.File — provided automatically by DASCore.
        """
        pa = _get_t1_patch(
            resource, self.name, self.version, time=time, distance=distance
        )
        if not pa.data.size:
            return dc.spool([])
        return dc.spool([pa])

    def read_array(
        self,
        resource: H5Reader,
        windows: dict[str, tuple[int, int]],
        snap: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Slice the ``Data/Temperature`` dataset directly.
        """
        raise_on_extra_kwargs(kwargs, "windows and snap")
        return slice_dataset(
            resource["Data/Temperature"], ("time", "distance"), windows
        )
