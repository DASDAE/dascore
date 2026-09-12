"""
IO module for reading Febus data.
"""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import snap_type
from dascore.core.source import PatchSource
from dascore.io import FiberIO
from dascore.io.utils import resolve_keyed_source, slice_dataset
from dascore.models import OptionalFiniteFloat, UTF8Str
from dascore.utils.hdf5 import H5Reader
from dascore.utils.io import TextReader

from .a1utils import (
    _flatten_febus_info,
    _get_febus_version_str,
    _get_source_patch_key,
    _read_febus_array,
    _yield_attrs_coords,
)
from .g1utils import (
    _BSL_DIMS,
    _MTX_DIMS,
    _bsl_version,
    _get_bsl_attrs,
    _get_bsl_coords,
    _get_g1_coords_and_attrs,
    _get_mtx_attrs,
    _get_mtx_coords,
    _is_g1_file,
    _mtx_version,
)
from .t1utils import _is_t1_file, _scan_t1


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

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        version_str = _get_febus_version_str(resource)
        if version_str:
            return version_str
        return None

    def get_metadata(
        self, resource: H5Reader, *, snap: snap_type = True
    ) -> list[dc.Patch]:
        """Scan a febus file, return summary information about the file's contents."""
        out = []
        for attr, cm, feb in _yield_attrs_coords(resource):
            attrs = FebusPatchAttrs.from_dict(attr)
            out.append(
                dc.Patch(
                    attrs=attrs,
                    coords=cm,
                    dtype=str(feb.zone[feb.data_name].dtype),
                    source=PatchSource(key=_get_source_patch_key(feb)),
                )
            )
        return out

    def read_array(
        self, resource: H5Reader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """
        Read one zone's window out of its block-structured data cube.

        ``source_patch_key`` is the ``group:source:zone`` name `scan`
        reports; a file holding several zones needs one.
        """
        zones = [
            (_get_source_patch_key(zone), zone)
            for zone in _flatten_febus_info(resource)
        ]
        where = str(getattr(resource, "filename", "the resource"))
        febus = resolve_keyed_source(zones, key, where=where)
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

    def get_version(self, resource: TextReader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        is_g1_file = _is_g1_file(resource)
        resource.seek(0)
        return self.version if is_g1_file else None

    def get_metadata(
        self, resource: TextReader, *, snap: snap_type = True
    ) -> list[dc.Patch]:
        """Get the coords and attrs of a G1 file."""
        coords, attrs = _get_g1_coords_and_attrs(resource)
        attrs_no_private = {i: v for i, v in attrs.items() if not i.startswith("_")}
        attrs = FebusBOTDRStrainAttrs(**attrs_no_private)
        return [dc.Patch(attrs=attrs, coords=coords, dtype="float64")]

    def read_array(
        self, resource: TextReader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """Decode CSV samples and select the requested positional window."""
        coords, attrs = _get_g1_coords_and_attrs(resource)
        resource.seek(0)
        data = np.loadtxt(resource, skiprows=int(attrs.get("_data_start_line", 0)))
        return slice_dataset(
            np.asarray(data).reshape(coords.shape), coords.dims, windows
        )


class FebusMTXH5V1(FiberIO):
    """
    HDF5 format used by Febus for storing Brillouin spectra.

    As with the BSL files, ``time`` holds the start of each acquisition
    window and the non-dimensional ``sample_span`` coord holds its length.
    """

    name = "febus_mtx_h5"
    preferred_extensions = ("h5", "hdf5")
    version = "1"

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        version = _mtx_version(resource)
        return self.version if version == self.version else None

    def get_metadata(
        self, resource: H5Reader, *, snap: snap_type = True
    ) -> list[dc.Patch]:
        """Scan a Febus MTX HDF5 file."""
        attrs = _get_mtx_attrs(resource)
        coords = _get_mtx_coords(resource, snap=snap)
        return [
            dc.Patch(
                attrs=FebusMTXAttrs(**attrs),
                coords=coords,
                dtype=str(resource["mtx"].dtype),
            )
        ]

    def read_array(
        self, resource: H5Reader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """
        Slice the ``mtx`` dataset directly.
        """
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

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        version = _bsl_version(resource)
        return self.version if version == self.version else None

    def get_metadata(
        self, resource: H5Reader, *, snap: snap_type = True
    ) -> list[dc.Patch]:
        """Scan a Febus BSL HDF5 file."""
        attrs = _get_bsl_attrs(resource)
        coords = _get_bsl_coords(resource, snap=snap)
        return [
            dc.Patch(
                attrs=FebusBOTDRStrainAttrs(**attrs),
                coords=coords,
                dtype=str(resource["bsl_data"].dtype),
            )
        ]

    def read_array(
        self, resource: H5Reader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """
        Slice the ``bsl_data`` dataset directly.
        """
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

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        return self.version if _is_t1_file(resource) else None

    def get_metadata(
        self, resource: H5Reader, *, snap: snap_type = True
    ) -> list[dc.Patch]:
        """Return a list with one PatchAttrs for the file's temperature data."""
        return [_scan_t1(resource, snap=snap)]

    def read_array(
        self, resource: H5Reader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """
        Slice the ``Data/Temperature`` dataset directly.
        """
        return slice_dataset(
            resource["Data/Temperature"], ("time", "distance"), windows
        )
