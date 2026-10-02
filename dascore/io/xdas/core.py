"""Read XDAS NetCDF files, one patch per signal."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import snap_type, windows_type
from dascore.io import FiberIO
from dascore.io.utils import resolve_keyed_source, windows_to_slices
from dascore.utils.hdf5 import H5Reader

from .utils import check_virtual_sources, get_pieces, is_xdas_file, unpack


class XdasV1(FiberIO):
    """
    Read signals stored in the XDAS NetCDF layout.

    Each signal, in any group, is a patch keyed by its path in the file.
    Tie-point coordinates are decoded exactly; a signal whose tie points
    leave a hole or change the step is one patch per evenly sampled
    stretch, keyed ``<path>#<number>``. Tile manifests are not supported.
    """

    name = "XDAS"
    version = "1"
    preferred_extensions = ("nc", "nc4", "netcdf")

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the version if a dataset carries an XDAS coordinate mapping."""
        return self.version if is_xdas_file(resource) else None

    def get_metadata(
        self, resource: H5Reader, *, snap: snap_type = True
    ) -> list[dc.PatchMeta]:
        """Describe each signal from its coordinates; no samples are read."""
        return [meta for meta, *_ in get_pieces(resource, snap).values()]

    def read_array(
        self, resource: H5Reader, windows: windows_type = (), key: str = ""
    ) -> np.ndarray:
        """Read the windows of one patch from its signal's dataset."""
        where = str(getattr(resource, "filename", "the resource"))
        meta, node, offsets = resolve_keyed_source(get_pieces(resource), key, where)
        slices = windows_to_slices(windows, meta.shape)
        check_virtual_sources(node)
        shifted = tuple(
            slice(x.start + off, x.stop + off)
            for x, off in zip(slices, offsets, strict=True)
        )
        return unpack(node, node[shifted])
