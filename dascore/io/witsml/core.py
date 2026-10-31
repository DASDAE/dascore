"""IO module for reading WITSML 1.4 DTS logs."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import snap_type, windows_type
from dascore.core.source import ArraySource
from dascore.io import FiberIO
from dascore.io.utils import resolve_keyed_source, slice_dataset
from dascore.utils.io import BinaryReader

from .utils import NS, get_attrs, get_coords, get_root, read_log


class WITSMLV1411(FiberIO):
    """WITSML 1.4.1.1 ``logs``/``log`` objects, one patch per curve."""

    name = "WITSML"
    preferred_extensions = ("xml",)
    version = "1.4.1.1"
    _root = "logs"

    def get_version(self, resource: BinaryReader, **kwargs) -> str | None:
        """Return the version when the root names this object and version."""
        tag, version = get_root(resource)
        ok = tag == f"{NS}{self._root}" and version == self.version
        return self.version if ok else None

    def get_metadata(
        self, resource: BinaryReader, *, snap: snap_type = ("distance",)
    ) -> list[dc.PatchMeta]:
        """Describe each curve, keyed and tagged by its mnemonic."""
        log = read_log(resource)
        coords = get_coords(log, snap)
        return [
            dc.PatchMeta(
                attrs=dc.PatchAttrs(**get_attrs(log, curve)),
                coords=coords,
                dtype="float64",
                source=ArraySource(key=curve),
            )
            for curve in log.curves
        ]

    def read_array(
        self, resource: BinaryReader, windows: windows_type = (), key: str = ""
    ) -> np.ndarray:
        """Convert the keyed curve's column and select its windows."""
        log = read_log(resource)
        curve = resolve_keyed_source({x: x for x in log.curves}, key, log.where)
        return slice_dataset(log.column(curve)[:, None], windows)


class WITSMLV1400(WITSMLV1411):
    """WITSML 1.4.0.0 ``wellLogs``/``wellLog`` objects, one patch per curve."""

    version = "1.4.0.0"
    _root = "wellLogs"
