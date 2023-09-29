"""IO module for reading SEGY file format support."""
from __future__ import annotations

import numpy as np
import segyio

import dascore as dc
from dascore.io.core import FiberIO

from .utils import get_attrs, get_coords


class SegyV2(FiberIO):
    """An IO class supporting version 1 of the SEGY format."""

    name = "segy"
    preferred_extensions = ("segy", "sgy")
    # also specify a version so when version 2 is released you can
    # just make another class in the same module named JingleV2.
    version = "2"

    def get_format(self, path) -> tuple[str, str] | bool:
        """Make sure input is segy."""
        try:
            with segyio.open(path, ignore_geometry=True):
                return self.name, self.version
        except Exception:
            return False

    def read(self, path, time=None, channel=None, **kwargs):
        """
        Read should take a path and return a patch or sequence of patches.

        It can also define its own optional parameters, and should always
        accept kwargs. If the format supports partial reads, these should
        be implemented as well.
        """
        with segyio.open(path, ignore_geometry=True) as fi:
            data = np.stack([x for x in fi.trace], axis=-1)
            coords = get_coords(fi)
            attrs = get_attrs(fi, coords, path, self)

        patch = dc.Patch(coords=coords, data=data, attrs=attrs)
        patch_trimmed = patch.select(time=time, channel=channel)
        return dc.spool([patch_trimmed])

    def scan(self, path) -> list[dc.PatchAttrs]:
        """
        Used to get metadata about a file without reading the whole file.

        This should return a list of
        [`PatchAttrs`](`dascore.core.attrs.PatchAttrs`) objects
        from the [dascore.core.attrs](`dascore.core.attrs`) module, or a
        format-specific subclass.
        """
        with segyio.open(path, ignore_geometry=True) as fi:
            coords = get_coords(fi)
            attrs = get_attrs(fi, coords, path, self)
        return [attrs]
