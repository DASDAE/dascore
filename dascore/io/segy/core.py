"""IO module for reading SEGY file format support."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import snap_type
from dascore.io.core import FiberIO
from dascore.io.utils import windows_to_slices
from dascore.utils.io import LocalBinaryReader, LocalPath
from dascore.utils.misc import optional_import

from .utils import (
    _get_coords,
    _get_segy_version,
    _write_segy,
)


class SegyV1_0(FiberIO):  # noqa
    """An IO class supporting version 1.0 of the SEGY format."""

    name = "segy"
    preferred_extensions = ("segy", "sgy")
    # also specify a version so when version 2 is released you can
    # just make another class in the same module named JingleV2.
    version = "1.0"
    # The name of the package to import. This is here so the class can be
    # subclassed and this changed for debugging reasons.
    _package_name = "segyio"

    def get_version(self, resource: LocalBinaryReader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        return _match[1] if (_match := _get_segy_version(resource)) else None

    def read_array(
        self, resource: LocalPath, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """Read selected trace samples in time/channel order."""
        segyio = optional_import(self._package_name)
        with segyio.open(str(resource), ignore_geometry=True) as stream:
            shape = (len(stream.samples), len(stream.header))
            time, channel = windows_to_slices(windows, ("time", "channel"), shape)
            channels = range(shape[1])[channel]
            if not channels:
                return np.empty((len(range(shape[0])[time]), 0), dtype=stream.dtype)
            return np.stack([stream.trace[index][time] for index in channels], axis=-1)

    def get_metadata(
        self, resource: LocalPath, *, snap: snap_type = True
    ) -> list[dc.Patch]:
        """
        Used to get metadata about a file without reading the whole file.

        Returns lightweight scan metadata without loading the data array.
        """
        segyio = optional_import(self._package_name)
        path_str = str(resource)
        with segyio.open(path_str, ignore_geometry=True) as fi:
            coords = _get_coords(fi)
            attrs = dc.PatchAttrs()
            dtype = str(fi.dtype)
        return [dc.Patch(attrs=attrs, coords=coords, dtype=dtype)]

    def write(self, spool: dc.Patch | dc.Spool, resource, **kwargs):
        """
        Create a segy file from length 1 spool or patch.

        Parameters
        ----------
        spool
            The patch or length 1 spool to write.
        resource
            The target for writing patch.

        Notes
        -----
        Based on the example from segyio:
        https://github.com/equinor/segyio/blob/master/python/examples/make-file.py
        """
        segyio = optional_import(self._package_name)
        _write_segy(spool, resource, self.version, segyio)


class SegyV0_0(SegyV1_0):  # noqa
    """
    An IO class supporting version 0.0 of the SEGY format.

    Or if the version is not set.
    """

    version = "0.0"


class SegyV0_100(SegyV1_0):  # noqa
    """
    An IO class supporting version 0.100 of the SEGY format.

    This odd version is output by some Silixa (Carina) files.
    """

    version = "0.100"


class SegyV2_0(SegyV1_0):  # noqa
    """An IO class supporting version 2.0 of the SEGY format."""

    version = "2.0"


class SegyV2_1(SegyV1_0):  # noqa
    """An IO class supporting version 2.1 of the SEGY format."""

    version = "2.1"
