"""IO module for reading SEGY file format support."""

from __future__ import annotations

import dascore as dc
from dascore.io.core import FiberIO
from dascore.utils.io import BinaryReader
from dascore.utils.misc import optional_import

from .utils import (
    _get_attrs,
    _get_coords,
    _get_filtered_data_and_coords,
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

    def get_format(self, fp: BinaryReader, **kwargs) -> tuple[str, str] | bool:
        """Make sure input is segy."""
        return _get_segy_version(fp)

    def read(self, path, time=None, channel=None, **kwargs):
        """
        Read should take a path and return a patch or sequence of patches.

        It can also define its own optional parameters, and should always
        accept kwargs. If the format supports partial reads, these should
        be implemented as well.
        """
        segyio = optional_import(self._package_name)
        with segyio.open(path, ignore_geometry=True) as fi:
            coords = _get_coords(fi)
            attrs = _get_attrs(fi, coords, path, self)
            data, coords = _get_filtered_data_and_coords(
                fi, coords, time=time, channel=channel
            )

        patch = dc.Patch(coords=coords, data=data, attrs=attrs)
        patch_trimmed = patch.select(time=time, channel=channel)
        return dc.spool([patch_trimmed])

    def scan(self, path, **kwargs) -> list[dc.PatchAttrs]:
        """
        Used to get metadata about a file without reading the whole file.

        This should return a list of
        [`PatchAttrs`](`dascore.core.attrs.PatchAttrs`) objects
        from the [dascore.core.attrs](`dascore.core.attrs`) module, or a
        format-specific subclass.
        """
        segyio = optional_import(self._package_name)
        with segyio.open(path, ignore_geometry=True) as fi:
            coords = _get_coords(fi)
            attrs = _get_attrs(fi, coords, path, self)
        return [attrs]

    def write(self, spool: dc.Patch | dc.BaseSpool, resource, **kwargs):
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
