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
    _get_segy_compatible_patch,
    _get_segy_version,
    _make_time_header_dict,
)


class SegyV1_0(FiberIO):  # noqa
    """An IO class supporting version 1.0 of the SEGY format."""

    name = "segy"
    preferred_extensions = ("segy", "sgy")
    # also specify a version so when version 2 is released you can
    # just make another class in the same module named JingleV2.
    version = "1.0"

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
        segyio = optional_import("segyio")
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
        segyio = optional_import("segyio")
        with segyio.open(path, ignore_geometry=True) as fi:
            coords = _get_coords(fi)
            attrs = _get_attrs(fi, coords, path, self)
        return [attrs]

    def write(self, spool, resource, **kwargs):
        """
        Create a segy file from length 1 spool.

        Based on the example from segyio:
        https://github.com/equinor/segyio/blob/master/python/examples/make-file.py
        """
        patch = _get_segy_compatible_patch(spool)
        time, distance = patch.get_coord("time"), patch.get_coord("distance")
        distance_step = distance.step

        time_dict = _make_time_header_dict(time)

        segyio = optional_import("segyio")
        spec = segyio.spec()

        spec.sorting = 2
        spec.format = 1
        spec.samples = [len(time)] * len(distance)
        spec.ilines = range(len(distance))
        spec.xlines = [1]

        with segyio.create(resource, spec) as f:
            # This works because we ensure dim order is (distance, time)
            for num, data in enumerate(patch.data):
                header = dict(time_dict)
                header.update(
                    {
                        segyio.su.offset: distance_step,
                        segyio.su.iline: num,
                        segyio.su.xline: 1,
                        segyio.su.yline: 1,
                    }
                )
                f.header[num] = header
                f.trace[num] = data

            f.bin.update(tsort=segyio.TraceSortingFormat.INLINE_SORTING)


class SegyV2_0(SegyV1_0):  # noqa
    """An IO class supporting version 2.0 of the SEGY format."""

    version = "2.0"


class SegyV2_1(SegyV1_0):  # noqa
    """An IO class supporting version 2.1 of the SEGY format."""

    version = "2.1"
