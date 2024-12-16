"""IO module for reading SEGY file format support."""

from __future__ import annotations

import numpy as np

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
        time, channel = patch.get_coord("time"), patch.get_coord("channel")
        chanel_step = channel.step

        time_dict = _make_time_header_dict(time)

        segyio = optional_import("segyio")
        bin_field = segyio.BinField

        spec = segyio.spec()
        version = self.version

        # spec.sorting = 2
        spec.format = 1  # 1 means float32 TODO look into supporting more
        spec.samples = np.ones(len(time)) * len(channel)
        spec.ilines = range(len(channel))
        spec.xlines = [1]
        # breakpoint()

        # For 32 bit float for now.
        data = patch.data.astype(np.float32)

        with segyio.create(resource, spec) as f:
            # Update the file header info.
            f.bin.update(tsort=segyio.TraceSortingFormat.INLINE_SORTING)
            f.bin.update(
                {
                    bin_field.Samples: time_dict[segyio.TraceField.TRACE_SAMPLE_COUNT],
                    bin_field.Interval: time_dict[
                        segyio.TraceField.TRACE_SAMPLE_INTERVAL
                    ],
                    bin_field.SEGYRevision: int(version.split(".")[0]),
                    bin_field.SEGYRevisionMinor: int(version.split(".")[1]),
                }
            )
            # Then iterate each channel and dump to segy.
            for num, data in enumerate(data):
                header = dict(time_dict)
                header.update(
                    {
                        segyio.su.offset: chanel_step,
                        segyio.su.iline: num,
                        segyio.su.xline: 1,
                    }
                )
                f.header[num] = header
                f.trace[num] = data


class SegyV2_0(SegyV1_0):  # noqa
    """An IO class supporting version 2.0 of the SEGY format."""

    version = "2.0"


class SegyV2_1(SegyV1_0):  # noqa
    """An IO class supporting version 2.1 of the SEGY format."""

    version = "2.1"
