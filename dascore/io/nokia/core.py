"""Core module for the Nokia Bell Labs multi-span (OFDR) formats."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import snap_type, windows_type
from dascore.core.source import ArraySource
from dascore.io import FiberIO
from dascore.io.utils import resolve_keyed_source, step_from_rate, windows_to_slices
from dascore.models import OptionalFiniteFloat
from dascore.utils.hdf5 import H5Reader
from dascore.utils.io import BinaryReader, LocalBinaryReader

from .utils import (
    _binary_time_step,
    _get_binary_version,
    _get_patch_infos,
    _h5_meta,
    _read_binary_header,
    _read_chunked,
)


class NokiaOFDRPatchAttrs(dc.PatchAttrs):
    """Patch attrs for the Nokia OFDR formats."""

    gauge_length: OptionalFiniteFloat = None


def _to_metas(infos, dtype) -> list[dc.PatchMeta]:
    """Turn (key, attrs, coords, row) tuples into patch metadata."""
    return [
        dc.PatchMeta(
            attrs=NokiaOFDRPatchAttrs.from_dict(attrs),
            coords=coords,
            dtype=dtype,
            source=ArraySource(key=key) if key else None,
        )
        for key, attrs, coords, _ in infos
    ]


def _resolve_rows(infos, key, where) -> slice:
    """Return the stored rows of the span a key names."""
    pairs = [(name, (row, coords.shape[0])) for name, _, coords, row in infos]
    row, count = resolve_keyed_source(pairs, key, where=where)
    return slice(row, row + count)


class NokiaOFDRV1(FiberIO):
    """Support for the Nokia multi-span binary format, version 1."""

    name = "NokiaOFDR"
    version = "1"

    def get_version(self, resource: BinaryReader, **kwargs) -> str | None:
        """Return the version when the file starts and ends with the magic."""
        version = _get_binary_version(resource)
        return version if version == self.version else None

    def _infos(self, resource):
        """Return the array info, start, chunk count and per-span info."""
        array_info, meta, array_start, nchunks = _read_binary_header(resource)
        _, nchan, nsamp = array_info["shape"]
        step = _binary_time_step(meta)
        infos = _get_patch_infos(meta, nchan, nchunks * nsamp, step)
        return array_info, array_start, nchunks, infos

    def get_metadata(
        self, resource: BinaryReader, *, snap: snap_type = ()
    ) -> list[dc.PatchMeta]:
        """Describe one patch per stored span of the file."""
        array_info, *_, infos = self._infos(resource)
        return _to_metas(infos, str(np.dtype(array_info["dtype"])))

    def read_array(
        self, resource: LocalBinaryReader, windows: windows_type = (), key: str = ""
    ) -> np.ndarray:
        """Memory-map the chunks and read only those the windows cover."""
        array_info, array_start, nchunks, infos = self._infos(resource)
        rows = _resolve_rows(infos, key, where=str(getattr(resource, "name", "")))
        _, nchan, nsamp = array_info["shape"]
        shape = (rows.stop - rows.start, nchunks * nsamp)
        dist, time = windows_to_slices(windows, shape)
        data = np.memmap(
            resource,
            dtype=array_info["dtype"],
            mode="r",
            offset=array_start,
            shape=(nchunks, nchan, nsamp),
        )
        dist = slice(rows.start + dist.start, rows.start + dist.stop)
        return _read_chunked(data, dist, time)


class NokiaOFDRH5V1(FiberIO):
    """Support for the downsampled Nokia multi-span HDF5 files, version 1."""

    name = "NokiaOFDRH5"
    preferred_extensions = ("h5",)
    version = "1"

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the version when the root data carries the stream attrs."""
        data = resource.get("data")
        attrs = getattr(data, "attrs", {})
        if "shm_type" in attrs and "stream.proc_start" in attrs:
            return self.version
        return None

    @staticmethod
    def _infos(resource):
        """Return the per-span info of the downsampled data."""
        data = resource["data"]
        meta = _h5_meta(data)
        nchan, ntime = data.shape
        return _get_patch_infos(meta, nchan, ntime, step_from_rate(meta["new_fs"]))

    def get_metadata(
        self, resource: H5Reader, *, snap: snap_type = ()
    ) -> list[dc.PatchMeta]:
        """Describe one patch per stored span of the file."""
        return _to_metas(self._infos(resource), str(resource["data"].dtype))

    def read_array(
        self, resource: H5Reader, windows: windows_type = (), key: str = ""
    ) -> np.ndarray:
        """Slice the stored rows of one span."""
        infos = self._infos(resource)
        rows = _resolve_rows(infos, key, where=str(resource.filename))
        shape = (rows.stop - rows.start, resource["data"].shape[1])
        dist, time = windows_to_slices(windows, shape)
        dist = slice(rows.start + dist.start, rows.start + dist.stop)
        return resource["data"][dist, time]
