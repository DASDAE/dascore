"""Utilities for the Nokia Bell Labs multi-span (OFDR) formats."""

from __future__ import annotations

import struct
from decimal import Decimal

import numpy as np
from scipy.constants import speed_of_light

from dascore.core import get_coord, get_coord_manager
from dascore.io.utils import step_from_rate
from dascore.utils.misc import optional_import

MAGIC = b"OFDR1MAGIC"
# version (uint8), header bytes, array start (uint32s), after the magic.
_PREAMBLE = struct.Struct("<BII")
_UINT = struct.Struct("<I")
# Each chunk holds this many time samples for every channel.
SAMPLES_PER_CHUNK = 32
# Group index of the fiber used in the gauge length formula.
_GROUP_INDEX = 1.4677


def _unpack_all(raw: bytes) -> list:
    """Decode every msgpack object in a byte string."""
    msgpack = optional_import("msgpack")  # optional dependency
    unpacker = msgpack.Unpacker(raw=False, strict_map_key=False)
    unpacker.feed(raw)
    return list(unpacker)


def _get_size(fi) -> int:
    """Return the size of an open binary file."""
    fi.seek(0, 2)
    return fi.tell()


def _get_binary_version(fi) -> str | None:
    """Return the version byte if the file starts and ends with the magic."""
    fi.seek(0)
    head = fi.read(len(MAGIC) + _PREAMBLE.size)
    if len(head) < len(MAGIC) + _PREAMBLE.size or not head.startswith(MAGIC):
        return None
    version = head[len(MAGIC)]
    fi.seek(_get_size(fi) - len(MAGIC))
    # A file without the end magic is still being written; skip it.
    if fi.read(len(MAGIC)) != MAGIC:
        return None
    return str(version)


def _read_binary_header(fi) -> tuple[dict, dict, int, int]:
    """
    Return the array info, chunk meta, array start and number of chunks.

    The header's second object describes the recording host and is ignored.
    """
    fi.seek(len(MAGIC))
    _, header_bytes, array_start = _PREAMBLE.unpack(fi.read(_PREAMBLE.size))
    array_info, _host, chunk_meta = _unpack_all(fi.read(header_bytes))[:3]
    # The trailer, framed by its length, sits before the end magic.
    size = _get_size(fi)
    fi.seek(size - len(MAGIC) - _UINT.size)
    (trailer_len,) = _UINT.unpack(fi.read(_UINT.size))
    fi.seek(size - len(MAGIC) - _UINT.size - trailer_len)
    bytes_written = _unpack_all(fi.read(trailer_len))[0]["bytes_written"]
    _, nchan, nsamp = array_info["shape"]
    itemsize = np.dtype(array_info["dtype"]).itemsize
    chunk_bytes = nchan * nsamp * itemsize
    assert nsamp == SAMPLES_PER_CHUNK and not bytes_written % chunk_bytes
    return array_info, chunk_meta, array_start, bytes_written // chunk_bytes


def _get_dx(meta) -> float:
    """Return the channel spacing in m from the acquisition parameters."""
    gauge, gamma = float(meta["gauge"]), float(meta["gamma"])
    return gauge * (0.25 / 2**24) / gamma * (speed_of_light / 2 / 1.45) * 4e-9


def _get_attrs(meta) -> dict:
    """Return the patch attrs shared by every span of a file."""
    gauge_length = float(meta["gauge"]) * speed_of_light / 250e6 / 2 / _GROUP_INDEX
    return {
        "data_category": "DAS",
        "data_type": "phase",
        "data_units": "rad",
        "gauge_length": gauge_length,
    }


def _get_spans(meta, nchan: int) -> list[tuple[int, int, int]]:
    """
    Return (first row, row count, first channel index) for each patch.

    A file whose ``masked`` spans add up to its channels stores only those
    spans, one after another; otherwise (no mask, or a mask recorded but not
    applied) the channels run contiguously from zero.
    """
    dx = _get_dx(meta)
    starts, counts = [], []
    for start, end in meta.get("masked") or ():
        first = int(float(start) / dx)
        starts.append(first)
        counts.append(int(float(end) / dx) - first)
    if not starts or sum(counts) != nchan:
        return [(0, nchan, 0)]
    rows = np.cumsum([0, *counts[:-1]])
    return list(zip(rows.tolist(), counts, starts))


def _get_patch_infos(meta, nchan: int, ntime: int, step) -> list:
    """Return (key, attrs, coords, first row) for each span of a file."""
    dx = _get_dx(meta)
    attrs = _get_attrs(meta)
    # Read the unix seconds as written, not the float's nanosecond expansion.
    seconds = Decimal(repr(float(meta["stream.proc_start"])))
    start = np.datetime64(int(seconds * 10**9), "ns")
    time = get_coord(start=start, step=step, shape=(ntime,))
    spans = _get_spans(meta, nchan)
    out = []
    for index, (row, count, first) in enumerate(spans):
        distance = get_coord(start=first * dx, step=dx, shape=(count,), units="m")
        coords = get_coord_manager(
            {"distance": distance, "time": time}, dims=("distance", "time")
        )
        key = str(index) if len(spans) > 1 else ""
        out.append((key, attrs, coords, row))
    return out


def _binary_time_step(meta):
    """Return the time step from the acquisition parameters."""
    gauge, tp = float(meta["gauge"]), float(meta["Tp"])
    return step_from_rate(gauge / (tp * float(meta["D"]) * float(meta["D2"])))


def _h5_meta(dataset) -> dict:
    """Return the downsampled dataset's attrs as plain python values."""
    return {
        name: value.tolist() if isinstance(value, np.ndarray | np.generic) else value
        for name, value in dataset.attrs.items()
    }


def _read_chunked(data, rows: slice, times: slice) -> np.ndarray:
    """
    Read rows and times out of a (chunk, channel, sample) array.

    Only the chunks covering the time window are read.
    """
    first = times.start // SAMPLES_PER_CHUNK
    stop = -(-times.stop // SAMPLES_PER_CHUNK)
    block = np.asarray(data[first:stop, rows, :])
    # (chunk, channel, sample) -> (channel, chunk * sample)
    out = block.transpose(1, 0, 2).reshape(block.shape[1], -1)
    offset = first * SAMPLES_PER_CHUNK
    return out[:, times.start - offset : times.stop - offset]
