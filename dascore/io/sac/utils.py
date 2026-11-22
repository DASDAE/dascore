"""Utilities for reading SAC binary files."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np

import dascore as dc
from dascore.constants import ONE_BILLION, snap_type
from dascore.core import get_coord, get_coord_manager
from dascore.exceptions import InvalidFiberFileError
from dascore.io.utils import _resource_name, stored_coord, windows_to_slices
from dascore.utils.io import BinaryReader, read_exactly
from dascore.utils.time import to_exact_fraction

# 70 floats, then 40 ints (the last five logicals), then 24 eight-byte strings.
_HEADER_SIZE = 632
_INT_OFFSET = 280
_STRING_OFFSET = 440
# Version 7 appends 22 doubles after the data, DELTA and B first.
_FOOTER_SIZE = 22 * 8
_VERSIONS = (6, 7)
_UNDEFINED = -12345
_ITIME = 1
# Spectral files hold two data sections, as unevenly sampled ones do.
_SPECTRAL = (2, 3)
# Positions in the int block, and of the identity strings among the strings.
_NVHDR, _NPTS, _IFTYPE, _LEVEN = 6, 9, 15, 35
_KSTNM, _KHOLE, _KCMPNM, _KNETWK = 0, 3, 20, 21


@dataclass(frozen=True)
class _Header:
    """The SAC header fields the reader uses, undefined text read as empty."""

    byte_order: str
    version: int
    npts: int
    delta: float
    begin: float
    reference: tuple[int, ...]
    iftype: int
    leven: int
    network: str
    station: str
    location: str
    seed_channel: str

    @property
    def dtype(self) -> np.dtype:
        """Return the stored sample dtype."""
        return np.dtype(f"{self.byte_order}f4")


def _text(head: bytes, index: int) -> str:
    """Return a string field, ending at a null, with the undefined value empty."""
    start = _STRING_OFFSET + 8 * index
    text = head[start : start + 8].split(b"\x00", 1)[0].decode("latin-1").strip()
    return "" if text == str(_UNDEFINED) else text


def _exact(value) -> Fraction:
    """Return a simple fraction a float holds exactly, else its shortest decimal."""
    return to_exact_fraction(value) or Fraction(str(value))


def _read_header(resource: BinaryReader) -> _Header | None:
    """
    Return the header of a SAC binary file, or None for any other file.

    Only the header is read. NVHDR must be 6 or 7 in either byte order, DELTA
    positive and finite, and the file the size NPTS samples make in the data
    sections IFTYPE and LEVEN call for (plus a version 7 footer).
    """
    resource.seek(0)
    head = read_exactly(resource, _HEADER_SIZE)
    if len(head) < _HEADER_SIZE:
        return None
    for order in "<>":
        ints = np.frombuffer(head, f"{order}i4", 40, _INT_OFFSET)
        if ints[_NVHDR] in _VERSIONS:
            break
    else:
        return None
    floats = np.frombuffer(head, f"{order}f4", 70)
    version, npts, delta = int(ints[_NVHDR]), int(ints[_NPTS]), floats[0]
    iftype, leven = int(ints[_IFTYPE]), int(ints[_LEVEN])
    sections = 2 if leven == 0 or iftype in _SPECTRAL else 1
    resource.seek(0, 2)
    footer = _FOOTER_SIZE if version == 7 else 0
    data_size = resource.tell() - _HEADER_SIZE - footer
    if npts < 0 or data_size != 4 * npts * sections or not 0 < delta < np.inf:
        return None
    return _Header(
        byte_order=order,
        version=version,
        npts=npts,
        delta=delta,
        begin=floats[5],
        reference=tuple(int(x) for x in ints[:6]),
        iftype=iftype,
        leven=leven,
        network=_text(head, _KNETWK),
        station=_text(head, _KSTNM),
        location=_text(head, _KHOLE),
        seed_channel=_text(head, _KCMPNM),
    )


def _read_trace_header(resource: BinaryReader) -> _Header:
    """
    Return the header of an evenly sampled SAC time series.

    A version 7 file's footer supplies DELTA and B in double precision.
    """
    if (header := _read_header(resource)) is None:
        msg = f"{_resource_name(resource)} is not a SAC binary file."
        raise InvalidFiberFileError(msg)
    if header.iftype != _ITIME or header.leven != 1:
        msg = (
            "Only evenly sampled SAC time series are read; this file has "
            f"IFTYPE={header.iftype} and LEVEN={header.leven}."
        )
        raise InvalidFiberFileError(msg)
    if header.version == 7:
        resource.seek(_HEADER_SIZE + 4 * header.npts)
        footer = read_exactly(resource, _FOOTER_SIZE)
        delta, begin = np.frombuffer(footer, f"{header.byte_order}f8", 2)
        if not 0 < delta < np.inf:
            msg = f"The SAC footer's DELTA ({delta}) is not a positive number."
            raise InvalidFiberFileError(msg)
        header = replace(header, delta=delta, begin=begin)
    return header


def _get_start(header: _Header) -> np.datetime64 | np.timedelta64:
    """
    Return the first sample's time: the reference time plus B.

    The reference fields are added up, so a value past its range rolls over.
    Without a reference time, the start is B as an offset.
    """
    begin = 0 if header.begin == _UNDEFINED else _exact(header.begin)
    offset = np.timedelta64(round(begin * ONE_BILLION), "ns")
    if _UNDEFINED in header.reference:
        return offset
    year, day, hour, minute, second, msec = header.reference
    # datetime64[ns] holds these years only; others would wrap silently.
    if not 1678 <= year <= 2261:
        msg = f"The SAC reference year {year} is outside the years DASCore holds."
        raise InvalidFiberFileError(msg)
    seconds = (((day - 1) * 24 + hour) * 60 + minute) * 60 + second
    start = np.datetime64(f"{year:04d}-01-01", "ns")
    return start + np.timedelta64(seconds * 1000 + msec, "ms") + offset


def _source_id(header: _Header) -> str:
    """Return the FDSN source ID, splitting a three-letter channel code."""
    channel = header.seed_channel
    channel = "_".join(channel) if len(channel) == 3 else channel
    return f"FDSN:{header.network}_{header.station}_{header.location}_{channel}"


def _get_metadata(header: _Header, snap: snap_type = ()) -> dc.PatchMeta:
    """Return the metadata of a SAC trace as a one-channel patch."""
    codes = (header.network, header.station, header.location, header.seed_channel)
    coords = {
        "channel": stored_coord(np.asarray([0]), snap, name="channel"),
        "time": get_coord(
            start=_get_start(header),
            step=_exact(header.delta),
            shape=(header.npts,),
        ),
        "source_id": (("channel",), (_source_id(header),)),
        "network": (("channel",), (header.network,)),
        "station": (("channel",), (header.station,)),
        "location": (("channel",), (header.location,)),
        "seed_channel": (("channel",), (header.seed_channel,)),
    }
    return dc.PatchMeta(
        attrs={"tag": ".".join(codes)},
        coords=get_coord_manager(coords, dims=("channel", "time")),
        dtype="float32",
    )


def _read_samples(resource: BinaryReader, header: _Header, windows) -> np.ndarray:
    """Read the windowed samples, in native byte order, as one row."""
    channels, time = windows_to_slices(windows, (1, header.npts))
    resource.seek(_HEADER_SIZE + 4 * time.start)
    raw = read_exactly(resource, 4 * (time.stop - time.start))
    data = np.frombuffer(raw, header.dtype).astype(np.float32)
    return data[np.newaxis][channels]
