"""Utilities for reading RSF headers and locating their binaries."""

from __future__ import annotations

import re
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np

import dascore as dc
from dascore.exceptions import InvalidFiberFileError
from dascore.warnings import DASCoreWarning

# The bytes between a packed header and its binary.
SEPARATOR = b"\x0c\x0c\x04"
# The longest header text read, so a large text file is not read whole.
_HEADER_MAX = 1 << 20
_BLOCK = 1 << 16
# A line binds when it opens with a key=value token.
_BINDS = re.compile(r"\s*[A-Za-z_]\w*=")
_TOKEN = re.compile(r"([A-Za-z_]\w*)=(\"[^\"]*\"|'[^']*'|\S*)")
_AXIS = re.compile(r"n([1-9])")
_ORDERS = {"native": "=", "xdr": ">"}
_KINDS = {"float": "f4", "double": "f8", "int": "i4", "short": "i2", "complex": "c8"}


def _read_header(handle) -> tuple[dict[str, str], int | None] | None:
    """
    Return a header's bindings and the offset of a packed binary.

    The offset is None when no separator follows the header text, and
    None is returned in place of both for anything but an RSF header.
    """
    raw = b""
    while len(raw) <= _HEADER_MAX and (block := handle.read(_BLOCK)):
        raw += block
        if b"\x00" in block or SEPARATOR in raw:
            break
    head, sep, _ = raw.partition(SEPARATOR)
    if b"\x00" in head or len(head) > _HEADER_MAX:
        return None
    try:
        text = head.decode()
    except UnicodeDecodeError:
        text = head.decode("latin-1")
    hdr = {}
    for line in text.split("\n"):
        if _BINDS.match(line):
            for key, value in _TOKEN.findall(line):
                quoted = value[:1] in {'"', "'"}
                hdr[key] = value[1:-1] if quoted else value
    if "in" not in hdr or "n1" not in hdr:
        return None
    return hdr, (len(head) + len(sep) if sep else None)


def _get_header(handle, path) -> tuple[dict[str, str], int | None]:
    """Return the header of an RSF file, or raise if it has none."""
    if (out := _read_header(handle)) is None:
        msg = f"{path} does not hold an RSF header."
        raise InvalidFiberFileError(msg)
    return out


def _get_dtype(hdr) -> np.dtype:
    """Return the stored dtype named by data_format."""
    fmt = hdr.get("data_format", "native_float")
    order, _, kind = fmt.partition("_")
    if order not in _ORDERS or kind not in _KINDS:
        msg = f"RSF data_format {fmt!r} is not supported."
        raise InvalidFiberFileError(msg)
    return np.dtype(_ORDERS[order] + _KINDS[kind])


def _get_shape(hdr) -> tuple[int, ...]:
    """Return the axis lengths, n1 first; an unbound n<i> is one sample."""
    ndim = max(int(m[1]) for key in hdr if (m := _AXIS.fullmatch(key)))
    return tuple(int(hdr.get(f"n{i}", 1)) for i in range(1, ndim + 1))


def _get_epoch(hdr, i, name, units) -> np.datetime64 | None:
    """Return the absolute time of axis i's zero, or None for no time axis."""
    epoch = hdr.get(f"epoch{i}")
    if epoch is not None and units != "s":
        msg = f"RSF epoch{i} is ignored because unit{i} is {units!r}, not 's'."
        warnings.warn(msg, DASCoreWarning, stacklevel=2)
        return None
    if epoch is not None:
        return np.datetime64(epoch.removesuffix("Z"), "ns")
    if name == "time" and "starttime" in hdr:
        return dc.to_datetime64(float(hdr["starttime"]))
    return None


def _get_coords(hdr) -> dict:
    """Return a coordinate for each axis, keyed by label and in n1 order."""
    coords = {}
    for i, n in enumerate(_get_shape(hdr), start=1):
        label = hdr.get(f"label{i}")
        name = label if label and label not in coords else f"axis{i}"
        units = hdr.get(f"unit{i}") or None
        origin, step = hdr.get(f"o{i}", "0"), hdr.get(f"d{i}", "1")
        epoch = _get_epoch(hdr, i, name, units)
        if epoch is None:
            coord = dc.get_coord(
                start=float(origin), step=float(step), shape=(n,), units=units
            )
        else:
            start = epoch + np.timedelta64(round(Fraction(origin) * 10**9), "ns")
            coord = dc.get_coord(start=start, step=Fraction(step), shape=(n,))
        coords[name] = coord
    return coords


def _get_layout(path) -> tuple[Path, int, np.dtype, tuple[int, ...]]:
    """
    Return the file holding the samples, their offset, dtype and shape.

    A relative in= resolves against the header's directory, then the
    working directory.
    """
    path = Path(path)
    with path.open("rb") as handle:
        hdr, offset = _get_header(handle, path)
    dtype, shape, source = _get_dtype(hdr), _get_shape(hdr), hdr["in"]
    if source != "stdin":
        # an absolute source replaces the parent when joined
        paths, offset = [path.parent / source, Path(source)], 0
    elif offset is None:
        msg = f"RSF header {path} has in=stdin but no binary separator."
        raise InvalidFiberFileError(msg)
    else:
        paths = [path]
    if (binary := next((x for x in paths if x.exists()), None)) is None:
        msg = f"RSF binary {source} named by {path} does not exist."
        raise InvalidFiberFileError(msg)
    size = int(np.prod(shape)) * dtype.itemsize
    if binary.stat().st_size - offset != size:
        msg = f"RSF binary {binary} must hold exactly {size} bytes."
        raise InvalidFiberFileError(msg)
    return binary, offset, dtype, shape
