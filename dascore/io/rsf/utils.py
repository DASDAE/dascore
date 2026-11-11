"""Utilities for reading RSF headers and locating their binaries."""

from __future__ import annotations

import json
import re
from contextlib import suppress
from fractions import Fraction
from pathlib import Path

import numpy as np

import dascore as dc
from dascore.core.attrs import drop_unheld_attrs
from dascore.exceptions import InvalidFiberFileError
from dascore.io.utils import parse_units, step_from_interval

# The bytes between a packed header and its binary.
SEPARATOR = b"\x0c\x0c\x04"
# The most header text read; it holds every header DASCore or Madagascar writes.
HEADER_MAX = 1 << 16
# A line binds when it opens with a key=value token; other lines are prose.
_BINDS = re.compile(r"\s*[A-Za-z_]\w*=")
_TOKEN = re.compile(r'([A-Za-z_]\w*)=("(?:[^"\\]|\\.)*"|\S*)')
_AXIS = re.compile(r"n([1-9])")
_ORDERS = {"native": "=", "xdr": ">"}
KINDS = {"float": "f4", "double": "f8", "int": "i4", "short": "i2", "complex": "c8"}
# Keys that describe the file rather than the patch; an attr of the same
# name is written with the prefix.
RESERVED = re.compile(
    r"(?:n|o|d|label|unit|epoch|coord)\d+|coord_\w+|esize|data_format|in|starttime"
)
PREFIX = "dascore_"
# What opens a history line DASCore wrote, before the version.
HISTORY = "# dascore "


def _unquote(value: str, escapes: bool = True) -> str:
    """Return a value's text, without its quotes and, optionally, escapes."""
    if not value.startswith('"'):
        return value
    return re.sub(r'\\([\\"])', r"\1", value[1:-1]) if escapes else value[1:-1]


def _get_bindings(text: str):
    """Yield each key and value token of the header lines that bind."""
    for line in text.split("\n"):
        if _BINDS.match(line):
            yield from _TOKEN.findall(line)


def read_header(handle, raise_on_fail=False):
    """
    Return a header's bindings, the offset of a packed binary, and its text.

    The offset is None when no separator follows the header text. Anything
    but an RSF header returns None, or raises when raise_on_fail.
    """
    head, sep, _ = handle.read(HEADER_MAX).partition(SEPARATOR)
    text = head.decode(errors="replace")
    # Madagascar keeps the backslashes of an in= path (a UNC share) as is.
    hdr = {k: _unquote(v, escapes=k != "in") for k, v in _get_bindings(text)}
    if b"\x00" not in head and "in" in hdr and "n1" in hdr:
        return hdr, (len(head) + len(sep) if sep else None), text
    if raise_on_fail:
        msg = f"{getattr(handle, 'name', handle)} does not hold an RSF header."
        raise InvalidFiberFileError(msg)
    return None


def get_dtype(hdr) -> np.dtype:
    """Return the stored dtype named by data_format."""
    fmt = hdr.get("data_format", "native_float")
    order, _, kind = fmt.partition("_")
    if order not in _ORDERS or kind not in KINDS:
        msg = f"RSF data_format {fmt!r} is not supported."
        raise InvalidFiberFileError(msg)
    return np.dtype(_ORDERS[order] + KINDS[kind])


def _get_shape(hdr) -> tuple[int, ...]:
    """Return the axis lengths, n1 first; an unbound n<i> is one sample."""
    ndim = max(int(m[1]) for key in hdr if (m := _AXIS.fullmatch(key)))
    return tuple(int(hdr.get(f"n{i}", 1)) for i in range(1, ndim + 1))


def get_coords(hdr) -> dict:
    """
    Return a coordinate for each axis, keyed by label and in n1 order.

    An axis with epoch<i>, or the time axis with the legacy starttime=, is
    datetime; a time axis in s without either holds offsets (timedelta).
    """
    coords = {}
    for i, n in enumerate(_get_shape(hdr), start=1):
        label = hdr.get(f"label{i}")
        name = label if label and label not in coords else f"axis{i}"
        # free text such as unit1="hz" or "Various" is not units; drop it
        units = parse_units(hdr.get(f"unit{i}") or None)
        origin, step = hdr.get(f"o{i}", "0"), float(hdr.get(f"d{i}", "1"))
        if (epoch := hdr.get(f"epoch{i}")) is not None:
            zero = np.datetime64(epoch.removesuffix("Z"), "ns")
        elif name == "time" and "starttime" in hdr:
            zero = dc.to_datetime64(float(hdr["starttime"]))
        elif name == "time" and units == "s":
            zero = np.timedelta64(0, "ns")
        else:
            coords[name] = dc.get_coord(
                start=float(origin), step=step, shape=(n,), units=units
            )
            continue
        start = zero + np.timedelta64(round(Fraction(origin) * 10**9), "ns")
        # the writer stores the double nearest an exact step such as 1/3000
        exact = step_from_interval(abs(step)) * (-1 if step < 0 else 1)
        coords[name] = dc.get_coord(start=start, step=exact, shape=(n,))
    return coords


def _get_attr_value(value: str):
    """Return a bare value as the JSON it spells, else its text."""
    text = _unquote(value)
    if not value.startswith('"'):
        with suppress(ValueError):
            return json.loads(text)
    return text


def get_attrs(text: str) -> dict:
    """
    Return the attrs and the history DASCore wrote into a header.

    Keys that describe the file are skipped and the prefix is removed. A
    PatchAttrs field takes the value's text, an extra its typed value. Only
    the history lines DASCore wrote are read; another program's are prose.
    """
    fields = dc.PatchAttrs.model_fields
    out = {}
    for key, value in _get_bindings(text):
        if not RESERVED.fullmatch(key):
            name = key.removeprefix(PREFIX)
            out[name] = _unquote(value) if name in fields else _get_attr_value(value)
    out = drop_unheld_attrs(out)
    lines = text.split("\n")
    if history := [x[len(HISTORY) :] for x in lines if x.startswith(HISTORY)]:
        # each line holds the version before the entry
        out["history"] = tuple(x.partition(" ")[2] for x in history)
    return out


def get_layout(path) -> tuple[Path, int, np.dtype, tuple[int, ...]]:
    """
    Return the file holding the samples, their offset, dtype and shape.

    A relative in= resolves against the header's directory, then the
    working directory, where Madagascar recorded it.
    """
    path = Path(path)
    with path.open("rb") as handle:
        hdr, offset, _ = read_header(handle, raise_on_fail=True)
    dtype, shape, source = get_dtype(hdr), _get_shape(hdr), hdr["in"]
    if source != "stdin":
        # an absolute source replaces the parent when joined
        binary, offset = path.parent / source, 0
        binary = binary if binary.exists() else Path(source)
    elif offset is None:
        msg = f"RSF header {path} has in=stdin but no binary separator."
        raise InvalidFiberFileError(msg)
    else:
        binary = path
    size = int(np.prod(shape)) * dtype.itemsize
    if binary.stat().st_size - offset != size:
        msg = f"RSF binary {binary} must hold exactly {size} bytes."
        raise InvalidFiberFileError(msg)
    return binary, offset, dtype, shape
