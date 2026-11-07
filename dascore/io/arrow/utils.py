"""
Encode patches as Arrow IPC and decode them back.

One patch is one IPC stream or file whose single column, ``data``, holds the
patch's values flattened in C order. Dims, shape, coordinates and attrs are a
JSON document in the schema metadata under ``dascore:patch``. The format is
specified in dastor's ``docs/arrow-patch-format.md``.
"""

from __future__ import annotations

import datetime
import functools
import json
import math
from typing import Any

import numpy as np

import dascore as dc
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.exceptions import InvalidFiberFileError, PatchError
from dascore.io import ArraySource
from dascore.io.core import STORED_ORIGIN_ID
from dascore.io.utils import should_snap, stored_coord
from dascore.models.registry import get_model_tag, resolve_tagged_model
from dascore.units import Quantity, Unit, get_quantity, get_quantity_str, get_unit
from dascore.utils.misc import optional_import
from dascore.utils.time import to_datetime64, to_timedelta64

METADATA_KEY = b"dascore:patch"
FORMAT_NAME = "dascore.patch"
FORMAT_VERSION = "1.0"
# Every batch but the last holds at least this many values (or the whole
# patch), which bounds how many batches a patch can have.
MIN_BATCH_VALUES = 65536
# Writers aim for batches of about this many bytes.
_BATCH_BYTES = 16 * 1024**2
_DATA_TIME_UNITS = ("s", "ms", "us", "ns")
_NUMPY_KINDS = "biufmMU"
_TAGS = ("$float", "$scalar", "$array", "$quantity", "$unit")


class _UnsupportedError(PatchError, ValueError):
    """Raised when a patch holds something the format can't store."""


def _bad(msg: str) -> InvalidFiberFileError:
    """Return the error a malformed stream or file raises."""
    return InvalidFiberFileError(f"Invalid DASCore Arrow patch: {msg}")


def _check(ok, msg: str):
    """Raise for a malformed stream or file unless `ok`."""
    if not ok:
        raise _bad(msg)


def _malformed_is_invalid(func):
    """Report the errors a malformed document causes as an invalid file."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (
            AttributeError,
            IndexError,
            KeyError,
            OverflowError,
            PatchError,
            RecursionError,
            TypeError,
            ValueError,
        ) as exc:
            raise _bad(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


# --- tagged JSON ---------------------------------------------------------


def _encode_numpy(value: np.ndarray | np.generic):
    """Tag a numpy scalar or array with its dtype; times go as int64 counts."""
    dtype = value.dtype.newbyteorder("=")
    if dtype.kind not in _NUMPY_KINDS or (dtype.kind == "f" and dtype.itemsize > 8):
        raise _UnsupportedError(f"can't store values of dtype {value.dtype}")
    native = np.asarray(value).astype(dtype, copy=False)
    if dtype.kind in "mM":
        native = native.view(np.int64)
    tag = "$array" if isinstance(value, np.ndarray) else "$scalar"
    return {tag: [dtype.str, _encode(native.tolist())]}


def _encode(value):
    """Encode a value as JSON, tagging what JSON can't hold exactly."""
    if isinstance(value, np.ndarray | np.generic):  # before float: float64 is one
        return _encode_numpy(value)
    if isinstance(value, datetime.datetime | datetime.timedelta):
        if getattr(value, "tzinfo", None) is not None:
            raise _UnsupportedError(f"time-zone-aware times can't be stored: {value}")
        is_delta = isinstance(value, datetime.timedelta)
        return _encode((to_timedelta64 if is_delta else to_datetime64)(value))
    if isinstance(value, float) and not math.isfinite(value):
        return {"$float": str(value)}
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_encode(x) for x in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, Quantity | Unit):
        tag = "$quantity" if isinstance(value, Quantity) else "$unit"
        return {tag: get_quantity_str(value)}
    raise _UnsupportedError(f"can't store value {value!r}")


def _decode(value: Any) -> Any:
    """Decode a value written by `_encode`."""
    if isinstance(value, list):
        return [_decode(x) for x in value]
    if not isinstance(value, dict):
        return value
    if len(value) != 1 or (tag := next(iter(value))) not in _TAGS:
        return {k: _decode(v) for k, v in value.items()}
    item = value[tag]
    if tag == "$float":
        return float(item)
    if tag in ("$quantity", "$unit"):
        return (get_quantity if tag == "$quantity" else get_unit)(item)
    dtype = np.dtype(item[0])
    _check(dtype.kind in _NUMPY_KINDS, f"unsupported dtype {item[0]!r}")
    if dtype.kind in "mM":
        out = np.array(item[1], dtype=np.int64).view(dtype)
    else:
        out = np.array(_decode(item[1]), dtype=dtype)
    return out if tag == "$array" else out[()]


# --- coordinates ---------------------------------------------------------


def _window(run: dict, sources: dict, shape: tuple):
    """Replace a stored-labels run with the labels it selects."""
    if "id" not in run:
        return run
    index = run["offset"] + run["stride"] * np.arange(run["count"])
    labels = np.atleast_1d(sources[run["id"]])[index]
    return labels if len(shape) == 1 else labels.reshape(shape)


def coord_to_json(coord, dims) -> dict:
    """Write a coordinate as the format's coordinate document."""
    dump = coord.model_dump()
    sources = dump.pop("sources", {})
    if "runs" in dump:
        dump["runs"] = [_window(run, sources, coord.shape) for run in dump["runs"]]
    return _encode({**dump, "dims": list(dims), "dtype": dump["dtype"].str})


def json_to_coord(doc: dict):
    """Rebuild a coordinate from its document."""
    doc = _decode(doc)
    if "values" in doc:  # strings
        return get_coord(data=doc["values"])
    kwargs = {k: doc[k] for k in ("dtype", "step", "units")}
    return get_coord(runs=tuple(doc["runs"]), **kwargs)


# --- the patch document --------------------------------------------------


def arrow_type(dtype: np.dtype):
    """Return the Arrow type for a data dtype, refusing unsupported ones."""
    pa = optional_import("pyarrow")
    if dtype.kind in "mM":
        unit, count = np.datetime_data(dtype)
        if unit not in _DATA_TIME_UNITS or count != 1:
            raise _UnsupportedError(f"can't store data of dtype {dtype}")
        return pa.timestamp(unit) if dtype.kind == "M" else pa.duration(unit)
    if dtype.kind not in "biuf" or (dtype.kind == "f" and dtype.itemsize > 8):
        raise _UnsupportedError(f"can't store data of dtype {dtype}")
    return pa.from_numpy_dtype(dtype)


def patch_document(patch: dc.Patch) -> dict:
    """Return the ``dascore:patch`` document for a patch."""
    if np.ma.isMaskedArray(patch.data):
        raise _UnsupportedError("masked data can't be stored; fill it first")
    if not patch.dims:
        raise _UnsupportedError("a patch with no dims can't be stored")
    dtype = np.dtype(patch.dtype).newbyteorder("=")
    arrow_type(dtype)
    coords = {
        name: coord_to_json(coord, patch.coords.dim_map[name])
        for name, coord in patch.coords.coord_map.items()
    }
    attrs = patch.attrs.model_dump()
    attrs.pop(STORED_ORIGIN_ID, None)
    cls = type(patch.attrs)
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "dims": list(patch.dims),
        "shape": list(patch.shape),
        "dtype": dtype.newbyteorder("<").str if dtype.kind != "b" else dtype.str,
        "coords": coords,
        "attrs_class": None if cls is PatchAttrs else get_model_tag(cls),
        "attrs": _encode(attrs),
    }


@_malformed_is_invalid
def read_document(schema) -> dict:
    """Return the checked ``dascore:patch`` document from an Arrow schema."""
    raw = [v for k, v in (schema.metadata or {}).items() if k == METADATA_KEY]
    _check(raw, "no dascore:patch metadata")
    doc = json.loads(raw[0])
    _check(doc["format"] == FORMAT_NAME, "not a dascore.patch document")
    major = FORMAT_VERSION.partition(".")[0]
    _check(doc["version"].partition(".")[0] == major, "unsupported version")
    dims, shape = doc["dims"], doc["shape"]
    names_ok = all(type(x) is str and x for x in dims) and len(set(dims)) == len(dims)
    shape_ok = all(type(x) is int and x >= 0 for x in shape) and len(shape) == len(dims)
    _check(dims and names_ok and shape_ok, "bad dims or shape")
    field = schema.field(0)
    simple = len(schema) == 1 and field.name == "data" and not field.metadata
    _check(simple and not field.nullable, "expected one non-nullable field, data")
    dtype = np.dtype(doc["dtype"])
    _check(field.type == arrow_type(dtype), f"data is {field.type}, not {dtype}")
    return doc


@_malformed_is_invalid
def document_meta(doc: dict, snap=()) -> dc.PatchMeta:
    """Return the data-less patch a checked document describes."""
    dims = tuple(doc["dims"])
    coords = {}
    for name, cdoc in doc["coords"].items():
        coord = json_to_coord(cdoc)
        if name in dims and should_snap(snap, name) and not coord.evenly_sampled:
            coord = stored_coord(coord.values, True, name=name, units=coord.units)
        coords[name] = (tuple(cdoc["dims"]), coord)
    manager = get_coord_manager(coords, dims=dims, shape=tuple(doc["shape"]))
    _check(manager.shape == tuple(doc["shape"]), "coordinates don't fit the shape")
    attrs = _decode(doc["attrs"])
    # The stored id survives the read, as DASDAE's does.
    if stored := attrs.get("origin_id", ""):
        attrs[STORED_ORIGIN_ID] = stored
    tag = doc["attrs_class"]
    cls = PatchAttrs if tag is None else resolve_tagged_model(tag)
    return dc.PatchMeta(
        coords=manager,
        dims=dims,
        attrs=cls.from_dict(attrs),
        dtype=doc["dtype"],
        source=ArraySource(key=""),
    )


# --- data ----------------------------------------------------------------


def data_batches(patch: dc.Patch, schema):
    """Yield the record batches holding a patch's values, in first-dim slices."""
    pa = optional_import("pyarrow")
    data = np.asarray(patch.data)
    data = np.ascontiguousarray(data.astype(data.dtype.newbyteorder("<"), copy=False))
    flat = data.reshape(-1)
    per_row = max(1, math.prod(data.shape[1:]))
    target = max(MIN_BATCH_VALUES, _BATCH_BYTES // flat.dtype.itemsize)
    step = per_row * -(-target // per_row)
    arrow = schema.field(0).type
    for start in range(0, flat.size, step):
        part = flat[start : start + step]
        if part.dtype.kind == "b":
            array = pa.array(part, type=arrow)
        else:
            # From the raw bytes, so NaT stays a value rather than a null.
            array = pa.Array.from_buffers(arrow, len(part), [None, pa.py_buffer(part)])
        yield pa.record_batch([array], schema=schema)


def batch_values(batch, dtype: np.dtype) -> np.ndarray:
    """Return a checked batch's values as a numpy array (a view where possible)."""
    column = batch.column(0)
    _check(not column.null_count, "data holds nulls")
    if dtype.kind == "b":
        return column.to_numpy(zero_copy_only=False)
    values = np.frombuffer(column.buffers()[1], dtype=dtype)
    return values[column.offset : column.offset + len(column)]
