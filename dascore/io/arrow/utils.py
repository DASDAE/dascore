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
from typing import Any, cast

import numpy as np
import pandas as pd

import dascore as dc
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import CoordString, get_coord
from dascore.exceptions import InvalidFiberFileError, PatchError
from dascore.io import ArraySource
from dascore.io.core import STORED_ORIGIN_ID
from dascore.io.utils import should_snap, stored_coord
from dascore.models.registry import get_model_tag, resolve_tagged_model
from dascore.units import get_quantity, get_quantity_str
from dascore.utils.misc import optional_import

METADATA_KEY = b"dascore:patch"
FORMAT_NAME = "dascore.patch"
FORMAT_VERSION = "1.0"
# Every batch but the last holds at least this many values (or the whole
# patch), which bounds how many batches a patch can have.
MIN_BATCH_VALUES = 65536
# Writers aim for batches of about this many bytes.
_BATCH_BYTES = 16 * 1024**2
_SAFE_INT = 2**53
_DATA_TIME_UNITS = ("s", "ms", "us", "ns")


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


# --- writing values ------------------------------------------------------


def _int_text(value) -> str:
    """Write an integer as decimal text."""
    return str(int(value))


def _float_json(value):
    """Write a float, tagging the values JSON lacks."""
    value = float(value)
    if math.isfinite(value):
        return value
    return {"$float": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}


def _time_json(value) -> list[str]:
    """Write a datetime64 or timedelta64 as its count and unit."""
    unit, multiple = np.datetime_data(value.dtype)
    if multiple != 1:  # e.g. 10ms; the count alone would read as 1 ms steps
        raise _UnsupportedError(f"can't store times in units of {multiple}{unit}")
    return [_int_text(value.view(np.int64)), unit]


def _scalar_json(value):
    """Write a coordinate scalar: a grid term, step or blank end."""
    if isinstance(value, np.datetime64 | np.timedelta64):
        return _time_json(value)
    if isinstance(value, int | np.integer):
        return _int_text(value)
    return _float_json(value)


def _attr_json(value):
    """Encode one attribute value in the format's tagged JSON."""
    if value is None or isinstance(value, bool | str):
        return value
    # Numpy scalars first: float64 is a float, and timedelta64 an integer.
    if isinstance(value, np.datetime64 | np.timedelta64):
        return {f"${type(value).__name__}": _time_json(value)}
    if isinstance(value, np.bool_ | np.integer | np.floating):
        return {"$scalar": [value.dtype.str, _attr_json(value.item())]}
    if isinstance(value, int):
        return value if abs(value) < _SAFE_INT else {"$int": _int_text(value)}
    if isinstance(value, float):
        return _float_json(value)
    if isinstance(value, pd.Timestamp | datetime.datetime):
        if value.tzinfo is not None:
            raise _UnsupportedError(f"time-zone-aware times can't be stored: {value}")
        unit = "ns" if isinstance(value, pd.Timestamp) else "us"
        value = pd.Timestamp(value).as_unit(unit).to_datetime64()
    if isinstance(value, pd.Timedelta | datetime.timedelta):
        unit = "ns" if isinstance(value, pd.Timedelta) else "us"
        value = cast(pd.Timedelta, pd.Timedelta(value)).as_unit(unit).to_timedelta64()
    if isinstance(value, np.datetime64 | np.timedelta64):
        return {f"${type(value).__name__}": _time_json(value)}
    if isinstance(value, bytes):
        return {"$bytes": value.hex()}
    if isinstance(value, list | tuple):
        return [_attr_json(x) for x in value]
    if isinstance(value, dict):
        if all(isinstance(k, str) and not k.startswith("$") for k in value):
            return {k: _attr_json(v) for k, v in value.items()}
        if not all(isinstance(k, str | int) and not isinstance(k, bool) for k in value):
            raise _UnsupportedError(f"unsupported dict keys in {value!r}")
        return {"$dict": [[_attr_json(k), _attr_json(v)] for k, v in value.items()]}
    if hasattr(value, "magnitude"):
        return {
            "$quantity": [_attr_json(value.magnitude), get_quantity_str(value.units)]
        }
    if type(value).__name__ == "Unit":
        return {"$unit": get_quantity_str(value)}
    raise _UnsupportedError(f"can't store attribute value {value!r}")


# --- reading values ------------------------------------------------------


def _text_int(value) -> int:
    """Read an integer written as decimal text."""
    _check(isinstance(value, str), f"expected an integer as text, got {value!r}")
    return int(value)


def _json_float(value) -> float:
    """Read a float, which may be tagged."""
    return float(value["$float"] if isinstance(value, dict) else value)


def _json_time(value, kind: str):
    """Read a datetime64 (kind M) or timedelta64 (kind m) from count and unit."""
    count, unit = value
    return np.array(_text_int(count), dtype=np.int64).view(f"{kind}8[{unit}]")[()]


def _json_scalar(value, dtype: np.dtype, time_kind: str):
    """Read a coordinate scalar of a coordinate with `dtype`, or None."""
    if value is None:
        return None
    if dtype.kind in "mM":
        return _json_time(value, time_kind)
    if dtype.kind in "iu":
        return _text_int(value)
    return _json_float(value)


def _json_attr(value: Any) -> Any:
    """Decode one attribute value written by `_attr_json`."""
    if isinstance(value, list):
        return [_json_attr(x) for x in value]
    if not isinstance(value, dict):
        return value
    if not any(k.startswith("$") for k in value):
        return {k: _json_attr(v) for k, v in value.items()}
    _check(len(value) == 1, f"a tagged value holds other keys: {value!r}")
    tag, item = cast(tuple[str, Any], next(iter(value.items())))
    if tag == "$int":
        return _text_int(item)
    if tag == "$float":
        return _json_float(value)
    if tag == "$scalar":
        dtype = np.dtype(item[0])
        _check(dtype.kind in "biuf", f"unsupported scalar type {item[0]!r}")
        return dtype.type(_json_attr(item[1]))
    if tag in ("$datetime64", "$timedelta64"):
        return _json_time(item, "M" if tag == "$datetime64" else "m")
    if tag == "$bytes":
        return bytes.fromhex(item)
    if tag == "$quantity":
        return _json_attr(item[0]) * get_quantity(item[1])
    if tag == "$unit":
        return cast(Any, get_quantity(item)).units
    _check(tag == "$dict", f"unknown tag {tag!r}")
    out = {}
    for key, val in item:
        key = _json_attr(key)
        ok = isinstance(key, str | int) and not isinstance(key, bool) and key not in out
        _check(ok, f"bad or repeated dict key {key!r}")
        out[key] = _json_attr(val)
    return out


# --- coordinates ---------------------------------------------------------


def _label_json(values: np.ndarray) -> list:
    """Write stored labels: integer and time labels as text, floats tagged."""
    flat = np.ravel(values)
    if flat.dtype.kind in "mM":
        return [_int_text(x) for x in flat.view(np.int64)]
    if flat.dtype.kind in "iu":
        return [_int_text(x) for x in flat]
    if flat.dtype.kind == "f":
        return [_float_json(x) for x in flat]
    return flat.tolist()  # bool and str


def _json_labels(values: list, dtype: np.dtype) -> np.ndarray:
    """Read stored labels written by `_label_json`."""
    if dtype.kind in "mM":
        return np.array([_text_int(x) for x in values], dtype=np.int64).view(dtype)
    if dtype.kind in "iu":
        return np.array([_text_int(x) for x in values], dtype=object).astype(dtype)
    if dtype.kind == "f":
        return np.array([_json_float(x) for x in values], dtype=dtype)
    # numpy would quietly truncate long strings and call any string True.
    expected = str if dtype.kind == "U" else bool
    width = dtype.itemsize // 4 if dtype.kind == "U" else 1
    ok = all(
        type(x) is expected and (expected is bool or len(x) <= width) for x in values
    )
    _check(ok, f"labels don't fit dtype {dtype}")
    return np.array(values, dtype=dtype)


def _scalar_type(value) -> str:
    """Name the numpy type of a float grid's origin or step."""
    return value.dtype.str if isinstance(value, np.generic) else "float"


def _json_typed(value, dtype_text: str):
    """Read a float grid's origin or step in its recorded numpy type."""
    if dtype_text == "float":
        return _json_float(value)
    dtype = np.dtype(dtype_text)
    if dtype.kind in "mM":
        return _json_time(value, dtype.kind)
    _check(dtype.kind == "f", f"unsupported float grid type {dtype_text!r}")
    return dtype.type(_json_float(value))


def _run_json(run: dict, sources: dict) -> dict:
    """Write one run of a dumped numeric coordinate."""
    if "id" in run:  # stored labels, sent as the window's values
        index = run["offset"] + run["stride"] * np.arange(run["count"])
        return {"values": _label_json(np.atleast_1d(sources[run["id"]])[index])}
    if "shape" in run:  # blank
        ends = {
            k: run[k] if run[k] is None else _scalar_json(run[k])
            for k in ("start", "stop")
        }
        return {"blank": ends}
    if run["step_den"]:  # exact grid; slicing folds k0 and stride in
        assert run["k0"] == 0 and run["stride"] == 1 and run["parent_count"] is None
        terms = {k: _int_text(run[k]) for k in ("origin", "step_num", "step_den")}
        return {
            "grid": {**terms, "phase": _int_text(run["phase"]), "count": run["count"]}
        }
    return {
        "float_grid": {
            "origin": _scalar_json(run["origin"]),
            "step": _scalar_json(run["step_num"]),
            "count": run["count"],
            "k0": run["k0"],
            "stride": run["stride"],
            "parent_count": run["parent_count"],
            "origin_dtype": _scalar_type(run["origin"]),
            "step_dtype": _scalar_type(run["step_num"]),
        }
    }


def coord_to_json(coord, dims) -> dict:
    """Write a coordinate as the format's coordinate document."""
    dtype = np.dtype(coord.dtype)
    unsized = dtype.kind in "mM" and np.datetime_data(dtype.str)[1] != 1
    if dtype.kind not in "biufmMU" or (dtype.kind == "f" and dtype.itemsize > 8):
        unsized = True
    if unsized:
        raise _UnsupportedError(f"can't store a coordinate of dtype {dtype}")
    dump = coord.model_dump()
    if isinstance(coord, CoordString):
        runs = [{"values": _label_json(np.asarray(coord.values))}]
    else:
        runs = [_run_json(run, dump["sources"]) for run in dump["runs"]]
    step, units = dump.get("step"), dump.get("units")
    return {
        "dims": list(dims),
        "shape": list(coord.shape),
        "dtype": dtype.newbyteorder("<").str if dtype.kind != "b" else dtype.str,
        "units": None if units is None else get_quantity_str(units),
        "step": None if step is None else _scalar_json(step),
        "runs": runs,
    }


def _count(value) -> int:
    """Read a JSON count."""
    ok = type(value) is int and 0 <= value < _SAFE_INT
    _check(ok, f"expected a count, got {value!r}")
    return value


def _json_run(run, dtype: np.dtype, shape: tuple):
    """Read one run into the form `get_coord(runs=...)` takes."""
    _check(isinstance(run, dict) and len(run) == 1, f"bad run {run!r}")
    kind, body = next(iter(run.items()))
    if kind == "values":
        labels = _json_labels(body, dtype)
        return labels if len(shape) == 1 else labels.reshape(shape)
    if kind == "blank":
        ends = {k: _json_scalar(body[k], dtype, dtype.kind) for k in ("start", "stop")}
        return {"shape": shape, **ends}
    _check(kind in ("grid", "float_grid"), f"unknown run kind {kind!r}")
    _check(dtype.kind in ("iumM" if kind == "grid" else "fmM"), f"no {kind} of {dtype}")
    if kind == "grid":
        terms = {k: _text_int(body[k]) for k in ("origin", "step_num", "step_den")}
        phase = _text_int(body["phase"])
        _check(0 <= phase < terms["step_den"], f"bad grid terms {body!r}")
        rest = {"k0": 0, "stride": 1, "parent_count": None}
        return {**terms, "phase": phase, "count": _count(body["count"]), **rest}
    return {
        "origin": _json_typed(body["origin"], body["origin_dtype"]),
        "step_num": _json_typed(body["step"], body["step_dtype"]),
        "step_den": 0,
        "count": _count(body["count"]),
        "k0": _count(body["k0"]),
        "phase": 0,
        "parent_count": body["parent_count"],
        "stride": int(body["stride"]),
    }


def json_to_coord(doc: dict):
    """Rebuild a coordinate from its document."""
    dtype = np.dtype(doc["dtype"])
    shape = tuple(_count(x) for x in doc["shape"])
    parts = [_json_run(run, dtype, shape) for run in doc["runs"]]
    if dtype.kind == "U":
        return get_coord(data=parts[0] if len(parts) == 1 else np.concatenate(parts))
    step = _json_scalar(doc["step"], dtype, "m")
    return get_coord(runs=tuple(parts), dtype=dtype, step=step, units=doc["units"])


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
        "attrs": {k: _attr_json(v) for k, v in attrs.items()},
    }


def _reject_repeats(pairs):
    """Build a JSON object, refusing repeated keys."""
    out = dict(pairs)
    _check(len(out) == len(pairs), "a JSON object repeats a key")
    return out


def _reject_constant(name):
    """Refuse the non-JSON constants Python's parser accepts."""
    raise _bad(f"{name} is not JSON")


@_malformed_is_invalid
def read_document(schema) -> dict:
    """Return the checked ``dascore:patch`` document from an Arrow schema."""
    raw = [v for k, v in (schema.metadata or {}).items() if k == METADATA_KEY]
    _check(raw, "no dascore:patch metadata")
    doc = json.loads(
        raw[0].decode("utf-8"),
        object_pairs_hook=_reject_repeats,
        parse_constant=_reject_constant,
    )
    _check(doc["format"] == FORMAT_NAME, "not a dascore.patch document")
    major = FORMAT_VERSION.partition(".")[0]
    _check(doc["version"].partition(".")[0] == major, "unsupported version")
    dims, shape = doc["dims"], [_count(x) for x in doc["shape"]]
    names_ok = all(type(x) is str and x for x in dims) and len(set(dims)) == len(dims)
    _check(dims and names_ok and len(shape) == len(dims), "bad dims or shape")
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
    attrs = {k: _json_attr(v) for k, v in doc["attrs"].items()}
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
