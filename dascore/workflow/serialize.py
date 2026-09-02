"""
Canonical encoding, decoding and hashing for workflow objects, and the files
those documents are written to.

A [`Task`](`dascore.workflow.task.Task`) is identified by a fingerprint: a
digest of what it is and what it was given. That only works if the same
parameters always encode to the same bytes, so this module turns arbitrary
python values into a JSON tree with a fixed shape, then hashes its canonical
text.

Two modes exist. ``"fingerprint"`` encodes values for hashing: an array
becomes a digest of its bytes and a value left at ``None`` is dropped.
``"document"`` encodes values for storage: an array becomes a nested list and
nothing is dropped, so most values read back. A function, a partial, or a
value with no encoding of its own is named rather than reproduced in either
mode, and a dataframe has no document form at all; decoding any of them
raises. `write_workflow` and `read_workflow` put a document on disk in the
format its suffix names, refusing a suffix which names none.

Stability rests on `json`, `repr` of a float, numpy's byte layout and blake2b,
none of which change between python versions, and -- for frames and quantities
only -- on pandas' object hash and pint's short unit format. Python's own
``hash`` is never used: it is salted per process.
"""

from __future__ import annotations

import datetime
import hashlib
import inspect
import json
import warnings
from collections.abc import Callable, Iterable, Mapping, Set
from enum import Enum
from functools import partial
from pathlib import Path, PurePath
from typing import Any, Literal

import numpy as np
import pandas as pd
from pint import Quantity

from dascore.exceptions import ParameterError
from dascore.models.base import DascoreBaseModel
from dascore.models.registry import TAG_FIELD, get_model_tag, resolve_tagged_model
from dascore.utils.array_api import is_foreign, to_numpy
from dascore.utils.documents import DocumentFormat, read_document, write_document
from dascore.utils.paths import quote_path
from dascore.warnings import DASCoreWarning

# The two encoding modes; see the module docstring.
FINGERPRINT = "fingerprint"
DOCUMENT = "document"

EncodeMode = Literal["fingerprint", "document"]

# Every tag is a single key starting with "$", which no python identifier and
# no dascore field name can be. A mapping whose keys come from data rather
# than from source can still hold one, so `_encode_mapping` writes any
# mapping with a "$" key as an escaped `$dict` instead; without that a lone
# {"$datetime64": 0} key would read back, and fingerprint, as a datetime.
_ARRAY = "$array"
_BOOL = "$bool"
_BYTES = "$bytes"
_CALLABLE = "$callable"
_COMPLEX = "$complex"
_DATAFRAME = "$dataframe"
_DATETIME = "$datetime64"
_DICT = "$dict"
_ELLIPSIS = "$ellipsis"
_FLOAT = "$float"
_MODEL = "$model"
_OPAQUE = "$opaque"
_PARTIAL = "$partial"
_QUANTITY = "$quantity"
_SLICE = "$slice"
TASK_TAG = "$task"
_TASK = TASK_TAG
_TIMEDELTA = "$timedelta64"

# The digest size used everywhere: 8 bytes, written as 16 hex characters.
DIGEST_SIZE = 8

# The suffixes a workflow is written and read as, enumerated rather than
# inferred: a path spelled `.txt` is a caller who meant something this does
# not do, and picking a format for them would hide it. A path with no
# suffix at all is JSON.
YAML_SUFFIXES = frozenset({".yaml", ".yml"})
JSON_SUFFIXES = frozenset({".json", ""})


def digest(obj: Any, mode: EncodeMode = FINGERPRINT) -> str:
    """
    Return a stable 16 character digest of any encodable object.

    Parameters
    ----------
    obj
        The object to hash.
    mode
        Either "fingerprint" (the default) or "document"; see the module
        docstring.

    Examples
    --------
    >>> from dascore.workflow.serialize import digest
    >>> assert digest({"dim": "time"}) == digest({"dim": "time"})
    >>> assert digest({"dim": "time"}) != digest({"dim": "distance"})
    """
    return _digest_bytes(canonical_json(obj, mode=mode).encode("ascii"))


def combine_hashes(hashes: Iterable[str]) -> str:
    """
    Return one digest standing for an ordered series of digests.

    Order matters: the digests of two patches combined the other way round
    give a different answer, because which patch came first is part of what
    was done.

    Examples
    --------
    >>> from dascore.workflow.serialize import combine_hashes
    >>> assert combine_hashes(["a", "b"]) != combine_hashes(["b", "a"])
    """
    return digest(list(hashes))


def canonical_json(obj: Any, mode: EncodeMode = FINGERPRINT) -> str:
    """
    Return the canonical JSON text of an object.

    Keys are sorted, whitespace is stripped and non-ascii characters are
    escaped, so that equal objects give equal text on any platform.
    """
    return json.dumps(
        encode(obj, mode=mode),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def encode(obj: Any, mode: EncodeMode = FINGERPRINT) -> Any:
    """
    Return a JSON-safe tree standing for an object.

    Parameters
    ----------
    obj
        The object to encode.
    mode
        Either "fingerprint" (the default) or "document"; see the module
        docstring.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.workflow.serialize import encode
    >>> encode(np.arange(3), mode="document")["$array"]["data"]
    [0, 1, 2]
    """
    return _encode_value(obj, mode)


def decode(obj: Any) -> Any:
    """
    Return the object a document-mode encoding stands for.

    Values which a document cannot carry -- an array reduced to its digest, a
    function, a dataframe -- raise rather than come back as the tagged
    mapping which stands for them.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.workflow.serialize import encode, decode
    >>> decode(encode(np.datetime64("2020-01-01"), mode="document"))
    np.datetime64('2020-01-01T00:00:00.000000000')
    """
    if isinstance(obj, Mapping):
        return _decode_mapping(obj)
    if isinstance(obj, list):
        return [decode(x) for x in obj]
    return obj


def write_workflow(document: Mapping, path: Path) -> Path:
    """
    Write a workflow document to a file, in the format its suffix names.

    ``.yaml`` and ``.yml`` write YAML, ``.json`` and a bare name write
    JSON, and anything else is refused.
    """
    return write_document(document, path, _file_format(path))


def read_workflow(path: Path) -> Any:
    """Return the workflow document a file holds; see `write_workflow`."""
    return read_document(
        path,
        _file_format(path),
        error=ParameterError,
        holds="describes no workflow",
    )


def _file_format(path: Path) -> DocumentFormat:
    """Return the format a path names, refusing a suffix which names none."""
    suffix = Path(path).suffix.lower()
    if suffix in YAML_SUFFIXES:
        return "yaml"
    if suffix in JSON_SUFFIXES:
        return "json"
    msg = (
        f"{quote_path(Path(path))} has a suffix which names no format. Use "
        f"one of {sorted(YAML_SUFFIXES | JSON_SUFFIXES - {''})}, or no "
        "suffix at all."
    )
    raise ParameterError(msg)


def _digest_bytes(data: bytes | memoryview) -> str:
    """Return the digest of some bytes."""
    return hashlib.blake2b(data, digest_size=DIGEST_SIZE).hexdigest()


def _encode(obj: Any, mode: EncodeMode) -> Any:
    """Encode anything but the scalars `encode` handles itself."""
    # Ordered by how specific each check is, not by how common: bool is an
    # int, np.bool_ is an np.generic, and a Quantity holds an array.
    if isinstance(obj, bool | np.bool_):
        return {_BOOL: bool(obj)}
    if isinstance(obj, float | np.floating):
        return _encode_float(float(obj))
    if isinstance(obj, np.datetime64 | np.timedelta64):
        return _encode_time(obj)
    if isinstance(obj, np.generic):
        return _encode_value(obj.item(), mode)
    # A pandas Timestamp is a datetime and a Timedelta is a timedelta, so
    # both are covered here. Without them a time DASCore accepts everywhere
    # would fall through to the opaque tag and lose its value.
    if isinstance(obj, datetime.datetime | datetime.date):
        return _encode_time(_to_datetime64(obj))
    if isinstance(obj, datetime.timedelta):
        return _encode_time(np.timedelta64(obj))
    if isinstance(obj, complex):
        return {_COMPLEX: [_encode_float(obj.real), _encode_float(obj.imag)]}
    if isinstance(obj, bytes | bytearray):
        return {_BYTES: bytes(obj).hex()}
    if isinstance(obj, Quantity):
        return _encode_quantity(obj, mode)
    if isinstance(obj, Enum):
        return _encode_value(obj.value, mode)
    if isinstance(obj, PurePath):
        return obj.as_posix()
    if isinstance(obj, slice):
        parts = (obj.start, obj.stop, obj.step)
        return {_SLICE: [_encode_value(x, mode) for x in parts]}
    if obj is Ellipsis:
        return {_ELLIPSIS: True}
    if isinstance(obj, _task_class()):
        # Checked before the model branch below, which every task also
        # answers to: a task carries a version its fields do not show.
        return {_TASK: obj.fingerprint if mode == FINGERPRINT else obj.to_dict()}
    if isinstance(obj, DascoreBaseModel):
        return _encode_model(obj, mode)
    if isinstance(obj, pd.DataFrame | pd.Series):
        return _encode_dataframe(obj, mode)
    if _is_array(obj):
        return _encode_array(obj, mode)
    if isinstance(obj, Mapping):
        return _encode_mapping(obj, mode)
    if isinstance(obj, Set):
        return _encode_set(obj, mode)
    if isinstance(obj, list | tuple):
        return [_encode_value(x, mode) for x in obj]
    if isinstance(obj, partial):
        return _encode_partial(obj, mode)
    if callable(obj):
        return _encode_callable(obj, mode)
    return _encode_opaque(obj)


def _encode_value(obj: Any, mode: EncodeMode) -> Any:
    """Encode one value, short-circuiting the ones JSON already spells."""
    # None, strings and ints are much the most common parameter values, and
    # the check is cheaper than walking the dispatch chain in `_encode`.
    if obj is None or (isinstance(obj, str | int) and not isinstance(obj, bool)):
        return obj
    return _encode(obj, mode)


def _encode_float(value: float) -> Any:
    """Encode a float, tagging the three JSON cannot spell."""
    if value == value and abs(value) != np.inf:
        return value
    if value != value:
        return {_FLOAT: "nan"}
    return {_FLOAT: "inf" if value > 0 else "-inf"}


def _encode_time(value: np.datetime64 | np.timedelta64) -> Any:
    """Encode a numpy time as nanoseconds, so the unit it was written in
    does not change the answer.
    """
    tag = _DATETIME if isinstance(value, np.datetime64) else _TIMEDELTA
    unit = "datetime64[ns]" if tag == _DATETIME else "timedelta64[ns]"
    try:
        out = value.astype(unit)
    except OverflowError:
        out = None
    # DASCore works in nanoseconds throughout, and a time outside that range
    # wraps silently -- to a value centuries away -- on some numpy versions
    # and raises on others. Either way it is refused rather than hashed as
    # whatever it wrapped to, which is checked by converting it back.
    if out is None or (not np.isnat(value) and out.astype(value.dtype) != value):
        msg = f"{value} cannot be represented in nanoseconds."
        raise ParameterError(msg)
    return {tag: int(out.astype(np.int64))}


def _to_datetime64(value: datetime.datetime | datetime.date) -> np.datetime64:
    """Return a python date or datetime as a numpy one."""
    # An aware datetime is moved onto UTC, which is the only zone numpy has;
    # converting one directly is deprecated and then dropped.
    if isinstance(value, datetime.datetime) and value.tzinfo is not None:
        value = value.astimezone(datetime.UTC).replace(tzinfo=None)
    return np.datetime64(value)


def _encode_quantity(value: Quantity, mode: EncodeMode) -> Any:
    """
    Encode a quantity as its magnitude and the unit it was written in.

    The unit is not normalized: ``1 m`` and ``100 cm`` are the same length
    but not the same call, and a task is identified by the call.
    """
    return {_QUANTITY: [_encode_value(value.magnitude, mode), f"{value.units:~}"]}


def model_values(model: DascoreBaseModel) -> dict[str, Any]:
    """
    Return a model's fields and extras, as the objects they are.

    Not model_dump: a dump turns a nested model into a plain mapping, losing
    which class it was, and its json mode adds a key which is not a field.
    """
    out = {name: getattr(model, name) for name in type(model).model_fields}
    out.update(model.__pydantic_extra__ or {})
    return out


def _encode_model(model: DascoreBaseModel, mode: EncodeMode) -> Any:
    """Encode a dascore model as its tag and its fields."""
    fields = model_values(model)
    # A class no tag can name -- a parametrized generic -- is still spelled
    # out, so that two of them do not fingerprint alike; a document holding
    # one refuses to decode rather than rebuilding the wrong class.
    tag = get_model_tag(type(model)) or _spell_type(type(model))
    return {_MODEL: {TAG_FIELD: tag, "fields": _encode_mapping(fields, mode)}}


def _encode_dataframe(df: pd.DataFrame | pd.Series, mode: EncodeMode) -> Any:
    """
    Encode a dataframe as a digest of its labels, dtypes and values.

    The labels are hashed as well as the values because a frame's columns
    are parameters in their own right: ``coords_from_df`` names the coords
    it builds after them.

    There is no document form: a frame, with its dtypes and its index, is a
    storage format rather than a parameter encoding.
    """
    if mode == DOCUMENT:
        msg = (
            "A dataframe parameter cannot be written to a document. Give the "
            "task the values it needs instead of the frame holding them."
        )
        raise ParameterError(msg)
    frame = df.to_frame() if isinstance(df, pd.Series) else df
    values = pd.util.hash_pandas_object(df, index=True).to_numpy()
    payload = {
        "columns": [str(x) for x in frame.columns],
        "dtypes": [str(x) for x in frame.dtypes],
        "values": _digest_bytes(values.tobytes()),
    }
    return {_DATAFRAME: digest(payload)}


def _encode_array(array: Any, mode: EncodeMode) -> Any:
    """Encode an array by its dtype, shape and contents."""
    # hash_array lives in dascore.utils.array, which imports dascore itself,
    # so naming it at module scope is a cycle; it is the tree's one array
    # hash and is used rather than repeated.
    from dascore.utils.array import hash_array  # noqa: PLC0415

    array = to_numpy(array) if is_foreign(array) else np.asarray(array)
    if array.dtype == object:
        # An object array holds python values, which have no bytes to hash;
        # each element is encoded on its own terms instead.
        return _encode_value(array.tolist(), mode)
    array = _normalize_array(array)
    out = {"dtype": array.dtype.str, "shape": list(array.shape)}
    if mode == DOCUMENT:
        out["data"] = _encode_value(_array_data(array), mode)
    else:
        out["hash"] = hash_array(array)
    return {_ARRAY: out}


def _normalize_array(array: np.ndarray) -> np.ndarray:
    """Return the array in the layout its values are hashed and written in."""
    dtype = array.dtype
    # Times normalize to nanoseconds for the same reason scalar ones do: the
    # unit an array of times was built with is not part of its values.
    if dtype.kind in "Mm":
        return _times_as_nanoseconds(array)
    # A big-endian array holds the same values as its little-endian twin, so
    # the byte order it happens to be stored in is normalized away.
    elif dtype.byteorder == ">":
        dtype = dtype.newbyteorder("<")
    return array.astype(dtype, copy=False)


def _times_as_nanoseconds(array: np.ndarray) -> np.ndarray:
    """Return an array of times as nanoseconds, refusing any which wrap."""
    # An out of range time raises on some numpy versions and wraps silently
    # -- to a value centuries away -- on others, so both are refused; see
    # `_encode_time`.
    try:
        out = array.astype(np.dtype(f"<{array.dtype.kind}8[ns]"))
        kept = ~np.isnat(array)
        wrapped = not np.array_equal(out[kept].astype(array.dtype), array[kept])
    except OverflowError:
        wrapped = True
    if wrapped:
        msg = "Some times in the array cannot be represented in nanoseconds."
        raise ParameterError(msg)
    return out


def _array_data(array: np.ndarray) -> Any:
    """Return an array's values in a form a document can hold."""
    # tolist gives a datetime array back as python datetimes, which lose
    # their unit; the counts they are stored as do not.
    if array.dtype.kind in "Mm":
        return array.view(np.int64).tolist()
    return array.tolist()


def _encode_mapping(mapping: Mapping, mode: EncodeMode) -> Any:
    """
    Encode a mapping, dropping every None in fingerprint mode.

    A parameter left at its None default is then the same call as one left
    out. It applies at every depth, so a dict parameter loses its None
    values too: ``{"time": None, "distance": (1, 2)}`` encodes as
    ``{"distance": (1, 2)}``.
    """
    if not all(isinstance(key, str) and not key.startswith("$") for key in mapping):
        return _encode_odd_keyed_mapping(mapping, mode)
    items = mapping.items()
    if mode == FINGERPRINT:
        items = [(key, value) for key, value in items if value is not None]
    return {key: _encode_value(value, mode) for key, value in items}


def _encode_odd_keyed_mapping(mapping: Mapping, mode: EncodeMode) -> Any:
    """
    Encode a mapping as sorted pairs, which any key can survive.

    Used for a mapping whose keys are not all strings, and for one holding a
    key which would otherwise be read back as a tag.
    """
    pairs = [
        [_encode_value(key, mode), _encode_value(value, mode)]
        for key, value in mapping.items()
    ]
    return {_DICT: sorted(pairs, key=_sort_key)}


def _encode_set(values: Set, mode: EncodeMode) -> Any:
    """Encode a set as a sorted list; a set has no order of its own."""
    return sorted((_encode_value(x, mode) for x in values), key=_sort_key)


def _encode_partial(value: partial, mode: EncodeMode) -> Any:
    """Encode a partial as the function it wraps and what it wraps it with."""
    return {
        _PARTIAL: {
            "func": _encode_value(value.func, mode),
            "args": [_encode_value(x, mode) for x in value.args],
            "kwargs": _encode_mapping(value.keywords, mode),
        }
    }


def _encode_callable(func: Callable, mode: EncodeMode = FINGERPRINT) -> Any:
    """
    Encode a callable by where it is defined.

    A function which cannot be named -- a lambda, or one defined inside
    another function -- carries a digest of its source as well, since its
    path names every one of them alike. Two written on the same line share
    that source, and so are one parameter as far as a fingerprint is
    concerned.
    """
    if mode == DOCUMENT:
        msg = (
            "A function parameter cannot be written to a document. Give the "
            "task values it can carry, or keep the operation in code."
        )
        raise ParameterError(msg)
    module = getattr(func, "__module__", None) or "<unknown>"
    qualname = getattr(func, "__qualname__", None) or repr(func)
    out = {"path": f"{module}:{qualname}"}
    if "<lambda>" in qualname or "<locals>" in qualname:
        out["source"] = _source_digest(func)
    return {_CALLABLE: out}


def _source_digest(func: Callable) -> str | None:
    """Return a digest of a function's source, or None if it has none."""
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        # Defined in a shell, or built in: there is no text to read.
        return None
    return _digest_bytes(source.encode("utf8"))


def _encode_opaque(obj: Any) -> Any:
    """
    Encode a value nothing else describes by naming its class.

    Two values of such a class are indistinguishable, so this warns: a
    fingerprint which cannot see a parameter's value is a fingerprint two
    different calls can share.
    """
    name = _spell_type(type(obj))
    msg = (
        f"A value of type {name} has no encoding of its own, so only its "
        "type is hashed. Two different values of it share one fingerprint."
    )
    warnings.warn(msg, DASCoreWarning, stacklevel=2)
    return {_OPAQUE: name}


def _spell_type(cls: type) -> str:
    """Spell a class the way an opaque encoding names it."""
    # Never repr(obj): the default holds the object's address, which changes
    # from run to run and would make every fingerprint unrepeatable.
    return f"{cls.__module__}.{cls.__qualname__}"


def _sort_key(value: Any) -> str:
    """Return a total order over encoded values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _is_array(obj: Any) -> bool:
    """Return True for anything which should encode as an array."""
    return isinstance(obj, np.ndarray) or is_foreign(obj)


def _task_class() -> type:
    """Return the Task class, which cannot be named at module scope."""
    # task.py imports this module, so importing it back is deferred; the
    # module object is in sys.modules by the time any value is encoded.
    from dascore.workflow.task import Task  # noqa: PLC0415

    return Task


def _decode_mapping(obj: Mapping) -> Any:
    """Decode a mapping, inverting whichever tag it carries."""
    if len(obj) == 1:
        key = next(iter(obj))
        if (decoder := _DECODERS.get(key)) is not None:
            return decoder(obj[key])
    return {key: decode(value) for key, value in obj.items()}


def _decode_datetime(value: int) -> np.datetime64:
    """Decode a datetime from its nanoseconds."""
    return np.datetime64(value, "ns")


def _decode_timedelta(value: int) -> np.timedelta64:
    """Decode a timedelta from its nanoseconds."""
    return np.timedelta64(value, "ns")


def _decode_slice(value: list) -> slice:
    """Decode a slice from its three parts."""
    return slice(*(decode(x) for x in value))


def _decode_quantity(value: list) -> Quantity:
    """Decode a quantity from its magnitude and unit."""
    # dascore.units imports dascore, which imports this module once patch
    # functions are processors, so the registry is fetched when it is needed.
    from dascore.units import get_quantity  # noqa: PLC0415

    magnitude, units = decode(value[0]), value[1]
    # An empty unit spells "no units" to get_quantity, which returns None,
    # so a dimensionless quantity names its dimensionlessness instead.
    return magnitude * get_quantity(units or "dimensionless")


def _decode_array(value: Mapping) -> np.ndarray:
    """Decode an array from its dtype, shape and data."""
    if "data" not in value:
        msg = (
            "An array encoded for a fingerprint holds only a digest of its "
            "values, so the array itself cannot be read back from it."
        )
        raise ParameterError(msg)
    dtype = np.dtype(value["dtype"])
    data = decode(value["data"])
    # A time array is written as the counts it is stored as, so it is read
    # back as those before it is given its dtype again.
    source = np.int64 if dtype.kind in "Mm" else dtype
    array = np.asarray(data, dtype=source).astype(dtype)
    return array.reshape(value["shape"])


def _decode_model(value: Mapping) -> DascoreBaseModel:
    """Decode a dascore model from its tag and fields."""
    # resolve_tagged_model rather than a check of its own: a document naming
    # a class nothing registers fails the same way here as it does for the
    # task holding it.
    cls = resolve_tagged_model(value[TAG_FIELD])
    return cls(**{key: decode(val) for key, val in value["fields"].items()})


def _decode_task(value: Any) -> Any:
    """Decode a task from its document."""
    if not isinstance(value, Mapping):
        msg = (
            "A task encoded for a fingerprint holds only its digest, so the "
            "task itself cannot be read back from it."
        )
        raise ParameterError(msg)
    # Imported here rather than at module scope: task.py imports this module.
    from dascore.workflow.task import Task  # noqa: PLC0415

    return Task.from_dict(value)


def _decode_dict(pairs: list) -> dict:
    """Decode a mapping whose keys are not strings."""
    # A key which was a tuple comes back as a list, which cannot be a key
    # again, so any decoded sequence is made hashable on the way in.
    return {_hashable(decode(key)): decode(value) for key, value in pairs}


def _hashable(value: Any) -> Any:
    """Return a value which can key a mapping."""
    return tuple(value) if isinstance(value, list) else value


def _refuse(kind: str):
    """Return a decoder which refuses a value a document cannot carry."""

    def _decoder(value):
        msg = (
            f"A {kind} was encoded by name rather than by value, so it cannot "
            "be rebuilt from a document. Give the task a value it can carry."
        )
        raise ParameterError(msg)

    return _decoder


_DECODERS: dict[str, Callable[[Any], Any]] = {
    _ARRAY: _decode_array,
    _BOOL: bool,
    _BYTES: bytes.fromhex,
    _CALLABLE: _refuse("function"),
    _COMPLEX: lambda pair: complex(*(decode(x) for x in pair)),
    _DATAFRAME: _refuse("dataframe"),
    _DATETIME: _decode_datetime,
    _DICT: _decode_dict,
    _ELLIPSIS: lambda _: Ellipsis,
    _FLOAT: float,
    _MODEL: _decode_model,
    _OPAQUE: _refuse("value"),
    _PARTIAL: _refuse("partial"),
    _QUANTITY: _decode_quantity,
    _SLICE: _decode_slice,
    _TASK: _decode_task,
    _TIMEDELTA: _decode_timedelta,
}
