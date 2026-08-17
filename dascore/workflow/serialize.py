"""
Canonical encoding, decoding and hashing for workflow objects.

A [`Task`](`dascore.workflow.task.Task`) is identified by a fingerprint: a
digest of what it is and what it was given. That only works if the same
parameters always encode to the same bytes, so this module turns arbitrary
python values into a JSON tree with a fixed shape, then hashes its canonical
text.

Two modes exist. ``"fingerprint"`` encodes values for hashing: an array
becomes a digest of its bytes, a parameter left at ``None`` is dropped, and
values which cannot be reconstructed are named rather than reproduced.
``"document"`` encodes values for storage: an array becomes a nested list, and
nothing is dropped, so that reading the document back gives the value again.

Stability rests on four things which do not change between python versions:
`json`, `repr` of a float, numpy's byte layout, and blake2b. Python's own
``hash`` is never used -- it is salted per process.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import threading
import weakref
from collections.abc import Callable, Iterable, Mapping, Set
from enum import Enum
from functools import partial
from pathlib import PurePath
from typing import Any, Literal

import numpy as np
import pandas as pd
from pint import Quantity

from dascore.exceptions import ParameterError
from dascore.models.base import DascoreBaseModel
from dascore.models.registry import TAG_FIELD, get_model_tag, resolve_model_tag
from dascore.utils.array_api import is_foreign, to_numpy

# The two encoding modes; see the module docstring.
FINGERPRINT = "fingerprint"
DOCUMENT = "document"

EncodeMode = Literal["fingerprint", "document"]

# Every tag is a single key starting with "$", which no python identifier and
# no dascore field name can be, so a tagged value is never mistaken for a
# plain mapping.
_ARRAY = "$array"
_BOOL = "$bool"
_CALLABLE = "$callable"
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
_TASK = "$task"
_TIMEDELTA = "$timedelta64"

# The digest size used everywhere: 8 bytes, written as 16 hex characters.
DIGEST_SIZE = 8

# Digests of arrays already seen, keyed by id. An array passed to the same
# call in a loop -- a mask given to `where`, an inventory given to `enrich` --
# is then hashed once rather than once per patch. Entries are dropped when
# the array is collected, so the cache holds no array alive and cannot grow
# past what the caller is already holding.
_ARRAY_DIGESTS: dict[int, str] = {}
_ARRAY_REFS: dict[int, weakref.ref] = {}
_ARRAY_LOCK = threading.Lock()


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
    if _is_task(obj):
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
        return _encode_callable(obj)
    return {_OPAQUE: _spell_type(type(obj))}


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
    return {tag: int(value.astype(unit).astype(np.int64))}


def _encode_quantity(value: Quantity, mode: EncodeMode) -> Any:
    """
    Encode a quantity as its magnitude and the unit it was written in.

    The unit is not normalized: ``1 m`` and ``100 cm`` are the same length
    but not the same call, and a task is identified by the call.
    """
    return {_QUANTITY: [_encode_value(value.magnitude, mode), f"{value.units:~}"]}


def _encode_model(model: DascoreBaseModel, mode: EncodeMode) -> Any:
    """Encode a dascore model as its tag and its fields."""
    fields = {name: getattr(model, name) for name in type(model).model_fields}
    fields.update(model.__pydantic_extra__ or {})
    # A class no tag can name -- a parametrized generic -- is still spelled
    # out, so that two of them do not fingerprint alike; a document holding
    # one refuses to decode rather than rebuilding the wrong class.
    tag = get_model_tag(type(model)) or _spell_type(type(model))
    return {_MODEL: {TAG_FIELD: tag, "fields": _encode_mapping(fields, mode)}}


def _encode_dataframe(df: pd.DataFrame | pd.Series, mode: EncodeMode) -> Any:
    """
    Encode a dataframe as a digest of its values.

    A frame is a parameter to exactly one patch function today
    (``coords_from_df``) and hashing it is all a fingerprint needs. It has no
    document form: writing a frame into a pipe file, with its dtypes and its
    index, is a storage format rather than a parameter encoding.
    """
    if mode == DOCUMENT:
        msg = (
            "A dataframe parameter cannot be written to a document. Give the "
            "task the values it needs instead of the frame holding them."
        )
        raise ParameterError(msg)
    hashed = pd.util.hash_pandas_object(df, index=True).to_numpy()
    return {_DATAFRAME: _digest_bytes(hashed.tobytes())}


def _encode_array(array: Any, mode: EncodeMode) -> Any:
    """Encode an array by its dtype, shape and contents."""
    original = array
    array = to_numpy(array) if is_foreign(array) else np.asarray(array)
    if array.dtype == object:
        # An object array holds python values, which have no bytes to hash;
        # each element is encoded on its own terms instead.
        return [_encode_value(x, mode) for x in array.tolist()]
    # A big-endian array holds the same values as its little-endian twin, so
    # the byte order it happens to be stored in is normalized away.
    array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    out = {"dtype": array.dtype.str, "shape": list(array.shape)}
    if mode == DOCUMENT:
        out["data"] = _encode_value(array.tolist(), mode)
    else:
        out["hash"] = _array_digest(array, original)
    return {_ARRAY: out}


def _array_digest(array: np.ndarray, original: Any) -> str:
    """
    Return the digest of an array's bytes, hashing each array once.

    The cache is keyed on the array the caller passed rather than on the
    normalized copy, which is a fresh object every call and would never hit.
    """
    key = id(original)
    with _ARRAY_LOCK:
        if (out := _ARRAY_DIGESTS.get(key)) is not None:
            return out
    # A view, and any datetime array, cannot export a buffer directly, so
    # the values are copied into a layout numpy can hand out as bytes.
    out = _digest_bytes(np.ascontiguousarray(array).view(np.uint8).data)
    _remember_array(key, original, out)
    return out


def _remember_array(key: int, array: np.ndarray, value: str) -> None:
    """Cache an array's digest for as long as the array itself lives."""

    def _forget(_ref, key=key):
        with _ARRAY_LOCK:
            _ARRAY_DIGESTS.pop(key, None)
            _ARRAY_REFS.pop(key, None)

    # A view of another array, or an array a backend owns, may refuse a weak
    # reference; the digest is then simply not cached.
    try:
        ref = weakref.ref(array, _forget)
    except TypeError:
        return
    with _ARRAY_LOCK:
        _ARRAY_DIGESTS[key] = value
        _ARRAY_REFS[key] = ref


def _encode_mapping(mapping: Mapping, mode: EncodeMode) -> Any:
    """
    Encode a mapping, dropping what a fingerprint should not see.

    A parameter left at None is dropped in fingerprint mode: a call which
    passes a default explicitly is the same call as one which leaves it out.
    """
    if not all(isinstance(key, str) for key in mapping):
        return _encode_odd_keyed_mapping(mapping, mode)
    items = mapping.items()
    if mode == FINGERPRINT:
        items = [(key, value) for key, value in items if value is not None]
    return {key: _encode_value(value, mode) for key, value in items}


def _encode_odd_keyed_mapping(mapping: Mapping, mode: EncodeMode) -> Any:
    """Encode a mapping whose keys are not all strings as sorted pairs."""
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


def _encode_callable(func: Callable) -> Any:
    """
    Encode a callable by where it is defined.

    A function which cannot be named -- a lambda, or one defined inside
    another function -- carries a digest of its source as well, since its
    path names every one of them alike. Two written on the same line share
    that source, and so are one parameter as far as a fingerprint is
    concerned.
    """
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


def _is_task(obj: Any) -> bool:
    """Return True if an object is a Task, without importing one."""
    # Not an isinstance check: task.py imports this module, so naming Task
    # here is a cycle. Every Task is a DascoreBaseModel and is checked
    # before this, so only a Task reaches it with a `to_dict`.
    return hasattr(obj, "fingerprint") and hasattr(obj, "to_dict")


def _decode_mapping(obj: Mapping) -> Any:
    """Decode a mapping, inverting whichever tag it carries."""
    if len(obj) == 1:
        key = next(iter(obj))
        if (decoder := _DECODERS.get(key)) is not None:
            return decoder(obj[key])
    return {key: decode(value) for key, value in obj.items()}


def _decode_float(value: str) -> float:
    """Decode one of the floats JSON cannot spell."""
    return float(value)


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
    array = np.asarray(decode(value["data"]), dtype=np.dtype(value["dtype"]))
    return array.reshape(value["shape"])


def _decode_model(value: Mapping) -> DascoreBaseModel:
    """Decode a dascore model from its tag and fields."""
    cls = resolve_model_tag(value[TAG_FIELD])
    if cls is None:
        msg = (
            f"Nothing registers the {TAG_FIELD} {value[TAG_FIELD]!r}, so the "
            "model it names cannot be rebuilt."
        )
        raise ParameterError(msg)
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
    _CALLABLE: _refuse("function"),
    _DATAFRAME: _refuse("dataframe"),
    _DATETIME: _decode_datetime,
    _DICT: _decode_dict,
    _ELLIPSIS: lambda _: Ellipsis,
    _FLOAT: _decode_float,
    _MODEL: _decode_model,
    _OPAQUE: _refuse("value"),
    _PARTIAL: _refuse("partial"),
    _QUANTITY: _decode_quantity,
    _SLICE: _decode_slice,
    _TASK: _decode_task,
    _TIMEDELTA: _decode_timedelta,
}
