"""Hash things into ids, and say which data a patch is.

One hash, [`H`](`dascore.utils.identity.H`), digests the canonical JSON of a
domain and a payload. Everything which has an id gets it from there: a
coordinate, an operation, and the two ids a patch carries.

``origin_id`` says which stored data a patch came from. It survives every
operation and folds when patches are combined. ``data_id`` says which array
this is: a freshly read patch takes its origin's id, and an operation's
result takes the digest of its inputs' ids and the operation's id, so the id
commits to everything upstream of it.

An id is never knowingly false. A parameter the encoder cannot spell, or an
input which carries no id, gives the result a random ``data_id`` rather than
one another result could share.

Examples
--------
>>> from dascore.utils.identity import H, derive, fold_origin_ids
>>>
>>> # Domains keep equal payloads apart.
>>> assert H("operation", {"dim": "time"}) != H("coord", {"dim": "time"})
>>>
>>> # Which array: the inputs, in order, and what was done to them.
>>> assert derive(["a", "b"], "op") != derive(["b", "a"], "op")
>>>
>>> # Which stored data: one origin stays itself however often it is met.
>>> assert fold_origin_ids(["a", "a"]) == "a"
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import inspect
import json
import re
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import PurePath
from types import ModuleType
from typing import TYPE_CHECKING, Any, NoReturn
from uuid import uuid4

import numpy as np
import pandas as pd
from pint import Quantity, Unit

from dascore.exceptions import ParameterError
from dascore.models.base import DascoreBaseModel, model_values
from dascore.models.registry import TAG_FIELD, get_model_tag
from dascore.utils.array_api import is_foreign, to_numpy
from dascore.warnings import DASCoreWarning

if TYPE_CHECKING:
    from dascore.core.source import ArraySource

# The version of the whole scheme, hashed into every id. Bump it when the
# encoding or a payload changes meaning, so old and new ids cannot meet.
SCHEME = 1

# The digest size used everywhere: 16 bytes, written as 32 hex characters.
DIGEST_SIZE = 16

# Every tag is a single key starting with "$", which no python identifier and
# no dascore field name can be. A mapping whose keys come from data rather
# than from source can still hold one, so `_encode_mapping` writes any
# mapping with a "$" key as an escaped `$dict` instead.
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
_ID = "$id"
_MODEL = "$model"
_PARTIAL = "$partial"
_PATCH = "$patch"
_QUANTITY = "$quantity"
_REGEX = "$regex"
_SET = "$set"
_SLICE = "$slice"
_TIMEDELTA = "$timedelta64"
_UNIT = "$unit"

# The attrs which say which data a patch is. Never compared where patches
# are combined: the result's are worked out from its members'.
_ID_FIELDS = ("origin_id", "data_id")


@dataclass(frozen=True, slots=True)
class PatchMarker:
    """
    Stands for a patch handed to an operation as an argument.

    A patch argument is an input, not a parameter. `index` is its place in
    the operation's inputs, so which argument each input filled is part of
    the operation: `where(cond=a, other=b)` is not `where(cond=b, other=a)`.
    """

    index: int


def H(domain: str, payload: Any, *, encoded: bool = False) -> str:  # noqa: N802
    """
    Return the 32 character id of a payload within a domain.

    Parameters
    ----------
    domain
        What kind of thing is being named. Equal payloads in different
        domains get different ids.
    payload
        Anything [`encode`](`dascore.utils.identity.encode`) can spell.
    encoded
        True if the payload is already a canonical JSON-safe tree, which
        skips encoding it; for hot paths which build one themselves.

    Examples
    --------
    >>> from dascore.utils.identity import H
    >>> assert H("operation", {"dim": "time"}) == H("operation", {"dim": "time"})
    >>> assert H("operation", {"dim": "time"}) != H("operation", {"dim": "distance"})
    """
    text = json.dumps(
        [SCHEME, domain, payload if encoded else encode(payload)],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return _hash_bytes(text.encode("ascii"))


def encode(obj: Any) -> Any:
    """
    Return the canonical JSON-safe tree standing for an object.

    Every value is kept, containers are framed so that no two of them read
    alike, and a value which cannot be spelled faithfully raises
    `ParameterError` rather than being named by its type.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.utils.identity import encode
    >>> encode(np.timedelta64(1, "s"))
    {'$timedelta64': 1000000000}
    """
    return _encode_value(obj)


def new_id() -> str:
    """Return a random id, for data nothing can be derived for."""
    return uuid4().hex


def ids_enabled() -> bool:
    """Whether this process is keeping track of what a patch is."""
    # Imported here rather than at module scope: `dascore.config` is not
    # built yet when this module is first imported.
    from dascore import get_config  # noqa: PLC0415

    return get_config().patch_provenance != "disabled"


def operation_id(name: str, params: Mapping[str, Any], version: str = "1.0") -> str:
    """
    Return the id of an operation and its parameters.

    Parameters
    ----------
    name
        What was done: a patch function's registry tag, or a capitalized
        kind for the operations which are not patch functions.
    params
        What it was done with, any patch already replaced by its
        [`PatchMarker`](`dascore.utils.identity.PatchMarker`).
    version
        The operation's version; bump it when the same parameters mean a
        different result, a changed default included.
    """
    return H("operation", {"name": name, "version": version, "params": params})


def derive(inputs: Sequence[str], operation: str, output: int | None = None) -> str:
    """
    Return the data id of an operation's result.

    Parameters
    ----------
    inputs
        The data ids of the patches the operation was given, in order.
    operation
        The operation's id; see
        [`operation_id`](`dascore.utils.identity.operation_id`).
    output
        The result's position among the patches the operation returned.
        None for an operation which returns one patch.
    """
    payload: list[Any] = [list(inputs), operation]
    if output is not None:
        payload.append(int(output))
    return H("derived", payload)


def fold_origin_ids(origin_ids: Sequence[str]) -> str:
    """
    Return the origin id of a patch combined from several.

    The distinct ids, in the order first met: windows of one file merged
    back together still come from that file, and how often an input was
    used is the data id's to say. Data which never named an origin does not
    acquire one by being combined.
    """
    seen = tuple(dict.fromkeys(x for x in origin_ids if x))
    if len(seen) <= 1:
        return seen[0] if seen else ""
    return H("origin-fold", list(seen))


def origin_id_for(
    source: ArraySource,
    size_bytes: int | None = None,
    mtime_ns: int | None = None,
    *,
    ordinal: object = None,
) -> str:
    """
    Return the origin id of a stored patch without touching the filesystem.

    Parameters
    ----------
    source
        The format, version, canonical path, and native logical patch key.
    size_bytes
        The size of the source, when available.
    mtime_ns
        The source's modification time, when available.
    ordinal
        The fallback for a source without a native key.

    Notes
    -----
    An id stored in the file remains authoritative. A derived id is only as
    good as its fields: changing the path, size, or modification time
    changes it, and a rewrite which keeps all three does not.
    """
    return H(
        "origin",
        {
            "format": source.format,
            "version": source.version,
            "path": source.path,
            "key": source.key or ordinal,
            "size_bytes": size_bytes,
            "mtime_ns": mtime_ns,
        },
    )


def narrowed_data_id(
    attrs, before: ArraySource | None, after: ArraySource | None
) -> str | None:
    """
    Return the id of a window of the array a patch already is, or None.

    A patch nothing has been done to *is* its source, so narrowing it to
    something the source can still load names a window of the same array
    rather than the result of an operation. Windows compose, so which
    selections led there does not matter.

    Parameters
    ----------
    attrs
        The attrs of the patch being narrowed.
    before
        The source of that patch, if it has one.
    after
        The source the narrowed patch would load from, if any.

    Returns
    -------
    The window's id, or None when the result must be derived as usual:
    the patch is no longer what its source loads (anything was done to
    it), or one of the two describes no array (a stepped, fancy or
    non-contiguous selection detaches the source).
    """
    if before is None or after is None or not (before.loadable and after.loadable):
        return None
    if (getattr(attrs, "data_id", "") or "") != before.data_id:
        return None
    return after.data_id


def read_operation_id(snap) -> str | None:
    """
    Return the id of decoding a resource under non-default options, or None.

    `snap` changes which coordinates the same bytes decode to, so a
    non-default value makes a different array; the default reads the
    stored thing itself and derives nothing. None means the reader was
    left to its own default, which is the default.
    """
    if snap is True or snap is None:
        return None
    return operation_id("Read", {"snap": snap})


def with_ids(attrs):
    """
    Return attrs which name which data they belong to.

    Minted rather than derived when there is nothing to derive from: a
    patch built in memory is not the same data as anything else. A reader
    which knows better stamps over this.
    """
    # `getattr`: attrs unpickled from before these fields existed lack them.
    if getattr(attrs, "origin_id", None) or not ids_enabled():
        return attrs
    minted = new_id()
    update = {"origin_id": minted}
    if not getattr(attrs, "data_id", None):
        update["data_id"] = minted
    # Validation restores an old pickle's missing fields; otherwise copy.
    if not all(hasattr(attrs, name) for name in _ID_FIELDS):
        return attrs.update(**update)
    return attrs.model_copy(update=update)


def merge_operation(params: Mapping[str, Any] | None = None) -> str:
    """Return the id of merging patches into one, under these merge options."""
    return operation_id("Merge", dict(params or {}))


def result_ids(
    members,
    operation: str | None,
    output: int | None = None,
    *,
    data_id: str | None = None,
) -> dict:
    """
    Return the two ids of an operation's result.

    Parameters
    ----------
    members
        The attrs of the patches which went into it, in order. A member
        dropped for being incompatible did not contribute its data.
    operation
        The operation's id. None says it could not be derived -- a parameter
        the encoder refused -- and gives the result a random data id.
    output
        The result's position among several; see
        [`derive`](`dascore.utils.identity.derive`).
    data_id
        The result's data id where it is known outright, such as a window
        of an input's array; nothing is derived for it.
    """
    members = list(members)
    if data_id is None:
        parents = [getattr(x, "data_id", "") or "" for x in members]
        # A parent which names no data cannot be derived from either: two
        # such inputs would otherwise lead to one id.
        derivable = operation and parents and all(parents)
        data_id = derive(parents, operation, output) if derivable else new_id()
    origin = fold_origin_ids([getattr(x, "origin_id", "") or "" for x in members])
    return {"origin_id": origin, "data_id": data_id}


def stamp(attrs, members, operation: str | None, output: int | None = None):
    """
    Return attrs carrying the ids of an operation's result.

    See [`result_ids`](`dascore.utils.identity.result_ids`). With ids
    disabled the result claims none, rather than keeping its input's.
    """
    if not ids_enabled():
        return _without_ids(attrs)
    return attrs.update(**result_ids(members, operation, output))


def try_operation_id(name: str, params: Mapping[str, Any], version="1.0"):
    """Return an operation's id, or None (and warn) if a parameter is refused."""
    try:
        return operation_id(name, params, version)
    except Exception as error:
        warn_random_id(name, error)
        return None


def warn_random_id(name: str, error: Exception) -> None:
    """Say that an operation worked but its result's id could not be derived."""
    msg = (
        f"No id could be derived for {name}: {error} Its result carries a "
        "random data_id, so it cannot be matched to the same call made again."
    )
    warnings.warn(msg, DASCoreWarning, stacklevel=3)


def _without_ids(attrs):
    """Return attrs which claim no id; what changed them was not recorded."""
    if not any(getattr(attrs, name, "") for name in _ID_FIELDS):
        return attrs
    return attrs.update(**dict.fromkeys(_ID_FIELDS, ""))


def extract_patches(params: Mapping[str, Any]) -> tuple[dict[str, Any], list]:
    """
    Return parameters with each patch replaced by a marker, and the patches.

    One walk, in the order the encoder writes -- mapping keys sorted,
    sequences as given -- numbers the markers and orders the patches, so a
    marker always points at the input which filled that argument.
    """
    # Imported here rather than at module scope: this module is imported
    # while `dascore` is still being built.
    import dascore as dc  # noqa: PLC0415

    found: list = []

    def _walk(value):
        if isinstance(value, dc.PatchMeta):
            found.append(value)
            return PatchMarker(len(found) - 1)
        if isinstance(value, Mapping):
            if not _holds_patch(value):
                return value
            keys = sorted(value, key=lambda key: _sort_key(_encode_value(key)))
            return {key: _walk(value[key]) for key in keys}
        # A list, whatever it was: the encoder spells every sequence alike,
        # and a namedtuple cannot be rebuilt from an iterable.
        if isinstance(value, list | tuple) and _holds_patch(value):
            return [_walk(x) for x in value]
        return value

    def _holds_patch(value) -> bool:
        items = value.values() if isinstance(value, Mapping) else value
        return any(
            isinstance(x, dc.PatchMeta)
            or (isinstance(x, Mapping | list | tuple) and _holds_patch(x))
            for x in items
        )

    out = {key: _walk(params[key]) for key in sorted(params)}
    return out, found


def _hash_bytes(data: bytes | memoryview) -> str:
    """Return the digest of some bytes."""
    return hashlib.blake2b(data, digest_size=DIGEST_SIZE).hexdigest()


def _encode_value(obj: Any) -> Any:
    """Encode one value, short-circuiting the ones JSON already spells."""
    # None, strings and ints are much the most common parameter values, and
    # the check is cheaper than walking the dispatch chain in `_encode`.
    if obj is None or (isinstance(obj, str | int) and not isinstance(obj, bool)):
        return obj
    return _encode(obj)


def _encode(obj: Any) -> Any:
    """Encode anything but the scalars `_encode_value` handles itself."""
    # Ordered by how specific each check is, not by how common: bool is an
    # int, np.bool_ is an np.generic, and a Quantity holds an array.
    if isinstance(obj, bool | np.bool_):
        return {_BOOL: bool(obj)}
    if isinstance(obj, float | np.floating):
        return _encode_float(float(obj))
    if isinstance(obj, np.datetime64 | np.timedelta64):
        return _encode_time(obj)
    if isinstance(obj, np.generic):
        return _encode_value(obj.item())
    # A pandas Timestamp is a datetime and a Timedelta is a timedelta, so
    # both are covered here.
    if isinstance(obj, datetime.datetime | datetime.date):
        return _encode_time(_to_datetime64(obj))
    if isinstance(obj, datetime.timedelta):
        return _encode_time(np.timedelta64(obj))
    if isinstance(obj, complex):
        return {_COMPLEX: [_encode_float(obj.real), _encode_float(obj.imag)]}
    if isinstance(obj, bytes | bytearray):
        return {_BYTES: bytes(obj).hex()}
    if isinstance(obj, Quantity):
        return _encode_quantity(obj)
    if isinstance(obj, Unit):
        return {_UNIT: f"{obj:~}"}
    if isinstance(obj, Enum):
        return _encode_value(obj.value)
    if obj is pd.NA:
        return {_FLOAT: "na"}
    if isinstance(obj, re.Pattern):
        return {_REGEX: [_encode_value(obj.pattern), int(obj.flags)]}
    if isinstance(obj, PurePath):
        return obj.as_posix()
    if isinstance(obj, slice):
        parts = (obj.start, obj.stop, obj.step)
        return {_SLICE: [_encode_value(x) for x in parts]}
    if obj is Ellipsis:
        return {_ELLIPSIS: True}
    if isinstance(obj, PatchMarker):
        return {_PATCH: obj.index}
    # Something with an id of its own is that id wherever it appears.
    if (identity := getattr(obj, "_identity", None)) is not None:
        return {_ID: list(identity())}
    if isinstance(obj, DascoreBaseModel):
        return _encode_model(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = type(obj)
        fields = {x.name: getattr(obj, x.name) for x in dataclasses.fields(obj)}
        tag = f"{cls.__module__}.{cls.__qualname__}"
        return {_MODEL: {TAG_FIELD: tag, "fields": _encode_mapping(fields)}}
    if isinstance(obj, pd.DataFrame | pd.Series):
        return _encode_dataframe(obj)
    if _is_array(obj):
        return _encode_array(obj)
    if isinstance(obj, Mapping):
        return _encode_mapping(obj)
    if isinstance(obj, Set):
        return {_SET: sorted((_encode_value(x) for x in obj), key=_sort_key)}
    # A list and a tuple are one spelling on purpose: a patch function
    # given `(1, 2)` and one given `[1, 2]` made the same call.
    if isinstance(obj, list | tuple):
        return [_encode_value(x) for x in obj]
    if isinstance(obj, partial):
        return _encode_partial(obj)
    if callable(obj):
        return _encode_callable(obj)
    return _refuse(obj, "has no encoding")


def _refuse(obj: Any, why: str) -> NoReturn:
    """Raise for a value whose id would not be its own."""
    cls = type(obj)
    msg = (
        f"A value of type {cls.__module__}.{cls.__qualname__} {why}, so an id "
        "which includes it cannot be derived."
    )
    raise ParameterError(msg)


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
    # Out of range wraps silently on some numpy versions and raises on
    # others; converting back catches the wrap, and both are refused.
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


def _encode_quantity(value: Quantity) -> Any:
    """
    Encode a quantity as its magnitude and the unit it was written in.

    The unit is not normalized: ``1 m`` and ``100 cm`` are the same length
    but not the same call, and an operation is identified by the call.
    """
    return {_QUANTITY: [_encode_value(value.magnitude), f"{value.units:~}"]}


def _encode_model(model: DascoreBaseModel) -> Any:
    """Encode a dascore model as its tag and its fields."""
    fields = model_values(model)
    # A class no tag can name -- a parametrized generic -- is still spelled
    # out, so that two of them do not read alike.
    cls = type(model)
    tag = get_model_tag(cls) or f"{cls.__module__}.{cls.__qualname__}"
    return {_MODEL: {TAG_FIELD: tag, "fields": _encode_mapping(fields)}}


def _encode_dataframe(df: pd.DataFrame | pd.Series) -> Any:
    """
    Encode a dataframe by its labels, dtypes and a hash of its values.

    The labels count as well as the values because a frame's columns are
    parameters in their own right: ``coords_from_df`` names the coords it
    builds after them.
    """
    # hash_array lives in dascore.utils.array, which imports dascore itself.
    from dascore.utils.array import hash_array  # noqa: PLC0415

    frame = df.to_frame() if isinstance(df, pd.Series) else df
    dtypes = [*frame.dtypes, *frame.index.to_frame().dtypes]
    # pandas hashes an object through `str`, so 1 and "1" read alike; a
    # categorical holds its objects in its categories.
    kinds = [getattr(x, "categories", pd.Index([], dtype=x)).dtype for x in dtypes]
    if any(isinstance(x, np.dtype) and x.hasobject for x in kinds):
        _refuse(df, "holds object columns")
    values = pd.util.hash_pandas_object(df, index=True).to_numpy()
    return {
        _DATAFRAME: {
            "columns": [_encode_value(x) for x in frame.columns],
            "index": [_encode_value(x) for x in frame.index.names],
            "dtypes": [str(x) for x in dtypes],
            "values": hash_array(values),
        }
    }


def _encode_array(array: Any) -> Any:
    """Encode an array by its dtype, shape and contents."""
    # A module-scope import is a cycle: dascore.utils.array imports dascore.
    from dascore.utils.array import hash_array  # noqa: PLC0415

    array = to_numpy(array) if is_foreign(array) else np.asarray(array)
    if array.dtype == object:
        # An object array holds python values, which have no bytes to hash;
        # each element is encoded on its own terms, inside the array's frame.
        data = _encode_value(array.tolist())
        return {_ARRAY: {"dtype": "object", "shape": list(array.shape), "data": data}}
    array = _normalize_array(array)
    out = {"dtype": array.dtype.str, "shape": list(array.shape)}
    out["hash"] = hash_array(array)
    return {_ARRAY: out}


def _normalize_array(array: np.ndarray) -> np.ndarray:
    """Return the array in the layout its values are hashed in."""
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
    # on others, so both are refused; see `_encode_time`.
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


def _encode_mapping(mapping: Mapping) -> Any:
    """Encode a mapping; every value is kept, a None included."""
    if not all(isinstance(key, str) and not key.startswith("$") for key in mapping):
        return _encode_odd_keyed_mapping(mapping)
    return {key: _encode_value(value) for key, value in mapping.items()}


def _encode_odd_keyed_mapping(mapping: Mapping) -> Any:
    """
    Encode a mapping as sorted pairs, which any key can survive.

    Used for a mapping whose keys are not all strings, and for one holding a
    key which would otherwise read as a tag.
    """
    pairs = [
        [_encode_value(key), _encode_value(value)] for key, value in mapping.items()
    ]
    return {_DICT: sorted(pairs, key=_sort_key)}


def _encode_partial(value: partial) -> Any:
    """Encode a partial as the function it wraps and what it wraps it with."""
    return {
        _PARTIAL: {
            "func": _encode_value(value.func),
            "args": [_encode_value(x) for x in value.args],
            "kwargs": _encode_mapping(value.keywords),
        }
    }


def _encode_callable(func: Callable) -> Any:
    """Encode a callable by the name which says which one it is."""
    return {_CALLABLE: callable_name(func)}


def callable_name(func: Any) -> str:
    """
    Return a name which says which function or class this is.

    Its import path, when the path leads back to it. Otherwise -- defined
    inside a call -- the path, a digest of its source and its encoded
    defaults. Anything holding state that leaves out is refused: closure
    cells, a bound instance, a callable object, or a lambda, which shares
    its source line with its neighbours.
    """
    module = getattr(func, "__module__", None) or "<unknown>"
    qualname = getattr(func, "__qualname__", None)
    if not isinstance(qualname, str):
        _refuse(func, "is a callable object, whose state is not its name")
    path = f"{module}:{qualname}"
    if _resolves(func, module, qualname):
        return path
    # A builtin's `__self__` is its module. Anything else it is bound to --
    # an instance, or the class an inherited classmethod reads -- is state.
    bound = getattr(func, "__self__", None)
    if bound is not None and not isinstance(bound, ModuleType):
        _refuse(func, "is a method bound to an object")
    if getattr(func, "__closure__", None):
        _refuse(func, "closes over values its source does not show")
    if "<lambda>" in qualname:
        _refuse(func, "is a lambda, which nothing tells from its neighbours")
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        source = ""
    # Text given to `exec` lives in a file called "<string>", whose source
    # reads back blank or as whatever was last compiled under that name.
    filename = getattr(getattr(func, "__code__", None), "co_filename", "")
    if not source.strip() or filename.startswith("<"):
        _refuse(func, "has no import path and no source")
    defaults = [
        getattr(func, "__defaults__", None),
        getattr(func, "__kwdefaults__", None),
    ]
    state = json.dumps(encode(defaults), sort_keys=True)
    return f"{path}@{_hash_bytes((source + state).encode('utf8'))}"


def _resolves(func: Any, module: str, qualname: str) -> bool:
    """Return True if a module and qualified name lead back to the object."""
    found: Any = sys.modules.get(module)
    for part in qualname.split("."):
        found = getattr(found, part, None)
    return found is func


def _sort_key(value: Any) -> str:
    """Return a total order over encoded values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _is_array(obj: Any) -> bool:
    """Return True for anything which should encode as an array."""
    return isinstance(obj, np.ndarray) or is_foreign(obj)
