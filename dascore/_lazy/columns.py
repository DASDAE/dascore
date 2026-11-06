"""
Dictionary encoded columns, string buffers and the sources table.

Members of a recipe keep what they read as columns: a column holds each
distinct value once and a code per row, and the sources table holds one row
per distinct stored array or constant.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype, is_list_like

from dascore.exceptions import ParameterError
from dascore.utils.identity import dtype_description

# Each layout column of a source, with the value one recording no offsets
# has: space separated offsets, rows per offset, and the file stat they were
# taken under.
_LAYOUT_DEFAULTS = MappingProxyType(
    {"offsets": "", "chunk_rows": 0, "mtime_ns": None, "size_bytes": None}
)

_LAYOUT_NUMBERS = ("chunk_rows", "mtime_ns", "size_bytes")

# The columns of the sources table besides `filled`, in the frame's order.
SOURCE_FIELDS = (
    "base_uri",
    "path",
    "format",
    "version",
    "key",
    "origin_id",
    "dtype",
    *_LAYOUT_DEFAULTS,
    "value",
)

# The sources columns kept in one utf-8 buffer; nearly every file or patch has
# its own value, so a dictionary would be no smaller.
_STRING_FIELDS = ("path", "origin_id")

# Below this many rows a dict encodes a column faster than a hash pass.
_FACTORIZE_ROWS = 256

# The fewest rows a column of one value must have to be kept as that value.
_FOLD_ROWS = 256


def _dict_key(value: Any) -> Any:
    """Return the key a value is deduplicated under; type and sign count."""
    if isinstance(value, bool | int | float):
        # repr round trips a float exactly, and keeps -0.0 from 0.0.
        return type(value).__name__, repr(value)
    return value


def _dtype_text(dtype: Any) -> str:
    """Return the text a dtype is stored and named by, fields and all."""
    description = dtype_description(dtype)
    return description if isinstance(description, str) else json.dumps(description)


def _dtype_of(text: str) -> np.dtype:
    """Return the dtype one stored description names."""
    if not text.startswith("["):
        return np.dtype(text)
    return np.dtype([_dtype_field(x) for x in json.loads(text)])


def _dtype_field(entry: Sequence) -> tuple:
    """Return one field of a description, as numpy spells it."""
    name, spec, *shape = entry
    name = tuple(name) if isinstance(name, list) else name
    spec = spec if isinstance(spec, str) else [_dtype_field(x) for x in spec]
    return (name, spec, tuple(shape[0])) if shape else (name, spec)


class _Column:
    """A dictionary encoded column: the distinct values and a code per row."""

    __slots__ = ("codes", "values")

    def __init__(self, values: Sequence, codes):
        self.values = tuple(values)
        self.codes = np.asarray(codes, dtype=np.int32)

    @classmethod
    def of(cls, values: Sequence | np.ndarray) -> _Column:
        """Encode a sequence of values."""
        if not isinstance(values, list | tuple):
            # Positional, whatever the index, and python str for numpy text.
            values = np.asarray(values, object)
        many = len(values) >= _FACTORIZE_ROWS
        if many and infer_dtype(values, skipna=False) == "string":
            codes, distinct = pd.factorize(np.asarray(values, object), sort=False)
            # A missing value gets no code, and hashing stops at a NUL, which
            # merges strings the NUL alone tells apart: either keeps the loop,
            # which encodes both exactly.
            if codes.min() >= 0 and "\0" not in "".join(values):
                return cls([str(x) for x in distinct], codes)
        if len(values) and values[0] is None and all(x is None for x in values):
            return cls.constant(None, len(values))
        index: dict = {}
        distinct: list = []
        codes = np.empty(len(values), np.int32)
        for row, value in enumerate(values):
            code = index.get(key := _dict_key(value))
            if code is None:
                index[key] = code = len(distinct)
                distinct.append(value)
            codes[row] = code
        return cls(distinct, codes)

    @classmethod
    def constant(cls, value: Any, rows: int) -> _Column:
        """Encode one value repeated over rows."""
        return cls((value,), np.zeros(rows, np.int32))

    def map(self, convert) -> _Column:
        """Return the column of each value converted, once per distinct value."""
        out = _Column.of([convert(x) for x in self.values])
        return _Column(out.values, out.codes[self.codes])

    def __getitem__(self, row: int) -> Any:
        """Return the value of one row."""
        return self.values[self.codes[row]]

    def at(self, rows) -> list:
        """Return the values of a selection of rows."""
        values = self.values
        return [values[x] for x in self.codes[rows].tolist()]

    def take(self, rows) -> _Column:
        """Return the column of a selection of rows; the values are kept."""
        return _Column(self.values, self.codes[rows])


def _broadcast(values: Any, rows: int, name: str) -> _Column:
    """Encode one value per row, or a scalar which every row shares."""
    if not is_list_like(values):
        return _Column.constant(values, rows)
    if len(values) != rows:
        msg = f"Give one {name} for every source or member, or one for all."
        raise ParameterError(msg)
    return _Column.of(values)


def _combine(columns: Sequence[_Column]) -> tuple[np.ndarray, np.ndarray]:
    """Return the first row of each distinct combination, and each row's code."""
    count = len(columns[0].codes)
    # A column of one value can split no rows.
    columns = [x for x in columns if len(x.values) > 1]
    if count < _FACTORIZE_ROWS:
        index: dict = {}
        first: list[int] = []
        codes = []
        rows = zip(*[x.codes.tolist() for x in columns]) if columns else [()] * count
        for row, combination in enumerate(rows):
            codes.append(code := index.setdefault(combination, len(index)))
            if code == len(first):
                first.append(row)
        return np.array(first, np.intp), np.array(codes, np.int32)
    key = np.zeros(count, np.int64)
    for column in columns:
        key = pd.factorize(key * len(column.values) + column.codes)[0]
    # Codes number values as they are first met, so each new one is a new max.
    first = np.flatnonzero(np.diff(np.maximum.accumulate(key), prepend=-1))
    return first, key.astype(np.int32)


def _merge_columns(columns: Sequence[_Column], rows: Sequence | None = None) -> _Column:
    """Concatenate columns, or a selection of each one's rows, merging dictionaries."""
    flat: list = []
    starts: dict[int, int] = {}
    found, offsets = [np.empty(0, np.int32)], [0]
    for column, taken in zip(columns, rows or itertools.repeat(slice(None))):
        codes, dictionary = column.codes[taken], column.values
        start = starts.get(id(dictionary))
        if start is None:
            start = len(flat)
            # Columns cut from one table share a dictionary, which may be far
            # bigger than the rows which are left; both are worth skipping.
            if len(dictionary) > len(codes):
                used, codes = np.unique(codes, return_inverse=True)
                dictionary = [dictionary[x] for x in used.tolist()]
            else:
                starts[id(dictionary)] = start
            flat.extend(dictionary)
        found.append(codes)
        offsets.append(start)
    # Every dictionary is deduplicated at once, values first met first.
    merged = _Column.of(flat)
    codes = np.concatenate(found) + np.repeat(offsets, [len(x) for x in found])
    return _Column(merged.values, merged.codes[codes])


class _Strings:
    """A column of strings: one utf-8 buffer, and where each string starts."""

    __slots__ = ("data", "offsets")

    def __init__(self, data: np.ndarray, offsets: np.ndarray):
        self.data = data
        self.offsets = offsets

    @classmethod
    def of(cls, values: Sequence[str]) -> _Strings:
        """Encode a sequence of strings; a missing one is empty."""
        try:
            text = "".join(values)
        except TypeError:
            values = [x if isinstance(x, str) else "" for x in values]
            text = "".join(values)
        data = text.encode("utf-8", "surrogatepass")
        # When each character is one byte, the strings' lengths are the bytes'.
        sized: Sequence = values
        if len(data) != len(text):
            sized = [x.encode("utf-8", "surrogatepass") for x in values]
        ends = itertools.accumulate(map(len, sized), initial=0)
        offsets = np.fromiter(ends, np.int64, len(values) + 1)
        return cls(np.frombuffer(data, np.uint8), offsets)

    def at(self, rows) -> list[str]:
        """Return the strings of a selection of rows."""
        view, offsets = memoryview(self.data), self.offsets
        starts, stops = offsets[:-1][rows].tolist(), offsets[1:][rows].tolist()
        return [str(view[a:b], "utf-8", "surrogatepass") for a, b in zip(starts, stops)]


@dataclass(frozen=True, eq=False)
class _Sources:
    """
    The stored arrays and constants members read, one per row.

    Building and joining give each distinct source one row, so each is
    named once; a view may hold rows none of its members read.
    """

    base_uri: _Column
    path: _Strings
    format: _Column
    version: _Column
    key: _Column
    origin_id: _Strings
    dtype: _Column
    offsets: _Column
    chunk_rows: _Column
    mtime_ns: _Column
    size_bytes: _Column
    value: _Column
    filled: np.ndarray
    # Whether the columns of one value have been folded; see RecipeTable.
    folded: bool = field(default=False, repr=False)

    def __len__(self) -> int:
        return len(self.filled)


def _sources_of_columns(
    columns: Mapping[str, _Column], filled: np.ndarray
) -> tuple[_Sources, np.ndarray]:
    """Return the distinct rows of source columns, and which one each row is."""
    first, rows = _combine([*columns.values(), _Column((False, True), filled)])
    # Rows which all differ are the sources, in order, as they stand.
    fields: dict[str, Any] = {
        x: y if len(first) == len(filled) else y.take(first) for x, y in columns.items()
    }
    for name in _STRING_FIELDS:
        fields[name] = _Strings.of(fields[name].at(slice(None)))
    return _Sources(filled=filled[first], **fields), rows


def _join_sources(
    parts: Sequence[tuple[_Sources, np.ndarray]],
) -> tuple[_Sources, np.ndarray]:
    """Return selected rows of several tables as one table, and where each went."""
    columns: dict[str, _Column] = {}
    rows = [x for _, x in parts]
    for name in SOURCE_FIELDS:
        found = [getattr(x, name) for x, _ in parts]
        if name in _STRING_FIELDS:
            text = itertools.chain.from_iterable(map(_Strings.at, found, rows))
            columns[name] = _Column.of(list(text))
        else:
            columns[name] = _merge_columns(found, rows)
    filled = [x.filled[rows] for x, rows in parts]
    return _sources_of_columns(columns, np.concatenate([np.empty(0, bool), *filled]))


@dataclass(frozen=True, eq=False)
class _Members:
    """Which source row each member reads, and the dtype it passes through."""

    sources: _Sources
    source_row: np.ndarray
    cast: _Column

    def __len__(self) -> int:
        return len(self.source_row)

    @property
    def filled(self) -> np.ndarray:
        """Whether each member is a constant."""
        return self.sources.filled[self.source_row]

    def take(self, rows) -> _Members:
        """Return the members of a selection of rows; the sources are kept."""
        return _Members(self.sources, self.source_row[rows], self.cast.take(rows))


def _merge_members(members: Sequence[_Members]) -> _Members:
    """
    Concatenate members.

    Members of one table share it while they read at least half its rows.
    Otherwise the tables are joined into a new one, which keeps one row of
    each source; a table read less than that is cut to the rows read first.
    """
    if len(members) == 1:
        return members[0]
    cast = _merge_columns([x.cast for x in members])
    groups: dict[int, list[int]] = {}
    for index, member in enumerate(members):
        groups.setdefault(id(member.sources), []).append(index)
    rows = [x.source_row for x in members]
    parts, start = [], 0
    for indices in groups.values():
        sources = members[indices[0]].sources
        kept: Any = slice(None)
        if 2 * sum(len(rows[x]) for x in indices) < len(sources):
            kept = np.unique(np.concatenate([rows[x] for x in indices]))
        elif len(groups) == 1:
            return _Members(sources, np.concatenate(rows), cast)
        for index in indices:
            read = rows[index]
            if not isinstance(kept, slice):
                read = np.searchsorted(kept, read)
            rows[index] = read + start
        parts.append((sources, kept))
        start += len(sources) if isinstance(kept, slice) else len(kept)
    sources, remap = _join_sources(parts)
    source_row = remap[np.concatenate([np.empty(0, np.intp), *rows])]
    return _Members(sources, source_row, cast)


def _members_of(frame: pd.DataFrame) -> _Members:
    """Return the members one array's rows of a member frame describe."""
    columns = {
        x: _Column.of(frame[x].astype(str).to_numpy(object)) for x in SOURCE_FIELDS[:4]
    }
    # A frame written before members stated these reads them as empty, or
    # the layout default.
    for name in (*SOURCE_FIELDS[4:], "cast"):
        default = _LAYOUT_DEFAULTS.get(name, "")
        values = (
            frame[name].to_numpy(object) if name in frame else [default] * len(frame)
        )
        if name in _LAYOUT_DEFAULTS:
            # a join with an older frame leaves NaN, and parquet gives floats
            values = [_layout_value(name, x, default) for x in values]
        columns[name] = _Column.of(values)
    cast = columns.pop("cast")
    sources, rows = _sources_of_columns(columns, frame["filled"].to_numpy(bool))
    return _Members(sources, rows, cast)


def _layout_value(name: str, value: Any, default: Any) -> Any:
    """Return a layout cell of a frame as a source states it."""
    if value is None or pd.isna(value):
        return default
    return int(value) if name in _LAYOUT_NUMBERS else value


def _folded(array: np.ndarray) -> np.ndarray:
    """Return an array of one value throughout as that value, broadcast."""
    # A short column costs more to check than folding it saves.
    if len(array) < _FOLD_ROWS or not array.strides[-1] or (array != array[0]).any():
        return array
    return np.broadcast_to(array[:1].copy(), array.shape)
