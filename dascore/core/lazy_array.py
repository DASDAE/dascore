"""
A recipe for an array which is read a member at a time.

A [`LazyArray`](`dascore.core.lazy_array.LazyArray`) holds no data. It holds
members, and each member says "this window of this source goes in this box of
the output". Everything is positional: axis numbers and sample indices, never
dimension names, coordinates or units.

Many arrays share one [`LazyTable`](`dascore.core.lazy_array.LazyTable`),
which owns the storage; an array is a view of one of its rows. Members of an
array are stored together, in canonical placement order, so slicing, joining
and rechunking are vectorized over members and never open a file.

Examples
--------
>>> import numpy as np
>>> from dascore.core.source import ArraySource
>>> from dascore.core.lazy_array import LazyArray, concat
>>>
>>> left = LazyArray.from_source(ArraySource.full((4, 3), 1.0))
>>> right = LazyArray.from_source(ArraySource.full((4, 3), 2.0))
>>> array = concat([left, right], axis=0)
>>> assert array.shape == (8, 3) and len(array) == 2
>>> assert np.array_equal(array[0:5].load()[:4], np.ones((4, 3)))
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from dascore.core.source import ArraySource
from dascore.exceptions import ParameterError
from dascore.utils.identity import DIGEST_SIZE, H, dtype_description

# The src_axis of an output axis which no stored axis feeds.
NEW_AXIS = -1

# The per member per axis matrices, in the order the digest takes them.
AXIS_FIELDS = ("out_start", "out_stop", "src_axis", "src_start", "src_extent")

# The fields of the sources dictionary, which is the sources table.
SOURCE_FIELDS = ("base_uri", "path", "format", "version")

# The dictionary encoded member columns; `filled` is a plain bool array.
MEMBER_FIELDS = ("source", "key", "origin_id", "dtype", "value")

# The version of the byte layout the data_id digest is taken over. Bump it
# when the layout changes, so ids from two layouts cannot meet.
DIGEST_LAYOUT = 2

# The bytes the digest starts with, so no other payload can read alike.
_DIGEST_TAG = b"dascore-lazy-blocks\0"

# The name the array api reports for a lazy array.
BACKEND_NAME = "lazy"


def _dict_key(value: Any) -> Any:
    """Return the key a value is deduplicated under; type and sign count."""
    if isinstance(value, bool | int | float):
        # repr round trips a float exactly, and keeps -0.0 from 0.0.
        return type(value).__name__, repr(value)
    return value


def _dtype_text(dtype: np.dtype) -> str:
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
    def of(cls, values: Sequence) -> _Column:
        """Encode a sequence of values."""
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

    def __getitem__(self, row: int) -> Any:
        """Return the value of one row."""
        return self.values[self.codes[row]]

    def take(self, rows) -> _Column:
        """Return the column of a selection of rows; the values are kept."""
        return _Column(self.values, self.codes[rows])


def _merge_columns(columns: Sequence[_Column]) -> _Column:
    """Concatenate columns, merging their dictionaries."""
    index: dict = {}
    values: list = []
    seen: dict[int, np.ndarray] = {}
    parts = []
    for column in columns:
        codes, dictionary = column.codes, column.values
        # Columns cut from one table share a dictionary, which may be far
        # bigger than the rows which are left; both are worth skipping.
        remap = seen.get(id(dictionary))
        if remap is None and len(dictionary) > len(codes):
            used, codes = np.unique(codes, return_inverse=True)
            dictionary = [dictionary[x] for x in used.tolist()]
        if remap is None:
            remap = np.empty(len(dictionary), np.int32)
            for code, value in enumerate(dictionary):
                key = _dict_key(value)
                if key not in index:
                    index[key] = len(values)
                    values.append(value)
                remap[code] = index[key]
            if dictionary is column.values:
                seen[id(dictionary)] = remap
        parts.append(remap[codes])
    codes = np.concatenate(parts) if parts else np.empty(0, np.int32)
    return _Column(values, codes)


@dataclass(frozen=True, eq=False)
class _Members:
    """The columns which say what each member reads."""

    source: _Column
    key: _Column
    origin_id: _Column
    dtype: _Column
    filled: np.ndarray
    value: _Column

    def __len__(self) -> int:
        return len(self.filled)

    def take(self, rows) -> _Members:
        """Return the members of a selection of rows."""
        columns = {x: getattr(self, x).take(rows) for x in MEMBER_FIELDS}
        return _Members(filled=self.filled[rows], **columns)


def _merge_members(members: Sequence[_Members]) -> _Members:
    """Concatenate members, merging every dictionary in one pass."""
    if len(members) == 1:
        return members[0]
    columns = {
        name: _merge_columns([getattr(x, name) for x in members])
        for name in MEMBER_FIELDS
    }
    filled = [x.filled for x in members]
    stacked = np.concatenate(filled) if filled else np.empty(0, bool)
    return _Members(filled=stacked, **columns)


@dataclass(frozen=True, eq=False)
class _Block:
    """One array's header and members, before they are stacked in a table."""

    shape: tuple[int, ...]
    dtype: np.dtype
    concat_axis: int
    members: _Members
    axes: dict[str, np.ndarray]

    @property
    def ndim(self) -> int:
        """The number of output axes."""
        return len(self.shape)

    def __len__(self) -> int:
        return len(self.members)

    def take(self, rows) -> _Block:
        """Return the block holding a selection of members."""
        axes = {name: matrix[rows] for name, matrix in self.axes.items()}
        return replace(self, members=self.members.take(rows), axes=axes)


def _check_axis(axis: int, ndim: int) -> int:
    """Return a positive axis number, refusing one outside the array."""
    out = int(axis)
    out = out + ndim if out < 0 else out
    if not 0 <= out < ndim:
        msg = f"Axis {axis} is outside an array of {ndim} dimensions."
        raise ParameterError(msg)
    return out


def _is_canonical(out_start: np.ndarray) -> bool:
    """Whether the corners are in lexicographic order; nothing is sorted."""
    if len(out_start) < 2:
        return True
    difference = np.diff(out_start, axis=0)
    first = np.argmax(difference != 0, axis=1)
    lead = np.take_along_axis(difference, first[:, None], axis=1)[:, 0]
    return bool(np.all(lead >= 0))


def _canonical(block: _Block) -> _Block:
    """Return the block in canonical placement order."""
    out_start = block.axes["out_start"]
    if _is_canonical(out_start):
        return block
    return block.take(np.lexsort(out_start.T[::-1]))


def _merge_sorted_unique(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Merge two sorted arrays without sorting them again, dropping repeats."""
    out = np.empty(len(first) + len(second), first.dtype)
    out[np.arange(len(first)) + np.searchsorted(second, first, "left")] = first
    out[np.arange(len(second)) + np.searchsorted(first, second, "right")] = second
    keep = np.ones(len(out), bool)
    keep[1:] = out[1:] != out[:-1]
    return out[keep]


@dataclass(frozen=True, eq=False)
class LazyTable:
    """
    The storage many lazy arrays share.

    Members of array `k` are the rows `member_offsets[k]:member_offsets[k+1]`
    of every member column, and its axis rows start at `axis_offsets[k]` in
    each flat placement array. Arrays of different `ndim` therefore sit in
    one table: the placement array is flat and ragged, one row per member per
    output axis, which is the shape the database tables take.

    Every array here is read only, so views may share all of them and an
    id worked out once stays true.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.core.source import ArraySource
    >>> from dascore.core.lazy_array import LazyArray, LazyTable
    >>>
    >>> flat = LazyArray.from_source(ArraySource.full((4, 3), 1.0))
    >>> cube = LazyArray.from_source(ArraySource.full((2, 3, 4), 2.0))
    >>> table = LazyTable.from_arrays([flat, cube])
    >>> assert [x.ndim for x in table] == [2, 3]
    """

    member_offsets: np.ndarray
    axis_offsets: np.ndarray
    shape_offsets: np.ndarray
    shapes: np.ndarray
    dtypes: _Column
    concat_axes: np.ndarray
    members: _Members
    axes: Mapping[str, np.ndarray]
    _ids: dict[int, str] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Freeze the storage; each flag is set once, whatever the size."""
        members = self.members
        for array in (
            self.member_offsets,
            self.axis_offsets,
            self.shape_offsets,
            self.shapes,
            self.concat_axes,
            self.dtypes.codes,
            members.filled,
            *self.axes.values(),
            *[getattr(members, x).codes for x in MEMBER_FIELDS],
        ):
            # A view can be written through whatever it is a view of.
            array.setflags(write=False)
            base = array.base
            while base is not None:
                base.setflags(write=False)
                base = base.base
        object.__setattr__(self, "axes", MappingProxyType(dict(self.axes)))

    @classmethod
    def from_arrays(cls, arrays: Sequence[LazyArray]) -> LazyTable:
        """Stack arrays, which may differ in ndim, into one table."""
        return _table([x._block() for x in arrays])

    def __len__(self) -> int:
        """The number of arrays in the table."""
        return len(self.member_offsets) - 1

    def __getitem__(self, row: int) -> LazyArray:
        """Return the array stored in one row."""
        index = int(row)
        row = index + len(self) if index < 0 else index
        if not 0 <= row < len(self):
            msg = f"Row {index} is outside a table of {len(self)} arrays."
            raise IndexError(msg)
        return LazyArray(self, row)

    def __iter__(self):
        """Iterate over the arrays in the table."""
        return (LazyArray(self, row) for row in range(len(self)))

    @property
    def n_members(self) -> int:
        """The number of members in the whole table."""
        return len(self.members)


def _members_of(frame: pd.DataFrame) -> _Members:
    """Return the members one array's rows of a member frame describe."""
    columns = {x: _Column.of(list(frame[x])) for x in MEMBER_FIELDS[1:]}
    source = zip(*[frame[x].astype(str) for x in SOURCE_FIELDS])
    columns["source"] = _Column.of(list(source))
    return _Members(filled=frame["filled"].to_numpy(bool, copy=True), **columns)


def _table(blocks: Sequence[_Block]) -> LazyTable:
    """Stack blocks into one table."""
    counts = np.array([len(x) for x in blocks], np.int64)
    ndim = np.array([x.ndim for x in blocks], np.int64)
    shapes = [np.asarray(x.shape, np.int64) for x in blocks]
    if len(blocks) == 1:
        axes = {name: blocks[0].axes[name].reshape(-1) for name in AXIS_FIELDS}
    else:
        axes = {
            name: np.concatenate([x.axes[name].reshape(-1) for x in blocks])
            if blocks
            else np.empty(0, np.int64)
            for name in AXIS_FIELDS
        }
    return LazyTable(
        member_offsets=_offsets(counts),
        axis_offsets=_offsets(counts * ndim),
        shape_offsets=_offsets(ndim),
        shapes=np.concatenate(shapes) if shapes else np.empty(0, np.int64),
        dtypes=_Column.of([x.dtype for x in blocks]),
        concat_axes=np.array([x.concat_axis for x in blocks], np.int64),
        members=_merge_members([x.members for x in blocks]),
        axes=axes,
    )


def _offsets(counts: np.ndarray) -> np.ndarray:
    """Return the start of each run, and the total, from run lengths."""
    out = np.zeros(len(counts) + 1, np.int64)
    np.cumsum(counts, out=out[1:])
    return out


class LazyArray:
    """
    An array which says where each of its members is read from.

    A view of one row of a [`LazyTable`](`dascore.core.lazy_array.LazyTable`).
    The shape, ndim and dtype are stored rather than derived, so an array
    which selects nothing still knows what it is. Boxes may not overlap and
    must cover the whole output; a hole is an explicit constant member.
    [`validate`](`dascore.core.lazy_array.LazyArray.validate`) checks every
    rule, on demand.

    A member is cast once, from the dtype it is stored at to the array's, so
    a chain of joins does not round at each step as numpy would.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.core.source import ArraySource
    >>> from dascore.core.lazy_array import LazyArray
    >>>
    >>> array = LazyArray.from_source(ArraySource.full((6, 4), np.nan))
    >>> assert array.shape == (6, 4) and array.ndim == 2
    >>> window = array[2:5]
    >>> assert window.shape == (3, 4)
    >>> assert np.isnan(window.load()).all()
    """

    __slots__ = ("_row", "_table")

    def __init__(self, table: LazyTable, row: int):
        self._table = table
        self._row = int(row)

    @classmethod
    def from_source(cls, source: ArraySource, base_uri: str = "") -> LazyArray:
        """
        Return the array one source loads, placed at the origin.

        Parameters
        ----------
        source
            A loadable source which states the extent of its whole array.
        base_uri
            A prefix the source's path is stored relative to.
        """
        return cls.from_sources([source], base_uri=base_uri)

    @classmethod
    def from_sources(
        cls,
        sources: Sequence[ArraySource],
        starts=None,
        axis: int = 0,
        shape: tuple[int, ...] | None = None,
        base_uri: str = "",
    ) -> LazyArray:
        """
        Return an array which reads one member from each source.

        Parameters
        ----------
        sources
            The loadable sources, each of which states the extent of its
            whole array.
        starts
            The corner each source is placed at, as an `(n, ndim)` array of
            sample numbers. By default the sources are laid end to end.
        axis
            The axis the sources are laid along when `starts` is not given.
        shape
            The shape of the output; the smallest which holds every
            member by default.
        base_uri
            A prefix the sources' paths are stored relative to. A path
            which does not start with it is stored whole, and a member's
            path is always the two joined.
        """
        sources = list(sources)
        if not sources:
            msg = "A lazy array takes at least one source."
            raise ParameterError(msg)
        block = _block_of_sources(sources, base_uri)
        ndim = block.ndim
        axis = _check_axis(axis, ndim)
        lengths = block.axes["out_stop"]
        if starts is None:
            corners = np.zeros_like(lengths)
            corners[1:, axis] = np.cumsum(lengths[:-1, axis])
            concat_axis = axis
        else:
            # A copy, so the table never holds an array the caller keeps.
            corners = np.array(starts, np.int64).reshape(len(sources), ndim)
            concat_axis = NEW_AXIS
        block.axes["out_start"] = corners
        block.axes["out_stop"] = corners + lengths
        if shape is None:
            shape = block.axes["out_stop"].max(axis=0)
        shape = tuple(int(x) for x in shape)
        block = replace(block, shape=shape, concat_axis=concat_axis)
        if not lengths.all():
            # A source with no samples fills no box, so it is not a member.
            block = block.take(np.flatnonzero(lengths.all(axis=1)))
        return _array(_canonical(block))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, shape, dtype) -> LazyArray:
        """
        Return the array `to_frame` wrote.

        Parameters
        ----------
        frame
            One row per member per output axis, as `to_frame` returns.
        shape
            The shape of the array, which its members do not state.
        dtype
            The dtype of the array.
        """
        shape = tuple(int(x) for x in shape)
        ndim = len(shape)
        keys = ["ordinal", "out_axis"]
        order = frame[keys].to_numpy(np.int64)
        if not _is_canonical(order):
            frame = frame.sort_values(keys, kind="stable")
        count = len(frame) // ndim
        # Copies, so that editing the frame afterwards changes no array.
        matrices = {
            name: frame[name].to_numpy(np.int64, copy=True).reshape(count, ndim)
            for name in AXIS_FIELDS
        }
        block = _Block(
            shape=shape,
            dtype=np.dtype(dtype),
            concat_axis=_concat_axis_of(matrices),
            members=_members_of(frame.iloc[::ndim]),
            axes=matrices,
        )
        return _array(block)

    @property
    def table(self) -> LazyTable:
        """The table which holds this array."""
        return self._table

    @property
    def row(self) -> int:
        """Which row of the table this array is."""
        return self._row

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the array the members cover."""
        table = self._table
        start, stop = table.shape_offsets[self._row : self._row + 2]
        return tuple(table.shapes[start:stop].tolist())

    @property
    def ndim(self) -> int:
        """The number of dimensions of the array."""
        offsets = self._table.shape_offsets
        return int(offsets[self._row + 1] - offsets[self._row])

    @property
    def dtype(self) -> np.dtype:
        """The dtype of the loaded array."""
        return np.dtype(self._table.dtypes[self._row])

    @property
    def size(self) -> int:
        """The number of elements in the array."""
        return math.prod(self.shape)

    def __len__(self) -> int:
        """The number of members."""
        offsets = self._table.member_offsets
        return int(offsets[self._row + 1] - offsets[self._row])

    def __repr__(self) -> str:
        name = type(self).__name__
        return f"{name}(shape={self.shape}, dtype={self.dtype}, members={len(self)})"

    def _block(self) -> _Block:
        """Return this array's storage, as views of the table's arrays."""
        table, row = self._table, self._row
        first, last = table.member_offsets[row : row + 2]
        start = int(table.axis_offsets[row])
        count, ndim = int(last - first), self.ndim
        stop = start + count * ndim
        axes = {
            name: table.axes[name][start:stop].reshape(count, ndim)
            for name in AXIS_FIELDS
        }
        return _Block(
            shape=self.shape,
            dtype=self.dtype,
            concat_axis=int(table.concat_axes[row]),
            members=table.members.take(slice(int(first), int(last))),
            axes=axes,
        )

    def source(self, member: int) -> ArraySource:
        """
        Return the source one member reads, resolved against its base uri.

        Parameters
        ----------
        member
            Which member, by its place in the array.
        """
        return _member_source(self._block(), int(member))

    @property
    def sources(self) -> tuple[ArraySource, ...]:
        """The source each member reads, in placement order."""
        block = self._block()
        return tuple(_member_source(block, x) for x in range(len(block)))

    def __getitem__(self, index) -> LazyArray:
        """
        Return the array a selection of samples describes; nothing is read.

        Only slices with a step of one are taken, and trailing axes may be
        left out. Each box is clipped to the request, its window moved by
        what was clipped, and members the request misses are dropped.
        """
        index = index if isinstance(index, tuple) else (index,)
        shape = self.shape
        if len(index) > len(shape):
            msg = f"Cannot index a {self.ndim} dimensional array with {index}."
            raise ParameterError(msg)
        spans = [_resolve(item, size) for item, size in zip(index, shape)]
        spans += [(0, size) for size in shape[len(index) :]]
        starts = np.array([x[0] for x in spans], np.int64)
        stops = np.array([x[1] for x in spans], np.int64)
        if not starts.any() and np.array_equal(stops, np.asarray(shape)):
            return self
        return _array(_clip(self._block(), starts, stops))

    def transpose(self, order: Sequence[int] | None = None) -> LazyArray:
        """
        Return the array with its output axes permuted.

        Parameters
        ----------
        order
            The output axis each new axis takes; reversed by default.
        """
        ndim = self.ndim
        if order is None:
            order = tuple(reversed(range(ndim)))
        order = tuple(_check_axis(x, ndim) for x in order)
        if sorted(order) != list(range(ndim)):
            msg = f"{order} is not a permutation of {ndim} axes."
            raise ParameterError(msg)
        block = self._block()
        shape = tuple(self.shape[x] for x in order)
        concat_axis = block.concat_axis
        concat_axis = order.index(concat_axis) if concat_axis >= 0 else NEW_AXIS
        axes = {name: matrix[:, order] for name, matrix in block.axes.items()}
        moved = replace(block, shape=shape, concat_axis=concat_axis, axes=axes)
        return _array(_canonical(moved))

    def rechunk(self, bounds, axis: int = 0) -> LazyTable:
        """
        Cut the array at new bounds along one axis, giving one array each.

        Parameters
        ----------
        bounds
            The sample numbers the pieces are cut at, ascending; `n` bounds
            give `n - 1` arrays, and samples outside them are dropped.
        axis
            The axis to cut along. The members must already be full width
            slabs stacked along it.

        Raises
        ------
        NotImplementedError
            If the bounds overlap, or the members are not full width slabs
            stacked along `axis`.
        ParameterError
            If the bounds fall outside the array.
        """
        axis = _check_axis(axis, self.ndim)
        block = self._block()
        _check_stack(block, axis)
        bounds = np.asarray(bounds, np.int64).reshape(-1)
        if len(bounds) < 2:
            msg = "Rechunking takes at least two bounds, which are the ends."
            raise ParameterError(msg)
        if np.any(np.diff(bounds) <= 0):
            msg = "Rechunking into overlapping or empty pieces is not supported."
            raise NotImplementedError(msg)
        if bounds[0] < 0 or bounds[-1] > self.shape[axis]:
            msg = f"Bounds {bounds} fall outside the array."
            raise ParameterError(msg)
        return _rechunk(block, bounds, axis)

    def validate(self) -> LazyArray:
        """
        Check the array's rules, raising `ParameterError` for the first broken.

        The boxes must be inside the array, may not overlap, and must cover
        every sample; each window must be inside its source; and the members
        must be in canonical placement order. Operations do not validate,
        because they start from arrays which already hold.
        """
        _validate(self._block())
        return self

    def load(self) -> np.ndarray:
        """
        Read every member and return the array they make.

        The rules are checked first, so no sample of the result is left
        uninitialized; see
        [`validate`](`dascore.core.lazy_array.LazyArray.validate`).
        """
        block = self._block()
        _validate(block)
        out = np.empty(self.shape, self.dtype)
        start, stop = block.axes["out_start"], block.axes["out_stop"]
        src_axis = block.axes["src_axis"]
        for row in range(len(block)):
            data = _member_source(block, row).load()
            data = _to_output(data, src_axis[row])
            box = tuple(map(slice, start[row].tolist(), stop[row].tolist()))
            out[box] = data
        return out

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Load the array for numpy; each call reads every member."""
        out = self.load()
        return out if dtype is None else out.astype(dtype, copy=False)

    def __array_namespace__(self, api_version: str | None = None):
        """Return the namespace which owns this array."""
        return _NAMESPACE

    @property
    def data_id(self) -> str:
        """
        The id of the array the members make; nothing is read to work it out.

        It says which samples of which sources fill which region of the
        output, and nothing else: an array cut into members any way at all
        keeps the id of the array it tiles. Each source is named exactly as
        an [`ArraySource`](`dascore.core.source.ArraySource`) of it is, so
        how the paths are split under a base uri does not reach the id
        either, and a member is cast once, so the array's own dtype does.

        Examples
        --------
        >>> from dascore.core.source import ArraySource
        >>> from dascore.core.lazy_array import LazyArray
        >>>
        >>> array = LazyArray.from_source(ArraySource.full((8, 2), 1.0))
        >>> assert array[0:4].data_id == array[0:8][0:4].data_id
        """
        cache = self._table._ids
        out = cache.get(self._row)
        if out is None:
            cache[self._row] = out = self._data_id()
        return out

    def _identity(self) -> tuple[str, str]:
        """Return the id this array has as an operation's parameter."""
        return "array", self.data_id

    def _data_id(self) -> str:
        """Work out the id of the array the members make."""
        block = self._block()
        signature = _signature(block)
        out = _whole_source_id(block, signature)
        if out is not None:
            return out
        header = [list(self.shape), _dtype_text(self.dtype)]
        return H("blocks", [*header, _digest(block, signature)])

    def to_frame(self) -> pd.DataFrame:
        """
        Return one row per member per output axis, for inspection.

        The columns are those of the member and axis database tables, joined:
        `ordinal`, `out_axis`, the placement, and what the member reads.
        """
        block = self._block()
        count, ndim = len(block), self.ndim
        members = block.members
        out = {
            "ordinal": np.repeat(np.arange(count, dtype=np.int64), ndim),
            "out_axis": np.tile(np.arange(ndim, dtype=np.int64), count),
            **{name: block.axes[name].reshape(-1) for name in AXIS_FIELDS},
        }
        source = members.source
        for index, name in enumerate(SOURCE_FIELDS):
            out[name] = _spread(source, [x[index] for x in source.values], ndim)
        for name in ("key", "origin_id", "dtype"):
            column = getattr(members, name)
            out[name] = _spread(column, column.values, ndim)
        out["filled"] = np.repeat(members.filled, ndim)
        # The values are python scalars of several types, which stay objects.
        out["value"] = _spread(members.value, members.value.values, ndim, object)
        return pd.DataFrame(out)


def _spread(column: _Column, values: Sequence, ndim: int, dtype=None) -> np.ndarray:
    """Return one row per member per output axis of a column's values."""
    return np.asarray(values, dtype)[np.repeat(column.codes, ndim)]


def _array(block: _Block) -> LazyArray:
    """Return the array one block describes, in a table of its own."""
    return LazyArray(_table([block]), 0)


def _resolve(item, size: int) -> tuple[int, int]:
    """Return the samples one index selects, refusing what cannot be a box."""
    if not isinstance(item, slice):
        msg = f"A lazy array takes slices with a step of one, not {item!r}."
        raise ParameterError(msg)
    if item.step is not None and item.step != 1:
        msg = f"A lazy array takes a step of one, not {item.step}."
        raise ParameterError(msg)
    start, stop, _ = item.indices(size)
    return start, max(start, stop)


def _clip(block: _Block, starts: np.ndarray, stops: np.ndarray) -> _Block:
    """Return the members a request selects, each clipped to it."""
    rows = _candidates(block, starts, stops)
    start = np.maximum(block.axes["out_start"][rows], starts)
    stop = np.minimum(block.axes["out_stop"][rows], stops)
    keep = np.flatnonzero(np.all(stop > start, axis=1))
    start, stop = start[keep], stop[keep]
    # The candidates are one run, so every column is gathered once.
    out = block.take(keep + (rows.start or 0))
    axes = out.axes
    axes["src_start"] += np.where(axes["src_axis"] >= 0, start - axes["out_start"], 0)
    axes["out_start"] = start - starts
    axes["out_stop"] = stop - starts
    _flatten_constants(axes, out.members.filled)
    out = replace(out, shape=tuple((stops - starts).tolist()))
    # Stacked along the first axis, only one corner can be clipped, so the
    # members stay in order; any other placement may tie two of them.
    return out if block.concat_axis == 0 else _canonical(out)


def _concat_axis_of(axes: dict[str, np.ndarray]) -> int:
    """Return an axis the members are laid along, which _candidates needs."""
    start, stop = axes["out_start"], axes["out_stop"]
    if len(start) < 2:
        return 0
    ordered = np.flatnonzero(np.all(start[1:] >= stop[:-1], axis=0))
    return int(ordered[0]) if len(ordered) else NEW_AXIS


def _candidates(block: _Block, starts: np.ndarray, stops: np.ndarray) -> slice:
    """Return the rows a request can touch, by binary search where it can."""
    axis = block.concat_axis
    if axis < 0 or len(block) == 0:
        return slice(None)
    first = np.searchsorted(block.axes["out_stop"][:, axis], starts[axis], "right")
    last = np.searchsorted(block.axes["out_start"][:, axis], stops[axis], "left")
    return slice(int(first), int(max(first, last)))


def _block_of_sources(sources: Sequence[ArraySource], base_uri: str = "") -> _Block:
    """Return a block which reads each source whole, placed at the origin."""
    ndim = sources[0].ndim
    count = len(sources)
    for source in sources:
        if not source.loadable:
            msg = f"{source} does not say enough to load an array."
            raise ParameterError(msg)
        if source.ndim != ndim:
            msg = "Every source of one lazy array must have the same ndim."
            raise ParameterError(msg)
        if len(source.extent) != ndim:
            msg = f"{source} does not state the extent of its whole array."
            raise ParameterError(msg)
    if ndim == 0:
        msg = "A lazy array takes sources of at least one dimension."
        raise ParameterError(msg)
    axes = {
        "out_start": np.zeros((count, ndim), np.int64),
        "out_stop": np.array([x.shape for x in sources], np.int64),
        "src_axis": np.tile(np.arange(ndim, dtype=np.int64), (count, 1)),
        "src_start": np.array([[w[0] for w in x.windows] for x in sources], np.int64),
        "src_extent": np.array([x.extent for x in sources], np.int64),
    }
    members = _Members(
        source=_Column.of(
            [(*_split_path(x.path, base_uri), x.format, x.version) for x in sources]
        ),
        key=_Column.of([x.key for x in sources]),
        origin_id=_Column.of([x.origin_id for x in sources]),
        dtype=_Column.of([_dtype_text(np.dtype(x.dtype)) for x in sources]),
        filled=np.array([x.filled for x in sources], bool),
        value=_Column.of([x.value for x in sources]),
    )
    _flatten_constants(axes, members.filled)
    dtype = np.result_type(*[np.dtype(x.dtype) for x in sources])
    return _Block((0,) * ndim, dtype, NEW_AXIS, members, axes)


def _flatten_constants(axes: dict[str, np.ndarray], filled: np.ndarray) -> None:
    """Set the source fields of every constant member to its box."""
    rows = np.flatnonzero(filled)
    if not len(rows):
        return
    lengths = axes["out_stop"][rows] - axes["out_start"][rows]
    axes["src_start"][rows] = 0
    axes["src_extent"][rows] = np.where(axes["src_axis"][rows] >= 0, lengths, 0)


def _split_path(path: str, base_uri: str) -> tuple[str, str]:
    """Split a path into the prefix it is stored under and the rest."""
    if base_uri and path.startswith(base_uri):
        return base_uri, path[len(base_uri) :]
    return "", path


def _member_source(block: _Block, member: int) -> ArraySource:
    """Return the source one member of a block reads."""
    members = block.members
    base_uri, path, format_, version = members.source[member]
    src_axis = block.axes["src_axis"][member]
    src_start = block.axes["src_start"][member]
    lengths = block.axes["out_stop"][member] - block.axes["out_start"][member]
    extent = block.axes["src_extent"][member]
    stored = np.flatnonzero(src_axis >= 0)
    windows: list = [()] * len(stored)
    sizes: list = [0] * len(stored)
    for out_axis in stored.tolist():
        axis = int(src_axis[out_axis])
        start = int(src_start[out_axis])
        windows[axis] = (start, start + int(lengths[out_axis]))
        sizes[axis] = int(extent[out_axis])
    return ArraySource(
        path=base_uri + path,
        format=format_,
        version=version,
        key=members.key[member],
        windows=tuple(windows),
        shape=tuple(stop - start for start, stop in windows),
        dtype=_dtype_of(members.dtype[member]),
        origin_id=members.origin_id[member],
        extent=tuple(sizes),
        filled=bool(members.filled[member]),
        value=members.value[member],
    )


def _to_output(data: np.ndarray, src_axis: np.ndarray) -> np.ndarray:
    """Return a member's array with its stored axes on the output's."""
    axes = src_axis.tolist()
    order = [x for x in axes if x >= 0]
    if order != sorted(order):
        data = np.transpose(data, order)
    new = [index for index, axis in enumerate(axes) if axis < 0]
    return np.expand_dims(data, tuple(new)) if new else data


def concat(arrays: Sequence[LazyArray], axis: int = 0) -> LazyArray:
    """
    Join arrays end to end along one axis, in one pass over their members.

    Parameters
    ----------
    arrays
        The arrays to join; they must agree on every other axis.
    axis
        The axis to join along.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.core.source import ArraySource
    >>> from dascore.core.lazy_array import LazyArray, concat
    >>>
    >>> parts = [LazyArray.from_source(ArraySource.full((2, 3), x)) for x in (1, 2)]
    >>> joined = concat(parts, axis=1)
    >>> assert joined.shape == (2, 6)
    >>> expected = np.concatenate([x.load() for x in parts], axis=1)
    >>> assert np.array_equal(joined.load(), expected)
    """
    blocks, ndim = _blocks_of(list(arrays))
    return _array(_canonical(_join(blocks, _check_axis(axis, ndim))))


def _join(blocks: Sequence[_Block], axis: int) -> _Block:
    """Return the block which lays every block end to end along one axis."""
    shapes = [x.shape for x in blocks]
    for other in range(len(shapes[0])):
        sizes = {x[other] for x in shapes}
        if other != axis and len(sizes) > 1:
            msg = f"Arrays with shapes {shapes} cannot be joined on axis {axis}."
            raise ParameterError(msg)
    lengths = np.array([x[axis] for x in shapes], np.int64)
    axes = {
        name: np.concatenate([x.axes[name] for x in blocks]) for name in AXIS_FIELDS
    }
    counts = np.array([len(x) for x in blocks], np.int64)
    shift = np.repeat(_offsets(lengths)[:-1], counts)
    axes["out_start"][:, axis] += shift
    axes["out_stop"][:, axis] += shift
    shape = list(shapes[0])
    shape[axis] = int(lengths.sum())
    slabs = all(len(x) <= 1 or x.concat_axis == axis for x in blocks)
    return _Block(
        shape=tuple(shape),
        dtype=np.result_type(*[x.dtype for x in blocks]),
        concat_axis=axis if slabs else NEW_AXIS,
        members=_merge_members([x.members for x in blocks]),
        axes=axes,
    )


def stack(arrays: Sequence[LazyArray], axis: int = 0) -> LazyArray:
    """
    Join arrays along a new axis, in one pass over their members.

    Parameters
    ----------
    arrays
        The arrays to stack; they must all have the same shape.
    axis
        Where the new axis goes in the output.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.core.source import ArraySource
    >>> from dascore.core.lazy_array import LazyArray, stack
    >>>
    >>> parts = [LazyArray.from_source(ArraySource.full((2, 3), x)) for x in (1, 2)]
    >>> stacked = stack(parts, axis=1)
    >>> assert stacked.shape == (2, 2, 3)
    >>> assert np.array_equal(stacked.load(), np.stack([x.load() for x in parts], 1))
    """
    blocks, ndim = _blocks_of(list(arrays))
    axis = _check_axis(axis, ndim + 1)
    shapes = {x.shape for x in blocks}
    if len(shapes) > 1:
        msg = f"Arrays with shapes {sorted(shapes)} cannot be stacked."
        raise ParameterError(msg)
    expanded = [_expand(x, axis) for x in blocks]
    return _array(_canonical(_join(expanded, axis)))


def _expand(block: _Block, axis: int) -> _Block:
    """Return the block with a length one axis no stored axis feeds."""
    fills = {"out_stop": 1, "src_axis": NEW_AXIS}
    axes = {
        name: np.insert(block.axes[name], axis, fills.get(name, 0), axis=1)
        for name in AXIS_FIELDS
    }
    shape = list(block.shape)
    shape.insert(axis, 1)
    moved = block.concat_axis
    moved = moved + int(moved >= axis) if moved >= 0 else NEW_AXIS
    return replace(block, shape=tuple(shape), concat_axis=moved, axes=axes)


def _blocks_of(arrays: Sequence[LazyArray]) -> tuple[list[_Block], int]:
    """Return the block of each array, which must agree on ndim."""
    if not arrays:
        msg = "At least one array is needed."
        raise ParameterError(msg)
    blocks = [x._block() for x in arrays]
    ndims = {x.ndim for x in blocks}
    if len(ndims) > 1:
        msg = f"Arrays of {sorted(ndims)} dimensions cannot be combined."
        raise ParameterError(msg)
    return blocks, blocks[0].ndim


def _check_stack(block: _Block, axis: int) -> None:
    """Refuse a block whose members are not full width slabs of one axis."""
    start, stop = block.axes["out_start"], block.axes["out_stop"]
    shape = np.asarray(block.shape, np.int64)
    others = [x for x in range(block.ndim) if x != axis]
    full = np.all(start[:, others] == 0) and np.all(stop[:, others] == shape[others])
    # An array of no samples has no member, and every cut of it is empty.
    tiles = np.array_equal(
        np.concatenate([[0], stop[:, axis]]),
        np.concatenate([start[:, axis], shape[axis : axis + 1]]),
    ) or not math.prod(block.shape)
    if not (full and tiles):
        msg = (
            f"Rechunking axis {axis} needs members which are full width "
            "slabs stacked along it."
        )
        raise NotImplementedError(msg)


def _rechunk(block: _Block, bounds: np.ndarray, axis: int) -> LazyTable:
    """Cut a stack of slabs at new bounds, as one table of many arrays."""
    start = block.axes["out_start"][:, axis]
    stop = block.axes["out_stop"][:, axis]
    if len(block):
        edges = _merge_sorted_unique(np.concatenate([start, stop[-1:]]), bounds)
        # The pieces the bounds keep are one run of edges, so they are a view.
        first = np.searchsorted(edges, bounds[0])
        last = np.searchsorted(edges, bounds[-1])
        low, high = edges[first:last], edges[first + 1 : last + 1]
    else:
        # An array of no samples gives no member to any piece.
        low = high = np.empty(0, np.int64)
    member = np.searchsorted(start, low, "right") - 1
    chunk = np.searchsorted(bounds, low, "right") - 1
    axes = {name: matrix[member] for name, matrix in block.axes.items()}
    moved = np.where(axes["src_axis"][:, axis] >= 0, low - start[member], 0)
    axes["src_start"][:, axis] += moved
    axes["out_start"][:, axis] = low - bounds[chunk]
    axes["out_stop"][:, axis] = high - bounds[chunk]
    members = block.members.take(member)
    _flatten_constants(axes, members.filled)
    counts = np.diff(np.searchsorted(chunk, np.arange(len(bounds))))
    shapes = np.tile(np.asarray(block.shape, np.int64), (len(bounds) - 1, 1))
    shapes[:, axis] = np.diff(bounds)
    ndim = block.ndim
    return LazyTable(
        member_offsets=_offsets(counts),
        axis_offsets=_offsets(counts * ndim),
        shape_offsets=_offsets(np.full(len(counts), ndim, np.int64)),
        shapes=shapes.reshape(-1),
        dtypes=_Column.constant(block.dtype, len(counts)),
        concat_axes=np.full(len(counts), axis, np.int64),
        members=members,
        axes={name: matrix.reshape(-1) for name, matrix in axes.items()},
    )


def _validate(block: _Block) -> None:
    """Check one block's rules, raising for the first which is broken."""
    shape = np.asarray(block.shape, np.int64)
    if np.any(shape < 0):
        msg = f"A lazy array cannot have shape {block.shape}."
        raise ParameterError(msg)
    start, stop = block.axes["out_start"], block.axes["out_stop"]
    if np.any(start < 0) or np.any(stop > shape) or np.any(stop <= start):
        msg = f"Some boxes are empty or fall outside an array of {block.shape}."
        raise ParameterError(msg)
    src_axis = block.axes["src_axis"]
    stored = src_axis >= 0
    outside = (src_axis >= block.ndim) | (src_axis < NEW_AXIS)
    if outside.any():
        bad = int(src_axis[outside][0])
        msg = (
            f"A member reads stored axis {bad}, which an array of "
            f"{block.ndim} dimensions does not have."
        )
        raise ParameterError(msg)
    if _repeats(src_axis):
        msg = "A member reads one stored axis onto two output axes."
        raise ParameterError(msg)
    # An array whose members read every axis is a permutation already.
    if not stored.all() and np.any(
        src_axis.max(axis=1, initial=NEW_AXIS) != stored.sum(axis=1) - 1
    ):
        msg = "A member's stored axes are not the first axes of its source."
        raise ParameterError(msg)
    window = block.axes["src_start"]
    extent = block.axes["src_extent"]
    inside = (window >= 0) & (window + (stop - start) <= extent)
    if not np.all(np.where(stored, inside, (window == 0) & (extent == 0))):
        msg = "Some windows fall outside the source they read."
        raise ParameterError(msg)
    if not _is_canonical(start):
        msg = "The members are not in canonical placement order."
        raise ParameterError(msg)
    axis = block.concat_axis
    if axis != NEW_AXIS and not 0 <= axis < block.ndim:
        msg = (
            f"The stacking axis {axis} is outside an array of {block.ndim} dimensions."
        )
        raise ParameterError(msg)
    if axis >= 0 and len(block) > 1 and np.any(start[1:, axis] < stop[:-1, axis]):
        msg = f"The members are not stacked along axis {axis}, which is claimed."
        raise ParameterError(msg)
    _check_cover(block, shape)


def _repeats(src_axis: np.ndarray) -> bool:
    """Whether any member names one stored axis twice."""
    ordered = np.sort(src_axis, axis=1)
    repeated = (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)
    return bool(repeated.any())


def _check_cover(block: _Block, shape: np.ndarray) -> None:
    """Check the boxes cover the whole array exactly once."""
    size = math.prod(block.shape)
    start, stop = block.axes["out_start"], block.axes["out_stop"]
    if size == 0:
        # Every box was checked above and none can be empty, so there are none.
        assert not len(block)
        return
    covered = int(np.prod(stop - start, axis=1).sum())
    if covered != size:
        why = "overlap" if covered > size else "leave a hole, which must be a member"
        msg = f"The members cover {covered} of {size} samples, so they {why}."
        raise ParameterError(msg)
    if not _tiles(start, stop, tuple(block.shape)):
        msg = "Some members overlap, so the boxes do not tile the array."
        raise ParameterError(msg)


def _tiles(start: np.ndarray, stop: np.ndarray, sizes: tuple[int, ...]) -> bool:
    """Whether boxes tile a region exactly; the ndim is cut down as it goes."""
    if not len(start):
        return not any(sizes)
    keep = [
        axis
        for axis, size in enumerate(sizes)
        if not (np.all(start[:, axis] == 0) and np.all(stop[:, axis] == size))
    ]
    if not keep:
        return len(start) == 1
    axis = keep[0]
    if len(keep) == 1:
        low, high = start[:, axis], stop[:, axis]
        if _chains(low, high, sizes[axis]):
            return True
        # A slab holds an arbitrary subset of the members, in no order.
        order = np.argsort(low, kind="stable")
        return _chains(low[order], high[order], sizes[axis])
    others = [x for x in range(len(sizes)) if x != axis]
    shape = tuple(sizes[x] for x in others)
    # The region's own ends are edges, so a gap is a slab with no members.
    edges = np.unique(np.concatenate([start[:, axis], stop[:, axis], [0, sizes[axis]]]))
    for low, high in zip(edges[:-1].tolist(), edges[1:].tolist()):
        rows = (start[:, axis] <= low) & (stop[:, axis] >= high)
        if not _tiles(start[rows][:, others], stop[rows][:, others], shape):
            return False
    return True


def _chains(low: np.ndarray, high: np.ndarray, size: int) -> bool:
    """Whether spans laid in this order run from zero to size with no gap."""
    return bool(low[0] == 0 and high[-1] == size and np.array_equal(low[1:], high[:-1]))


@dataclass(frozen=True, eq=False)
class _Signature:
    """
    What an array is, whatever the cuts which made it.

    `names` is the kind and id of each distinct source and `name_code` says
    which one each group reads. The rest hold one row per group, in
    canonical order: a member of it, the rest of its key, how many corners
    it kept, and those corners, lexicographic within the group.
    """

    names: list[tuple[int, str]]
    name_code: np.ndarray
    rows: np.ndarray
    fields: np.ndarray
    counts: np.ndarray
    corners: np.ndarray
    signs: np.ndarray


def _group_codes(columns: Sequence[_Column], rows: np.ndarray) -> np.ndarray | None:
    """Return one code per row for the combination of several columns."""
    out = None
    for column in columns:
        codes = column.codes[rows]
        if codes.max() == codes.min():
            continue
        codes = codes.astype(np.int64)
        if out is None:
            out = codes
            continue
        # Renumbered each time, so the packed key stays well inside int64.
        width = int(codes.max()) + 1
        out = np.unique(out * width + codes, return_inverse=True)[1].reshape(-1)
    return out


def _distinct(columns: Sequence[_Column], rows: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return one row of each distinct combination, and which one each row is."""
    keys = _group_codes(columns, rows)
    if keys is None:
        return np.zeros(1, np.int64), np.zeros(len(rows), np.int64)
    _, index, inverse = np.unique(keys, return_index=True, return_inverse=True)
    return index, inverse.reshape(-1)


def _base_ids(block: _Block, rows: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Return the id each stored member builds on, and which one it uses."""
    members = block.members
    columns = (members.key, members.source, members.origin_id)
    index, inverse = _distinct(columns, rows)
    # One row per distinct source, key and origin, so none is worked twice.
    out = []
    for row in rows[index].tolist():
        origin = members.origin_id[row]
        if not origin:
            base_uri, path, format_, version = members.source[row]
            location = {
                "path": base_uri + path,
                "format": format_,
                "version": version,
                "key": members.key[row],
            }
            origin = H("location", location)
        out.append(origin)
    return out, inverse


def _constant_ids(block: _Block, rows: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Return the id of each distinct constant, and which one each row is."""
    members = block.members
    index, inverse = _distinct((members.value, members.dtype), rows)
    out = []
    for row in rows[index].tolist():
        # The shape is the region, which the corners hold, so it is not here.
        content = {"value": members.value[row], "dtype": members.dtype[row]}
        out.append(H("constant", content))
    return out, inverse


def _group_names(block: _Block) -> tuple[list[tuple[int, str]], np.ndarray]:
    """Return the id of each distinct source, and which one each member reads."""
    members = block.members
    code = np.zeros(len(block), np.int64)
    names: list[tuple[int, str]] = []
    rows = np.flatnonzero(~members.filled)
    if len(rows):
        found, inverse = _base_ids(block, rows)
        code[rows] = inverse
        names += [(0, x) for x in found]
    rows = np.flatnonzero(members.filled)
    if len(rows):
        found, inverse = _constant_ids(block, rows)
        code[rows] = inverse + len(names)
        names += [(1, x) for x in found]
    # One id reached two ways is one source, and the order is the ids' own.
    ranks = {name: rank for rank, name in enumerate(sorted(set(names)))}
    lookup = np.array([ranks[x] for x in names], np.int64)
    return list(ranks), lookup[code]


def _placement(block: _Block) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the axis map, offset and extent each member is laid by."""
    axes = block.axes
    # A constant is its value, however it is laid; a broadcast axis reads
    # nothing, so where it sits in the output says nothing either.
    reads = (axes["src_axis"] >= 0) & ~block.members.filled[:, None]
    src_axis = np.where(reads, axes["src_axis"], NEW_AXIS)
    offset = np.where(reads, axes["out_start"] - axes["src_start"], 0)
    extent = np.where(reads, axes["src_extent"], 0)
    return src_axis, offset, extent


def _grouped(keys: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return the rows in key order, and where each run of equal keys starts."""
    count = len(keys[-1])
    # A key which is one value throughout can neither sort nor split.
    active = [x for x in keys if x.max() != x.min()]
    if not active:
        return np.arange(count), np.zeros(1, np.int64)
    order = np.lexsort(active)
    new = np.zeros(count, bool)
    new[0] = True
    for key in active:
        values = key[order]
        new[1:] |= values[1:] != values[:-1]
    return order, np.flatnonzero(new)


def _is_slab(block: _Block) -> bool:
    """Whether the members are full width slabs stacked along the first axis."""
    if block.concat_axis != 0:
        return False
    start, stop = block.axes["out_start"], block.axes["out_stop"]
    for axis in range(1, block.ndim):
        size = block.shape[axis]
        if not size or start[:, axis].any() or (stop[:, axis] != size).any():
            return False
    return bool(np.all(start[1:, 0] >= stop[:-1, 0]))


def _slab_corners(block: _Block, order: np.ndarray, owner: np.ndarray) -> tuple:
    """
    Return the corners of a stack of slabs, which are in order already.

    The boxes of one group run up the first axis and are full width on the
    rest, so their corners meet only where the first axis bounds do, and
    each surviving bound stands for the corners of every other axis. That
    gives the corners sorting them all would, in the order it would.
    """
    axes = block.axes
    count = len(order)
    points = np.empty(2 * count, np.int64)
    points[0::2] = axes["out_start"][order, 0]
    points[1::2] = axes["out_stop"][order, 0]
    signs = np.empty(2 * count, np.int64)
    signs[0::2] = 1
    signs[1::2] = -1
    group = np.repeat(owner, 2)
    new = np.ones(2 * count, bool)
    new[1:] = (points[1:] != points[:-1]) | (group[1:] != group[:-1])
    index = np.flatnonzero(new)
    total = np.add.reduceat(signs, index)
    keep = total != 0
    points, group, total = points[index][keep], group[index][keep], total[keep]
    ndim = block.ndim
    rest = _patterns(ndim - 1)
    width = len(rest)
    corners = np.empty((len(points) * width, ndim), np.int64)
    corners[:, 0] = np.repeat(points, width)
    corners[:, 1:] = np.tile(
        rest * np.asarray(block.shape[1:], np.int64), (len(points), 1)
    )
    signs = np.repeat(total, width) * np.tile(_corner_signs(rest), len(points))
    return corners, signs, np.repeat(group, width)


def _all_corners(block: _Block, order: np.ndarray, owner: np.ndarray) -> tuple:
    """Return every box's corners, summed where they meet, in sorted order."""
    ndim = block.ndim
    low = block.axes["out_start"][order]
    high = block.axes["out_stop"][order]
    count = len(order)
    corners = np.empty((count << ndim, ndim), np.int64)
    signs = np.empty(count << ndim, np.int64)
    for index, bits in enumerate(itertools.product((0, 1), repeat=ndim)):
        piece = slice(index * count, (index + 1) * count)
        for axis, bit in enumerate(bits):
            corners[piece, axis] = high[:, axis] if bit else low[:, axis]
        signs[piece] = -1 if sum(bits) % 2 else 1
    group = np.tile(owner, 1 << ndim)
    place = np.lexsort([*[corners[:, x] for x in reversed(range(ndim))], group])
    corners, signs, group = corners[place], signs[place], group[place]
    new = np.ones(len(group), bool)
    new[1:] = (corners[1:] != corners[:-1]).any(axis=1) | (group[1:] != group[:-1])
    index = np.flatnonzero(new)
    total = np.add.reduceat(signs, index)
    keep = total != 0
    return corners[index][keep], total[keep], group[index][keep]


def _patterns(count: int) -> np.ndarray:
    """Return every choice of a lower or an upper bound, in order."""
    return np.array(list(itertools.product((0, 1), repeat=count)), np.int64)


def _corner_signs(bits: np.ndarray) -> np.ndarray:
    """Return the sign of each corner: minus one for every upper bound."""
    return np.where(bits.sum(axis=1) % 2, -1, 1)


def _signature(block: _Block) -> _Signature:
    """
    Return what an array is, whatever the cuts which made it.

    Members are grouped by everything except the region they cover: which
    source they read, how its axes lie on the output, how far the output is
    from the source on each of them, and how long the whole source is. Two
    members of one group read one sample wherever their boxes meet one
    output sample, so the region alone says what a group holds.

    A region is named by the signed corners of its boxes: each box gives
    each of its `2 ** ndim` corners the sign `(-1) ** upper bounds`, the
    signs of corners which fall together are summed, and a corner which
    sums to zero drops out. That is the mixed difference of how often the
    boxes cover each sample, and summing it back gives the region, so the
    corners say what is covered and never how it was cut up.
    """
    ndim = block.ndim
    count = len(block)
    if not count:
        return _Signature(
            names=[],
            name_code=np.zeros(0, np.int64),
            rows=np.zeros(0, np.int64),
            fields=np.zeros((0, 3 * ndim + 1), np.int64),
            counts=np.zeros(0, np.int64),
            corners=np.zeros((0, ndim), np.int64),
            signs=np.zeros(0, np.int64),
        )
    names, code = _group_names(block)
    src_axis, offset, extent = _placement(block)
    keys = [extent[:, x] for x in reversed(range(ndim))]
    keys += [offset[:, x] for x in reversed(range(ndim))]
    keys += [src_axis[:, x] for x in reversed(range(ndim))]
    keys.append(code)
    order, starts = _grouped(keys)
    groups = len(starts)
    owner = np.repeat(np.arange(groups), np.diff(np.append(starts, count)))
    maker = _slab_corners if _is_slab(block) else _all_corners
    corners, signs, holder = maker(block, order, owner)
    rows = order[starts]
    name_code = code[rows]
    kinds = np.array([x[0] for x in names], np.int64)[name_code]
    fields = np.concatenate(
        [kinds[:, None], src_axis[rows], offset[rows], extent[rows]], axis=1
    )
    return _Signature(
        names=names,
        name_code=name_code,
        rows=rows,
        fields=fields,
        counts=np.bincount(holder, minlength=groups),
        corners=corners,
        signs=signs,
    )


def _whole_source_id(block: _Block, signature: _Signature) -> str | None:
    """Return the id of the one source an array is, if that is what it is."""
    if len(signature.counts) != 1 or signature.counts[0] != 1 << block.ndim:
        return None
    shape = np.asarray(block.shape, np.int64)
    bits = _patterns(block.ndim)
    # The signs say how often the boxes cover a sample, which must be once.
    if not np.array_equal(signature.corners, bits * shape) or not np.array_equal(
        signature.signs, _corner_signs(bits)
    ):
        return None
    row = int(signature.rows[0])
    if block.dtype != _dtype_of(block.members.dtype[row]):
        return None
    kind, name = signature.names[int(signature.name_code[0])]
    if kind:
        # A constant which fills the array is the array of its value.
        value = block.members.value[row]
        source = ArraySource(filled=True, value=value)
        return source.describe(block.shape, block.dtype).data_id
    src_axis, offset, extent = signature.fields[0, 1:].reshape(3, block.ndim)
    if not np.array_equal(src_axis, np.arange(block.ndim)):
        return None
    windows = tuple((-int(x), -int(x) + int(y)) for x, y in zip(offset, shape))
    whole = tuple((0, int(x)) for x in extent)
    return name if windows == whole else H("window", [name, windows])


def _fold(value: str) -> bytes:
    """Return the 16 bytes of one id, hashing one which is not 32 hex."""
    if len(value) == 32:
        try:
            return bytes.fromhex(value)
        except ValueError:
            pass
    return hashlib.blake2b(value.encode(), digest_size=DIGEST_SIZE).digest()


def _digest(block: _Block, signature: _Signature) -> str:
    r"""
    Return the digest of what an array is.

    The bytes hashed are, in order: the tag `dascore-lazy-blocks\0`; the
    layout version, the ndim, the number of groups and the number of
    corners, as four little-endian int64; then five tables, each in
    canonical group order, which the four counts above give the length of.
    The 16 bytes naming each group's source; the rest of each group's key,
    as `3 * ndim + 1` little-endian int64, which are its kind and then the
    `src_axis`, the offset and the `src_extent` of each output axis; how
    many corners each group kept; every corner, `ndim` int64 each; and the
    sign of every corner. Paths never appear in these bytes; a group's name
    may be derived from one.
    """
    out = hashlib.blake2b(digest_size=DIGEST_SIZE)
    out.update(_DIGEST_TAG)
    groups, corners = len(signature.counts), len(signature.signs)
    out.update(struct.pack("<4q", DIGEST_LAYOUT, block.ndim, groups, corners))
    raw = b"".join(_fold(x) for _, x in signature.names)
    ids = np.frombuffer(raw, np.uint8).reshape(len(signature.names), 16)
    out.update(np.ascontiguousarray(ids[signature.name_code]))
    for table in (
        signature.fields,
        signature.counts,
        signature.corners,
        signature.signs,
    ):
        out.update(np.ascontiguousarray(table, dtype="<i8"))
    return out.hexdigest()


# The array api namespace of a lazy array, which only names itself.
_NAMESPACE = SimpleNamespace(__name__=BACKEND_NAME)
