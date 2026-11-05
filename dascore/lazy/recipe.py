"""
A recipe for an array which is read a member at a time.

A [`Recipe`](`dascore.lazy.recipe.Recipe`) holds no data. It holds
members, and each member says "this window of this source goes in this box of
the output". Everything is positional: axis numbers and sample indices, never
dimension names, coordinates or units.

Many arrays share one [`RecipeTable`](`dascore.lazy.recipe.RecipeTable`),
which owns the storage; an array is a view of one of its rows. Members of an
array are stored together, in canonical placement order, so slicing, joining
and rechunking are vectorized over members and never open a file.

Examples
--------
>>> import numpy as np
>>> from dascore.core.source import ArraySource
>>> from dascore.lazy.recipe import Recipe, concat
>>>
>>> left = Recipe.from_source(ArraySource.full((4, 3), 1.0))
>>> right = Recipe.from_source(ArraySource.full((4, 3), 2.0))
>>> array = concat([left, right], axis=0)
>>> assert array.shape == (8, 3) and len(array) == 2
>>> assert np.array_equal(array[0:5].load()[:4], np.ones((4, 3)))
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, DTypeLike
from pandas.api.types import is_list_like

import dascore as dc
from dascore.core.source import ArraySource
from dascore.exceptions import ParameterError
from dascore.lazy.block import (
    AXIS_FIELDS,
    NEW_AXIS,
    _Block,
    _block_of_columns,
    _block_of_sources,
    _candidates,
    _canonical,
    _clip,
    _flatten_constants,
    _is_canonical,
    _transposed,
    _View,
)
from dascore.lazy.columns import (
    _LAYOUT_NUMBERS,
    _STRING_FIELDS,
    SOURCE_FIELDS,
    _Column,
    _dtype_of,
    _dtype_text,
    _folded,
    _Members,
    _members_of,
    _merge_members,
)
from dascore.lazy.identity import _digest, _signature, _whole_source_id
from dascore.lazy.read import (
    _coalesced,
    _member_source,
    _source_groups,
    _sources_of,
    _to_output,
)
from dascore.utils.identity import H

# The name the array api reports for a lazy array.
BACKEND_NAME = "lazy"


def _check_axis(axis: int, ndim: int) -> int:
    """Return a positive axis number, refusing one outside the array."""
    out = int(axis)
    out = out + ndim if out < 0 else out
    if not 0 <= out < ndim:
        msg = f"Axis {axis} is outside an array of {ndim} dimensions."
        raise ParameterError(msg)
    return out


def _merge_sorted_unique(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Merge two sorted arrays without sorting them again, dropping repeats."""
    out = np.empty(len(first) + len(second), first.dtype)
    out[np.arange(len(first)) + np.searchsorted(second, first, "left")] = first
    out[np.arange(len(second)) + np.searchsorted(first, second, "right")] = second
    keep = np.ones(len(out), bool)
    keep[1:] = out[1:] != out[:-1]
    return out[keep]


@dataclass(frozen=True, eq=False)
class RecipeTable:
    """
    The storage many lazy arrays share.

    Members of array `k` are the rows `member_offsets[k]:member_offsets[k+1]`
    of `source_row` and `cast`, whose sources sit in one shared table, and
    its axis rows start at `axis_offsets[k]` in each flat placement array.
    Arrays of different `ndim` therefore sit in one table: the placement
    array is flat and ragged, one row per member per output axis, which is
    the shape the database tables take.

    Every array here is read only, so views may share all of them and an
    id worked out once stays true.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.core.source import ArraySource
    >>> from dascore.lazy.recipe import Recipe, RecipeTable
    >>>
    >>> flat = Recipe.from_source(ArraySource.full((4, 3), 1.0))
    >>> cube = Recipe.from_source(ArraySource.full((2, 3, 4), 2.0))
    >>> table = RecipeTable.from_arrays([flat, cube])
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
        """Fold and freeze the storage; each flag is set once, whatever the size."""
        members, sources = self.members, self.members.sources
        # Columns which hold one value throughout keep only that value. The
        # columns may be shared with other tables, which see the same values.
        axes = {name: _folded(array) for name, array in self.axes.items()}
        object.__setattr__(self, "axes", axes)
        members.cast.codes = _folded(members.cast.codes)
        # A sources table is shared by the tables cut from it: folded once.
        if not sources.folded:
            object.__setattr__(sources, "filled", _folded(sources.filled))
            for name in SOURCE_FIELDS:
                column = getattr(sources, name)
                if name in _STRING_FIELDS:
                    column.offsets = _folded(column.offsets)
                else:
                    column.codes = _folded(column.codes)
            object.__setattr__(sources, "folded", True)
        for array in self._storage():
            # A view can be written through whatever it is a view of.
            array.setflags(write=False)
            base = array.base
            while isinstance(base, np.ndarray):
                base.setflags(write=False)
                base = base.base
        object.__setattr__(self, "axes", MappingProxyType(dict(self.axes)))

    def _storage(self) -> list[np.ndarray]:
        """Return every array the table holds."""
        members = self.members
        sources = members.sources
        strings = [getattr(sources, x) for x in _STRING_FIELDS]
        coded = [x for x in SOURCE_FIELDS if x not in _STRING_FIELDS]
        return [
            self.member_offsets,
            self.axis_offsets,
            self.shape_offsets,
            self.shapes,
            self.concat_axes,
            self.dtypes.codes,
            members.source_row,
            members.cast.codes,
            sources.filled,
            *self.axes.values(),
            *[getattr(sources, x).codes for x in coded],
            *[x.data for x in strings],
            *[x.offsets for x in strings],
        ]

    @classmethod
    def from_arrays(cls, arrays: Sequence[Recipe]) -> RecipeTable:
        """Stack arrays, which may differ in ndim, into one table."""
        return _table([x._block() for x in arrays])

    def __len__(self) -> int:
        """The number of arrays in the table."""
        return len(self.member_offsets) - 1

    def __getitem__(self, row: int) -> Recipe:
        """Return the array stored in one row."""
        index = int(row)
        row = index + len(self) if index < 0 else index
        if not 0 <= row < len(self):
            msg = f"Row {index} is outside a table of {len(self)} arrays."
            raise IndexError(msg)
        return Recipe(self, row)

    def __iter__(self):
        """Iterate over the arrays in the table."""
        return (Recipe(self, row) for row in range(len(self)))

    @property
    def n_members(self) -> int:
        """The number of members in the whole table."""
        return len(self.members)


def _table(blocks: Sequence[_Block]) -> RecipeTable:
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
    return RecipeTable(
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


class Recipe:
    """
    An array which says where each of its members is read from.

    A view of one row of a [`RecipeTable`](`dascore.lazy.recipe.RecipeTable`).
    Slicing and transposing keep a window and an axis order instead, and
    the members are clipped to them only when something needs them.
    The shape, ndim and dtype are stored rather than derived, so an array
    which selects nothing still knows what it is. Boxes may not overlap and
    must cover the whole output; a hole is an explicit constant member.
    [`validate`](`dascore.lazy.recipe.Recipe.validate`) checks every
    rule, on demand.

    A member is cast once, from the dtype it is stored at to the array's, so
    a chain of joins does not round at each step as numpy would. A caller
    reproducing a buffer which did round states where, per member, with
    `cast_via`.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.core.source import ArraySource
    >>> from dascore.lazy.recipe import Recipe
    >>>
    >>> array = Recipe.from_source(ArraySource.full((6, 4), np.nan))
    >>> assert array.shape == (6, 4) and array.ndim == 2
    >>> window = array[2:5]
    >>> assert window.shape == (3, 4)
    >>> assert np.isnan(window.load()).all()
    """

    # A view clips and permutes its base's members only when they are needed;
    # its lock makes every thread asking at once get the one table.
    __slots__ = ("_lock", "_row", "_table", "_view")

    def __init__(self, table: RecipeTable, row: int):
        self._table: RecipeTable | None = table
        self._row = int(row)
        self._view: _View | None = None
        self._lock: threading.Lock | None = None

    @classmethod
    def from_source(cls, source: ArraySource, base_uri: str = "") -> Recipe:
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
        dtype=None,
        cast_via: Sequence | None = None,
    ) -> Recipe:
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
        dtype
            The dtype the members are cast to and the array loads as; by
            default the one they all promote to together. Promotion is
            not associative, so a caller which must match another
            order's result states it.
        cast_via
            A dtype per source which its samples pass through on the way
            to the array's, or None for a member cast straight there. A
            caller reproducing a buffer which promoted a member at a time
            states where that buffer rounded each of them.
        """
        sources = list(sources)
        if not sources:
            msg = "A lazy array takes at least one source."
            raise ParameterError(msg)
        block = _block_of_sources(sources, base_uri, cast_via)
        return Recipe._placed(block, starts, axis, shape, dtype)

    @staticmethod
    def _placed(block: _Block, starts=None, axis=0, shape=None, dtype=None):
        """Return the array laying out a block's members, as `from_sources` does."""
        if dtype is not None:
            block = replace(block, dtype=np.dtype(dtype))
        ndim = block.ndim
        axis = _check_axis(axis, ndim)
        lengths = block.axes["out_stop"]
        if starts is None:
            corners = np.zeros_like(lengths)
            corners[1:, axis] = np.cumsum(lengths[:-1, axis])
            concat_axis = axis
        else:
            # A copy, so the table never holds an array the caller keeps.
            corners = np.array(starts, np.int64).reshape(len(block), ndim)
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
    def from_columns(
        cls,
        path: str | ArrayLike,
        shape: ArrayLike,
        *,
        format: str | ArrayLike,
        version: str | ArrayLike,
        source_dtype: DTypeLike | ArrayLike,
        key: str | ArrayLike = "",
        origin_id: str | ArrayLike = "",
        start: ArrayLike | None = None,
        extent: ArrayLike | None = None,
        axis: int = 0,
        base_uri: str = "",
        dtype: DTypeLike | None = None,
        cast_via: DTypeLike | ArrayLike | None = None,
    ) -> Recipe:
        """
        Return an array which reads one member per row, laid end to end.

        The array and `data_id` `from_sources` builds from the equivalent
        sources, with no object made per member. `axis`, `base_uri` and
        `dtype` are single values; every other argument is one per member,
        or one all share. The member count is the length of any per member
        argument, else the rows of `shape`.

        Parameters
        ----------
        path
            Where each member is stored.
        shape
            Each member's window lengths as an `(n, ndim)` array. A flat
            `shape` is one shape every member shares, so the lengths of 1-D
            members are given as `(n, 1)`.
        format, version
            The FiberIO which reads the members, and its version.
        source_dtype
            The dtype the members are stored at.
        key, origin_id
            Which array of each resource, and its id; see `ArraySource`.
        start, extent
            Where each window starts, and the shape of the whole stored
            array; zeros and `shape` by default.
        axis, base_uri, dtype, cast_via
            As `from_sources` takes them.

        Examples
        --------
        >>> import numpy as np
        >>> from dascore.lazy.recipe import Recipe
        >>>
        >>> paths = np.array(["/data/f0.h5", "/data/f1.h5", "/data/f2.h5"])
        >>> array = Recipe.from_columns(
        ...     paths, (100, 10), format="DASDAE", version="1", source_dtype="f4"
        ... )
        >>> assert array.shape == (300, 10) and len(array) == 3
        """
        lengths = np.atleast_1d(np.asarray(shape, np.int64))
        ndim = lengths.shape[-1]
        if ndim > 64:  # the most axes numpy gives an array
            msg = f"A shape of {ndim} axes; give per member lengths as (n, ndim)."
            raise ParameterError(msg)
        columns: dict[str, Any] = {"path": path, "format": format, "version": version}
        columns |= {"key": key, "origin_id": origin_id, "source_dtype": source_dtype}
        columns["cast_via"] = cast_via
        listed = [len(x) for x in columns.values() if is_list_like(x)]
        windows = (np.asarray(x) for x in (lengths, start, extent) if x is not None)
        listed += [len(x) for x in windows if x.ndim == 2]
        rows = listed[0] if listed else 1
        if not rows:
            msg = "A lazy array takes at least one member."
            raise ParameterError(msg)
        if any(np.any(pd.isna(x) | (np.asarray(x) == "")) for x in (path, format)):
            msg = "Every member needs a path and a format to be read from."
            raise ParameterError(msg)
        block = _block_of_columns(
            columns,
            _rows_of(lengths, rows, ndim, "shape"),
            _rows_of(0 if start is None else start, rows, ndim, "start"),
            _rows_of(lengths if extent is None else extent, rows, ndim, "extent"),
            base_uri,
        )
        return cls._placed(block, axis=axis, dtype=dtype)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, shape, dtype) -> Recipe:
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
    def table(self) -> RecipeTable:
        """The table which holds this array."""
        if self._table is None:
            assert self._lock is not None, "only a view has no table"
            with self._lock:
                if self._table is None:
                    self._table = _resolved(self._view)
                    # The members are resolved, so the base they came from can go.
                    self._view = None
        return self._table

    @property
    def row(self) -> int:
        """Which row of the table this array is."""
        return self._row

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the array the members cover."""
        if (view := self._view) is not None:
            return tuple(view.stops[x] - view.starts[x] for x in view.order)
        table = self.table
        start, stop = table.shape_offsets[self._row : self._row + 2]
        return tuple(table.shapes[start:stop].tolist())

    @property
    def ndim(self) -> int:
        """The number of dimensions of the array."""
        if (view := self._view) is not None:
            return len(view.order)
        offsets = self.table.shape_offsets
        return int(offsets[self._row + 1] - offsets[self._row])

    @property
    def dtype(self) -> np.dtype:
        """The dtype of the loaded array."""
        if (view := self._view) is not None:
            return view.block.dtype
        return np.dtype(self.table.dtypes[self._row])

    @property
    def size(self) -> int:
        """The number of elements in the array."""
        return math.prod(self.shape)

    def __len__(self) -> int:
        """The number of members."""
        offsets, row = self.table.member_offsets, self.row
        return int(offsets[row + 1] - offsets[row])

    def __repr__(self) -> str:
        name = type(self).__name__
        return f"{name}(shape={self.shape}, dtype={self.dtype}, members={len(self)})"

    def _block(self) -> _Block:
        """Return this array's storage, as views of the table's arrays."""
        table, row = self.table, self._row
        first, last = table.member_offsets[row : row + 2].tolist()
        low, high = table.shape_offsets[row : row + 2].tolist()
        start, count, ndim = int(table.axis_offsets[row]), last - first, high - low
        stop = start + count * ndim
        axes = {
            name: table.axes[name][start:stop].reshape(count, ndim)
            for name in AXIS_FIELDS
        }
        return _Block(
            shape=tuple(table.shapes[low:high].tolist()),
            dtype=np.dtype(table.dtypes[row]),
            concat_axis=int(table.concat_axes[row]),
            members=table.members.take(slice(first, last)),
            axes=axes,
        )

    def _viewed(self) -> _View:
        """Return the view this array is; an array which is none views itself."""
        if (view := self._view) is not None:
            return view
        block, ndim = self._block(), self.ndim
        rows = slice(0, len(block))
        return _View(block, (0,) * ndim, self.shape, tuple(range(ndim)), rows)

    @staticmethod
    def _of_view(view: _View) -> Recipe:
        """Return the array a view names, whose members are clipped on demand."""
        out = Recipe.__new__(Recipe)
        out._table, out._row, out._view = None, 0, view
        out._lock = threading.Lock()
        return out

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
        return tuple(_sources_of(self._block()))

    def __getitem__(self, index) -> Recipe:
        """
        Return the array a selection of samples describes; nothing is read.

        Only slices with a step of one are taken, and trailing axes may be
        left out. Each box is clipped to the request, its window moved by
        what was clipped, and members the request misses are dropped; the
        clipping waits until something needs the members.
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
        view = self._viewed()
        low, high = list(view.starts), list(view.stops)
        for axis, (start, stop) in enumerate(spans):
            base = view.order[axis]
            low[base], high[base] = view.starts[base] + start, view.starts[base] + stop
        rows = _candidates(view.block, low, high, view.rows)
        return self._of_view(
            view._replace(starts=tuple(low), stops=tuple(high), rows=rows)
        )

    def transpose(self, order: Sequence[int] | None = None) -> Recipe:
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
        view = self._viewed()
        order = tuple(view.order[x] for x in order)
        return self._of_view(view._replace(order=order))

    def rechunk(self, bounds, axis: int = 0) -> RecipeTable:
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

    def validate(self) -> Recipe:
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
        [`validate`](`dascore.lazy.recipe.Recipe.validate`).
        Abutting windows of one source are read as one. Members are read a
        resource at a time: each resource is opened once for all of its
        members, and closed before the next is opened.
        """
        block = self._block()
        _validate(block)
        block = _coalesced(block)
        out = np.empty(self.shape, self.dtype)
        # The placement is taken apart once rather than a row at a time,
        # which numpy charges for however few samples a member holds.
        start = block.axes["out_start"].tolist()
        stop = block.axes["out_stop"].tolist()
        src_axis = block.axes["src_axis"].tolist()
        casts = block.members.cast
        sources = _sources_of(block)

        def place(row: int, data: np.ndarray):
            data = _to_output(data, src_axis[row])
            if via := casts[row]:
                data = data.astype(_dtype_of(via), copy=False)
            out[tuple(map(slice, start[row], stop[row]))] = data

        for row in np.flatnonzero(block.members.filled).tolist():
            place(row, sources[row].load())
        for rows in _source_groups(block):
            with dc.io.core._open_array_reader(sources[rows[0]]) as load:
                for row in rows:
                    place(row, load(sources[row]))
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
        either. A member is cast once, so the array's own dtype reaches it,
        as does an intermediate dtype a member is stated to pass through.

        Naming costs what the cuts did: an axis every member spans alike is
        an interval rather than a pair of corners, so an array of any
        number of dimensions is named as cheaply as one which is cut the
        same way in two. One cut on more than `MAX_CUT_AXES` distinct axes
        is refused, since every axis cut doubles the corners.

        Examples
        --------
        >>> from dascore.core.source import ArraySource
        >>> from dascore.lazy.recipe import Recipe
        >>>
        >>> array = Recipe.from_source(ArraySource.full((8, 2), 1.0))
        >>> assert array[0:4].data_id == array[0:8][0:4].data_id
        """
        cache, row = self.table._ids, self.row
        out = cache.get(row)
        if out is None:
            cache[row] = out = self._data_id()
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
        sources, rows = members.sources, members.source_row
        # Only the strings of the rows this array reads are decoded.
        used, inverse = np.unique(rows, return_inverse=True)
        for name in SOURCE_FIELDS[:-1]:
            column = getattr(sources, name)
            if name in _STRING_FIELDS:
                out[name] = _spread(inverse, column.at(used), ndim)
            else:
                out[name] = _spread(column.codes[rows], column.values, ndim)
                if name in _LAYOUT_NUMBERS:  # exact through parquet, nulls too
                    out[name] = pd.array(out[name], dtype="Int64")
        out["cast"] = _spread(members.cast.codes, members.cast.values, ndim)
        out["filled"] = np.repeat(members.filled, ndim)
        # The values are python scalars of several types, which stay objects.
        value = sources.value
        out["value"] = _spread(value.codes[rows], value.values, ndim, object)
        return pd.DataFrame(out)


def _spread(codes: np.ndarray, values: Sequence, ndim: int, dtype=None) -> np.ndarray:
    """Return one row per member per output axis of a column's values."""
    # An array of no members still has text columns, whatever it was cut from.
    dtype = dtype or (None if values else str)
    return np.asarray(values, dtype)[np.repeat(codes, ndim)]


def _array(block: _Block) -> Recipe:
    """Return the array one block describes, in a table of its own."""
    return Recipe(_table([block]), 0)


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


def _resolved(view: _View | None) -> RecipeTable:
    """Return the table of the members a view shows, in no storage of its base's."""
    assert view is not None, "an array is a table row or a view"
    block, starts, stops, order = view.block, view.starts, view.stops, view.order
    identity = order == tuple(range(len(order)))
    if any(starts) or stops != block.shape:
        block = _clip(block.take(view.rows), np.array(starts), np.array(stops))
    elif identity:
        # The whole block, unmoved: copied, so the base's storage can go, and
        # put in placement order as a transpose would.
        block = _canonical(block.take(np.arange(len(block))))
    # A window of the whole block clips nothing, so a bare transpose keeps
    # each member as it is stored.
    return _table([block if identity else _transposed(block, order)])


def _concat_axis_of(axes: dict[str, np.ndarray]) -> int:
    """Return an axis the members are laid along, which _candidates needs."""
    start, stop = axes["out_start"], axes["out_stop"]
    if len(start) < 2:
        return 0
    ordered = np.flatnonzero(np.all(start[1:] >= stop[:-1], axis=0))
    return int(ordered[0]) if len(ordered) else NEW_AXIS


def _rows_of(values, rows: int, ndim: int, name: str) -> np.ndarray:
    """Return a new `(rows, ndim)` integer array from a scalar or rows."""
    array = np.asarray(values, np.int64)
    if array.shape not in ((), (ndim,), (rows, ndim)):
        msg = f"The {name} must be a scalar, or of shape ({ndim},) or ({rows}, {ndim})."
        raise ParameterError(msg)
    return np.array(np.broadcast_to(array, (rows, ndim)))


def concat(arrays: Sequence[Recipe], axis: int = 0) -> Recipe:
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
    >>> from dascore.lazy.recipe import Recipe, concat
    >>>
    >>> parts = [Recipe.from_source(ArraySource.full((2, 3), x)) for x in (1, 2)]
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
    # Repeat counts are intp; a 32-bit platform will not take int64 for them.
    counts = np.array([len(x) for x in blocks], np.intp)
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


def stack(arrays: Sequence[Recipe], axis: int = 0) -> Recipe:
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
    >>> from dascore.lazy.recipe import Recipe, stack
    >>>
    >>> parts = [Recipe.from_source(ArraySource.full((2, 3), x)) for x in (1, 2)]
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


def _blocks_of(arrays: Sequence[Recipe]) -> tuple[list[_Block], int]:
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


def _rechunk(block: _Block, bounds: np.ndarray, axis: int) -> RecipeTable:
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
    return RecipeTable(
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


# The array api namespace of a lazy array, which only names itself.
_NAMESPACE = SimpleNamespace(__name__=BACKEND_NAME)
