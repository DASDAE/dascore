"""
A lazy xarray index over a DASCore coordinate.

xarray materializes every dimension coordinate into a numpy array and a
pandas index, which for a long merged time coordinate costs 8 bytes a
sample before any data loads — a year of millisecond sampling is a
quarter terabyte of labels. `CoordIndex` holds the DASCore coordinate
itself instead: an evenly sampled range (an exact integer grid for
nanosecond time and integers, a scalar step for floats) or a segmented
coordinate, whose segments may be ranges or monotonic arrays. Labels are
the coordinate's own, computed for the positions asked about; selection
follows the rules `Patch.sel` follows, which are pandas'; and a slice or
an abutting concatenation of an integer grid is again a coordinate, so
laziness survives selection chains.

Everything here is imported only behind the optional xarray dependency.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from xarray import DataArray, Variable
from xarray.core.indexes import IndexSelResult
from xarray.indexes import CoordinateTransform, CoordinateTransformIndex, PandasIndex

from dascore.core.coords import (
    BaseCoord,
    CoordArray,
    CoordMonotonicArray,
    CoordRange,
    CoordSegmented,
    concat_coords,
)
from dascore.exceptions import CoordError
from dascore.utils.indexing import label_indexer, positional_indexer
from dascore.utils.misc import is_strictly_monotonic


def is_servable(coord) -> bool:
    """Whether `CoordIndex` can serve a coordinate's labels."""
    if isinstance(coord, CoordSegmented):
        return True
    # a zero step repeats one label, which no index can look up
    return isinstance(coord, CoordRange) and bool(coord.step)


def _relabels_exactly(coord) -> bool:
    """Whether slices of a coordinate keep exactly the labels they select."""
    # a float range recomputes a slice's labels from its new start, which
    # can move them in the last bits, and then they no longer align
    if isinstance(coord, CoordSegmented):
        return all(_relabels_exactly(x) for x in coord.segments)
    return not isinstance(coord, CoordRange) or coord._exact


def _array_coord(labels, units) -> BaseCoord:
    """Labels held as they are, never re-inferred as a range."""
    monotonic = len(labels) > 1 and is_strictly_monotonic(labels)
    cls = CoordMonotonicArray if monotonic else CoordArray
    return cls(values=labels, units=units)


def _as_pandas(index) -> PandasIndex:
    """The materialized form of any one-dimensional index."""
    if isinstance(index, PandasIndex):
        return index
    return PandasIndex(index.to_pandas_index(), index.dim)


def _same_labels(first: BaseCoord, second: BaseCoord) -> bool:
    """Whether two coordinates label their samples alike, units aside."""
    if first is second:
        return True
    if len(first) != len(second) or first.dtype != second.dtype:
        return False
    if first.units != second.units:
        # xarray states units as an attribute beside the labels, so an
        # index compares labels only, as a materialized index does
        first, second = first.set_units(None), second.set_units(None)
    return first.fingerprint() == second.fingerprint()


def _chained(coords: list[BaseCoord]) -> BaseCoord | None:
    """The coordinates end to end, or None if they do not chain in this order."""
    try:
        out = concat_coords(*coords)
    except CoordError:
        return None
    # concat_coords orders its inputs; xarray's order is the data's
    starts = [x.min() if out.sorted else x.max() for x in coords]
    ordered = all((a < b) if out.sorted else (a > b) for a, b in pairwise(starts))
    return out if ordered and is_servable(out) and _relabels_exactly(out) else None


class CoordTransform(CoordinateTransform):
    """
    Positions to the labels of a DASCore coordinate, and back.

    ``forward`` evaluates only the positions asked for; ``reverse``
    returns the nearest sample as a float position, as xarray's transform
    contract asks. `CoordIndex` selects through `label_indexer` instead.
    """

    def __init__(self, name, coord: BaseCoord, dim: str | None = None):
        dim = name if dim is None else dim
        super().__init__((name,), {dim: len(coord)}, dtype=np.dtype(coord.dtype))
        self.coord = coord

    @property
    def dim(self) -> str:
        """The dimension the coordinate labels."""
        return self.dims[0]

    def forward(self, dim_positions) -> dict:
        """Return the labels for the given positions."""
        pos = np.asarray(dim_positions[self.dim])
        if pos.dtype.kind == "f":
            pos = np.rint(pos)
        labels = self.coord._get_index_values(pos.astype(np.int64))
        return {self.coord_names[0]: labels}

    def reverse(self, coord_labels) -> dict:
        """Return the (float) positions of the samples nearest the labels."""
        labels = np.atleast_1d(np.asarray(coord_labels[self.coord_names[0]]))
        positions = label_indexer(self.coord, labels, method="nearest")
        return {self.dim: np.asarray(positions, dtype=np.float64)}

    def equals(self, other, exclude=None, **kwargs) -> bool:
        """Two transforms are equal when they label every sample alike, units aside."""
        return (
            isinstance(other, CoordTransform)
            and self.dims == other.dims
            and _same_labels(self.coord, other.coord)
        )


class CoordIndex(CoordinateTransformIndex):
    """
    An xarray index serving a DASCore coordinate's labels.

    Selection answers as `Patch.sel` answers, which is as a materialized
    pandas index answers: partial datetime strings name their periods,
    slices include both endpoints, and ``method`` and ``tolerance`` work
    as pandas has them. A slice ``isel`` or a concatenation whose parts
    chain returns a new lazy index; fancy indexing, an empty slice, a
    slice whose labels a coordinate would recompute (a float range, or a
    stride over segments), or a concatenation which reorders or overlaps
    holds just the labels concerned, still as a `CoordIndex`, so arrays
    derived from one another align. Aligning with a `CoordIndex` whose
    labels differ materializes both, costing what aligning materialized
    indexes costs.

    xarray matches an index only with indexes of its own type: combining
    a lazy array with one whose index is materialized, or reindexing it
    to new labels, raises xarray's AlignmentError even where the labels
    agree. Convert with ``lazy_coords=False`` for those.
    """

    transform: CoordTransform

    @classmethod
    def from_coord(cls, name: str, coord: BaseCoord) -> CoordIndex:
        """Serve a range or segmented DASCore coordinate as ``name``."""
        assert is_servable(coord), f"{type(coord).__name__} is not servable"
        return cls(CoordTransform(name, coord))

    @property
    def coordinate(self) -> BaseCoord:
        """The DASCore coordinate this index serves."""
        return self.transform.coord

    @property
    def dim(self) -> str:
        """The dimension the index labels."""
        return self.transform.dim

    @property
    def index(self) -> pd.Index:
        """The materialized pandas form, as a `PandasIndex` offers it."""
        # xarray's own indexes read these when a materialized part comes
        # first in a concatenation
        return self.to_pandas_index()

    @property
    def coord_dtype(self) -> np.dtype:
        """The label dtype, as a `PandasIndex` offers it."""
        return self.transform.dtype

    def _with(self, coord: BaseCoord) -> CoordIndex:
        """An index serving another coordinate under these names."""
        name = self.transform.coord_names[0]
        return type(self)(CoordTransform(name, coord, self.dim))

    def _labels(self, positions) -> np.ndarray:
        """The labels at these positions."""
        name = self.transform.coord_names[0]
        return self.transform.forward({self.dim: np.asarray(positions)})[name]

    def _picked(self, positions) -> CoordIndex:
        """An index holding the labels at these positions."""
        # still a CoordIndex, which xarray aligns with this one
        return self._with(_array_coord(self._labels(positions), self.coordinate.units))

    def to_pandas_index(self) -> pd.Index:
        """The materialized pandas form, computed on demand."""
        labels = self._labels(np.arange(len(self.coordinate)))
        return pd.Index(labels, name=self.transform.coord_names[0])

    def isel(self, indexers) -> Any:
        """A slice keeps a lazy index where its labels stay exact; else materialize."""
        idx = indexers.get(self.dim)
        coord = self.coordinate
        if isinstance(idx, slice):
            start, stop, stride = idx.indices(len(coord))
            positions = range(start, stop, stride)
            # a strided segmented coordinate is an array of its labels
            lazy = isinstance(coord, CoordRange) or stride == 1
            if len(positions) and lazy and _relabels_exactly(coord):
                return self._with(coord[idx])
            return self._picked(np.asarray(positions, dtype=np.int64))
        if getattr(idx, "dims", (self.dim,)) != (self.dim,):
            # vectorized onto another dimension: the labels no longer
            # index this one, so xarray drops the index
            return None
        raw = idx.values if isinstance(idx, Variable | DataArray) else idx
        if np.ndim(raw) != 1:
            return None
        return self._picked(positional_indexer(raw, len(coord)))

    def sel(self, labels, method=None, tolerance=None) -> IndexSelResult:
        """Resolve label selection as `Patch.sel` does."""
        label = labels[self.transform.coord_names[0]]
        coord = self.coordinate
        if isinstance(label, Variable | DataArray) and label.ndim == 0:
            label = label.values[()]  # a scalar, strings naming their periods
        if not isinstance(label, Variable | DataArray):
            return IndexSelResult(
                {self.dim: label_indexer(coord, label, method, tolerance)}
            )
        # vectorized selection: look up the values, keep the label's dims;
        # a mask selects as it stands
        values = np.asarray(label.values)
        pos = values
        if values.dtype != bool:
            found = label_indexer(coord, values.ravel(), method, tolerance)
            pos = np.asarray(found).reshape(values.shape)
        if isinstance(label, DataArray):
            return IndexSelResult({self.dim: label.copy(data=pos)})
        return IndexSelResult({self.dim: Variable(label.dims, pos)})

    @classmethod
    def concat(cls, indexes, dim, positions=None) -> Any:
        """
        Concatenate indexes, staying lazy when their coordinates chain.

        Parts in ascending (or descending) order with no overlap chain
        into one range when they continue each other's grid and into a
        segmented coordinate otherwise; anything else — a reordering, an
        overlap, a materialized part, float ranges whose fused labels
        would be recomputed — comes back as a pandas index over the
        concatenated labels.
        """
        if not all(isinstance(x, CoordIndex) for x in indexes):
            return PandasIndex.concat([_as_pandas(x) for x in indexes], dim, positions)
        first = indexes[0]
        coords = [x.coordinate for x in indexes]
        if positions is None and (out := _chained(coords)) is not None:
            return first._with(out)
        labels = np.concatenate(
            [x._labels(np.arange(len(x.coordinate))) for x in indexes]
        )
        if positions is not None:
            labels = labels[np.argsort(np.concatenate([list(x) for x in positions]))]
        return first._with(_array_coord(labels, first.coordinate.units))

    def join(self, other, how="inner") -> PandasIndex:
        """Join as materialized indexes join."""
        # xarray folds several indexes through this, so self may already
        # be the materialized result of an earlier join
        return _as_pandas(self).join(_as_pandas(other), how=how)

    def reindex_like(self, other, method=None, tolerance=None) -> dict:
        """Reindex as a materialized index reindexes."""
        return _as_pandas(self).reindex_like(_as_pandas(other), method, tolerance)

    def _repr_inline_(self, max_width) -> str:
        coord = self.coordinate
        return f"{type(self).__name__} ({type(coord).__name__}, size={len(coord)})"

    def __repr__(self) -> str:
        return self._repr_inline_(None)
