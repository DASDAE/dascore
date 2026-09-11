"""
A lazy xarray index over a DASCore coordinate.

xarray materializes every dimension coordinate into a numpy array and a
pandas index, which for a long merged time coordinate costs 8 bytes a
sample before any data loads — a year of millisecond sampling is a
quarter terabyte of labels. `CoordIndex` holds the DASCore coordinate
itself instead: an evenly sampled range (an exact integer grid for
nanosecond time and integers, a scalar step for floats) or a segmented
coordinate of such runs. Labels are the coordinate's own, computed for
the positions asked about; selection follows the rules `Patch.sel`
follows, which are pandas'; and a slice or an abutting concatenation is
again a coordinate, so laziness survives selection chains.

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

from dascore.core.coords import BaseCoord, CoordRange, CoordSegmented, concat_coords
from dascore.exceptions import CoordError
from dascore.utils.indexing import label_indexer


def is_servable(coord) -> bool:
    """Whether `CoordIndex` can serve a coordinate's labels."""
    if isinstance(coord, CoordSegmented):
        return True
    # a zero step repeats one label, which no index can look up
    return isinstance(coord, CoordRange) and bool(coord.step)


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
    return out if ordered and is_servable(out) else None


class CoordTransform(CoordinateTransform):
    """
    Positions to the labels of a DASCore coordinate, and back.

    ``forward`` evaluates only the positions asked for; ``reverse``
    returns the nearest sample as a float position, as xarray's transform
    contract asks. `CoordIndex` selects through `label_indexer` instead.
    """

    def __init__(self, name: str, coord: BaseCoord):
        super().__init__((name,), {name: len(coord)}, dtype=np.dtype(coord.dtype))
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
        """Two transforms are equal when they label every sample alike."""
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
    slices keep their endpoints, and ``method`` and ``tolerance`` work as
    pandas has them. A slice ``isel`` or a concatenation whose parts
    chain returns a new lazy index; fancy indexing, or a concatenation
    which reorders or overlaps, falls back to a materialized pandas index
    over just the labels concerned. Aligning with a different index
    materializes both, as aligning materialized indexes costs.

    An index matches only indexes of its own type: combining a lazy array
    with one whose index is materialized raises xarray's AlignmentError
    even where the labels agree. Convert with ``lazy_coords=False`` to
    combine with such arrays.

    A segmented coordinate's labels are evaluated in full for each label
    selection on it, and not kept. A slice of a float range is labeled as
    DASCore labels it, which can differ from the parent's labels in the
    last bits.
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

    def _positions(self, positions) -> PandasIndex:
        """A materialized index over the labels at these positions."""
        labels = self.transform.forward({self.dim: np.asarray(positions)})
        return PandasIndex(pd.Index(labels[self.transform.coord_names[0]]), self.dim)

    def to_pandas_index(self) -> pd.Index:
        """The materialized pandas form, computed on demand."""
        return self._positions(np.arange(len(self.coordinate))).index

    def isel(self, indexers) -> Any:
        """A sliced view keeps a lazy index; fancy indexing materializes."""
        idx = indexers.get(self.dim)
        coord = self.coordinate
        if isinstance(idx, slice):
            start, stop, stride = idx.indices(len(coord))
            positions = range(start, stop, stride)
            # a strided segmented coordinate is an array of its labels
            if len(positions) and (isinstance(coord, CoordRange) or stride == 1):
                name = self.transform.coord_names[0]
                return type(self)(CoordTransform(name, coord[idx]))
            return self._positions(np.asarray(positions, dtype=np.int64))
        if getattr(idx, "dims", (self.dim,)) != (self.dim,):
            # vectorized onto another dimension: the labels no longer
            # index this one, so xarray drops the index
            return None
        positions = np.asarray(getattr(idx, "values", idx))
        if isinstance(idx, list | tuple) and not len(idx):
            positions = positions.astype(np.int64)  # an empty list picks nothing
        if positions.ndim != 1:
            return None
        if positions.dtype == bool:
            positions = np.flatnonzero(positions)
        if positions.dtype.kind not in "iu":
            # as numpy refuses them for the data; forward would round them
            msg = "arrays used as indices must be of integer (or boolean) type"
            raise IndexError(msg)
        return self._positions(positions)

    def sel(self, labels, method=None, tolerance=None) -> IndexSelResult:
        """Resolve label selection as `Patch.sel` does."""
        label = labels[self.dim]
        coord = self.coordinate
        if isinstance(coord, CoordSegmented):
            # evaluated for this lookup only; the coordinate keeps no values
            coord = coord.new(values=coord._get_index_values(np.arange(len(coord))))
        if not isinstance(label, Variable | DataArray):
            return IndexSelResult(
                {self.dim: label_indexer(coord, label, method, tolerance)}
            )
        # vectorized selection: look up the values, keep the label's dims
        values = np.asarray(label.values)
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
        overlap, a materialized part — comes back as a pandas index over
        the concatenated labels.
        """
        if positions is None and all(isinstance(x, CoordIndex) for x in indexes):
            coords = [x.coordinate for x in indexes]
            if (out := _chained(coords)) is not None:
                name = indexes[0].transform.coord_names[0]
                return cls(CoordTransform(name, out))
        return PandasIndex.concat([_as_pandas(x) for x in indexes], dim, positions)

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
