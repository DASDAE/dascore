"""
An xarray index which computes datetime64 and timedelta64 labels from a
start, a step and a size, rather than storing them.

xarray materializes every dimension coordinate into a numpy array and a
pandas index, which for a long merged time coordinate costs 8 bytes a
sample before any data loads — a year of millisecond sampling is a
quarter terabyte of labels. pandas has no computed-on-demand datetime
index (``freq`` is metadata on a fully allocated array), so xarray grew
the ``CoordinateTransform`` machinery instead; its own ``RangeIndex``
serves lazy float labels on top of it. This module supplies the
temporal counterpart: labels are ``start + i * step`` computed from the
position on demand, and selection inverts that arithmetic exactly in
python integers — int64 subtraction of a far-away label wraps around,
and float inversion loses sample precision once offsets pass 2**53 ns,
about 104 days.

Everything here is imported only behind the optional xarray dependency.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from xarray import DataArray, Variable
from xarray.core.indexes import IndexSelResult
from xarray.indexes import CoordinateTransform, CoordinateTransformIndex

_NAT_I8 = np.iinfo("int64").min

# pandas' Resolution ranks, finest first; a partial datetime string
# names a span only when its resolution is coarser than the index's
_PERIOD_RESO = {
    "ns": 0,
    "us": 1,
    "ms": 2,
    "s": 3,
    "min": 4,
    "h": 5,
    "D": 6,
    "M": 7,
    "Q": 8,
    "Y": 9,
}
_UNIT_NS = (1_000, 10**6, 10**9, 60 * 10**9, 3_600 * 10**9, 86_400 * 10**9)


def _ns_resolution(value: int) -> int:
    """The finest unit with a nonzero component in ``value`` (day at most)."""
    for rank, divisor in enumerate(_UNIT_NS):
        if value % divisor:
            return rank
    return _PERIOD_RESO["D"]


def _label_ints(labels: np.ndarray, dtype) -> np.ndarray:
    """Labels as exact python-int nanoseconds, refusing NaT."""
    arr = np.atleast_1d(np.asarray(labels))
    if arr.dtype.kind in "biufc":
        # pandas never reads a number as a stamp; a materialized twin
        # raises here, so the lazy index must not select the epoch
        msg = f"Numeric label(s) {labels!r} do not name {np.dtype(dtype)} samples."
        raise KeyError(msg)
    if arr.dtype == np.dtype(dtype):
        ints = arr.view("int64")
    else:
        # pandas raises for values the ns range cannot represent, where a
        # raw astype would silently wrap a year-2500 label into 1915; the
        # explicit ns astype matters too, since pandas keeps a coarser
        # input unit and its int64 form would then be in that unit
        kind = np.dtype(dtype).kind
        converter = pd.to_timedelta if kind == "m" else pd.to_datetime
        converted = converter(arr.ravel()).astype(np.dtype(dtype))
        ints = np.asarray(converted.astype("int64")).reshape(arr.shape)
    if np.any(ints == _NAT_I8):
        msg = "Cannot select with NaT labels on an evenly sampled coordinate."
        raise ValueError(msg)
    # python ints: label arithmetic must not wrap for far-away labels
    return ints.astype(object)


class TemporalRangeTransform(CoordinateTransform):
    """
    Positions to evenly sampled datetime64/timedelta64 labels and back.

    ``forward`` is exact int64 (positions are valid, so no overflow);
    ``reverse`` computes through python integers, so a label centuries
    away yields its true out-of-range position instead of wrapping — but
    returns float positions, as xarray's transform contract asks, so its
    callers accept float rounding; the index's own selection paths use
    `_exact_positions` instead.
    """

    def __init__(self, name: str, size: int, start_ns: int, step_ns: int, dtype):
        super().__init__((name,), {name: int(size)}, dtype=np.dtype(dtype))
        self.name = name
        self.start_ns = int(start_ns)
        self.step_ns = int(step_ns)

    def forward(self, dim_positions) -> dict:
        """Return the labels for the given positions."""
        pos = np.asarray(dim_positions[self.name])
        if pos.dtype.kind == "f":
            # rounding through float would corrupt exact positions past
            # 2**53, so only genuinely fractional input goes through it
            pos = np.rint(pos)
        ints = self.start_ns + pos.astype("int64") * self.step_ns
        return {self.name: ints.astype("int64").view(self.dtype)}

    def reverse(self, coord_labels) -> dict:
        """Return the (float) positions for the given labels."""
        ints = _label_ints(coord_labels[self.name], self.dtype)
        positions = (ints - self.start_ns) / self.step_ns
        return {self.name: positions.astype("float64")}

    def equals(self, other, **kwargs) -> bool:
        """Two transforms are equal when they label every sample alike."""
        return (
            isinstance(other, TemporalRangeTransform)
            and self.start_ns == other.start_ns
            and self.step_ns == other.step_ns
            and self.dim_size == other.dim_size
            and self.dtype == other.dtype
        )


class TemporalRangeIndex(CoordinateTransformIndex):
    """
    An xarray index over a `TemporalRangeTransform`.

    Selection answers exactly as the materialized segments of the same
    tree answer: a scalar label must land on a sample — nearer than half
    a step past either end counts — or ``method="nearest"`` takes the
    nearest sample within the span; a slice keeps every sample within
    its (inclusive) endpoints; and a partial datetime string names its
    whole period, as pandas reads it. A contiguous or strided ``isel``
    returns a new lazy index, so laziness survives selection chains;
    anything fancier falls back to a materialized pandas index over just
    the selected labels. Asking for the pandas form (``.indexes``,
    ``.to_pandas()``) materializes the labels, as reading ``.values``
    does, and concatenating abutting segments stays lazy.

    Not yet supported: aligning with a differently indexed coordinate
    (xarray raises), and ``sel`` with ``tolerance``, with a ``method``
    other than nearest, or with ``method`` on a slice.
    """

    transform: TemporalRangeTransform

    def __init__(self, transform: TemporalRangeTransform):
        super().__init__(transform)
        self.dim = transform.name

    @classmethod
    def from_coord(cls, name: str, coord) -> TemporalRangeIndex:
        """Build from an evenly sampled temporal dascore coordinate."""
        start = np.asarray(coord.min())
        kind = (
            "datetime64[ns]"
            if np.issubdtype(start.dtype, np.datetime64)
            else ("timedelta64[ns]")
        )
        # explicit ns: a coarser-unit start or step would silently
        # relabel every sample 1000x too fine read as raw integers
        start_ns = start.astype(kind).view("int64")
        step_ns = np.asarray(coord.step).astype("timedelta64[ns]").view("int64")
        transform = TemporalRangeTransform(
            name, len(coord), int(start_ns), int(step_ns), kind
        )
        return cls(transform)

    @property
    def size(self) -> int:
        """The number of samples the index labels."""
        return self.transform.dim_size[self.dim]

    def _exact_positions(self, values) -> tuple[np.ndarray, np.ndarray]:
        """Exact (quotient, remainder) sample positions for labels."""
        t = self.transform
        # object (python-int) arrays lack a divmod ufunc; // and % map
        # to the exact python operators
        offset = _label_ints(values, t.dtype) - t.start_ns
        quotient = offset // t.step_ns
        return quotient, offset - quotient * t.step_ns

    def rename(self, name_dict, dims_dict) -> TemporalRangeIndex:
        """A renamed coordinate keeps its lazy index under the new name."""
        t = self.transform
        new = dims_dict.get(self.dim, name_dict.get(self.dim, self.dim))
        if new == self.dim:
            return self
        return type(self)(
            TemporalRangeTransform(new, self.size, t.start_ns, t.step_ns, t.dtype)
        )

    def to_pandas_index(self) -> pd.Index:
        """The materialized pandas form, computed on demand."""
        labels = self.transform.forward({self.dim: np.arange(self.size)})[self.dim]
        return pd.Index(labels)

    @classmethod
    def concat(cls, indexes, dim, positions=None) -> TemporalRangeIndex | Any:
        """
        Concatenate segment indexes, staying lazy when the grids chain.

        Segments whose ranges abut exactly (same step, each starting one
        step past the previous end) merge into one lazy index; anything
        else — a gap, a step change, an explicit reordering — comes back
        as an ordinary materialized pandas index over the concatenated
        labels.
        """
        transforms = [index.transform for index in indexes]
        first = transforms[0]
        chained = positions is None and all(
            t.step_ns == first.step_ns
            and t.dtype == first.dtype
            and t.start_ns == prev.start_ns + prev.dim_size[prev.name] * prev.step_ns
            for prev, t in pairwise(transforms)
        )
        if chained:
            size = sum(t.dim_size[t.name] for t in transforms)
            return cls(
                TemporalRangeTransform(
                    first.name, size, first.start_ns, first.step_ns, first.dtype
                )
            )
        from xarray.core import nputils  # noqa: PLC0415
        from xarray.indexes import PandasIndex  # noqa: PLC0415

        values = np.concatenate([index.to_pandas_index().values for index in indexes])
        if positions is not None:
            indices = nputils.inverse_permutation(np.concatenate(positions))
            values = values[indices]
        return PandasIndex(pd.Index(values), dim)

    def isel(self, indexers) -> Any:
        """A sliced view keeps a lazy index; fancy indexing materializes.

        Falling back to a pandas index over just the selected labels
        (rather than returning None, which would drop the index) keeps
        label selection working on the result.
        """
        idx = indexers.get(self.dim)
        if isinstance(idx, slice):
            start, stop, stride = idx.indices(self.size)
            if stride > 0:
                size = max((stop - start + stride - 1) // stride, 0)
                t = self.transform
                new = TemporalRangeTransform(
                    self.dim,
                    size,
                    t.start_ns + start * t.step_ns,
                    t.step_ns * stride,
                    t.dtype,
                )
                return type(self)(new)
            positions = np.arange(start, stop, stride)
        else:
            if getattr(idx, "dims", (self.dim,)) != (self.dim,):
                # vectorized onto another dimension: the labels no
                # longer index this one, so xarray drops the index
                return None
            positions = np.asarray(getattr(idx, "values", idx))
            if positions.ndim != 1:
                # multi-dimensional (vectorized) indexing has no 1-d
                # labels to index; let xarray drop the index
                return None
            if positions.dtype == bool:
                positions = np.flatnonzero(positions)
            # the data reads a negative position from the end; so must
            # its label
            positions = np.where(positions < 0, positions + self.size, positions)
        from xarray.indexes import PandasIndex  # noqa: PLC0415

        labels = self.transform.forward({self.dim: positions})[self.dim]
        return PandasIndex(pd.Index(labels), self.dim)

    def _positions_for(self, values, method):
        """Validated integer positions for on-grid/nearest label values."""
        step = self.transform.step_ns
        quot, rem = self._exact_positions(values)
        if method == "nearest":
            pos = quot + (2 * rem >= step)
        else:
            if np.any(rem != 0):
                msg = (
                    f"Label(s) {values} do not fall on the coordinate's "
                    "sample grid; pass method='nearest' to take the "
                    "nearest sample."
                )
                raise KeyError(msg)
            pos = quot
        if np.any((pos < 0) | (pos >= self.size)):
            msg = (
                f"Label(s) {values} fall outside the coordinate's sampled "
                "span; nothing to select."
            )
            raise KeyError(msg)
        return pos.astype("int64")

    def _period_bounds(self, label: str):
        """The inclusive span a partial datetime string names, or None.

        pandas answers ``sel(time="2020-01-01")`` on an hourly index with
        the whole day; the lazy index must answer identically, so a
        string label resolves through its period's start and end.
        """
        if np.dtype(self.transform.dtype).kind != "M":
            return None
        try:
            period = pd.Period(label)
        except Exception:
            return None
        if not isinstance(period, pd.Period):  # NaT parses without erroring
            return None
        reso = _PERIOD_RESO.get(period.freqstr.split("-")[0])
        assert reso is not None, f"unexpected period frequency {period.freqstr!r}"
        return (
            period.start_time.to_datetime64(),
            period.end_time.to_datetime64(),
            reso,
        )

    @property
    def _resolution(self) -> int:
        """The finest unit any label carries, as pandas infers it.

        pandas infers a DatetimeIndex's resolution from its values; the
        samples here are ``start + k * step``, so the start's finest unit
        and (past one sample) the step's finest unit decide it.
        """
        t = self.transform
        reso = _ns_resolution(t.start_ns)
        if self.size > 1:
            reso = min(reso, _ns_resolution(t.step_ns))
        return reso

    def sel(self, labels, method=None, tolerance=None) -> IndexSelResult:
        """Resolve label selection arithmetically."""
        if method not in (None, "nearest"):
            msg = (
                "TemporalRangeIndex resolves labels to their nearest "
                f"sample; method={method!r} is not supported."
            )
            raise ValueError(msg)
        if tolerance is not None:
            msg = "TemporalRangeIndex does not support tolerance in sel."
            raise ValueError(msg)
        label = labels[self.dim]
        if isinstance(label, slice):
            if method is not None:
                msg = "cannot use ``method`` argument with a slice, as pandas."
                raise ValueError(msg)
            return IndexSelResult({self.dim: self._sel_slice(label)})
        if isinstance(label, Variable | DataArray):
            # same validation as plain arrays, keeping the label's dims
            # so vectorized selection stays vectorized
            pos = self._positions_for(np.asarray(label.values), method)
            if isinstance(label, DataArray):
                return IndexSelResult({self.dim: label.copy(data=pos)})
            return IndexSelResult({self.dim: Variable(label.dims, pos)})
        if isinstance(label, str) and (bounds := self._period_bounds(label)):
            # a datetime string coarser than the index names a span and
            # keeps the dimension, even over one sample; one at least as
            # fine names a single stamp, exactly as pandas resolves it
            lo, hi, reso = bounds
            if reso <= self._resolution:
                pos = self._positions_for(lo, method)
                return IndexSelResult({self.dim: int(pos[0])})
            indexer = self._sel_slice(slice(lo, hi))
            if indexer.stop == indexer.start:
                msg = f"No samples fall within the period named by {label!r}."
                raise KeyError(msg)
            return IndexSelResult({self.dim: indexer})
        # scalar and plain-array labels: exact by default, exactly as the
        # materialized segments of the same tree answer; nearest opts in.
        pos = self._positions_for(label, method)
        if np.ndim(label) == 0:
            return IndexSelResult({self.dim: int(pos[0])})
        return IndexSelResult({self.dim: pos})

    def _sel_slice(self, label: slice) -> slice:
        """Resolve a label slice to a positional slice, endpoints inclusive."""
        # np.timedelta64 subclasses np.integer, so exclude it by name
        step_ok = isinstance(label.step, int | np.integer) and not isinstance(
            label.step, np.timedelta64
        )
        if label.step is not None and (not step_ok or label.step <= 0):
            msg = (
                "A label slice step must be a positive integer stride "
                f"of samples, got {label.step!r}."
            )
            raise ValueError(msg)

        def endpoint(value, edge):
            """A slice endpoint, with partial strings naming their period."""
            if value is None:
                return None
            if isinstance(value, str) and (bounds := self._period_bounds(value)):
                return bounds[edge]
            return value

        start = endpoint(label.start, 0)
        stop = endpoint(label.stop, 1)
        # inclusive endpoints, like pandas label slicing: every sample
        # with start <= label <= stop stays.
        if start is None:
            first = 0
        else:
            quot, rem = self._exact_positions(start)
            first = int(quot[0]) + int(rem[0] > 0)  # ceil
        if stop is None:
            last = self.size - 1
        else:
            quot, _ = self._exact_positions(stop)
            last = int(quot[0])  # floor
        first = max(first, 0)
        last = min(last, self.size - 1)
        return slice(first, max(last + 1, first), label.step)
