"""Dimension indexing shared by Patch.sel and Patch.isel."""

from __future__ import annotations

import operator
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from dascore.core.coords import BaseCoord, CoordRange
from dascore.utils.time import to_timedelta64


def get_indexers(
    indexers: Mapping[str, Any] | None,
    kwargs: Mapping[str, Any],
    dims: Sequence[str],
    missing_dims: str = "raise",
) -> dict[str, Any]:
    """Normalize dictionary/keyword indexers and validate dimension names."""
    if indexers is not None and not isinstance(indexers, Mapping):
        raise ValueError("indexers must be a mapping of dimension names to indexers.")
    if indexers and kwargs:
        raise ValueError("Provide either indexers or keyword indexers, not both.")
    if missing_dims not in {"raise", "warn", "ignore"}:
        raise ValueError("missing_dims must be 'raise', 'warn', or 'ignore'.")
    out = dict(indexers or kwargs)
    missing = set(out) - set(dims)
    if missing:
        msg = f"Dimensions {sorted(missing)} do not exist. Expected one of {dims}."
        if missing_dims == "raise":
            raise ValueError(msg)
        if missing_dims == "warn":
            warnings.warn(msg, UserWarning, stacklevel=3)
        out = {key: value for key, value in out.items() if key not in missing}
    return out


def _unlabelled_array(value):
    """Refuse labelled/vectorized indexers instead of discarding their meaning."""
    if hasattr(value, "dims") or isinstance(value, Mapping):
        raise TypeError("Only scalar, slice, and unlabelled 1D indexers are supported.")
    out = np.asarray(value)
    if out.ndim > 1:
        raise IndexError("Only scalar and 1D array indexers are supported.")
    return out


def positional_indexer(value: Any, size: int) -> int | slice | np.ndarray:
    """Validate one positional indexer without allocating a full coordinate."""
    if isinstance(value, slice):
        # Validate bounds without replacing open/negative bounds with normalized
        # ones: slice(None, None, -1) must still reach the first sample.
        value.indices(size)
        return value
    indexer = _unlabelled_array(value)
    if indexer.dtype.kind == "b" and indexer.ndim == 1:
        if len(indexer) != size:
            raise IndexError(
                f"Boolean indexer has length {len(indexer)}, expected {size}."
            )
        return np.flatnonzero(indexer)
    # An empty Python sequence is a valid integer indexer despite numpy's
    # default float dtype. Explicitly typed float arrays remain invalid.
    if isinstance(value, (list, tuple)) and not len(value):
        indexer = indexer.astype(np.intp)
    if indexer.dtype.kind not in "iu":
        error = TypeError if indexer.ndim == 0 else IndexError
        raise error("Positional indexers must be integers, slices, or boolean masks.")
    if np.any(indexer >= size) or (
        indexer.dtype.kind == "i" and np.any(indexer < -size)
    ):
        raise IndexError(f"Index is out of bounds for a dimension of size {size}.")
    indexer = indexer.astype(np.intp)
    if indexer.ndim == 0:
        return operator.index(indexer)
    return np.where(indexer < 0, indexer + size, indexer)


def _range_estimate(coord, bounds):
    """Reuse native selection arithmetic without casting non-finite positions."""
    clipped = np.clip(bounds, coord.min(), coord.max())
    clipped = np.where(np.isfinite(clipped), clipped, coord.min())
    return np.clip(coord._get_index(clipped), 0, len(coord) - 1)


def _range_searchsorted(coord, bounds, side):
    """Verify native range estimates, with bounded binary search as a fallback."""
    size = len(coord)
    low = np.zeros(len(bounds), dtype=np.intp)
    high = np.full(len(bounds), size, dtype=np.intp)
    compare = np.less if side == "left" else np.less_equal

    def values_at(positions):
        positions = np.clip(positions, 0, size - 1)
        if coord.reverse_sorted:
            positions = size - 1 - positions
        return coord._get_index_values(positions)

    if bounds.dtype.kind in "iufmM" and coord.step:
        # Reuse select's arithmetic lookup, but verify its bracket against
        # actual labels: grid rounding may move an estimate by a sample.
        estimate = _range_estimate(coord, bounds)
        if coord.reverse_sorted:
            estimate = size - 1 - estimate
        lower, upper = np.maximum(estimate - 1, 0), np.minimum(estimate + 2, size)
        low = np.where(compare(values_at(lower - 1), bounds), lower, low)
        high = np.where(
            (upper == size) | ~compare(values_at(upper), bounds), upper, high
        )
    while np.any(low < high):
        middle = low + (high - low) // 2
        right = compare(values_at(middle), bounds) & (low < high)
        low = np.where(right, middle + 1, low)
        high = np.where(right, high, middle)
    return low


def _require_unique_range(coord):
    """Check range uniqueness without an unbounded scan of floating labels."""
    size = len(coord)
    if size > 1 and np.dtype(coord.dtype).kind not in "mM":
        endpoints = np.asarray([coord.start, coord.stop - coord.step])
        dtype = np.result_type(endpoints, 0.0)
        resolution = np.max(np.abs(np.spacing(endpoints.astype(dtype))))
        # Numeric ranges, including integer ranges, use linspace arithmetic.
        grid_step = np.subtract(endpoints[1], endpoints[0], dtype=dtype) / (size - 1)
        if abs(grid_step) < resolution:
            # A bounded sample handles short grids. Long grids below floating
            # precision cannot promise unique labels without a full-grid scan.
            sample = coord._get_index_values(np.arange(min(size, 32)))
            if size > 32 or not pd.Index(sample).is_unique:
                raise pd.errors.InvalidIndexError(
                    "Range labels may repeat at this floating-point precision; "
                    "use positional indexing instead."
                )


def _label_index(coord, probes, require_unique=False):
    """Use stored labels or query-sized samples; never expand a compact grid."""
    if not isinstance(coord, CoordRange):
        return pd.Index(coord.values), None
    size = len(coord)
    positions = np.unique([0, min(1, size - 1), max(0, size - 2), size - 1])
    anchor = pd.Index(coord._get_index_values(positions))
    if require_unique:
        _require_unique_range(coord)
    pieces = [positions]
    values = np.asarray(probes)
    for side in ("left", "right"):
        # Pandas supplies partial-date precision and invalid-bound errors.
        # Ordinary numeric and temporal arrays need no per-label Python work.
        if values.dtype.kind in "iufmM":
            bounds = values
        else:
            bounds = []
            for probe in probes:
                if probe is None:
                    continue
                try:
                    bound = anchor._maybe_cast_slice_bound(probe, side)
                    anchor[0] < bound  # Validate mixed types before batching.
                except (TypeError, ValueError):
                    continue  # The final pandas lookup reports invalid labels.
                bounds.append(bound)
            bounds = pd.Index(bounds).to_numpy()
        try:
            found = _range_searchsorted(coord, bounds, side)
        except (TypeError, ValueError):
            continue
        neighbors = np.concatenate([found - 1, found])
        neighbors = neighbors[(neighbors >= 0) & (neighbors < size)]
        if coord.reverse_sorted:
            neighbors = size - 1 - neighbors
        pieces.append(neighbors)
    positions = np.unique(np.concatenate(pieces))
    return pd.Index(coord._get_index_values(positions)), positions


def _restore_indexer(indexer, positions):
    """Map a sparse range lookup back to positions on the original coordinate."""
    if positions is None:
        return indexer
    if not isinstance(indexer, slice):
        return positions[indexer]
    step = 1 if indexer.step is None else indexer.step
    direction = 1 if step > 0 else -1
    span = positions[slice(indexer.start, indexer.stop, direction)]
    if not len(span):
        return slice(0, 0)
    stop = int(span[-1]) + direction
    return slice(int(span[0]), None if stop < 0 else stop, step)


def label_indexer(
    coord: BaseCoord,
    value: Any,
    method: Literal["nearest"] | None = None,
    tolerance: Any = None,
) -> int | slice | np.ndarray:
    """Resolve labels with the pandas index semantics used by xarray."""
    if coord._partial:
        if method is not None or tolerance is not None:
            raise ValueError("Inexact matching requires coordinate labels.")
        return positional_indexer(value, len(coord))

    def compatible(label):
        # Keep datetime strings intact: pandas understands their precision and
        # includes the whole stated interval when they are used as slice bounds.
        if isinstance(label, str) or (
            np.dtype(coord.dtype).kind in "mM" and not hasattr(label, "units")
        ):
            return label
        return coord._get_compatible_value(label)

    if isinstance(value, slice):
        if method is not None or tolerance is not None:
            raise NotImplementedError(
                "method and tolerance are not supported with slices."
            )
        start, stop = compatible(value.start), compatible(value.stop)
        index, positions = _label_index(coord, (start, stop))
        result = index.slice_indexer(start, stop, value.step)
        if not isinstance(result, slice):
            raise KeyError("Label slice cannot be represented as a positional slice.")
        return _restore_indexer(result, positions)
    if hasattr(value, "units"):
        value = compatible(value)
    labels = _unlabelled_array(value)
    if labels.ndim == 1 and labels.dtype.kind == "b":
        return positional_indexer(labels, len(coord))
    if labels.dtype.kind not in "US":
        labels = np.asarray(compatible(labels))
    if np.dtype(coord.dtype).kind == "f":
        labels = labels.astype(coord.dtype)
    if tolerance is not None and hasattr(tolerance, "units"):
        # Tolerances are durations even when labels are absolute datetimes.
        if np.dtype(coord.dtype).kind in "mM":
            tolerance = to_timedelta64(tolerance.to("s").magnitude)
        else:
            tolerance = compatible(tolerance)
    if (
        isinstance(coord, CoordRange)
        and labels.ndim == 1
        and method is None
        and tolerance is None
        and labels.dtype.kind in "iufmM"
        and (
            labels.dtype.kind == np.dtype(coord.dtype).kind
            if np.dtype(coord.dtype).kind in "mM"
            else labels.dtype.kind in "iuf"
        )
    ):
        # Numeric exact arrays already carry their labels. Resolve positions
        # directly instead of sorting a second candidate index for dense queries.
        _require_unique_range(coord)
        positions = (
            _range_estimate(coord, labels)
            if coord.step
            else np.zeros(len(labels), dtype=np.intp)
        )
        missing = coord._get_index_values(positions) != labels
        if np.any(missing):
            found = _range_searchsorted(coord, labels[missing], "left")
            if np.any(found == len(coord)):
                raise KeyError("Not all requested labels were found in the coordinate.")
            positions[missing] = (
                len(coord) - 1 - found if coord.reverse_sorted else found
            )
        if not np.array_equal(coord._get_index_values(positions), labels):
            raise KeyError("Not all requested labels were found in the coordinate.")
        return positions
    index, positions = _label_index(
        coord,
        np.atleast_1d(labels),
        require_unique=labels.ndim != 0 or method is not None,
    )
    if labels.ndim == 0 and method is None:
        result = index.get_loc(labels[()])
        if positions is not None and isinstance(result, np.ndarray):
            # Pandas returns arrays for partial dates on descending indexes.
            # A range's matching interval is contiguous, including omitted samples.
            matches = np.flatnonzero(result) if result.dtype.kind == "b" else result
            if not len(matches):
                return slice(0, 0)
            result = slice(int(matches[0]), int(matches[-1]) + 1)
        return _restore_indexer(result, positions)
    result = index.get_indexer(
        np.atleast_1d(labels), method=method, tolerance=tolerance
    )
    if np.any(result < 0):
        raise KeyError("Not all requested labels were found in the coordinate.")
    result = _restore_indexer(result, positions)
    return int(result[0]) if labels.ndim == 0 else result
