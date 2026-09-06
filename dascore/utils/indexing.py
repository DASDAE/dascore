"""Dimension indexing shared by Patch.sel and Patch.isel."""

from __future__ import annotations

import operator
import warnings
from bisect import bisect_left, bisect_right
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


def _label_index(coord, probes):
    """Build pandas' index from stored labels or just the needed range samples."""
    if not isinstance(coord, CoordRange):
        return pd.Index(coord.values), None
    size = len(coord)
    # Vectorized pandas lookup is cheaper when many probes would each require
    # two Python binary searches. Small queries still avoid expanding large grids.
    if 4 * len(probes) * size.bit_length() > size:
        return pd.Index(coord.values), None
    if np.dtype(coord.dtype).kind not in "mM":
        endpoints = np.asarray([coord.start, coord.stop - coord.step])
        dtype = np.result_type(endpoints, 0.0)
        resolution = np.max(np.abs(np.spacing(endpoints.astype(dtype))))
        if abs(coord.step) < resolution:
            # Rounding can introduce duplicate labels anywhere on this grid.
            # Pandas must see the whole index to enforce its uniqueness rules.
            return pd.Index(coord.values), None
    positions = {0, min(1, size - 1), max(0, size - 2), size - 1}
    anchor = pd.Index(coord._get_index_values(np.array(sorted(positions))))

    def label_at(position):
        """Expose the compact coordinate in increasing label order for bisect."""
        return coord[size - 1 - position if coord.reverse_sorted else position]

    for probe in probes:
        if probe is None:
            continue
        for side, search in (("left", bisect_left), ("right", bisect_right)):
            try:
                # Reuse pandas' datetime precision and slice-bound type rules.
                bound = anchor._maybe_cast_slice_bound(probe, side)
                position = search(range(size), bound, key=label_at)
            except (TypeError, ValueError):
                # Let the final pandas operation report invalid labels using
                # the same error category as a materialized index.
                continue
            for neighbor in (position - 1, position):
                if 0 <= neighbor < size:
                    positions.add(
                        size - 1 - neighbor if coord.reverse_sorted else neighbor
                    )
    positions = np.array(sorted(positions), dtype=np.intp)
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
    index, positions = _label_index(coord, np.atleast_1d(labels))
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
