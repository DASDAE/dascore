"""Dimension indexing shared by Patch.sel and Patch.isel."""

from __future__ import annotations

import operator
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from dascore.utils.array_api import array_namespace, device
from dascore.utils.time import to_timedelta64

if TYPE_CHECKING:
    from dascore.core.coords import BaseCoord


def get_indexers(
    indexers: Mapping[str, Any] | None,
    kwargs: Mapping[str, Any],
    dims: Sequence[str],
    missing_dims: str = "raise",
) -> dict[str, Any]:
    """Normalize dictionary/keyword indexers and validate dimension names."""
    if indexers is not None and not isinstance(indexers, Mapping):
        raise TypeError("indexers must be a mapping of dimension names to indexers.")
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


def apply_indexers(array: Any, indexers: tuple) -> Any:
    """Index axes independently, preserving backend and Cartesian semantics."""
    if array is None:
        return None
    xp = array_namespace(array)
    # Work backwards so scalar indexing cannot shift an axis still to index.
    for axis in reversed(range(len(indexers))):
        indexer = indexers[axis]
        if isinstance(indexer, np.ndarray):
            inds = xp.asarray(indexer, dtype=xp.int64, device=device(array))
            array = xp.take(array, inds, axis=axis)
        elif indexer != slice(None):
            key = tuple(
                indexer if i == axis else slice(None) for i in range(array.ndim)
            )
            array = array[key]
    return xp.asarray(array)


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
    index = pd.Index(coord.values)

    def compatible(label):
        # Keep datetime strings intact: pandas understands their precision and
        # includes the whole stated interval when they are used as slice bounds.
        if isinstance(label, str):
            return label
        return coord._get_compatible_value(label)

    if isinstance(value, slice):
        if method is not None or tolerance is not None:
            raise NotImplementedError(
                "method and tolerance are not supported with slices."
            )
        result = index.slice_indexer(
            compatible(value.start), compatible(value.stop), value.step
        )
        if not isinstance(result, slice):
            raise KeyError("Label slice cannot be represented as a positional slice.")
        return result
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
    if labels.ndim == 0 and method is None:
        return index.get_loc(labels[()])
    result = index.get_indexer(
        np.atleast_1d(labels), method=method, tolerance=tolerance
    )
    if np.any(result < 0):
        raise KeyError("Not all requested labels were found in the coordinate.")
    return int(result[0]) if labels.ndim == 0 else result
