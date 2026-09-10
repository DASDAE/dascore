"""Utilities for detecting coordinate gaps and constructing cell edges."""

from __future__ import annotations

import warnings

import numpy as np

from dascore.utils.time import is_datetime64, is_timedelta64, to_float


def _to_numeric(values):
    """Convert timedeltas to seconds while retaining other numeric values."""
    values = np.asarray(values)
    return to_float(values) if is_timedelta64(values) else values


def _normalize_coord_values(values):
    """Normalize coordinate values for gap and edge calculations."""
    values = np.asarray(values)
    if is_datetime64(values):
        return values.astype("datetime64[ns]")
    return _to_numeric(values)


def is_monotonic_and_finite(values) -> bool:
    """Return True when values are finite and strictly monotonic."""
    values = _normalize_coord_values(values)
    if not np.all(np.isfinite(values)):
        return False
    diffs = _to_numeric(np.diff(values))
    return bool(not len(diffs) or np.all(diffs > 0) or np.all(diffs < 0))


def get_gap_edges(values, gap_factor: float | None = None):
    """
    Return cell edges and coordinate gap locations.

    Timedelta coordinates are converted to seconds. Datetime coordinates are
    retained so plotting libraries with datetime support can convert them.
    When ``gap_factor`` is None, adjacent cell edges meet halfway between
    coordinate centers. Otherwise, intervals larger than ``gap_factor`` times
    the median interval are expanded into a gap.

    Parameters
    ----------
    values
        One-dimensional, monotonic coordinate centers.
    gap_factor
        Factor of the median interval above which an interval is a gap. If
        None, no intervals are considered gaps.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Cell edges and a Boolean array marking gaps after each input value.

    See Also
    --------
    [`Spool.get_gaps`](`dascore.core.spool.Spool.get_gaps`) answers the
    other gap question: where whole patches fail to meet, read from the
    index rather than from a coordinate's values.
    """
    values = _normalize_coord_values(values)
    if len(values) == 1:
        msg = "Singleton coordinate has no inferred cell width; using a default width."
        warnings.warn(msg, UserWarning, stacklevel=2)
        if is_datetime64(values):
            step = np.asarray(np.timedelta64(1, "D")).astype("timedelta64[ns]")[()]
        else:
            step = 1
        return np.asarray([values[0] - step / 2, values[0] + step / 2]), np.zeros(
            0, dtype=bool
        )

    diffs = np.diff(values)
    numeric_diffs = _to_numeric(diffs)
    gap_mask = np.zeros(len(diffs), dtype=bool)
    if gap_factor is not None:
        representative_interval = np.median(np.abs(numeric_diffs))
        gap_mask = np.abs(numeric_diffs) > representative_interval * gap_factor

    if not np.any(gap_mask):
        edges = np.concatenate(
            (
                [values[0] - diffs[0] / 2],
                values[:-1] + diffs / 2,
                [values[-1] + diffs[-1] / 2],
            )
        )
        return edges, gap_mask

    representative_step = np.median(np.abs(diffs))
    direction = 1 if numeric_diffs[0] > 0 else -1
    signed_step = direction * representative_step
    first_step = signed_step if gap_mask[0] else diffs[0]
    last_step = signed_step if gap_mask[-1] else diffs[-1]
    edges = [values[0] - first_step / 2]
    for index, diff in enumerate(diffs):
        if gap_mask[index]:
            edges.extend(
                [
                    values[index] + signed_step / 2,
                    values[index + 1] - signed_step / 2,
                ]
            )
        else:
            edges.append(values[index] + diff / 2)
    edges.append(values[-1] + last_step / 2)
    return np.asarray(edges), gap_mask
