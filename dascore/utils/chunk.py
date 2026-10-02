"""Utilities for chunking dataframes.

The interval math here is consumed by the chunk planner
(`dascore.utils.chunk_plan`), which replaced the old ChunkManager.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dascore.exceptions import ChunkError, ParameterError
from dascore.utils.time import (
    is_datetime64,
    is_timedelta64,
    to_datetime64,
    to_timedelta64,
)

# Slack when deciding which side of a grid position a window edge falls on,
# so an edge a float rounding error short of a position still holds it.
_GRID_SNAP_RTOL = 1e-9
# The fraction of a step a float member bound is widened by when read, so a
# label known to rounding (float32 most of all) is not lost.
_READ_PAD = 0.1


def get_intervals(
    start,
    stop,
    length,
    overlap=None,
    step=None,
    keep_partials=False,
    tail=None,
):
    """
    Create a range of values with optional overlaps.

    Parameters
    ----------
    start
        The start of the interval.
    stop
        The end of the interval.
    length
        The length of the segments.
    overlap
        The overlap of the start of each interval with the end
        of the previous interval.
    step
        The sampling interval; a window is full when it ends no later than
        one step past ``stop``.
    keep_partials
        If True, keep the segments which are smaller than chunksize.
    tail
        The gap past ``stop`` to the next sample, where it is not ``step``
        (an exact fractional grid); used only to say which windows are full.

    Returns
    -------
    A 2D array of half-open ``[start, end)`` windows: the first column is
    the start, the second the exclusive end.
    """
    if is_datetime64(start):
        # need to ensure we have numpy datetimes, not pandas
        start, stop = to_datetime64(start), to_datetime64(stop)
        length = to_timedelta64(length)
    elif is_timedelta64(start):
        # a span of a timedelta64 coordinate is itself a duration, so a
        # numeric chunk length must be coerced to timedelta64 (otherwise the
        # duration < length comparison mixes Timedelta and float).
        start, stop = to_timedelta64(start), to_timedelta64(stop)
        length = to_timedelta64(length)
    # get variable and perform checks
    overlap = length * 0 if not overlap else overlap
    step = length * 0 if step is None or pd.isnull(step) else step
    tail = step if tail is None or pd.isnull(tail) else tail
    # Check for errors. Overlap equal to length would produce zero-stride
    # segments, so it is also rejected.
    if overlap >= length:
        msg = "Cant chunk when overlap is greater than or equal to chunk size"
        raise ParameterError(msg)
    if length < step:
        msg = "Cant chunk when chunk length is shorter than one sample step."
        raise ChunkError(msg)
    # A window is full when it ends no later than one step past the last
    # sample, to within the step's float rounding (see #474).
    top = stop + step * _GRID_SNAP_RTOL if isinstance(step, float) else stop
    if top - start + tail < length and not keep_partials:
        msg = "Cant chunk when data interval is less than chunk size. "
        raise ChunkError(msg)
    reference = np.arange(start, stop + length - overlap, step=length - overlap)
    # windows stop at the first to hold the last sample
    reach = top - overlap if overlap > 0 * overlap else top
    starts = reference[: max(np.searchsorted(reference, reach, "right"), 1)]
    # each window is half-open: [start, start + length)
    ends = starts + length
    if not keep_partials:
        full = np.searchsorted(ends, top + tail, "right")
        starts, ends = starts[:full], ends[:full]
    return np.stack([starts, ends]).T
