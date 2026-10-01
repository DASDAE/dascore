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


def get_intervals(
    start,
    stop,
    length,
    overlap=None,
    step=None,
    keep_partials=False,
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
    step = length * 0 if pd.isnull(step) else step
    # Check for errors. Overlap equal to length would produce zero-stride
    # segments, so it is also rejected.
    if overlap >= length:
        msg = "Cant chunk when overlap is greater than or equal to chunk size"
        raise ParameterError(msg)
    if length < step:
        msg = "Cant chunk when chunk length is shorter than one sample step."
        raise ChunkError(msg)
    # If the step is known, we need to account for it in the total duration
    # See 474.
    _raw_duration = stop - start
    duration = _raw_duration + step if step is not None else _raw_duration
    if duration < length and not keep_partials:
        msg = "Cant chunk when data interval is less than chunk size. "
        raise ChunkError(msg)
    # reference with no overlap
    new_step = length - overlap
    reference = np.arange(start, stop + new_step, step=new_step)
    # every window holding a sample starts at or before the last one
    starts = reference[reference <= stop]
    # each window is half-open: [start, start + length)
    ends = starts + length
    if not keep_partials:
        full = ends <= stop + step
        starts, ends = starts[full], ends[full]
    return np.stack([starts, ends]).T
