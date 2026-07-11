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
        If not None, subtract step (the sampling interval) from the end
        values so that the intervals do not overlap by one sample.
    keep_partials
        If True, keep the segments which are smaller than chunksize.

    Returns
    -------
    A 2D array where first column is start and second column is end.
    """
    # when length is null just use entire length
    if pd.isnull(length):
        out = np.asarray([start, stop])
        if is_datetime64(start):
            out = to_datetime64(out)
        return np.atleast_2d(out)

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
    # Since we just add to get stop values we need to remove anything
    # that is within a sample of stopping value (otherwise that segment
    # will have no data).
    reference = reference[(reference + step) <= stop]
    # we subtract step to avoid overlaps in segments. This can mean segments
    # are ~ one sample shorter than those requested.
    ends = reference + length - step
    starts = reference
    # trim end to not surpass stop
    bad_ends = ends > stop
    if bad_ends.any():
        if not keep_partials:
            ends_filt = ends <= stop
            ends, starts = ends[ends_filt], starts[ends_filt]
        else:
            ends[bad_ends] = stop
    return np.stack([starts, ends]).T
