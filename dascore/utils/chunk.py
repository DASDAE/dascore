"""
Utilities for chunking dataframes.
"""
from typing import Collection, Optional

import numpy as np
import pandas as pd

from dascore.constants import timeable_types
from dascore.utils.time import is_datetime64, to_timedelta64


def get_intervals(
    start,
    stop,
    length,
    overlap=None,
    step=None,
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
        The length of the segements.
    overlap
        The overlap of the start of each interval with the end
        of the previous interval.
    step
        If not None, subtract step (the sampling interval) from the end
        values so that the intervals do not overlap by one sample.

    Returns
    -------
    A 2D array where first column is start and second column is end.
    """
    if is_datetime64(start):
        length = to_timedelta64(length)
    # get variable and perform checks
    overlap = length * 0 if not overlap else overlap
    step = length * 0 if not step else step
    assert overlap < length, "Overlap must be less than step"
    assert (stop - start) > length, "Range must be greater than step"
    # reference with no overlap
    new_step = length - overlap
    reference = np.arange(start, stop + new_step, step=new_step)
    ends = reference[:-1] + length - step
    starts = reference[:-1]
    # trim end to not surpass stop
    if ends[-1] > stop:
        ends, starts = ends[:-1], starts[:-1]
    return np.stack([starts, ends]).T


def _get_columns(df, name):
    """Return a series of start, stop, step for columns."""
    names = f"{name}_min", f"{name}_max", f"d_{name}"
    missing_cols = set(names) - set(df.columns)
    if missing_cols:
        msg = f"Dataframe is missing {missing_cols} to chunk on {name}"
        raise KeyError(msg)
    return df[names[0]], df[names[1]], df[names[2]]


def _get_continuity_group_number(start, stop, step) -> pd.Series:
    """Return a series of ints indicating continuity group."""
    # start by sorting according to start time
    args = start.argsort()
    start_sorted, stop_sorted, step_sorted = start[args], stop[args], step[args]
    # next get cummax of endtimes and detect gaps
    stop_cum_max = stop_sorted.cummax()
    has_gap = start_sorted.shift() > (stop_cum_max + step_sorted)
    group_num = has_gap.astype(int).cumsum()
    return group_num[start.index]


def _get_duration_overlap(duration, start, step, overlap=None):
    """
    Get duration and overlap from kwargs.
    """
    # cast step to time delta if start is datetime
    if is_datetime64(start):
        step = to_timedelta64(step)
    over = overlap if overlap is not None else (step * 0).iloc[0]
    return duration, over


def _get_df_dropped_columns(df, name):
    """Drop columns related to name."""


def _create_df(df, name, start_stop):
    """Reconstruct the dataframe."""
    out = pd.DataFrame(start_stop, columns=[f"{name}_min", f"{name}_max"])
    merger = df.drop(columns=out.columns)
    for col in merger:
        vals = merger[col].unique()
        assert len(vals) == 1, "Havent yet implemented non-homogenous merging"
        out[col] = vals[0]
    return out


def chunk(
    df: pd.DataFrame,
    overlap: Optional[timeable_types] = None,
    group_columns: Optional[Collection[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Chunk a datafarme based on columns of the dataframe for one column.

    Parameters
    ----------
    df
        Input dataframe to chunk.
    overlap
        The amount of overlap between each segment. Negative values can
        be used for inducing gaps.
    group_columns
        A sequence of column names which should be used for sorting groups.
    **kwargs
        Used to specify the dimensions to chunk. The name of the kwarg
        should be a column
    """
    assert len(kwargs) == 1, "Chunking must be for a single variable"
    name = list(kwargs)[0]
    # get series of start/stop along requested dimension
    start, stop, step = _get_columns(df, name)
    dur, overlap = _get_duration_overlap(kwargs[name], start, step, overlap)
    # Find continuity group numbers (ints which indicate if a row belongs
    # to a continuity block) then apply column filters.
    group_cont = _get_continuity_group_number(start, stop, step)
    cols = [f"d_{name}"] + list(group_columns or [])
    col_groups = df.groupby(cols).ngroup()
    group = group_cont.astype(str) + "_" + col_groups.astype(str)
    # get max, min for each group and expand
    group_mins = start.groupby(group).min()
    group_maxs = stop.groupby(group).max()
    # split/group dataframe into new chunks by iterating over each group.
    out = []
    for gnum in group.unique():
        start, stop = group_mins[gnum], group_maxs[gnum]
        current_df = df.loc[group[group == gnum].index]
        # Reconstruct DF
        new_start_stop = get_intervals(
            start,
            stop,
            dur,
            overlap=overlap,
            step=step.iloc[0],
        )
        out.append(_create_df(current_df, name, new_start_stop))
    return pd.concat(out, axis=0).reset_index(drop=True)
