"""
Utilities for chunking dataframes.
"""
from typing import Collection, Optional, Tuple, Union

import numpy as np
import pandas as pd

from dascore.constants import numeric_types, timeable_types
from dascore.utils.time import is_datetime64, to_timedelta64


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
        if not keep_partials:
            ends, starts = ends[:-1], starts[:-1]
        else:
            ends[-1] = stop
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
        overlap = to_timedelta64(overlap)
    over = overlap if not pd.isnull(overlap) else (step * 0).iloc[0]
    return duration, over


def _create_df(df, name, start_stop, index_count=0):
    """Reconstruct the dataframe."""
    index = np.arange(index_count, index_count + len(start_stop))
    out = pd.DataFrame(start_stop, columns=[f"{name}_min", f"{name}_max"], index=index)
    merger = df.drop(columns=out.columns)
    for col in merger:
        vals = merger[col].unique()
        assert len(vals) == 1, "Havent yet implemented non-homogenous merging"
        out[col] = vals[0]
    return out


def _create_connecting_df(current_df, sub_new_df, name):
    """
    Create a connecting dataframe which contains instructions to make
    the new dataframe out of the old one.
    """
    min_name, max_name = f"{name}_min", f"{name}_max"
    c_start, c_stop = current_df[min_name], current_df[max_name]
    new_start, new_stop = sub_new_df[min_name], sub_new_df[max_name]
    out = []
    # TODO need to think more about this, a naive implementation for now
    for start, stop, ind in zip(new_start.values, new_stop.values, new_stop.index):
        too_late = c_start > stop
        too_early = c_stop < start
        in_range = ~(too_early | too_late)
        assert in_range.sum() > 0, "no original data source found!"
        sub_df = current_df[in_range]
        sub_df.loc[sub_df[min_name] < start, min_name] = start
        sub_df.loc[sub_df[max_name] > stop, max_name] = stop
        sub_df.loc[:, "original_index"] = sub_df.index.values
        sub_df.loc[:, "new_index"] = ind
        out.append(sub_df)
    df = pd.concat(out, axis=0).reset_index(drop=True)
    return df


def chunk(
    df: pd.DataFrame,
    overlap: Optional[Union[timeable_types, numeric_types]] = None,
    group_columns: Optional[Collection[str]] = None,
    keep_partial=False,
    tolerance=1.5,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chunk a dataframe into new contiguous segments.

    The dataframe must have column names {key}_max, {key}_min, and d_{key}
    where {key} is the key used in the kwargs.

    Parameters
    ----------
    df
        Input dataframe to chunk.
    overlap
        The amount of overlap between each segment. Negative values can
        be used for inducing gaps.
    group_columns
        A sequence of column names which should be used for sorting groups.
    keep_partial
        If True, keep segments which are shorter than chunk size (at end of
        contiguous blocks)
    tolerance
        The upper limit of a gap to tolerate in terms of the sampling
        along the desired dimension. E.G., the default value means entities
        with gaps <= 1.5 * d_{name} will be merged.
    **kwargs
        Used to specify the dimensions to chunk.

    Returns
    -------
    A tuple of intermediate dataframe and output dataframe. The intermediate
    dataframe provides instructions on how to form chunked dataframe from first
    dataframe.
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
    new_dfs = []
    connecting_dfs = []
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
            keep_partials=keep_partial,
        )
        # create the newly chunked dataframe
        sub_new_df = _create_df(current_df, name, new_start_stop, len(new_dfs))
        # and dataframe connecting it to original dataframe
        sub_connecting_df = _create_connecting_df(
            current_df,
            sub_new_df,
            name,
        )
        new_dfs.append(sub_new_df)
        connecting_dfs.append(sub_connecting_df)
    return pd.concat(new_dfs, axis=0), pd.concat(connecting_dfs, axis=0)
