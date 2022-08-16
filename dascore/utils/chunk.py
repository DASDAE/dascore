"""
Utilities for chunking dataframes.
"""
from typing import Collection, Optional, Union

import numpy as np
import pandas as pd

from dascore.constants import numeric_types, timeable_types
from dascore.exceptions import ParameterError
from dascore.utils.pd import (
    get_column_names_from_dim,
    get_dim_names_from_columns,
    get_interval_columns,
)
from dascore.utils.time import is_datetime64, to_datetime64, to_timedelta64


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
        out = np.array([start, stop])
        if is_datetime64(start):
            out = to_datetime64(out)
        return np.atleast_2d(out)

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


class ChunkManager:
    """
    A class for managing the chunking of data defined in a dataframe.

    The chunk manager handles both splitting and joining of contiguous,
    or near-contiguous, blocks of data.

    Parameters
    ----------
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
    **kawrgs
        kwargs specify the column along which to chunk. The key specifies the
        column along which to chunk, typically, `time` or `distance`, and the
        value specifies the chunk size. A value of None means to chunk on all
        available data (e.g. merge all data).
    """

    def __init__(
        self,
        overlap: Optional[Union[timeable_types, numeric_types]] = None,
        group_columns: Optional[Collection[str]] = None,
        keep_partial=False,
        tolerance=1.5,
        **kwargs,
    ):
        self._overlap = overlap
        self._group_columns = group_columns
        self._keep_partials = keep_partial
        self._tolerance = tolerance
        self._name, self._value = self._validate_kwargs(kwargs)
        self._validate_chunker()

    def _validate_kwargs(self, kwargs):
        """Ensure kwargs is len one and has a valid"""
        assert len(kwargs) == 1
        ((key, value),) = kwargs.items()
        return key, value

    def _validate_chunker(self):
        """Ensure selected parameters are compatible."""
        # chunker is used for merging
        if pd.isnull(self._value):
            if self._keep_partials or self._overlap:
                msg = (
                    "When chunk value is None (ie Chunker is used for merging) "
                    "both _keep_partials and self._overlap must not be selected."
                )
                raise ParameterError(msg)

    def _get_continuity_group_number(self, start, stop, step) -> pd.Series:
        """Return a series of ints indicating continuity group."""
        # start by sorting according to start time
        args = start.argsort()
        start_sorted, stop_sorted, step_sorted = start[args], stop[args], step[args]
        # next get cummax of endtimes and detect gaps
        stop_cum_max = stop_sorted.cummax()
        has_gap = start_sorted.shift() > (stop_cum_max + step_sorted)
        group_num = has_gap.astype(int).cumsum()
        return group_num[start.index]

    def _get_duration_overlap(self, duration, start, step, overlap=None):
        """
        Get duration and overlap from kwargs.
        """
        overlap = overlap if overlap is not None else self._overlap
        # cast step to time delta if start is datetime
        if is_datetime64(start):
            step = to_timedelta64(step)
            overlap = to_timedelta64(overlap)
        over = overlap if not pd.isnull(overlap) else (step * 0).iloc[0]
        return duration, over

    def _create_df(self, df, name, start_stop, index_count=0):
        """Reconstruct the dataframe."""
        index = np.arange(index_count, index_count + len(start_stop))
        out = pd.DataFrame(
            start_stop, columns=[f"{name}_min", f"{name}_max"], index=index
        )
        merger = df.drop(columns=out.columns)
        for col in merger:
            vals = merger[col].unique()
            assert len(vals) == 1, "Havent yet implemented non-homogenous merging"
            out[col] = vals[0]
        return out

    def get_instruction_df(self, origin_df, chunked_df, **kwargs):
        """
        Get a dataframe connecting the chunked dataframe to its origin.

        This is useful for describing how data chunking on patches should
        be performed.

        Parameters
        ----------
        origin_df
            The origin dataframe before chunking
        chunked_df
            The chunked dataframe (output of `chunk` method)
        """
        min_name, max_name = f"{self._name}_min", f"{self._name}_max"
        c_start, c_stop, c_step = get_interval_columns(origin_df, self._name)
        new_start, new_stop, new_step = get_interval_columns(chunked_df, self._name)
        dims = get_dim_names_from_columns(origin_df)
        cols2keep = get_column_names_from_dim(dims)
        out = []
        # TODO need to think more about this, a naive implementation for now
        for start, stop, ind in zip(new_start.values, new_stop.values, new_stop.index):
            too_late = c_start > stop
            too_early = c_stop < start
            in_range = ~(too_early | too_late)
            assert in_range.sum() > 0, "no original data source found!"
            sub_df = origin_df[in_range][cols2keep]
            sub_df.loc[sub_df[min_name] < start, min_name] = start
            sub_df.loc[sub_df[max_name] > stop, max_name] = stop
            sub_df.loc[:, "source_index"] = sub_df.index.values
            sub_df.loc[:, "current_index"] = ind
            out.append(sub_df)
        df = pd.concat(out, axis=0).reset_index(drop=True)
        return df.set_index("source_index")

    def chunk(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Chunk a dataframe into new contiguous segments.

        The dataframe must have column names {key}_max, {key}_min, and d_{key}
        where {key} is the key used in the kwargs.

        Parameters
        ----------
        df
            Input dataframe to chunk.

        Returns
        -------
        A tuple of intermediate dataframe and output dataframe. The intermediate
        dataframe provides instructions on how to form chunked dataframe from first
        dataframe.
        """
        # get series of start/stop along requested dimension
        start, stop, step = get_interval_columns(df, self._name)
        dur, overlap = self._get_duration_overlap(self._value, start, step)
        # Find continuity group numbers (ints which indicate if a row belongs
        # to a continuity block) then apply column filters.
        group_cont = self._get_continuity_group_number(start, stop, step)
        cols = [f"d_{self._name}"] + list(self._group_columns or [])
        columns = [x for x in cols if x in df.columns]
        col_groups = df.groupby(columns).ngroup()
        group = group_cont.astype(str) + "_" + col_groups.astype(str)
        # get max, min for each group and expand
        group_mins = start.groupby(group).min()
        group_maxs = stop.groupby(group).max()
        # split/group dataframe into new chunks by iterating over each group.
        out = []
        for gnum in group.unique():
            start, stop = group_mins[gnum], group_maxs[gnum]
            current_df = df.loc[group[group == gnum].index]
            # reconstruct DF
            new_start_stop = get_intervals(
                start,
                stop,
                dur,
                overlap=overlap,
                step=step.iloc[0],
                keep_partials=self._keep_partials,
            )
            # create the newly chunked dataframe
            sub_new_df = self._create_df(
                current_df, self._name, new_start_stop, len(out)
            )
            out.append(sub_new_df)
        return pd.concat(out, axis=0)

    @staticmethod
    def get_range_columns(df):
        """
        Return the names of the columns used by the dataframe to represent
        a range. Should have a {name}_min, {name}_max, d_{name}.
        """
