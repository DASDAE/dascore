"""Utilities for chunking dataframes."""

from __future__ import annotations

import warnings
from collections.abc import Collection
from functools import reduce

import numpy
import numpy as np
import pandas as pd

from dascore.constants import attr_conflict_description, numeric_types, timeable_types
from dascore.exceptions import ChunkError, CoordMergeError, ParameterError
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import get_middle_value
from dascore.utils.pd import (
    _instructions_modified,
    _remove_overlaps,
    get_column_names_from_dim,
    get_dim_names_from_columns,
    get_interval_columns,
)
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
    # get variable and perform checks
    overlap = length * 0 if not overlap else overlap
    step = length * 0 if pd.isnull(step) else step
    # Check for errors
    if overlap > length:
        msg = "Cant chunk when overlap is greater than chunk size"
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


@compose_docstring(attr_conflict=attr_conflict_description)
class ChunkManager:
    """
    A class for managing the chunking of data defined in a dataframe.

    The chunk manager handles both splitting and joining of contiguous,
    or near-contiguous, blocks of data.

    Parameters
    ----------
    overlap
        The amount of overlap between each segment, starting with the end of
        first row. Negative values can be used for inducing gaps.
    group_columns
        A sequence of column names which should be used for sorting groups.
    keep_partial
        If True, keep segments which are shorter than chunk size (at end of
        contiguous blocks)
    tolerance
        The upper limit of a gap to tolerate in terms of the sampling
        along the desired dimension. E.G., the default value means entities
        with gaps <= 1.5 * {name}_step will be merged.
    conflict
        {attr_conflict}
    **kawrgs
        kwargs specify the column along which to chunk. The key specifies the
        column along which to chunk, typically, `time` or `distance`, and the
        value specifies the chunk size. A value of None means to chunk on all
        available data (e.g. merge all data).

    Notes
    -----
    This class is used internally by `dc.BaseSpool.chunk`.
    """

    def __init__(
        self,
        overlap: timeable_types | numeric_types | None = None,
        group_columns: Collection[str] | None = None,
        keep_partial=False,
        snap_coords=True,
        tolerance=1.5,
        conflict="raise",
        **kwargs,
    ):
        self._overlap = overlap
        self._group_columns = group_columns
        self._keep_partials = keep_partial
        self._snap_coords = snap_coords
        self._tolerance = tolerance
        self._name, self._value = self._validate_kwargs(kwargs)
        self._attr_conflict = conflict
        self._validate_chunker()

    def _validate_kwargs(self, kwargs):
        """Ensure kwargs is len one and has a valid."""
        if not len(kwargs) == 1:
            msg = (
                f"Chunking only supported along one dimension. You passed "
                f"kwargs: {kwargs}"
            )
            raise ParameterError(msg)
        ((key, value),) = kwargs.items()
        value = None if value is ... else value
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
            return
        # ensure chunk values are greater than 0
        zero = to_timedelta64(0) if is_timedelta64(self._value) else 0
        if self._value <= zero:
            msg = "Chunk value must be greater than 0."
            raise ParameterError(msg)

    def _get_continuity_group_number(self, start, stop, step) -> pd.Series:
        """Return a series of ints indicating continuity group."""
        # start by sorting according to start time
        arg_ser = start.argsort()
        args = arg_ser.index[arg_ser.values]
        start_sorted, stop_sorted, step_sorted = start[args], stop[args], step[args]
        # next get cummax of endtimes and detect gaps
        stop_cum_max = stop_sorted.cummax()
        end_markers = stop_cum_max.shift() + step_sorted * self._tolerance
        has_gap = start_sorted > end_markers
        if has_gap.any():
            msg = (
                f"There is a gap in the patch along dimension {self._name}. "
                f"As a result, some patches in the chunked spool may be "
                f"unevenly sampled. However, they are still considered "
                f"contiguous because a tolerance of {self._tolerance} "
                f"was used in the chunk function."
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
        group_num = has_gap.astype(np.int64).cumsum()
        return group_num[start.index]

    def _get_sampling_group_num(self, step, tolerance=0.05) -> pd.Series:
        """
        Because sampling can be off a little, this adds some tolerance for
        how sampling affects groups.

        Tolerance affects how close samples have to be in order to count as
        the same. 5% is used here.
        """
        col = step.values
        sort_args = np.argsort(col)
        sorted_col = col[sort_args]
        roll_forward = np.roll(sorted_col, shift=1)
        diff = (sorted_col - roll_forward) / sorted_col
        out_of_threshold = diff > tolerance
        group_number = numpy.cumsum(out_of_threshold)
        # undo sorting
        out = pd.Series(group_number[np.argsort(sort_args)], index=step.index)
        return out

    def _get_duration_overlap(self, duration, start, step, overlap=None):
        """Get duration and overlap from kwargs."""
        overlap = overlap if overlap is not None else self._overlap
        # cast step to time delta if start is datetime
        if is_datetime64(start):
            step = to_timedelta64(step)
            overlap = to_timedelta64(overlap)
        if pd.isnull(overlap):
            overlap = np.asarray([0], dtype=step.dtype)[0]
        return duration, overlap

    def _create_df(self, df, name, start_stop, gnum):
        """Reconstruct the dataframe."""
        cols = f"{name}_min", f"{name}_max"
        out = pd.DataFrame(start_stop, columns=list(cols))
        out[f"{name}_step"] = get_middle_value(df[f"{name}_step"].values)
        merger = df.drop(columns=out.columns)
        # get dims to determine which columns are still compared. Some test
        # dfs don't have dims though, so it should still work without dims col.
        dims = set(df.iloc[0].get("dims", "").split(","))
        # We exclude private columns for considering if merge can happen.
        for col in set(x for x in merger.columns if not x.startswith("_")):
            prefix = col.split("_")[0]
            # If we have specified to ignore or remove conflicting attrs
            # we don't need to check them here, but we do still check dims.
            if self._attr_conflict != "raise" and prefix not in dims:
                continue
            vals = merger[col].unique()
            if len(vals) > 1:
                msg = (
                    f"Cannot merge on dim {self._name} because all values for "
                    f"{col} are not equal. Consider using the `conflict` "
                    f"argument to loosen this restriction."
                )
                raise CoordMergeError(msg)

            assert len(vals) == 1, "Haven't yet implemented non-homogenous merging"
            out[col] = vals[0]
        # add the group number for getting instruction df later
        out["_group"] = gnum
        return out

    def _get_chunk_overlap_inds(self, src1, src2, chu1, chu2):
        """Get an index mapping from source to chunk."""
        chunk_starts = np.searchsorted(src1, chu1, side="right") - 1
        chunk_ends = np.searchsorted(src2, chu2, side="left")
        # Ensure no chunks run off the end of the source.
        assert np.all(chunk_ends < len(src1)), "Invalid chunk range found"
        # add 1 to end so it is an exclusive end range
        return np.stack([chunk_starts, chunk_ends + 1], axis=1)

    def _get_source_and_chunk_inds(self, chunk2src_inds, s_index, c_index):
        """Get ndarrays of chunk index, source index."""
        # get indices for sorted arrays
        source_inds_ = np.concatenate(
            [np.arange(x[0], x[1], dtype=np.int64) for x in chunk2src_inds]
        )
        chunk_inds_ = np.concatenate(
            [
                np.ones((x[1] - x[0]), dtype=np.int64) * num
                for num, x in enumerate(chunk2src_inds)
            ]
        )
        # use pandex index to map back to actual indices
        source_inds = s_index.values[source_inds_]
        chunk_inds = c_index.values[chunk_inds_]
        out = {
            "source_sorted": source_inds_,
            "source": source_inds,
            "chunk_sorted": chunk_inds_,
            "chunk": chunk_inds,
        }
        return out

    def _get_instructions(self, sub_source, sub_chunk):
        """Get source mapping to chunk."""
        min_name, max_name = f"{self._name}_min", f"{self._name}_max"
        # sort inputs based on start of range, as long as we don't reset index
        # we should be ok.
        sub_source = sub_source.sort_values(min_name)
        sub_chunk = sub_chunk.sort_values(min_name)
        # need to make sure we don't have overlaps in source df. This implicitly
        # handles merging.
        sub_source = _remove_overlaps(sub_source, self._name)
        src1, src2, src_step = get_interval_columns(sub_source, self._name, arrays=True)
        chu1, chu2, chu_step = get_interval_columns(sub_chunk, self._name, arrays=True)
        dims = get_dim_names_from_columns(sub_source)
        cols2keep = get_column_names_from_dim(dims)
        # next get index range for which chunk times belong to.
        chunk2src_inds = self._get_chunk_overlap_inds(src1, src2, chu1, chu2)
        # total length of source to chunk mapping
        inds = self._get_source_and_chunk_inds(
            chunk2src_inds,
            sub_source.index,
            sub_chunk.index,
        )
        source_inds, chunk_inds = inds["source_sorted"], inds["chunk_sorted"]
        # get potential start/stop times.
        starts = np.stack([src1[source_inds], chu1[chunk_inds]], axis=1)
        ends = np.stack([src2[source_inds], chu2[chunk_inds]], axis=1)
        end_values = np.min(ends, axis=1)
        start_values = np.max(starts, axis=1)
        data_dict = {
            min_name: start_values,
            max_name: end_values,
            "source_index": inds["source"],
            "current_index": inds["chunk"],
        }
        out = pd.DataFrame(data_dict)
        # populate the rest of the columns needed in instruction df.
        for col in cols2keep:
            if col in out.columns:
                continue
            out[col] = sub_source[col].values[source_inds]
        out = out.sort_index()
        out["_modified"] = _instructions_modified(out, sub_source)
        return out

    def get_instruction_df(self, source_df, chunked_df):
        """
        Get a dataframe connecting the chunked dataframe to its origin.

        This is used to connect source data to desired data after chunking
        operation.

        Parameters
        ----------
        source_df
            The dataframe before chunking
        chunked_df
            The chunked dataframe (output of `chunk` method)
        """
        # the group column should exist and the chunked groups should be subset
        # of the source groups
        assert "_group" in source_df.columns and "_group" in chunked_df.columns
        chunked_groups = set(chunked_df["_group"])
        if not chunked_groups:
            return pd.DataFrame(columns=[*list(source_df.columns), "_modified"])
        # chunk groups should be a subset of source groups
        assert chunked_groups.issubset(set(source_df["_group"]))
        # iterate each group and create instruction df
        out = []
        for group in chunked_groups:
            sub_source = source_df[source_df["_group"] == group]
            sub_chunk = chunked_df[chunked_df["_group"] == group]
            out.append(self._get_instructions(sub_source, sub_chunk))
        df = pd.concat(out, axis=0).reset_index(drop=True).set_index("source_index")
        return df

    def _get_col_group(self, df, cont_g):
        """Get group columns based on common columns."""
        cols = list(self._group_columns or [])
        columns = [x for x in cols if x in df.columns]
        col_g = cont_g * 0 if not columns else df.groupby(columns).ngroup()
        return col_g

    def _get_group(self, df, start, stop, step):
        """
        Get the group designation for df. This accounts for both time intervals
        being consistent and group columns matching.
        """
        cont_g = self._get_continuity_group_number(start, stop, step)
        samp_g = self._get_sampling_group_num(step)
        col_g = self._get_col_group(df, cont_g)
        group_series = [x.astype(str) for x in [samp_g, col_g, cont_g]]
        group = reduce(lambda x, y: x + "_" + y, group_series)
        return group

    def _get_group_dfs(self, group, dur, overlap, group_mins, group_maxs, df, step):
        """Get the new dataframe for a given group."""
        out = []
        for gnum in group.unique():
            g_start, g_stop = group_mins[gnum], group_maxs[gnum]
            current_df = df.loc[group[group == gnum].index]
            # reconstruct DF
            try:
                new_start_stop = get_intervals(
                    g_start,
                    g_stop,
                    dur,
                    overlap=overlap,
                    step=step.iloc[0],
                    keep_partials=self._keep_partials,
                )
            except ChunkError:  # this chunk is too short, skip.
                continue
            # create the newly chunked dataframe
            sub_new_df = self._create_df(current_df, self._name, new_start_stop, gnum)
            out.append(sub_new_df)
        return out

    def _filter_nan_dfs(self, df, start, stop):
        """Filter NaN out of dataframe if they occur in start/stop."""

    def chunk(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chunk a dataframe into new contiguous segments.

        The dataframe must have column names {key}_max, {key}_min, and {key}_step
        where {key} is the key used in the kwargs.

        Parameters
        ----------
        df
            Input dataframe to chunk.

        Returns
        -------
        A tuple of the original dataframe with added column '_group' and an
        output dataframe with column '_group'. The _group column is used
        to link the two dataframes together.
        """
        if df.empty:  # empy df, do nothing
            return df.assign(_group=None), df.assign(_group=None)
        # get series of start/stop along requested dimension
        start, stop, step = get_interval_columns(df, self._name)
        # Filter out any NaN in start or stop.
        keep = ~(pd.isnull(start) | pd.isnull(stop))
        df, start, stop, step = df[keep], start[keep], stop[keep], step[keep]
        if df.empty:  # Need to check again since NaN can wipe out df.
            return df.assign(_group=None), df.assign(_group=None)
        dur, overlap = self._get_duration_overlap(self._value, start, step)
        # get group numbers
        group = self._get_group(df, start, stop, step)
        # get max, min for each group and expand
        group_mins = start.groupby(group).min()
        group_maxs = stop.groupby(group).max()
        # split/group dataframe into new chunks by iterating over each group.
        out = self._get_group_dfs(group, dur, overlap, group_mins, group_maxs, df, step)
        if not len(out):
            msg = "Could not chunk. No segments with sufficient length found."
            raise ChunkError(msg)
        out = pd.concat(out, axis=0).reset_index(drop=True)
        return df.assign(_group=group), out
