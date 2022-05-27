"""
Pandas utilities.
"""
import fnmatch
import os
from collections import defaultdict
from functools import cache
from typing import Collection

import numpy as np
import pandas as pd

from dascore.utils.time import to_timedelta64, to_datetime64


@cache
def get_regex(seed_str):
    """Compile, and cache regex for str queries."""
    return fnmatch.translate(seed_str)  # translate to re


def _remove_base_path(series: pd.Series, base="") -> pd.Series:
    """
    Ensure paths stored in column name use unix style paths and have base
    path removed.
    """
    if series.empty:
        return series
    unix_paths = series.str.replace(os.sep, "/")
    unix_base_path = str(base).replace(os.sep, "/")
    return unix_paths.str.replace(unix_base_path, "")


def _get_min_max(kwargs, df):
    """
    Get a dict of {column_name: Optional[min_val], Optional[max_val]}.

    Handles {column}_max, column_{min} type queries. Pop keys out of kwargs
    once they are in the return dict.
    """
    out = defaultdict(lambda: [None, None])
    col_set = set(df.columns)
    to_kill = []
    for key, val in kwargs.items():
        if key.endswith("_max") and key not in col_set:
            out[key.replace("_max", "")][1] = val
            to_kill.append(key)
        elif key.endswith("_min") and key not in col_set:
            out[key.replace("_min", "")][0] = val
            to_kill.append(key)
    # remove keys with min/max suffix
    for key in to_kill:
        kwargs.pop(key, None)
    return out


def _add_range_query(kwargs, df):
    """
    Add a range query that spans two columns.

    For example, if columns 'time_min' and 'time_max' exist but 'time'
    does not, time=(time_1, time_2) will filter df to only include columns
    which have a range in specified time.
    """
    col_set = set(df)
    unknown_cols = set(kwargs) - col_set
    bad_keys = set()
    range_query = {}
    for key in unknown_cols:
        min_key, max_key = f"{key}_min", f"{key}_max"
        val = kwargs[key]
        if {min_key, max_key}.issubset(col_set) and len(val) == 2:
            range_query[key] = val
            kwargs.pop(key, None)
        else:
            bad_keys.add(key)
    if len(bad_keys):
        msg = f"columns: {bad_keys} are not found in df"
        raise ValueError(msg)
    return range_query


def _get_flat_and_collection_queries(kwargs):
    """Divide kwargs into flat and sequence queries."""
    flat_query = {
        k: v
        for k, v in kwargs.items()
        if isinstance(v, str) or not isinstance(v, Collection)
    }
    sequence_query = {
        k: v for k, v in kwargs.items() if k not in flat_query and v is not None
    }
    return flat_query, sequence_query


def _filter_equality(query_dict, df, bool_index):
    """Filter based on equality checks."""
    # filter on non-collection queries
    for key, val in query_dict.items():
        if isinstance(val, str):
            regex = get_regex(val)
            new = df[key].str.match(regex).values
            bool_index = np.logical_and(bool_index, new)
        else:
            new = (df[key] == val).values
            bool_index = np.logical_and(bool_index, new)
    return bool_index


def _filter_contains(query_dict, df, bool_index):
    """Filter based on rows containing specified values."""
    for key, val in query_dict.items():
        bool_index = np.logical_and(bool_index, df[key].isin(val))
    return bool_index


def _filter_range(query_dict, df, bool_index):
    """Filter based on ranges for columns."""
    for key, (min_val, max_val) in query_dict.items():
        col = df[key]
        if min_val is not None:
            bool_index = np.logical_and(bool_index, col >= min_val)
        if max_val is not None:
            bool_index = np.logical_and(bool_index, col <= max_val)
    return bool_index


def _filter_multicolumn_range(query_dict, df, bool_index):
    """Filter based on inclusive ranges in multiple columns."""
    for key, val in query_dict.items():
        min_key, max_key = f"{key}_min", f"{key}_max"
        min_col, max_col = df[min_key], df[max_key]
        if val[0] is not None:
            max_too_big = max_col < val[0]
            bool_index = np.logical_and(~max_too_big, bool_index)
        if val[1] is not None:
            min_too_small = min_col > val[1]
            bool_index = np.logical_and(~min_too_small, bool_index)
    return bool_index


def _convert_times(df, some_dict):
    """Convert query values to datetime/timedelta values."""
    if not some_dict:
        return some_dict
    # convert queries related to datetime into datetime64
    datetime_cols = set(df.select_dtypes(include=np.datetime64).columns)
    non_min_max_cols = {x.replace("_min", "") for x in datetime_cols}
    datetime_keys = (datetime_cols & set(some_dict)) | (
        non_min_max_cols & set(some_dict)
    )
    for key in datetime_keys:
        some_dict[key] = to_datetime64(some_dict[key])
    # convert queries related to time delta into timedelta64
    timedelta_cols = set(df.select_dtypes(include=np.timedelta64).columns)
    timedelta_keys = timedelta_cols & set(some_dict)
    for key in timedelta_keys:
        some_dict[key] = to_timedelta64(some_dict[key])
    return some_dict


def filter_df(df: pd.DataFrame, **kwargs) -> np.array:
    """
    Determine if each row of the index meets some filter requirements.

    Parameters
    ----------
    df
        The input dataframe.
    kwargs
        kwargs are used to filter columns.

        Any condition to check against columns of df. Can be a single value
        or a collection of values (to check isin on columns). Str arguments
        can also use unix style matching. Additionally, queries of the form
        {column_name}_min or {column_name}_max can be used.

    Returns
    -------
    A boolean array of the same len as df indicating if each row meets the
    requirements.
    """
    min_max_query = _convert_times(df, _get_min_max(kwargs, df))
    multicolumn_range_query = _convert_times(df, _add_range_query(kwargs, df))
    equality_query, collection_query = _get_flat_and_collection_queries(kwargs)
    # get a blank index of True for filters
    bool_index = np.ones(len(df), dtype=bool)
    # filter on non-collection queries
    bool_index = _filter_equality(equality_query, df, bool_index)
    # filter on collection queries using isin
    bool_index = _filter_contains(collection_query, df, bool_index)
    # filter based on min/max query
    bool_index = _filter_range(min_max_query, df, bool_index)
    # filter based on ranges
    bool_index = _filter_multicolumn_range(multicolumn_range_query, df, bool_index)
    return bool_index
