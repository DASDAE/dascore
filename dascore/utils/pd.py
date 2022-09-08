"""
Pandas utilities.
"""
import fnmatch
import os
from collections import defaultdict
from functools import cache
from typing import Collection, Sequence, Tuple, Type

import numpy as np
import pandas as pd
from pydantic import BaseModel

from dascore.utils.time import to_datetime64, to_timedelta64


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


def _get_min_max_query(kwargs, df):
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


def _add_range_query(kwargs, df, ignore_bad_kwargs=False):
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
        if not ignore_bad_kwargs:
            msg = f"columns: {bad_keys} are not found in df"
            raise KeyError(msg)
        else:
            for key in bad_keys:
                kwargs.pop(key, None)
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


def get_interval_columns(df, name):
    """Return a series of start, stop, step for columns."""
    names = f"{name}_min", f"{name}_max", f"d_{name}"
    missing_cols = set(names) - set(df.columns)
    if missing_cols:
        msg = f"Dataframe is missing {missing_cols} to chunk on {name}"
        raise KeyError(msg)
    return df[names[0]], df[names[1]], df[names[2]]


def yield_slice_from_kwargs(df, kwargs) -> Tuple[str, slice]:
    """
    For each slice keyword, yield the name and slice.

    Will also convert slice values based on dtypes in dataframe, eg
    time=(1, 10) will convert to
    time=(np.timedelta64(1, 's'), np.timedelta64(10, 's')) provided columns
    'time_min' and 'time_max' are datetime columns.
    """

    def _get_slice(value):
        """Ensure the value can rep. a slice."""
        assert isinstance(value, (slice, Sequence)) and len(value) == 2
        if not isinstance(value, slice):
            value = slice(*value)
        return value

    def _maybe_convert_dtype(sli, name, df):
        """Convert dtypes of slice if needed."""
        datetime_cols = set(df.select_dtypes(include=np.datetime64).columns)
        if {f"{name}_min", f"{name}_max"}.issubset(datetime_cols):
            sli = slice(
                to_datetime64(sli.start) if sli.start is not None else None,
                to_datetime64(sli.stop) if sli.stop is not None else None,
                to_timedelta64(sli.step) if sli.step is not None else None,
            )
        return sli

    # find keys which correspond to column ranges
    col_set = set(df.columns)
    valid_minmax_kwargs = {
        x
        for x in kwargs
        if {f"{x}_min", f"{x}_max"}.issubset(col_set) and x not in col_set
    }
    # ensure exactly one column is found
    for name in valid_minmax_kwargs:
        out_slice = _maybe_convert_dtype(_get_slice(kwargs[name]), name, df)
        yield name, out_slice


def adjust_segments(df, ignore_bad_kwargs=False, **kwargs):
    """
    Filter a dataframe and adjust its limits.

    Parameters
    ----------
    df
        The input dataframe
    ignore_bad_kwargs
        Ignore kwargs that dont apply to df, else raise.
    kwargs
        The keyword arguments for filtering.
    """
    # apply filtering, this creates a copy so we *should* be ok to update inplace.
    out = df[filter_df(df, ignore_bad_kwargs=ignore_bad_kwargs, **kwargs)]
    # find slice kwargs, get series corresponding to interval columns
    for (name, qs) in yield_slice_from_kwargs(out, kwargs):
        start, stop, step = get_interval_columns(out, name)
        min_val = qs.start if qs.start is not None else start.min()
        max_val = qs.stop if qs.stop is not None else stop.max()
        too_small = start < min_val
        too_large = stop > max_val
        out.loc[too_large, too_large.name] = max_val
        out.loc[too_small, too_small.name] = min_val
    return out


def filter_df(df: pd.DataFrame, ignore_bad_kwargs=False, **kwargs) -> np.array:
    """
    Determine if each row of the index meets some filter requirements.

    Parameters
    ----------
    df
        The input dataframe.
    ignore_bad_kwargs
        If True, silently drop incompatible kwargs with dataframe.

    kwargs
        Used to filter columns.

        Any condition to check against columns of df. Can be a single value
        or a collection of values (to check isin on columns). Str arguments
        can also use unix style matching. Additionally, queries of the form
        {column_name}_min or {column_name}_max can be used, provided columns
        with the same name don't already exist.

    Returns
    -------
    A boolean array of the same len as df indicating if each row meets the
    requirements.
    """
    min_max_query = _convert_times(df, _get_min_max_query(kwargs, df))
    range_query = _add_range_query(kwargs, df, ignore_bad_kwargs)
    multicolumn_range_query = _convert_times(df, range_query)
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


def _convert_min_max_in_kwargs(kwargs, df):
    """
    Convert the min/max values in kwargs to single key form.

    For example, {'time_min': 10, 'time_max': 20} would be converted
    to {'time': (10, 20)}
    """
    out = dict(kwargs)
    minmax = defaultdict(lambda: [None, None])
    col_set = set(df.columns)
    max_kwargs = {x for x in col_set & set(out) if x.endswith("_max")}
    min_kwargs = {x for x in col_set & set(out) if x.endswith("_min")}
    datetime_cols = set(df.select_dtypes(include=np.datetime64).columns)
    iterable = zip([min_kwargs, max_kwargs], ["_min", "_max"], [0, 1])
    for minmax_kwargs, suffix, ind in iterable:
        for key in minmax_kwargs:
            val = out.pop(key)
            if key in datetime_cols:
                val = to_datetime64(val)
            minmax[key.replace(suffix, "")][ind] = val
    out.update(minmax)
    return out


def get_dim_names_from_columns(df: pd.DataFrame) -> list[str]:
    """
    Returns the names of columns which represent and range in the dataframe.

    For example, time_min, time_max, d_time would be returned if in dataframe.
    """
    cols = set(df.columns)
    possible_dims = {
        x.replace("_min", "").replace("_max", "").replace("d_", "") for x in cols
    }
    out = {
        x for x in possible_dims if {f"{x}_min", f"{x}_max", f"d_{x}"}.issubset(cols)
    }
    return sorted(out)


def get_column_names_from_dim(dims: Sequence[str]) -> list:
    """
    Get column names from a sequence of dimensions.
    """
    out = []
    for name in dims:
        out.append(f"{name}_min")
        out.append(f"{name}_max")
        out.append(f"d_{name}")
    return out


def fill_defaults_from_pydantic(df, base_model: Type[BaseModel]):
    """
    Fill missing columns in dataframe with defaults from base_model.

    If the missing column has no default value, raise ValueError.

    Parameters
    ----------
    df
        A dataframe
    base_model
        A pydantic BaseModel
    """
    fields = base_model.__fields__
    missing = set(fields) - set(df.columns)
    required = {x for x in missing if fields[x].required}
    if any(required):
        msg = f"Missing required value: {required}"
        raise ValueError(msg)
    fill = {x: fields[x].default for x in missing}
    return df.assign(**fill)


def list_ser_to_str(ser: pd.Series) -> pd.Series:
    """
    Convert a column of str sequences to a string with commas separating values.
    """
    values = [",".join(x) if not isinstance(x, str) else x for x in ser.values]
    return pd.Series(values, index=ser.index, dtype=object)
