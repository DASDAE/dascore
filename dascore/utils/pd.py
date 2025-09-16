"""Pandas utilities."""

from __future__ import annotations

import fnmatch
import os
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from functools import cache

import numpy as np
import pandas as pd
from pydantic import BaseModel

import dascore as dc
from dascore.constants import PatchType
from dascore.core.attrs import PatchAttrs
from dascore.exceptions import ParameterError
from dascore.utils.misc import sanitize_range_param
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
    assert not series.empty, "Series must be non-empty"
    unix_paths = series.str.replace(os.sep, "/")
    unix_base_path = (str(base) + "/").replace(os.sep, "/")
    out = unix_paths.str.replace(unix_base_path, "", regex=False)
    return out


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
        val = None if val is ... else val  # handle ...
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


def split_df_query(kwargs, df, ignore_bad_kwargs=False):
    """
    Split kwargs into normal, range, and unsupported kwargs.

    Normal query kwargs are the ones that apply directly to a single column.
    Range kwargs specify a range and the df must have {name}_min, {name}_max
    unsupported kwargs are the keys in kwargs that don't meet these reqs.

    For example, if columns 'time_min' and 'time_max' exist but 'time'
    does not, time=(time_1, time_2) will filter df to only include columns
    which have a range in specified time.
    """
    col_set = set(df)
    unknown_cols = set(kwargs) - col_set
    unsupported = {}
    range_query = {}
    out = dict(kwargs)
    for key in unknown_cols:
        min_key, max_key = f"{key}_min", f"{key}_max"
        val = kwargs[key]
        subset = {min_key, max_key}.issubset(col_set)
        if subset and val is not None and len(val) == 2:
            # handles ... as None.
            new_val = [None if x is ... else x for x in val]
            range_query[key] = tuple(new_val)
            out.pop(key, None)
        # If this is an empty range query just pop out key.
        elif val is None:
            out.pop(key, None)
        else:
            unsupported[key] = val
    # raise if bad keys are found and not ignored.
    if len(unsupported) and not ignore_bad_kwargs:
        bad_dict = {x: kwargs[x] for x in unsupported}
        msg = (
            "Bad filter parameter found. Either the column does not "
            f"exist or it's value is invalid. Keys/values are: {bad_dict}"
        )
        raise ParameterError(msg)
    # otherwise just pop out unsupported kwargs
    for key in unsupported:
        out.pop(key, None)
    return out, range_query, unsupported


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
        # remove null values in either end of query
        not_null = ~(pd.isnull(df[min_key]) | pd.isnull(df[max_key]))
        bool_index = np.logical_and(bool_index, not_null)

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


def get_interval_columns(df, name, arrays=False):
    """
    Return a series of start, stop, step for columns.

    Parameters
    ----------
    df
        The input dataframe.
    name
        The name of the coordinate (eg time).
    arrays
        If True, return output as numpy arrays, else pandas series.
    """
    names = f"{name}_min", f"{name}_max", f"{name}_step"
    missing_cols = set(names) - set(df.columns)
    if missing_cols:
        dims = get_dim_names_from_columns(df)
        msg = (
            f"Cannot chunk spool or dataframe on {missing_cols}, "
            f"valid dimensions or columns to chunk on are {dims}"
        )
        raise ParameterError(msg)
    start, stop, step = df[names[0]], df[names[1]], df[names[2]]
    if not arrays:
        return start, stop, step
    else:
        return start.values, stop.values, step.values


def yield_range_tuple_from_kwargs(df, kwargs) -> tuple[str, slice]:
    """
    For each slice keyword, yield the name and a tuple of (start, stop).

    Will also convert values based on dtypes in dataframe, eg
    time=(1, 10) will convert to
    time=(np.timedelta64(1, 's'), np.timedelta64(10, 's')) provided columns
    'time_min' and 'time_max' are datetime columns.
    """

    def _maybe_convert_dtype_to_date(range_tuple, name, df):
        """Convert dtypes of slice if needed."""
        datetime_cols = set(df.select_dtypes(include=np.datetime64).columns)
        if {f"{name}_min", f"{name}_max"}.issubset(datetime_cols):
            range_tuple = tuple(
                to_datetime64(x) if x is not None else None for x in range_tuple
            )
        return range_tuple

    # find keys which correspond to column ranges
    col_set = set(df.columns)
    valid_minmax_kwargs = {
        x
        for x in kwargs
        if {f"{x}_min", f"{x}_max"}.issubset(col_set) and x not in col_set
    }
    # ensure exactly one column is found
    for name in valid_minmax_kwargs:
        range_tuple = sanitize_range_param(kwargs[name])
        out = _maybe_convert_dtype_to_date(range_tuple, name, df)
        yield name, out


def adjust_segments(df, ignore_bad_kwargs=False, **kwargs):
    """
    Filter a dataframe and adjust its limits.

    Parameters
    ----------
    df
        The input dataframe
    ignore_bad_kwargs
        Ignore kwargs that don't apply to df, else raise.
    kwargs
        The keyword arguments for filtering.
    """
    # apply filtering, this creates a copy so we *should* be ok to update inplace.
    out = df[filter_df(df, ignore_bad_kwargs=ignore_bad_kwargs, **kwargs)]
    # Track which rows have been modified
    not_modified = ~_column_or_value(out, "_modified", False)
    # find slice kwargs, get series corresponding to interval columns
    for name, (val_min, val_max) in yield_range_tuple_from_kwargs(out, kwargs):
        start, stop, step = get_interval_columns(out, name)
        min_val = val_min if val_min is not None else start.min()
        max_val = val_max if val_max is not None else stop.max()
        too_small = start < min_val
        too_large = stop > max_val
        out.loc[too_large, too_large.name] = max_val
        out.loc[too_small, too_small.name] = min_val
        not_modified &= ~(too_small.values | too_large.values)
    return out.assign(_modified=~not_modified)


def filter_df(df: pd.DataFrame, ignore_bad_kwargs=False, **kwargs) -> np.ndarray:
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
    kwargs, range_query, _ = split_df_query(kwargs, df, ignore_bad_kwargs)
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

    For example, time_min, time_max, time_step would be returned if in dataframe.
    """
    cols = set(df.columns)
    possible_dims = {
        x.replace("_min", "").replace("_max", "").replace("_step", "") for x in cols
    }
    out = {
        x for x in possible_dims if {f"{x}_min", f"{x}_max", f"{x}_step"}.issubset(cols)
    }
    return sorted(out)


def get_column_names_from_dim(dims: Sequence[str]) -> list:
    """Get column names from a sequence of dimensions."""
    out = []
    for name in dims:
        out.append(f"{name}_min")
        out.append(f"{name}_max")
        out.append(f"{name}_step")
    return out


def fill_defaults_from_pydantic(df, base_model: type[BaseModel]):
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
    fields = base_model.model_fields
    missing = set(fields) - set(df.columns)
    required = {x for x in missing if fields[x].is_required()}
    if any(required):
        msg = f"Missing required value: {required}"
        raise ValueError(msg)
    fill = {x: fields[x].default for x in missing}
    return df.assign(**fill)


def list_ser_to_str(ser: pd.Series) -> pd.Series:
    """Convert a column of str sequences to a string with commas separating values."""
    values = [",".join(x) if not isinstance(x, str) else x for x in ser.values]
    return pd.Series(values, index=ser.index, dtype=object)


def _model_list_to_df(mod_list: Sequence[dc.PatchAttrs], exclude=None) -> pd.DataFrame:
    """
    Get a dataframe from a sequence of pydantic models.

    Optionally, exclude certain columns
    """
    df = pd.DataFrame([x.flat_dump(exclude=exclude) for x in mod_list])
    if "dims" in df.columns:
        df["dims"] = list_ser_to_str(df["dims"])
    return df


def _remove_overlaps(df, name):
    """.
    Remove overlaps in col, where col_min and col_max should exist in df.

    Assumes df is sorted.
    """

    def _get_step_values(df, step_name):
        array = np.roll(df[step_name].values, 1)
        isna = pd.isnull(array)
        zeros = np.zeros_like(array, dtype=array.dtype)
        return np.where(~isna, array, zeros)

    def _get_correct_starts(start, stop, df, step_name):
        step = _get_step_values(df, step_name)
        stop_roll = np.roll(stop, 1)
        start_lt_stop = start <= stop_roll
        start_lt_stop[0] = False  # remove roll artifact; [0] should be start[0]
        adjusted = stop_roll + step
        return np.where(start_lt_stop, adjusted, start)

    min_name, max_name, step_name = f"{name}_min", f"{name}_max", f"{name}_step"
    assert df[min_name].is_monotonic_increasing, "df must be sorted"
    start = df[min_name].values
    stop = df[max_name].values
    # Add step to stop roll so we don't get 1 sample of overlap
    corrected_starts = _get_correct_starts(start, stop, df, step_name)
    # wrap around in roll gives wrong start value, correct it.
    corrected_starts[0] = start[0]
    old_modified = _column_or_value(df, "_modified", False)
    _modified = old_modified | (corrected_starts != start)
    return df.assign(**{min_name: corrected_starts, "_modified": _modified})


def _column_or_value(df, col, value):
    """
    Return column values if present; else a numpy array broadcast of
    `value` to len(df).
    """
    if col in df.columns:
        return df[col].values
    out = np.broadcast_to(np.array(value), len(df))
    return out


def _instructions_modified(instruct_df, sub_source):
    """
    Determine if the instruction df columns are the same as the source.

    This is useful for determining which patches need select arguments.
    """
    # Get the source and desired output dfs broadcast together.
    names = set(sub_source.columns) & set(instruct_df.columns)
    source = sub_source.loc[instruct_df["source_index"].values]
    # not_modified = np.ones(len(instruct_df), dtype=bool)
    not_modified = ~_column_or_value(source, "_modified", False)
    for name in names:
        val1, val2 = source[name].values, instruct_df[name].values
        eq = val1 == val2
        null = pd.isnull(val1) & pd.isnull(val2)
        not_modified &= eq | null
    modified = ~not_modified
    return modified


def patch_to_dataframe(patch: PatchType) -> pd.DataFrame:
    """
    Convert a patch to a dataframe.

    Parameters
    ----------
    patch
        The input patch to convert.

    Notes
    -----
    - Patch attributes are attached to the experimental dataframe attribute
      called "attrs" as a dictionary
    """
    dims = patch.dims
    # ensure a 2D patch is passed
    assert (
        len(dims) == 2
    ), "Patch must have exactly 2 dimensions to convert to dataframe"
    # get arrays with dimensional values
    index_values = patch.get_coord(dims[0]).values
    col_values = patch.get_coord(dims[1]).values
    # create dataframe
    df = pd.DataFrame(patch.data, index=index_values, columns=col_values)
    # assign index names and attrs
    df.attrs = patch.attrs.model_dump()
    df.index.name = dims[0]
    df.columns.name = dims[1]
    return df


def dataframe_to_patch(
    df: pd.DataFrame, attrs: PatchAttrs | Mapping | None = None
) -> PatchType:
    """
    Convert a dataframe to a patch.

    Dimension names are either taken as the names of the index and columns or
    they must be provided in the attrs argument.

    Parameters
    ----------
    df
        The input dataframe to convert to a patch
    attrs
        Extra attributes to attach to the patch.
    """

    def _get_column_names(df, attrs):
        """Get columns names from dataframe or index."""
        dims = (df.index.name, df.columns.name)
        invalid_df_dims = any(x is None or x == "" for x in dims)
        if attrs is not None and invalid_df_dims:
            dims = attrs.get("dims", (None, None))
        if any(x is None or x == "" for x in dims):
            msg = (
                "Dimension names not found. Both columns and index must have "
                "a name or attrs must specify dimensions."
            )
            raise ValueError(msg)
        return dims

    # get data
    data = df.to_numpy()
    dims = _get_column_names(df, attrs)
    coords = {dims[0]: df.index.to_numpy(), dims[1]: df.columns.to_numpy()}
    return dc.Patch(data=data, dims=dims, coords=coords, attrs=attrs)


def rolling_df(df, window, step=None, axis=0, center=False):
    """
    A simple wrapper around pandas rolling to handle deprecated axis.

    See pandas.DataFrame.rolling for more details of arguments.
    """
    df = df if not axis else df.T  # silly deprecated axis argument.
    return df.rolling(window=window, step=step, center=center)
