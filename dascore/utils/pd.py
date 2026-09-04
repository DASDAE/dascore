"""Pandas utilities."""

from __future__ import annotations

import fnmatch
from collections import defaultdict
from collections.abc import Collection, Generator, Iterator, Mapping
from functools import cache
from typing import TypeVar, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel

import dascore as dc
from dascore.constants import PatchType, namespace_select_type
from dascore.core.attrs import PatchAttrs
from dascore.exceptions import InvalidSpoolQueryError, ParameterError
from dascore.utils.misc import is_range, order_range_tuple, sanitize_range_param
from dascore.utils.time import to_datetime64, to_timedelta64

_RowType = TypeVar("_RowType")


def iter_rows(df: pd.DataFrame, row_type: type[_RowType]) -> Iterator[_RowType]:
    """
    Iterate over a dataframe's rows as named tuples of a known shape.

    Parameters
    ----------
    df
        The dataframe to iterate.
    row_type
        A NamedTuple declaring the columns the caller reads. Pandas builds
        the row tuple dynamically, so this only names the shape for
        readers (and type checkers); it is never instantiated. The frame's
        index is left out so the declared fields line up with the row's.
    """
    return cast(Iterator[_RowType], df.itertuples(index=False))


def present_units_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expose private ``_{name}_units`` columns under public names.

    Coordinate envelopes are stored and presented in each coordinate's
    original units, so the unit belongs beside the values it scales:
    ``distance_min`` of 65.6 is self-explaining only with
    ``distance_units`` of ``ft`` in view. The private spellings exist
    for the planners (public columns are conflict-policed on merge), so
    only the presented frame renames them.

    An existing public column is never overwritten. For frames the index
    builds this cannot happen: an attr whose name would claim a
    coordinate's public units column is omitted from the flat view with
    a warning, the same rule that protects envelope columns. A frame
    assembled some other way keeps whatever it already had, and its
    private column simply stays private rather than silently replacing
    a value this function cannot vouch for.
    """
    renames = {}
    for col in df.columns:
        name = str(col)
        if not (name.startswith("_") and name.endswith("_units")):
            continue
        public = name[1:]
        if public not in df.columns:
            renames[col] = public
    return df.rename(columns=renames) if renames else df


def _present_private_column(df: pd.DataFrame, private: str) -> pd.DataFrame:
    """Rename one private column to its public spelling, if it is free."""
    public = private[1:]
    if private not in df.columns or public in df.columns:
        return df
    return df.rename(columns={private: public})


def present_dtype_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expose ``_dtype`` as ``dtype``.

    The element type of a patch's data is private in the relation
    because chunk groups and polices every public column, and patches of
    different element types must still merge. It is worth showing: with
    ``data_size`` it is what the patch costs to load.
    """
    return _present_private_column(df, "_dtype")


def present_data_size_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expose ``_data_size`` as ``data_size``.

    The sample count is private for the same reason ``_dtype`` is, and
    more sharply: patches of different lengths are the ordinary case,
    and a public column would keep chunk from merging any of them.

    A row states no size when it does not know one: a merged or
    subdivided chunk output, or a patch a selection trims. The column
    follows the index's convention
    of staying nullable (Int64) only when it holds nulls; a column of
    nothing else arrives from SQL untyped.
    """
    out = _present_private_column(df, "_data_size")
    if "data_size" not in out.columns or out is df:
        return out
    sizes = out["data_size"]
    return out.assign(data_size=sizes.astype("Int64" if sizes.isna().any() else int))


def drop_private_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every remaining private (leading underscore) column."""
    private = [col for col in df.columns if str(col).startswith("_")]
    return df.drop(columns=private) if private else df


# What `present_columns` runs, in order. Each step but the last gives one
# private column (or family) back its public spelling; dropping the rest
# is always last, so a private column no step claims simply does not
# leave. Add a step here to publish another one -- and only when a caller
# can act on it, since every column added is one more to read past.
PRESENTERS = (
    present_units_columns,
    present_dtype_column,
    present_data_size_column,
    drop_private_columns,
)


def present_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the public view of a flat relation.

    The relation carries private columns the spool needs and a caller
    does not: the row id, per-coordinate identity keys, the raw path
    attrs. A leading underscore means private everywhere else in
    DASCore, so a frame handed out publicly should not carry them. A few
    are private only because the planners police public columns, and
    those are renamed rather than dropped; `PRESENTERS` is the list.
    """
    for present in PRESENTERS:
        df = present(df)
    return df


@cache
def get_regex(seed_str):
    """Compile, and cache regex for str queries."""
    return fnmatch.translate(seed_str)  # translate to re


def relative_offset(gmin, gmax, value):
    """
    Resolve one relative bound against a global [gmin, gmax] envelope.

    Positive offsets measure from the start, negative from the end;
    None/Ellipsis bounds stay open. Datetime envelopes take numeric
    seconds offsets.
    """
    if value is None or value is Ellipsis:
        return None
    if isinstance(gmin, pd.Timestamp) or isinstance(gmin, np.datetime64):
        delta = to_timedelta64(abs(float(value)))
        return (gmin + delta) if value >= 0 else (gmax - delta)
    return (gmin + value) if value >= 0 else (gmax + value)


def relative_ranges_to_absolute(df, kwargs: dict) -> dict:
    """
    Resolve relative (start, stop) ranges against a frame's global envelopes.

    Operates only on the dataframe's `{name}_min`/`{name}_max` envelope
    columns, so both the generic dataframe select path and the catalog
    share one relative-select implementation without either depending on
    the index query builder.
    """
    out = {}
    for name, value in kwargs.items():
        lo_col, hi_col = f"{name}_min", f"{name}_max"
        if lo_col not in df.columns or hi_col not in df.columns or df.empty:
            msg = f"Cannot use relative select on {name!r}."
            raise InvalidSpoolQueryError(msg)
        if not is_range(value):
            # same vocabulary as the catalog path's selector shaping
            msg = (
                f"relative=True accepts range selectors only (a (start, stop) "
                f"tuple or slice), got {value!r}."
            )
            raise InvalidSpoolQueryError(msg)
        gmin, gmax = df[lo_col].min(), df[hi_col].max()
        lo, hi = value
        out[name] = (
            relative_offset(gmin, gmax, lo),
            relative_offset(gmin, gmax, hi),
        )
    return out


def normalize_range_forms(value):
    """
    Normalize the patch-level slice range form to a 2-tuple.

    Only slices are converted: bare None/Ellipsis keep their own errors,
    and a fully-open range is rejected downstream as having no usable
    bounds (per the selector spec).
    """
    if isinstance(value, slice):
        return sanitize_range_param(value)
    return value


def resolve_selector_namespaces(
    known_attrs: Collection[str],
    known_coords: Collection[str],
    _attrs: namespace_select_type = None,
    _coords: namespace_select_type = None,
    kwargs: Mapping | None = None,
) -> tuple[dict, dict]:
    """
    Split selector kwargs into (attrs, coords) per the selector spec.

    Bare kwargs resolve against attributes first, then coordinates;
    `_attrs`/`_coords` name their namespace explicitly and validate
    against that side only. Each accepts either a mapping of
    ``name -> selector`` (the fully general form — required when a name
    cannot be a Python keyword, e.g. it collides with a select parameter
    or is not an identifier) or a name/collection of names tagging which
    *bare kwargs* to interpret in that namespace. Unknown names, and
    names supplied in more than one namespace, raise (see #435).

    Both the catalog (which pushes predicates into SQL) and the generic
    dataframe select path resolve names here, so the two agree on which
    names are valid, what a bare name means, and which range forms are
    accepted — the paths differ only in how they *apply* a predicate.
    """

    def _tag_form(spec, kwargs, label):
        """Normalize a tag-form spec (names of bare kwargs) to a dict."""
        if spec is None or isinstance(spec, Mapping):
            return spec, kwargs
        names = [spec] if isinstance(spec, str) else list(spec)
        if not all(isinstance(n, str) for n in names):
            msg = (
                f"{label} must be a mapping of name -> selector, or a "
                "name/collection of names tagging bare keyword arguments."
            )
            raise InvalidSpoolQueryError(msg)
        kwargs = dict(kwargs or {})
        out = {}
        for n in names:
            if n not in kwargs:
                msg = f"{label}={n!r} names no bare keyword argument."
                raise InvalidSpoolQueryError(msg)
            out[n] = kwargs.pop(n)
        return out, kwargs

    _attrs, kwargs = _tag_form(_attrs, kwargs, "_attrs")
    _coords, kwargs = _tag_form(_coords, kwargs, "_coords")
    known_attrs, known_coords = set(known_attrs), set(known_coords)
    # A name in both explicit namespaces is a caller error whether or not
    # it is valid in either, so this precedes the membership checks.
    if duplicates := set(_attrs or {}) & set(_coords or {}):
        names = ", ".join(repr(x) for x in sorted(duplicates))
        raise InvalidSpoolQueryError(f"{names} given in both _attrs and _coords.")
    attrs: dict = {}
    coords: dict = {}
    for items, allowed, out, noun in (
        (_attrs, known_attrs, attrs, "an attribute"),
        (_coords, known_coords, coords, "a coordinate"),
    ):
        for name, value in (items or {}).items():
            if name not in allowed:
                msg = f"{name!r} is not {noun} of this spool."
                raise InvalidSpoolQueryError(msg)
            out[name] = normalize_range_forms(value)
    for name, value in (kwargs or {}).items():
        if name in attrs or name in coords:
            msg = f"{name!r} given as both a bare kwarg and in _attrs/_coords."
            raise InvalidSpoolQueryError(msg)
        value = normalize_range_forms(value)
        if name in known_attrs:
            attrs[name] = value
        elif name in known_coords:
            coords[name] = value
        else:
            msg = (
                f"{name!r} is neither an attribute nor a coordinate of this "
                f"spool. Attributes: {sorted(known_attrs)}; "
                f"coordinates: {sorted(known_coords)}."
            )
            raise InvalidSpoolQueryError(msg)
    return attrs, coords


def selector_spec_names(spec: namespace_select_type) -> set[str]:
    """
    Return the names an `_attrs`/`_coords` argument designates.

    Either form: a mapping of name -> selector, or a name (or names)
    tagging bare kwargs. A malformed spec keeps only what is a name, so
    `resolve_selector_namespaces` is left to complain about the rest
    properly.
    """
    if spec is None:
        return set()
    if isinstance(spec, str):
        return {spec}
    return {x for x in spec if isinstance(x, str)}


def requested_selector_names(_attrs, _coords, kwargs) -> set[str]:
    """
    Return every name a selection call names, whatever form it arrives in.

    A tag-form `_attrs`/`_coords` names bare kwargs, so a requested name
    is always either a bare kwarg or a key of a mapping form.
    """
    out = set(kwargs)
    for spec in (_attrs, _coords):
        if isinstance(spec, Mapping):
            out |= set(spec)
    return out


def drop_selector_names(spec: namespace_select_type, names) -> namespace_select_type:
    """
    Return an `_attrs`/`_coords` argument with some names taken out.

    Callers use this to peel off the names another namespace answers
    (e.g. coordinates an inventory defines along the fiber) — and the tag
    form as well as the mapping one, since a tag left behind designates a
    bare keyword which went with it.
    """
    if not names or spec is None:
        return spec
    if isinstance(spec, Mapping):
        return {str(k): v for k, v in spec.items() if k not in names}
    if isinstance(spec, str):
        return None if spec in names else spec
    kept = [x for x in spec if x not in names]
    return kept or None


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
        # Claim a min/max key only when the bare column exists; otherwise
        # leave it for the bad-kwarg policy (claiming it used to KeyError
        # deep in the range filter under its bare name).
        if key.endswith("_max") and key not in col_set:
            if key.removesuffix("_max") not in col_set:
                continue
            out[key.removesuffix("_max")][1] = val
            to_kill.append(key)
        elif key.endswith("_min") and key not in col_set:
            if key.removesuffix("_min") not in col_set:
                continue
            out[key.removesuffix("_min")][0] = val
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


def _check_misdirected_range_query(key, val, df):
    """
    Raise if a range query was aimed at an interval column.

    Columns like time_min/time_max hold the limits of each row, and a
    collection of values applied to one is a membership (isin) check. An open
    bound (... or None) in such a collection is meaningless, and signals a
    range query which belongs on the dimension instead, eg
    time_min=(t1, ...) should be time=(t1, ...). Without this the query would
    silently match nothing.
    """
    base = key[:-4] if key.endswith(("_min", "_max")) else None
    if base is None or not {f"{base}_min", f"{base}_max"}.issubset(set(df.columns)):
        return
    # Only a two element sequence can be a range. Anything else is a
    # membership check, where None may be a legitimate value to match.
    if len(val) != 2 or not any(x is ... or x is None for x in val):
        return
    msg = (
        f"An open bound (... or None) is not valid in the query for column "
        f"'{key}'; a collection of values is an isin check, not a range. "
        f"Use {base}=(min, max) to query a range of {base} values."
    )
    raise ParameterError(msg)


def _filter_contains(query_dict, df, bool_index):
    """Filter based on rows containing specified values."""
    for key, val in query_dict.items():
        _check_misdirected_range_query(key, val, df)
        # An ellipsis names no value, so it matches nothing and is dropped
        # before `isin` sees it: pandas' arrow-backed string columns refuse a
        # value arrow has no type for, where its numpy-backed ones quietly
        # never match. The range check above still sees it -- an open bound
        # in a membership query is what that check exists to name.
        wanted = [x for x in val if x is not ...]
        bool_index = np.logical_and(bool_index, df[key].isin(wanted))
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


def _convert_range_bounds(range_tuple, func):
    """
    Apply a time conversion to each bound of a range.

    Unbounded (None) ends are left alone; converting them would produce NaT,
    which compares False against everything and would silently empty the query.
    """
    return tuple(None if x is None else func(x) for x in range_tuple)


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
        some_dict[key] = _convert_range_bounds(some_dict[key], to_datetime64)
    # convert queries related to time delta into timedelta64
    timedelta_cols = set(df.select_dtypes(include=np.timedelta64).columns)
    timedelta_keys = timedelta_cols & set(some_dict)
    for key in timedelta_keys:
        some_dict[key] = _convert_range_bounds(some_dict[key], to_timedelta64)
    return some_dict


def get_interval_columns(df, name):
    """
    Return a series of start, stop, step for columns.

    Parameters
    ----------
    df
        The input dataframe.
    name
        The name of the coordinate (eg time).
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
    return start, stop, step


def yield_range_tuple_from_kwargs(df, kwargs) -> Generator[tuple[str, tuple]]:
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
    for name, range_tuple in yield_range_tuple_from_kwargs(out, kwargs):
        val_min, val_max = order_range_tuple(range_tuple)
        start, stop, _ = get_interval_columns(out, name)
        min_val = val_min if val_min is not None else start.min()
        max_val = val_max if val_max is not None else stop.max()
        too_small = start < min_val
        too_large = stop > max_val
        out.loc[too_large, too_large.name] = max_val
        out.loc[too_small, too_small.name] = min_val
        not_modified &= ~(too_small.values | too_large.values)
    return out.assign(_modified=~not_modified)


def filter_df(
    df: pd.DataFrame, ignore_bad_kwargs=False, **kwargs
) -> np.ndarray | pd.Series:
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
    A boolean mask of the same len as df indicating if each row meets the
    requirements. Whether it comes back as a bare array or a Series
    depends on which queries applied, so treat it as an opaque boolean
    container rather than relying on either.
    """
    min_max_query = _convert_times(df, _get_min_max_query(kwargs, df))
    kwargs, range_query, _ = split_df_query(kwargs, df, ignore_bad_kwargs)
    multicolumn_range_query = _convert_times(df, range_query)
    multicolumn_range_query = {
        key: order_range_tuple(val) for key, val in multicolumn_range_query.items()
    }
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

    def _is_null(val) -> bool:
        if pd.api.types.is_scalar(val):
            return bool(pd.isnull(val))
        if isinstance(val, (list | tuple | np.ndarray)):
            return bool(pd.isnull(np.asarray(val, dtype=object)).any())
        null = pd.isnull(val)
        return bool(null.any()) if isinstance(null, np.ndarray) else bool(null)

    values = [
        ""
        if _is_null(x)
        else ",".join(str(item) for item in x)
        if isinstance(x, (list | tuple | np.ndarray))
        else str(x)
        for x in ser.values
    ]
    return pd.Series(values, index=ser.index, dtype=object)


def _column_or_value(df, col, value):
    """
    Return column values if present; else a numpy array broadcast of
    `value` to len(df).
    """
    if col in df.columns:
        return df[col].values
    out = np.broadcast_to(np.array(value), len(df))
    return out


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
    assert len(dims) == 2, (
        "Patch must have exactly 2 dimensions to convert to dataframe"
    )
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
) -> dc.Patch:
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
    if isinstance(attrs, Mapping):
        attrs = dict(attrs)
        attrs.pop("dims", None)
    coords = {dims[0]: df.index.to_numpy(), dims[1]: df.columns.to_numpy()}
    return dc.Patch(data=data, dims=dims, coords=coords, attrs=attrs)


def rolling_df(df, window, step=None, axis=0, center=False):
    """
    A simple wrapper around pandas rolling to handle deprecated axis.

    See pandas.DataFrame.rolling for more details of arguments.
    """
    df = df if not axis else df.T  # silly deprecated axis argument.
    return df.rolling(window=window, step=step, center=center)
