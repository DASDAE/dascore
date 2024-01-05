"""Utilities for working with the Patch class."""
from __future__ import annotations

import functools
import inspect
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

import dascore as dc
from dascore.constants import FLOAT_PRECISION, PatchType, SpoolType, dascore_styles
from dascore.core.attrs import combine_patch_attrs
from dascore.core.coordmanager import merge_coord_managers
from dascore.exceptions import (
    CoordDataError,
    PatchAttributeError,
    PatchDimError,
)
from dascore.units import get_quantity
from dascore.utils.misc import all_diffs_close_enough, get_middle_value, iterate
from dascore.utils.time import to_float

attr_type = dict[str, Any] | str | Sequence[str] | None


def _format_values(val):
    """String formatting for values for history string."""
    if isinstance(val, list | tuple):
        out = ",".join(_format_values(x) for x in val)
        out = f"({out})" if isinstance(val, tuple) else f"[{out}]"
    elif isinstance(val, np.ndarray):
        # make sure numpy strings arent too long!
        out = np.array2string(
            val,
            precision=FLOAT_PRECISION,
            threshold=dascore_styles["patch_history_array_threshold"],
        )
    else:
        out = str(val)
    return out


def _func_and_kwargs_str(func: Callable, patch, *args, **kwargs) -> str:
    """Get a str rep of the function and input args."""
    callargs = inspect.getcallargs(func, patch, *args, **kwargs)
    callargs.pop("patch", None)
    callargs.pop("self", None)
    kwargs_ = callargs.pop("kwargs", {})
    arguments = []
    arguments += [
        f"{k}={_format_values(v)!r}" for k, v in callargs.items() if v is not None
    ]
    arguments += [
        f"{k}={_format_values(v)!r}" for k, v in kwargs_.items() if v is not None
    ]
    arguments.sort()
    out = f"{func.__name__}("
    if arguments:
        out += f"{','.join(arguments)}"
    return out + ")"


def _get_history_str(patch: PatchType, func, *args, _history="full", **kwargs) -> str:
    """
    Log history of a function being called on a Patch object.

    Parameters
    ----------
    patch
        The patch which will track history.
    func
        A callable which takes the patch as the first argument.
    *args
        The arguments passed to the function.
    _history
        String specifying how the history is to be recorded.
    **kwargs
        kwargs for func.
    """
    if _history is None:
        return None
    if _history == "full":
        history_str = _func_and_kwargs_str(func, patch, *args, **kwargs)
    else:
        history_str = str(func.__name__)
    return history_str


def check_patch_dims(
    patch: PatchType, required_dims: tuple[str, ...] | None
) -> PatchType:
    """
    Check if a patch object has required dimensions, else raise PatchDimError.

    Parameters
    ----------
    patch
        The input patch
    required_dims
        A tuple of required dimensions.
    """
    if required_dims is None:
        return patch
    dim_set = set(patch.dims)
    missing_dims = set(required_dims) - set(patch.dims)
    if not set(required_dims).issubset(dim_set):
        msg = f"patch is missing required dims: {tuple(missing_dims)}"
        raise PatchDimError(msg)
    return patch


def check_patch_attrs(patch: PatchType, required_attrs: attr_type) -> PatchType:
    """
    Check for expected attributes.

    Parameters
    ----------
    patch
        The patch to validate
    required_attrs
        The expected attrs. Can be a sequence or mapping. If sequence, only
        check that attrs exist. If mapping also check that values are equal.
    """
    if required_attrs is None:
        return patch
    # test that patch attr mapping is equal
    patch_attrs = patch.attrs
    if isinstance(required_attrs, Mapping):
        sub = {i: patch_attrs[i] for i in required_attrs}
        if sub != dict(required_attrs):
            msg = f"Patch's attrs {sub} are not required attrs: {required_attrs}"
            raise PatchAttributeError(msg)
    else:
        missing = set(required_attrs) - set(dict(patch.attrs))
        if missing:
            msg = f"Patch is missing the following attributes: {missing}"
            raise PatchAttributeError(msg)
    return patch


def patch_function(
    required_dims: tuple[str, ...] | Callable | None = None,
    required_attrs: attr_type = None,
    history: Literal["full", "method_name", None] = "full",
):
    """
    Decorator to mark a function as a patch method.

    Parameters
    ----------
    required_dims
        A tuple of dimensions which must be found in the Patch.
    required_attrs
        A dict of attributes which must be found in the Patch and whose
        values must be equal to those provided.
    history
        Specifies how to track history on Patch.
            Full - Records function name and str version of input arguments.
            method_name - Only records method name. Useful if args are long.
            None - Function call is not recorded in history attribute.

    Examples
    --------
    1. A patch method which requires dimensions (time, distance)
    >>> import dascore as dc
    >>> @dc.patch_function(required_dims=('time', 'distance'))
    ... def do_something(patch):
    ...     ...   # raises a PatchCoordsError if patch doesn't have time,
    ...     #  distance

    2. A patch method which requires an attribute 'data_type' == 'DAS'
    >>> import dascore as dc
    >>> @dc.patch_function(required_attrs={'data_type': 'DAS'})
    ... def do_another_thing(patch):
    ...     ...  # raise PatchAttributeError if patch doesn't have attribute
    ...     # called "data_type" or its values is not equal to "DAS".

    Notes
    -----
    The original function can still be accessed with the .func attribute.
    This may be useful for avoiding calling the patch_func machinery
    multiple times from within another patch function.
    """

    def _wrapper(func):
        @functools.wraps(func)
        def _func(patch, *args, **kwargs):
            check_patch_dims(patch, required_dims)
            check_patch_attrs(patch, required_attrs)
            hist_str = _get_history_str(patch, func, *args, _history=history, **kwargs)
            out: PatchType = func(patch, *args, **kwargs)
            # attach history string. Consider something a bit less hacky.
            if hist_str and hasattr(out, "attrs"):
                hist = list(out.attrs.history)
                hist.append(hist_str)
                out = out.update_attrs(history=hist)
            return out

        _func.func = func  # attach original function
        _func.__wrapped__ = func

        return _func

    if callable(required_dims):  # the decorator is used without parens
        return patch_function()(required_dims)

    return _wrapper


def patches_to_df(
    patches: Sequence[PatchType] | SpoolType | pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a dataframe.

    Parameters
    ----------
    patches
        A sequence of :class:`dascore.Patch`

    Returns
    -------
    A dataframe with the attrs of each patch converted to a columns
    plus a field called 'patch' which contains a reference to the patches.
    """
    if hasattr(patches, "_df"):
        df = patches._df
    # Handle spool case
    elif hasattr(patches, "get_contents"):
        df = patches.get_contents()
    elif isinstance(patches, pd.DataFrame):
        df = patches
    else:
        df = pd.DataFrame([x.flat_dump() for x in scan_patches(patches)])
        if df.empty:  # create empty df with appropriate columns
            cols = list(dc.PatchAttrs().model_dump())
            df = pd.DataFrame(columns=cols).assign(patch=None, history=None)
        else:  # else populate with patches and concat history
            history = df["history"].apply(lambda x: ",".join(x))
            df = df.assign(patch=patches, history=history)
    # Ensure history is in df
    if "history" not in df.columns:
        df = df.assign(history="")
    if "patch" not in df.columns:
        df["patch"] = [x for x in patches]
    return df


def merge_patches(
    patches: Sequence[PatchType] | pd.DataFrame | SpoolType,
    dim: str = "time",
    check_history: bool = True,
    tolerance: float = 1.5,
) -> Sequence[PatchType]:
    """
    Merge all compatible patches in spool or patch list together.

    Parameters
    ----------
    patches
        A sequence of patches to merge (if compatible)
    dim
        The dimension along which to merge
    check_history
        If True, only merge patches with common history. This will, for
        example, prevent merging filtered and unfiltered data together.
    tolerance
        The upper limit of a gap to tolerate in terms of the sampling
        along the desired dimension. e.g., the default value means any patches
        with gaps <= 1.5 * dt will be merged.
    """
    msg = (
        "merge_patches is deprecated. Use spool.chunk instead. "
        "For example, to merge a list of patches you can use: "
        "dascore.spool(patch_list).chunk(time=None) to merge on the time "
        "dimension."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return dc.spool(patches).chunk(**{dim: None}, tolerance=tolerance)


def _force_patch_merge(patch_dict_list, merge_kwargs, **kwargs):
    """
    Force a merge of the patches along a dimension.

    This function is used in conjunction with `spool.chunk`, which
    does all the compatibility checks beforehand.
    """

    def _get_merge_col(df):
        dims = df["dims"].unique()
        assert len(dims) == 1
        dims = dims[0].split(",")
        dims_vary = pd.Series({x: False for x in dims})
        for dim in dims:
            cols = [f"{dim}_min", f"{dim}_max", f"{dim}_step"]
            vals = df[cols].values
            vals_eq = vals == vals[[0], :]
            vals_null = pd.isnull(vals)
            columns_equal = (vals_eq | vals_null).all(axis=1)
            dims_vary[dim] = not np.all(columns_equal)
        assert dims_vary.sum() <= 1, "Only one dimension can vary for forced merge"
        if not dims_vary.any():  # the case of complete overlap.
            return None
        return dims_vary[dims_vary].index[0]

    def _maybe_step(df, dim):
        """Get the expected step if all steps are close, else None."""
        col = df[f"{dim}_step"].values
        if all_diffs_close_enough(col):
            return get_middle_value(col)
        return None

    def _get_new_coord(df, merge_dim, coords):
        """Get new coordinates, also validate anticipated sampling."""
        new_coord = merge_coord_managers(coords, dim=merge_dim)
        expected_step = _maybe_step(df, merge_dim)
        if not pd.isnull(expected_step):
            new_coord = new_coord.snap(merge_dim)[0]
            # TODO slightly different dt can be produced, let pass for now
            # need to think more about how the merging should work.
        return new_coord

    df = pd.DataFrame(patch_dict_list)
    merge_dim = _get_merge_col(df)
    merge_kwargs = merge_kwargs if merge_kwargs is not None else {}
    if merge_dim is None:  # nothing to merge, complete overlap
        return [patch_dict_list[0]]
    dims = df["dims"].iloc[0].split(",")
    # get patches, ensure they are oriented the same.
    patches = [x.transpose(*dims) for x in df["patch"]]
    axis = patches[0].dims.index(merge_dim)
    # get data, coords, attrs for merging patch together.
    datas = [x.data for x in patches]
    coords = [x.coords for x in patches]
    attrs = [x.attrs for x in patches]
    new_data = np.concatenate(datas, axis=axis)
    new_coord = _get_new_coord(df, merge_dim, coords)
    coord = new_coord.coord_map[merge_dim] if merge_dim in dims else None
    new_attrs = combine_patch_attrs(attrs, merge_dim, coord=coord, **merge_kwargs)
    patch = dc.Patch(data=new_data, coords=new_coord, attrs=new_attrs, dims=dims)
    new_dict = {"patch": patch}
    return [new_dict]


def scan_patches(patches: PatchType | Sequence[PatchType]) -> list[dc.PatchAttrs]:
    """
    Scan a sequence of patches and return a list of summaries.

    The summary dicts have the following fields:
        {fields}

    Parameters
    ----------
    patches
        A single patch or a sequence of patches.
    """
    if isinstance(patches, dc.Patch):
        patches = [patches]  # make sure we have an iterable
    out = [pa.attrs for pa in patches]
    return out


def get_start_stop_step(patch: PatchType, dim):
    """Convenience method for getting start, stop, step for a given coord."""
    assert dim in patch.dims, f"{dim} is not in Patch dimensions of {patch.dims}"
    coord = patch.get_coord(dim)
    start = coord.min()
    stop = coord.max()
    step = coord.step
    return start, stop, step


def get_default_patch_name(patch):
    """Generates the name of the node."""

    def _format_datetime64(dt):
        """Format the datetime string in a sensible way."""
        out = str(np.datetime64(dt).astype("datetime64[ns]"))
        return out.replace(":", "_").replace("-", "_").replace(".", "_")

    attrs = patch.attrs
    start = _format_datetime64(attrs.get("time_min", ""))
    end = _format_datetime64(attrs.get("time_max", ""))
    net = attrs.get("network", "")
    sta = attrs.get("station", "")
    tag = attrs.get("tag", "")
    return f"DAS__{net}__{sta}__{tag}__{start}__{end}"


def get_dim_value_from_kwargs(patch, kwargs):
    """
    Assert that kwargs contain one value and it is a dimension of patch.

    Several patch functions allow passing values via kwargs which are dimension
    specific. This function allows for some sane validation of such functions.

    Return the name of the dimension, its axis position, and its value.
    """
    dims = patch.dims
    overlap = set(dims) & set(kwargs)
    if len(kwargs) != 1 or not overlap:
        msg = (
            "You must use exactly one dimension name in kwargs. "
            f"You passed the following kwargs: {kwargs} to a patch with "
            f"dimensions {patch.dims}"
        )
        raise PatchDimError(msg)
    dim = next(iter(overlap))
    axis = dims.index(dim)
    return dim, axis, kwargs[dim]


def get_multiple_dim_value_from_kwargs(patch, kwargs):
    """Get multiple dim, axis and values from kwargs."""
    dims = patch.dims
    overlap = set(dims) & set(kwargs)
    if not overlap:
        msg = "You must specify one or more dimension in keyword args."
        raise PatchDimError(msg)
    out = {}
    for dim in overlap:
        axis = patch.dims.index(dim)
        out[dim] = {"dim": dim, "axis": axis, "value": kwargs[dim]}
    return out


def get_dim_sampling_rate(patch: PatchType, dim: str) -> float:
    """
    Get sampling rate, as a float from sampling period along a dimension.

    Parameters
    ----------
    patch
        The imput patch.
    dim
        Dimension to extract.

    Raises
    ------
    [CoordDataError](`dascore.exceptions.CoordDataError`) if patch is not
    evenly sampled along desired dimension.
    """
    d_dim = patch.coords.coord_map[dim].step
    if isinstance(d_dim, np.timedelta64):
        d_dim = d_dim / np.timedelta64(1, "s")
    if pd.isnull(d_dim):
        # get the name of the calling function
        calling_function = inspect.getframeinfo(sys._getframe(1))[2]
        msg = (
            f"Patch coordinate {dim} is not evenly sampled as required by "
            f"{calling_function}. This can be fixed with Patch.snap or "
            f"Patch.extrapolate. "
        )
        raise CoordDataError(msg)
    return 1.0 / d_dim


def _get_data_units_from_dims(patch, dims, operator):
    """Get new data units from some operation on dimensions."""
    if (data_units := get_quantity(patch.attrs.data_units)) is None:
        return
    dim_units = None
    for dim in iterate(dims):
        dim_unit = get_quantity(patch.get_coord(dim).units)
        if dim_unit is None:
            continue
        dim_units = dim_unit if dim_units is None else dim_unit * dim_units
    if dim_units is not None:
        data_units = operator(data_units, dim_units)
    return data_units


def _get_dx_or_spacing_and_axes(
    patch,
    dim,
    require_sorted=True,
    require_evenly_spaced=False,
) -> tuple[tuple[float | np.ndarray, ...], tuple[int, ...]]:
    """
    Return dx (spacing) or values for a list of dims and corresponding axes.

    Parameters
    ----------
    patch
        The input patch
    dim
        The dimension name or sequence of such
    require_sorted
        If True, raise an error if all requested dimensions are not sorted.
    require_evenly_spaced
        If True, raise an error if all requested dimensions are not evenly sampled.
    """
    dims = iterate(dim if dim is not None else patch.dims)
    out = []
    axes = []
    for dim_ in dims:
        coord = patch.get_coord(
            dim_,
            require_sorted=require_sorted,
            require_evenly_sampled=require_evenly_spaced,
        )
        if coord.evenly_sampled:
            val = coord.step
        else:
            val = coord.data
        # need to convert val to float so datetimes work
        out.append(to_float(val))
        axes.append(patch.dims.index(dim_))

    return tuple(out), tuple(axes)
