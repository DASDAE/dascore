"""
Utilities for working with the Patch class.
"""
import functools
import inspect
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

import dascore as dc
from dascore.constants import PATCH_MERGE_ATTRS, PatchType, SpoolType
from dascore.core.schema import PatchAttrs, PatchFileSummary
from dascore.exceptions import PatchAttributeError, PatchDimError
from dascore.utils.coordmanager import merge_coord_managers
from dascore.utils.docs import compose_docstring, format_dtypes
from dascore.utils.models import merge_models
from dascore.utils.time import to_timedelta64

attr_type = Union[Dict[str, Any], str, Sequence[str], None]


def _func_and_kwargs_str(func: Callable, patch, *args, **kwargs) -> str:
    """
    Get a str rep of the function and input args.
    """
    callargs = inspect.getcallargs(func, patch, *args, **kwargs)
    callargs.pop("patch", None)
    callargs.pop("self", None)
    kwargs_ = callargs.pop("kwargs", {})
    arguments = []
    arguments += [f"{k}={repr(v)}" for k, v in callargs.items() if v is not None]
    arguments += [f"{k}={repr(v)}" for k, v in kwargs_.items() if v is not None]
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
    patch: PatchType, required_dims: Optional[Tuple[str, ...]]
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
    required_dims: Optional[Union[Tuple[str, ...], Callable]] = None,
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


def copy_attrs(attrs) -> dict:
    """Make a copy of the attributes dict so it can be mutated."""
    out = dict(attrs)
    if "history" in out:
        out["history"] = list(out["history"])
    return out


@compose_docstring(fields=PatchAttrs.__annotations__)
def patches_to_df(
    patches: Union[Sequence[PatchType], SpoolType, pd.DataFrame]
) -> pd.DataFrame:
    """
    Return a dataframe

    Parameters
    ----------
    patches
        A sequence of :class:`dascore.Patch`

    Returns
    -------
    A dataframe with the following fields:
        {fields}
    plus a field called 'patch' which contains a reference to each patch.
    """

    if isinstance(patches, dc.BaseSpool):
        df = patches._df
    # Handle spool case
    elif hasattr(patches, "get_contents"):
        return patches.get_contents()
    elif isinstance(patches, pd.DataFrame):
        return patches
    else:
        df = pd.DataFrame([dict(x) for x in scan_patches(patches)])
        if df.empty:  # create empty df with appropriate columns
            cols = list(PatchAttrs().dict())
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
    patches: Union[Sequence[PatchType], pd.DataFrame, SpoolType],
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
    warnings.warn(msg, DeprecationWarning)
    return _merge_patches(patches, dim, check_history, tolerance)


def _merge_patches(
    patches: Union[Sequence[PatchType], pd.DataFrame, SpoolType],
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
        along the desired dimension. E.G., the default value means any patches
        with gaps <= 1.5 * dt will be merged.
    """

    def _get_sorted_df_sort_group_names(patches):
        """Return the sorted dataframe."""
        group_names = list(PATCH_MERGE_ATTRS) + [d_name]
        sort_names = group_names + [min_name, max_name]
        if check_history:
            sort_names += ["history"]
            patches = patches.assign(history=lambda x: x["history"].apply(str))
        return patches.sort_values(sort_names), sort_names, group_names

    def _merge_trimmed_patches(trimmed_patches):
        """Merge trimmed patches together."""
        axis = trimmed_patches[0].dims.index(dim)
        datas = [x.data for x in trimmed_patches]
        coords = [x.coords for x in trimmed_patches]
        attrs = [x.attrs for x in trimmed_patches]
        new_array = np.concatenate(datas, axis=axis)
        new_coord = merge_coord_managers(coords, dim=dim, snap_tolerance=tolerance)
        new_attrs = merge_models(attrs, dim)
        return dc.Patch(data=new_array, coords=new_coord, attrs=new_attrs)

    def _merge_compatible_patches(patch_df, dim):
        """perform merging after patch compatibility has been confirmed."""
        has_overlap = patch_df["_dist_to_previous"] <= to_timedelta64(0)
        overlap_start = patch_df[min_name] - patch_df["_dist_to_previous"]
        patches = list(patch_df["patch"])
        # this handles removing overlap in patches.
        trimmed_patches = [
            _trim_or_fill(x, start) if needs_action else x
            for x, start, needs_action in zip(patches, overlap_start, has_overlap)
        ]
        return _merge_trimmed_patches(trimmed_patches)

    def _trim_or_fill(patch, new_start):
        """Trim or fill data array."""
        out = patch.select(**{dim: (new_start, ...)})
        return out

    # get a dataframe
    if not isinstance(patches, pd.DataFrame):
        patches = patches_to_df(patches)
    assert dim in {"time", "distance"}, "merge must be on time/distance for now"
    out = []  # list of merged patches
    min_name, max_name, d_name = f"{dim}_min", f"{dim}_max", f"d_{dim}"
    # get sorted dataframe and group/sort column names
    df, sorted_names, group_names = _get_sorted_df_sort_group_names(patches)
    # get a boolean if each row is compatible with previous for merging
    gn = ~(df[group_names] == df.shift()[group_names]).all(axis=1)
    group_numbers = gn.astype(bool).cumsum()
    for _, sub_df in df.groupby(group_numbers):
        # get boolean indicating if patch overlaps with previous
        start, end = sub_df[min_name], sub_df[max_name]
        merge_dist = sub_df[d_name] * tolerance
        # get the dist between each patch
        dist_to_previous = start - end.shift()
        no_merge = ~(dist_to_previous <= merge_dist)
        sub_df["_dist_to_previous"] = dist_to_previous
        # determine if each patch should be merged with the previous one
        for _, merge_patch_df in sub_df.groupby(no_merge.astype(np.int64).cumsum()):
            out.append(_merge_compatible_patches(merge_patch_df, dim))
    return out

    assert False


@compose_docstring(fields=format_dtypes(PatchFileSummary.__annotations__))
def scan_patches(
    patches: Union[PatchType, Sequence[PatchType]]
) -> list[PatchFileSummary]:
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
    out = [PatchFileSummary(**dict(pa.attrs)) for pa in patches]
    return out


def get_start_stop_step(patch: PatchType, dim):
    """
    Convenience method for getting start, stop, step for a given dimension.
    """
    assert dim in patch.dims, f"{dim} is not in Patch dimensions of {patch.dims}"
    start = patch.attrs[f"{dim}_min"]
    end = patch.attrs[f"{dim}_max"]
    step = patch.attrs[f"d_{dim}"]
    return start, end, step


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
    dim = list(overlap)[0]
    axis = dims.index(dim)
    return dim, axis, kwargs[dim]
