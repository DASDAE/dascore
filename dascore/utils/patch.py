"""Utilities for working with the Patch class."""

from __future__ import annotations

import functools
import inspect
import sys
from collections import namedtuple
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal, Protocol, cast, overload

import numpy as np
import pandas as pd
import pydantic
from pint import DimensionalityError
from pydantic import TypeAdapter

import dascore as dc
from dascore.config import get_config
from dascore.constants import (
    WARN_LEVELS,
    PatchType,
    check_behavior_description,
)
from dascore.exceptions import (
    CoordDataError,
    CoordError,
    CoordMergeError,
    IncompatiblePatchError,
    ParameterError,
    PatchCoordinateError,
    UnitError,
)
from dascore.units import carries_units, convert_units, get_quantity
from dascore.utils.array_api import (
    asarray_like,
    backend_name,
    is_foreign,
    is_numpy,
    to_numpy,
    warn_numpy_fallback,
)
from dascore.utils.attrs import (
    _is_missing,
    _values_equal,
    combine_patch_attrs,
    warn_if_histories_differ,
)
from dascore.utils.coordmanager import merge_coord_managers
from dascore.utils.deprecate import deprecate
from dascore.utils.docs import compose_docstring
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import (
    _apply_union_indexers,
    _get_nullish,
    _merge_tuples,
    get_middle_value,
    iterate,
    to_object_array,
    validate_warn_level,
    warn_or_raise,
    yield_sub_sequences,
)
from dascore.utils.paths import is_memory_uri
from dascore.utils.time import to_float
from dascore.workflow.builtin import Concatenate, Stack
from dascore.workflow.checks import attr_type, check_patch_attrs, check_patch_coords
from dascore.workflow.identity import (
    _ID_FIELDS,
    advance,
    fold_patch_ids,
    fold_processing_ids,
    ids_enabled,
    patch_id_of,
    processing_id_of,
    stamp_combination,
)
from dascore.workflow.processor import (
    PatchOp,
    fingerprint_call,
    register_patch_function,
)

_DimAxisValue = namedtuple("_DimAxisValue", ["dim", "axis", "value"])

# Longest repr a single history argument may contribute.
_MAX_HISTORY_VALUE_LEN = 120


def _format_values(val):
    """String formatting for values for history string."""
    if isinstance(val, list | tuple):
        out = ",".join(_format_values(x) for x in val)
        out = f"({out})" if isinstance(val, tuple) else f"[{out}]"
    elif isinstance(val, np.ndarray):
        # make sure numpy strings aren't too long!
        config = get_config()
        out = np.array2string(
            val,
            precision=config.display_float_precision,
            threshold=config.display_patch_history_array_threshold,
        )
    elif isinstance(val, dc.Patch):
        # Truncate patch representations in history (issue #529)
        out = "Patch..."
    elif callable(val):
        # By name: a repr would carry an address which changes every run.
        out = getattr(val, "__qualname__", None) or repr(val)
    else:
        out = str(val)
        if len(out) > _MAX_HISTORY_VALUE_LEN:
            # An argument with a large repr (an inventory, say) would
            # otherwise be pasted into the history of every patch it touches.
            out = f"{type(val).__name__}..."
    return out


def _func_name(func: Callable) -> str:
    """Name a callable for the history string; partials have no __name__."""
    return getattr(func, "__name__", str(func))


def _func_and_kwargs_str(func: Callable, patch, *args, **kwargs) -> str:
    """Get a str rep of the function and input args."""
    # getcallargs is deprecated, but Signature.bind is not a drop-in
    # replacement (different handling of self and defaults); keep it for now.
    callargs = inspect.getcallargs(  # ty: ignore[deprecated]
        func, patch, *args, **kwargs
    )
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
    out = f"{_func_name(func)}("
    if arguments:
        out += f"{','.join(arguments)}"
    return out + ")"


def _get_history_str(
    patch: PatchType, func, *args, _history="full", **kwargs
) -> str | None:
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
    if _history is None or get_config().patch_history == "disabled":
        return None
    if _history == "full":
        history_str = _func_and_kwargs_str(func, patch, *args, **kwargs)
    else:
        history_str = _func_name(func)
    return history_str


def _maybe_add_history_str(attrs, hist_str):
    """Append a history string unless patch history recording is disabled."""
    if not hist_str or get_config().patch_history == "disabled":
        return attrs
    new_history = list(attrs.history)
    new_history.append(hist_str)
    return attrs.update(history=new_history)


class _HasDims(Protocol):
    """Anything with dimension names in axis order: a patch, or its metadata."""

    @property
    def dims(self) -> tuple[str, ...]:
        """The dimension names."""
        ...


class _PatchFunction(Protocol):
    """
    A function wrapped by `patch_function`.

    The decorator attaches references back to the function it wrapped, so
    callers can skip the patch-function machinery when calling it again. It
    also attaches `op`, which builds the operation this function is, and
    `__version__`, which that operation is declared at.
    """

    func: Callable
    raw_function: Callable
    op: Callable
    # What the decorator was told, so a registered processor can be held
    # to the same requirements rather than declaring its own in parallel.
    _declared: dict
    __version__: str
    __wrapped__: Callable

    def __call__(self, patch, *args, **kwargs): ...


def _stamp(patch, attrs, patch_func, args, kwargs):
    """
    Return attrs saying which data this is and what was just done to it.

    Every patch given to the call counts towards which data the result is
    -- `where(cond_patch, other_patch)` uses all three -- so their ids are
    folded rather than the first one being copied across. The ids are read
    with `getattr`, because attrs unpickled from before these fields
    existed have neither.
    """
    members = [patch.attrs]
    members += [x.attrs for x in (*args, *kwargs.values()) if isinstance(x, dc.Patch)]
    try:
        fingerprint = fingerprint_call(patch_func, args, kwargs)
    except Exception:
        # Provenance is metadata about the work, not the work. An argument
        # the serializer cannot encode is a reason to say nothing about
        # this call, never a reason to fail a call which otherwise worked.
        return attrs
    return attrs.update(
        # Carried from the inputs rather than from whatever the body
        # returned: filtering data does not make it other data, and a
        # function building its result from scratch would otherwise mint a
        # new id and claim it had.
        patch_id=fold_patch_ids([patch_id_of(x) for x in members]),
        processing_id=advance(
            fold_processing_ids([processing_id_of(x) for x in members]), fingerprint
        ),
    )


def _op_from_call(patch_func, *args, **kwargs):
    """Return the operation a call to a patch function is."""
    return PatchOp.from_call(patch_func, args, kwargs)


def _to_numpy_arg(obj):
    """Convert patches and non-numpy arrays to numpy, leaving the rest alone."""
    if isinstance(obj, dc.Patch):
        return obj if is_numpy(obj.data) else obj.new(data=to_numpy(obj.data))
    # Arrays which implement the standard (eg a boolean mask made from the
    # patch data) have to cross the boundary as well.
    if is_foreign(obj):
        return to_numpy(obj)
    return obj


def numpy_fallback(name, data, func, args=(), kwargs=None, stacklevel=4):
    """
    Apply a function numpy can perform but the patch's backend cannot.

    Every patch and array argument is converted to numpy, the function is
    applied, and a patch it returns is converted back to the backend of
    data. Used wherever dascore meets an operation the array API standard
    cannot express.

    Parameters
    ----------
    name
        The name of the operation, used in the warning.
    data
        The array whose backend the output should end up on.
    func
        The function to apply.
    args
        Positional arguments for func; patches and arrays are converted.
    kwargs
        Keyword arguments for func; patches and arrays are converted.
    stacklevel
        The stack level, as understood by warnings.warn, of the caller.
    """
    warn_numpy_fallback(name, backend_name(data), stacklevel=stacklevel + 1)
    converted = tuple(_to_numpy_arg(x) for x in args)
    kwargs = {i: _to_numpy_arg(v) for i, v in (kwargs or {}).items()}
    out = func(*converted, **kwargs)
    # Only patches carry data back to the original backend.
    if isinstance(out, dc.Patch):
        out = out.new(data=asarray_like(out.data, data))
    return out


def patch_function(
    required_dims: str | Sequence[str] | Callable | None = None,
    required_coords: str | Sequence[str] | None = None,
    required_attrs: attr_type = None,
    history: Literal["full", "method_name", None] = "full",
    validate_call: bool = False,
    data_type: str | None = None,
    version: str = "1.0",
):
    """
    Decorator to mark a function as a patch method.

    Parameters
    ----------
    required_dims
        A dimension name, or a sequence of them, which must be found in the
        Patch.
    required_coords
        A coordinate name, or a sequence of them, which must be found in the
        Patch.
    required_attrs
        An attr name, a sequence of them, or a mapping of names to the values
        the Patch must hold for them.
    history
        Specifies how to track history on Patch.
            Full - Records function name and str version of input arguments.
            method_name - Only records method name. Useful if args are long.
            None - Function call is not recorded in history attribute.
    validate_call
        If True, use pydantic to validate the function call. This can save
        quite a lot of code in validation checks, but does have some overhead.
        See [validate_call](https://docs.pydantic.dev/latest/api/validate_call/).
    data_type
        Controls the output patch's ``data_type`` attr. If None, leave the
        returned patch's ``data_type`` unchanged. Otherwise, set to specified
        value. Use an empty string ("") to clear.
    version
        The version of the operation. Bump it when the same arguments should
        mean a different answer, so that fingerprints recorded against the
        old behaviour do not name the new one.

    Examples
    --------
    >>> import dascore as dc
    >>>
    >>> # 1. A patch method which requires dimensions (time, distance)
    >>> @dc.patch_function(required_dims=('time', 'distance'))
    ... def do_something(patch):
    ...     ...   # raises a PatchCoordsError if patch doesn't have time,
    ...     #  distance
    >>>
    >>> # 2. A patch method which requires an attribute 'data_type' == 'DAS'
    >>> @dc.patch_function(required_attrs={'data_type': 'DAS'})
    ... def do_another_thing(patch):
    ...     ...  # raise PatchAttributeError if patch doesn't have attribute
    ...     # called "data_type" or its values is not equal to "DAS".
    >>>
    >>> # 3. A patch method which does type checking on inputs.
    >>> # The `Field` instance can require various data properties (like ranges)
    >>> from typing_extensions import Annotated, Literal
    >>> from pydantic import Field
    >>> @dc.patch_function(validate_call=True)
    ... def do_type_thing(
    ...     patch,
    ...     int_le_10_ge_1: int = Field(ge=1, le=10, default=1),
    ...     option: Literal["min", "max", None] = None,
    ... ):
    ...     ...
    >>>
    >>> # 4. A patch method which sets the output data_type.
    >>> @dc.patch_function(data_type="strain_rate")
    ... def do_strain_rate(patch):
    ...     ...
    >>>
    >>> # 5. A patch method which clears the output data_type.
    >>> @dc.patch_function(data_type="")
    ... def do_unknown_quantity(patch):
    ...     ...
    >>>
    >>> # 6. A patch method whose body is written to the array API standard,
    >>> # so it runs on any array backend rather than only on numpy.
    >>> from dascore.utils.array_api import array_namespace
    >>> @dc.patch_function()
    ... def do_portable_thing(patch):
    ...     xp = array_namespace(patch.data)
    ...     return patch.new(data=xp.abs(patch.data))
    >>>
    >>> # 7. Every patch function builds the operation it is with `.op`:
    >>> # the same call said as a task, which can be compared,
    >>> # fingerprinted, and written to a file.
    >>> patch = dc.get_example_patch()
    >>> op = dc.proc.normalize.op(dim="time")
    >>> assert op(patch).equals(patch.normalize(dim="time"))
    >>> assert op == dc.proc.normalize.op(dim="time")

    Notes
    -----
    - The original function can still be accessed with the raw_function
      attribute. This may be useful for avoiding calling the patch_func
      machinery multiple times from within another patch function.

    - The decorated function also carries ``op``, which builds the
      [PatchOp](`dascore.workflow.processor.PatchOp`) a call to it is:
      ``dc.proc.normalize.op(dim="time")``. The call is bound against the
      signature, so a positional and a keyword spelling of one call are one
      operation with one fingerprint.

    - If using `PatchType` or `SpoolType` type variables from the
      [constants module](`dascore.constants`), make sure dascore is imported
      as dc at the top of the file where the patch function is defined so
      the forward refs can be resolved properly for type checking.
    """
    # Handled before the wrapper is built so the rest of this function sees
    # required_dims as the dimension names it is everywhere else, not as the
    # decorated function.
    if callable(required_dims):  # the decorator is used without parens
        return patch_function()(required_dims)

    def _wrapper(func):
        if validate_call:
            config = pydantic.ConfigDict(arbitrary_types_allowed=True)
            func = pydantic.validate_call(config=config)(func)

        @functools.wraps(func)
        def _func(patch, *args, **kwargs):
            check_patch_coords(
                patch,
                dims=required_dims,
                coords=required_coords,
            )
            check_patch_attrs(patch, required_attrs)
            out = func(patch, *args, **kwargs)
            attr_updates = {}
            if data_type is not None:
                attr_updates["data_type"] = data_type
            # attach history string. Need to consider something a bit less hacky.
            if out is not patch and hasattr(out, "attrs"):
                hist_str = _get_history_str(
                    patch, func, *args, _history=history, **kwargs
                )
                attrs = _maybe_add_history_str(out.attrs, hist_str)
                # What was done, folded into what the input carried, into
                # the same attrs object the history went into: an operation
                # should cost one new patch, not one per thing it stamps.
                #
                # Only when something new came back: an operation which
                # handed the patch straight through did nothing, and nothing
                # is what it records.
                if ids_enabled():
                    attrs = _stamp(patch, attrs, patch_func, args, kwargs)
                if attrs is not out.attrs:
                    out = out.update(attrs=attrs)
            if attr_updates and hasattr(out, "attrs"):
                out = out.update_attrs(**attr_updates)
            return out

        # Attach original function. Although we want to encourage raw_function
        # for consistency with pydantic, we leave this to not break old code.
        patch_func = cast(_PatchFunction, _func)
        patch_func.func = getattr(func, "raw_function", func)
        # matches pydantic naming.
        patch_func.raw_function = getattr(func, "raw_function", func)
        patch_func.__wrapped__ = func
        patch_func.__version__ = version
        # What the decorator was told, kept where a registered processor
        # can find it. `register_implementation` reconciles the two, so
        # the class and the decorator cannot drift apart in silence.
        patch_func._declared = {
            "required_dims": required_dims,
            "required_coords": required_coords,
            "required_attrs": required_attrs,
            "data_type": data_type,
            "history": history,
            "validate_call": validate_call,
        }
        # Registered as it is decorated, so a function is nameable in a
        # document exactly when its module has been imported. A function
        # defined inside a call takes no tag; `op` says so if it is asked.
        register_patch_function(patch_func)
        # `op` builds the operation a call to this function is. A partial
        # rather than a lambda so it pickles, and so the wrapper carries no
        # closure of its own.
        patch_func.op = functools.partial(_op_from_call, patch_func)
        return patch_func

    return _wrapper


def patches_to_df(
    patches: Sequence[dc.Patch] | dc.Spool | pd.DataFrame,
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
    # Handle spool case (or anything else exposing spool-style get_contents)
    if callable(get_contents := getattr(patches, "get_contents", None)):
        df = get_contents()
        # get_contents() carries only metadata; embed the patches so the
        # flat-dump path can serve them (the "patch" column is the point).
        if "patch" not in df.columns:
            df = df.assign(patch=to_object_array(list(patches)))
    elif isinstance(patches, pd.DataFrame):
        df = patches
    else:
        df = dc.scan_to_df(
            patches,
            exclude=(),
        )
        if df.empty:  # create empty df with appropriate columns
            cols = list(dc.PatchAttrs().model_dump())
            df = pd.DataFrame(columns=cols).assign(patch=None, history=None)
        else:  # else populate with patches and concat history
            history = df["history"].apply(lambda x: ",".join(x))
            df = df.assign(patch=to_object_array(patches), history=history)
    # Ensure history is in df
    if "history" not in df.columns:
        df = df.assign(history="")
    if "patch" not in df.columns:
        df["patch"] = None
    return df


@deprecate(
    info=(
        "merge_patches is deprecated. Use spool.chunk instead. "
        "For example, to merge a list of patches you can use: "
        "dascore.spool(patch_list).chunk(time=None) to merge on the time "
        "dimension."
    ),
    removed_in="0.2.0",
)
def merge_patches(
    patches: Sequence[dc.Patch] | pd.DataFrame | dc.Spool,
    dim: str = "time",
    check_history: bool = True,
    tolerance: float = 1.5,
) -> dc.Spool:
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
    return dc.spool(patches).chunk(**{dim: None}, tolerance=tolerance)


def _get_merge_dim(df) -> str | None:
    """
    Get the merge dimension from a dataframe of patch dim summaries.

    The merge dimension is the single dimension whose range varies between
    rows; None means complete overlap (nothing to merge).
    """
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


def _middle_step(coords, dim, target_units):
    """Return the middle member step expressed in the merged coord's units."""
    steps = []
    for manager in coords:
        coord = manager.coord_map[dim]
        step = coord.step
        if pd.isnull(step):
            continue
        if target_units is not None and coord.units is not None:
            step = convert_units(step, to_units=target_units, from_units=coord.units)
        steps.append(step)
    if not steps:
        return None
    return get_middle_value(np.asarray(steps))


def _split_coord_merge_kwargs(merge_kwargs) -> tuple[dict, dict]:
    """Split spool merge kwargs into (attr kwargs, coord kwargs)."""
    merge_kwargs = dict(merge_kwargs or {})
    coord_kwargs = {
        "snap_coords": merge_kwargs.pop("snap_coords", True),
        "tolerance": merge_kwargs.pop("tolerance", 1.5),
    }
    return merge_kwargs, coord_kwargs


def _get_merged_coord(
    df, merge_dim, coords, drop_conflicting=False, snap_coords=True, tolerance=1.5
):
    """
    Get merged coordinates for patches combined along merge_dim.

    The merged dimension coordinate is built by truth-preserving
    concatenation of the member coords (exactly contiguous members fuse to
    a plain range; recorded seams otherwise), then — when `snap_coords` —
    simplified with bounded error: no value moves more than
    `tolerance * step`, or than the tolerance itself when it is a
    quantity or timedelta, which states the bound outright. Merges whose
    gaps exceed that stay segmented (honestly non-uniform) rather than
    being relabeled.
    """
    from dascore.core.coords import concat_coords  # noqa: PLC0415

    try:
        merged = concat_coords(*[cm.coord_map[merge_dim] for cm in coords])
    except CoordError:
        # Non-monotonic (or otherwise unsegmentable) member coordinates:
        # fall back to raw value concatenation of the dim coord.
        return merge_coord_managers(
            coords, dim=merge_dim, drop_conflicting=drop_conflicting
        )
    step = _middle_step(coords, merge_dim, merged.units)
    if snap_coords and carries_units(tolerance):
        # A tolerance which states its own units needs no step: simplify
        # reads it in the coordinate's units itself.
        merged = merged.simplify(tolerance)
    elif snap_coords and step is not None:
        merged = merged.simplify(tolerance * np.abs(step))
    # Passing the pre-built dim coord avoids materializing the members'
    # concatenated values only to discard them.
    return merge_coord_managers(
        coords, dim=merge_dim, drop_conflicting=drop_conflicting, dim_coord=merged
    )


def _force_patch_merge(patch_dict_list, merge_kwargs, **kwargs):
    """
    Force a merge of the patches along a dimension.

    This function is used in conjunction with `spool.chunk`, which
    does all the compatibility checks beforehand.
    """
    df = pd.DataFrame(patch_dict_list)
    merge_dim = _get_merge_dim(df)
    attr_kwargs, coord_kwargs = _split_coord_merge_kwargs(merge_kwargs)
    if merge_dim is None:  # nothing to merge, complete overlap
        return [patch_dict_list[0]]
    dims = df["dims"].iloc[0].split(",")
    # get patches, ensure they are oriented the same.
    dims_tuple = tuple(dims)
    patches = [x if x.dims == dims_tuple else x.transpose(*dims) for x in df["patch"]]
    axis = patches[0].get_axis(merge_dim)
    # get data, coords, attrs for merging patch together.
    data = [x.data for x in patches]
    coords = [x.coords for x in patches]
    attrs = [x.attrs for x in patches]
    new_data = np.concatenate(data, axis=axis)
    # Determine if conflicting non-dimensional coords should be dropped.
    conf = attr_kwargs.get("conflict", None)
    drop_conf_coords = True if conf in {"drop", "keep_first"} else False
    new_coord = _get_merged_coord(
        df, merge_dim, coords, drop_conf_coords, **coord_kwargs
    )
    warn_if_histories_differ(attrs, "Merging")
    new_attrs = combine_patch_attrs(attrs, **attr_kwargs)
    patch = dc.Patch(data=new_data, coords=new_coord, attrs=new_attrs, dims=dims)
    new_dict = {"patch": patch}
    return [new_dict]


def get_start_stop_step(patch: PatchType, dim):
    """Convenience method for getting start, stop, step for a given coord."""
    assert dim in patch.dims, f"{dim} is not in Patch dimensions of {patch.dims}"
    coord = patch.get_coord(dim)
    start = coord.min()
    stop = coord.max()
    step = coord.step
    return start, stop, step


def get_patch_names(
    # Forwarded straight to scan_to_df, so anything it scans works here,
    # including a plain list of patches. Spelled out rather than reusing
    # io.core.ScanInput: importing that is circular, and hiding it behind
    # TYPE_CHECKING leaves the annotation unresolvable at runtime, which
    # breaks get_type_hints and the API doc renderer.
    patch_data: pd.DataFrame | dc.Patch | dc.Spool | Iterable[dc.Patch],
    prefix="DAS",
    attrs=("acquisition_key", "tag"),
    coords=("time",),
    sep="__",
    strip_extension=True,
) -> pd.Series:
    """
    Generates the default name of patch data.

    Parameters
    ----------
    patch_data
        A container with patch data.
    prefix
        A string to prefix the names.
    attrs
        The Patch attrs to include in the name.
    coords
        The coordinate ranges to use in the names.
    sep
        The separator for each value.
    strip_extension
        If True, remove extensions when getting name from a file path.
        See the notes section for more details.

    Notes
    -----
    There are two special cases where the default logic is overwritten.
    The first one, is when a column called "name" already exists. This
    will simply be returned.

    The second is when a column called "source_path" exists. In this case, the
    output will be the file name with the extension removed (if
    strip_extension). The path must use '/' as a delimiter.

    See Also
    --------
    - [`Patch.get_patch_name`](`dascore.Patch.get_patch_name`)
    - [`Spool.get_patch_names`](`dascore.Spool.get_patch_names`)

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.utils.patch import get_patch_names
    >>>
    >>> # Get a series of names from a patch or spool
    >>> patch = dc.get_example_patch()
    >>> spool = dc.get_example_spool()
    >>> patch_name = get_patch_names(patch)
    >>> spool_name = get_patch_names(patch)
    >>>
    >>> # Use the Patch/Spool methods
    >>> spool_series = spool.get_patch_names()
    >>> patch_name = patch.get_patch_name() # a str w/ name.
    """

    def _format_time_column(ser):
        """Format the time column."""
        ser = ser.astype(str).str.split(".", expand=True)[0]
        chars_to_replace = (":", "-")
        for char in chars_to_replace:
            ser = ser.str.replace(char, "_")
        ser = ser.str.replace(" ", "T")
        return ser

    def _format_time_columns(df):
        """Format the datetime string in a sensible way."""
        sub = df.select_dtypes(include=["datetime64", "timedelta64"])
        out = {}
        for col in sub.columns:
            out[col] = _format_time_column(df[col])
        return df.assign(**out)

    def _get_filename(path_ser, strip_extension):
        """Get the file name from a path series."""
        ser = path_ser.astype(str)
        split_ser = ser.str.split("/")
        if strip_extension:
            # Only the extension: an acquisition_key puts dots in the name
            # itself, and splitting on the first one would collide every
            # patch of one source onto a single truncated name.
            file_names = [x[-1].rsplit(".", 1)[0] for x in split_ser]
        else:
            file_names = [x[-1] for x in split_ser]
        return pd.Series(file_names)

    # Validate inputs. Note we cannot use the validation decorator or
    # it introduces a circular import.
    prefix = TypeAdapter(str).validate_python(prefix)
    attrs = TypeAdapter(tuple[str, ...]).validate_python(attrs)
    coords = TypeAdapter(tuple[str, ...]).validate_python(coords)
    sep = TypeAdapter(str).validate_python(sep)

    # Ensure we are working with a dataframe.
    df = dc.scan_to_df(
        patch_data,
        exclude=(),
    )
    if df.empty:
        return pd.Series(dtype=str)
    col_set = set(df.columns)
    # Handle special cases.
    if "name" in col_set:
        return df["name"].astype(str)
    path_ser = df["source_path"].astype(str) if "source_path" in col_set else None
    if path_ser is not None:
        # synthetic in-memory identities are not real file names
        usable = path_ser.str.len().gt(0) & ~path_ser.map(is_memory_uri)
        if usable.all():
            return _get_filename(df["source_path"], strip_extension)
    # Determine the requested fields; absent columns render as empty so
    # names don't depend on which metadata engine produced the dataframe.
    coord_fields = zip([f"{x}_min" for x in coords], [f"{x}_max" for x in coords])
    fields = list(attrs) + [field for pair in coord_fields for field in pair]
    sub = df.reindex(columns=fields).pipe(_format_time_columns).fillna("").astype(str)
    out = f"{prefix}_{sep}" + sub[fields[0]].str.cat(sub[fields[1:]], sep=sep)
    return out


def get_dim_axis_value(
    patch: _HasDims,
    *,
    args: tuple = tuple(),
    kwargs: Mapping = FrozenDict(),
    arg_keys: tuple[str, ...] = ("dim", "coord", "dims", "coords"),
    allow_multiple: bool = False,
    allow_extra: bool = False,
) -> tuple[_DimAxisValue, ...]:
    """
    Get dimension name, index, and values from args/kwargs for a patch.

    This is helpful for implementing flexible fetching of dimension name,
    corresponding patch axis, and function specific values from args and
    kwargs as inputs.

    Parameters
    ----------
    patch
        The patch which contains desired dimensions.
    args
        A tuple of possible dimension names.
    kwargs
        A dict of dimension_name: value
    arg_keys
        Keys in the dictionary that indicate
    allow_multiple
        If True, allow multiple dimensions to be selected.
    allow_extra
        If True, do not raise an error if extra args or kwargs are found.

    Returns
    -------
    Returns a tuple of:
        ((dim, axis, value), (dim, axis, value), ...)
    To support retrieving multiple values from the same inputs. If dim name
    is found in args, its corresponding values is `None`.

    Examples
    --------
    import dascore.proc.coord    >>> import dascore as dc
    import dascore.proc.coords    >>> from dascore.utils.patch import get_dim_axis_value
    >>> import dascore as dc
    >>> from dascore.utils.patch import get_dim_axis_value
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Get tuple of dimension name, axis, and value from dict (eg kwargs)
    >>> (dim, ax, val) = get_dim_axis_value(patch, kwargs={"time": 10})[0]
    >>> assert dim == "time" and ax == patch.get_axis("time") and val == 10
    >>>
    >>> # Get dim name and axis from tuple (eg args)
    >>> (dim, ax, val) = get_dim_axis_value(patch, args=("time",))[0]
    >>> assert dim == "time" and ax == patch.get_axis("time") and val is None
    >>>
    >>> # Get list of dim, ax val from multiple kwargs and args
    >>> info = get_dim_axis_value(
    ...     patch, args=("time", ), kwargs={"distance": 10}, allow_multiple=True,
    ... )
    >>> assert len(info) == 2
    """
    kwargs = dict(kwargs)  # copy kwargs to avoid modifying the input dict
    dims: tuple[str, ...] = patch.dims
    # Pop out any args implicit in kwargs.
    args = args + tuple(kwargs.pop(x) for x in arg_keys if x in kwargs)
    input_set = set(args) | set(kwargs)
    patch_dim_set = set(dims)
    overlap = patch_dim_set & input_set
    # Determine if there is the right number of overlaps.
    if not overlap or (len(overlap) > 1 and not allow_multiple):
        expect = "at least one" if allow_multiple else "exactly one"
        msg = (
            f"You must specify {expect} dimension name in args or kwargs. "
            f"You passed the following kwargs: {kwargs} args: {args} "
            f"to a patch with dimensions {patch.dims}"
        )
        raise ParameterError(msg)
    # Handle the case of extra inputs
    if (remaining := input_set - patch_dim_set) and not allow_extra:
        msg = f"The following input dimensions are not found in the patch. {remaining}"
        raise PatchCoordinateError(msg)
    # Ensure order is preserved (eg args, then kwargs)
    dim_out = tuple(x for x in args + tuple(kwargs) if x in overlap)
    # Package everything up and return
    out = tuple(_DimAxisValue(x, dims.index(x), kwargs.get(x)) for x in dim_out)
    return out


def get_dim_sampling_rate(patch: PatchType, dim: str) -> float:
    """
    Get sampling rate, as a float from sampling period along a dimension.

    Parameters
    ----------
    patch
        The input patch.
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


@deprecate(
    info=(
        "get_patch_window_size is deprecated. Use "
        "dascore.utils.window.resolve_window(...).full_size() instead."
    ),
    removed_in="0.2.0",
)
def get_patch_window_size(
    patch: PatchType,
    kwargs: dict,
    samples: bool = False,
    *,
    require_odd: bool = False,
    warn_above: int | None = None,
    min_samples: int = 1,
    enforce_lt_coord: bool = False,
) -> tuple[int, ...]:
    """Return the window along every patch axis; see `resolve_window`."""
    # Deferred: dascore.utils.window imports this module.
    from dascore.utils.window import resolve_window  # noqa: PLC0415

    if not kwargs:
        return (1,) * len(patch.dims)
    return resolve_window(
        patch,
        kwargs,
        samples=samples,
        require_odd=require_odd,
        warn_above=warn_above,
        min_samples=min_samples,
        enforce_lt_coord=enforce_lt_coord,
    ).full_size()


@deprecate(
    info=(
        "get_window_axis_step is deprecated. Use "
        "dascore.utils.window.resolve_window instead."
    ),
    removed_in="0.2.0",
)
def get_window_axis_step(
    patch,
    overlap=None,
    step=None,
    samples=False,
    **kwargs,
) -> tuple[int, int, int | None]:
    """Return one dimension's window, axis, and step; see `resolve_window`."""
    # Deferred: dascore.utils.window imports this module.
    from dascore.utils.window import resolve_window  # noqa: PLC0415

    window = resolve_window(
        patch,
        kwargs,
        samples=samples,
        overlap=overlap,
        step=step,
        allow_multiple=False,
        min_samples=0,
        enforce_lt_coord=True,
    )
    stride = None if window.stride is None else window.stride[0]
    return window.size[0], window.axes[0], stride


# What a derivative along a dimension makes of the data. Read forward to
# differentiate and backward to integrate; the same physics either way,
# so one table states both.
_DATA_TYPE_DERIVATIVES = {
    "time": {
        "displacement": "velocity",
        "velocity": "acceleration",
        "strain": "strain_rate",
        "phase": "phase_rate",
    },
    # A derivative along the fiber is what makes strain out of motion,
    # which is the operation `velocity_to_strain_rate` performs.
    "distance": {
        "displacement": "strain",
        "velocity": "strain_rate",
    },
}

_DATA_TYPE_INTEGRALS = {
    dim: {v: k for k, v in table.items()}
    for dim, table in _DATA_TYPE_DERIVATIVES.items()
}


def _get_data_type_from_dims(patch, dims, differentiate: bool) -> str:
    """Get the data_type of a patch differentiated or integrated over dims."""
    tables = _DATA_TYPE_DERIVATIVES if differentiate else _DATA_TYPE_INTEGRALS
    data_type = patch.attrs.data_type
    for dim in iterate(dims):
        table = tables.get(dim, {})
        if data_type not in table:
            # A derivative is a different quantity than what it was taken
            # of, so a step the vocabulary cannot name leaves the patch
            # with no label it can honestly carry: the note on patch
            # attrs says a stale data_type is worse than an empty one.
            # It also settles the whole chain at once, which mapping only
            # the steps that are named would not -- velocity over time
            # then distance would be acceleration, and over distance then
            # time strain rate, for one and the same mixed derivative.
            return ""
        data_type = table[data_type]
    return data_type


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


@overload
def _get_dx_or_spacing_and_axes(
    patch,
    dim,
    require_sorted: bool = ...,
    *,
    require_evenly_spaced: Literal[True],
) -> tuple[tuple[float, ...], tuple[int, ...]]: ...


@overload
def _get_dx_or_spacing_and_axes(
    patch,
    dim,
    require_sorted: bool = ...,
    require_evenly_spaced: bool = ...,
) -> tuple[tuple[float | np.ndarray, ...], tuple[int, ...]]: ...


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
        Every returned value is then a scalar spacing rather than an array of
        values, which the overloads above make visible to callers.
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
        axes.append(patch.get_axis(dim_))

    return tuple(out), tuple(axes)


def align_patch_coords(
    patch1: PatchType, patch2: PatchType
) -> tuple[PatchType, PatchType]:
    """
    Align two patches of the same kind so their data broadcast together.

    The patches must be the same kind (see
    [`check_kind`](`dascore.utils.patch.check_kind`)). Dimensions only one
    patch has are appended to the other as length one; dimensions both have
    are aligned on the intersection of their coordinate values, and an
    empty intersection raises: sharing a dimension but none of its values
    is a conflict, not an empty answer.

    Parameters
    ----------
    patch1
        The first patch.
    patch2
        The second patch.
    """
    check_kind(patch1, patch2)
    # Fast path for no alignment needed
    if patch1.coords == patch2.coords:
        return patch1, patch2
    shared_dims = set(patch1.dims) & set(patch2.dims)
    if not shared_dims:
        msg = (
            "Cannot align patches with no shared dimensions. Dimensions are "
            f"patch1: {patch1.dims}, patch2: {patch2.dims}"
        )
        raise PatchCoordinateError(msg)
    # First ensure the patches have the same dims
    dims = _merge_tuples(patch1.dims, patch2.dims)
    dim_dict = {x: num for num, x in enumerate(dims)}
    patch1 = patch1.append_dims(*dims).transpose(*dims)
    patch2 = patch2.append_dims(*dims).transpose(*dims)
    # Next, find the common coordinates and align.
    align_1, align_2 = [slice(None)] * len(dims), [slice(None)] * len(dims)
    new_coords_1, new_coords_2 = {}, {}
    for dim in shared_dims:
        coord1, coord2 = patch1.get_coord(dim), patch2.get_coord(dim)
        if coord1 == coord2:
            continue
        dim_ind = dim_dict[dim]
        # We actually need to do some alignment here.
        ncoord1, ncoord2, sli1, sli2 = coord1.align_to(coord2)
        if not len(ncoord1):
            msg = (
                f"Cannot align patches: they share no values along dimension "
                f"{dim!r} ({coord1.min()} to {coord1.max()} and "
                f"{coord2.min()} to {coord2.max()})."
            )
            raise PatchCoordinateError(msg)
        new_coords_1[dim], new_coords_2[dim] = ncoord1, ncoord2
        align_1[dim_ind], align_2[dim_ind] = sli1, sli2
    # No alignment needed, just skip.
    if not (new_coords_1 or new_coords_2):
        return patch1, patch2
    # Update coordinate managers and reshape arrays, return new patches.
    coord1 = patch1.coords.update(**new_coords_1)
    coord2 = patch2.coords.update(**new_coords_2)
    array1 = _apply_union_indexers(tuple(align_1), patch1.data)
    array2 = _apply_union_indexers(tuple(align_2), patch2.data)
    out1 = patch1.new(data=array1, coords=coord1)
    out2 = patch2.new(data=array2, coords=coord2)
    return out1, out2


def get_patch_kind(patch: PatchType | dc.PatchAttrs) -> FrozenDict:
    """
    Return the attribute values which decide what kind of patch this is.

    The attribute names come from the config option `patch_kind_attrs`.
    A name the patch lacks, or holds a null or empty string for, maps to
    None: an attribute left at its empty default is the same as no
    attribute, which is also how spool metadata records it.
    [`check_kind`](`dascore.utils.patch.check_kind`) decides what None
    then compares equal to.

    Parameters
    ----------
    patch
        A patch or its attrs.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.utils.patch import get_patch_kind
    >>> patch = dc.get_example_patch()
    >>> kind = get_patch_kind(patch)
    >>> assert kind["tag"] == patch.attrs.tag
    >>> assert kind["acquisition_key"] is None  # not set
    """
    attrs = patch.attrs if isinstance(patch, dc.Patch) else patch
    names = get_config().patch_kind_attrs
    return FrozenDict({x: _kind_value(attrs.get(x)) for x in names})


def _kind_value(value):
    """Normalize one kind value: a missing one reads as None."""
    return None if _is_missing(value) else value


def check_kind(
    patch1, patch2, check_behavior: WARN_LEVELS = "raise", *, strict: bool = False
) -> bool:
    """
    Return True if two patches are the same kind.

    Kind is decided by the attributes named in the config option
    `patch_kind_attrs` and nothing else: coordinates, units, history, and
    the remaining attributes never enter. Patches of different kinds are
    never combined, whatever their coordinates.

    Two-operand callers -- operators, ufuncs,
    [`Patch.where`](`dascore.Patch.where`) -- leave `strict` False: a
    missing value is a wildcard matching anything, and the result carries
    the union of what the two knew. Callers combining a *collection* --
    concatenate, stack, the spool operations -- pass `strict`, because a
    wildcard is not transitive (`"a"` matches `""` matches `"b"`, yet
    `"a"` and `"b"` conflict) and a partition needs it to be.

    Parameters
    ----------
    patch1
        The first patch.
    patch2
        The second patch.
    check_behavior
        What to do when the kinds differ: 'raise' (default) raises
        [`IncompatiblePatchError`](`dascore.exceptions.IncompatiblePatchError`),
        'warn' warns and returns False, 'ignore' returns False quietly.
    strict
        If True, a missing value equals only another missing value.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.utils.patch import check_kind
    >>> patch = dc.get_example_patch()
    >>> assert check_kind(patch, patch.pass_filter(time=(None, 10)))
    >>> other = patch.update_attrs(tag="other")
    >>> assert not check_kind(patch, other, check_behavior="ignore")
    >>> # An unset attribute matches any value for a two-patch operation,
    >>> keyed = patch.update_attrs(acquisition_key="A.B.C.D")
    >>> assert check_kind(patch, keyed)
    >>> # but is a value of its own where patches are partitioned.
    >>> assert not check_kind(patch, keyed, check_behavior="ignore", strict=True)
    """
    validate_warn_level(check_behavior, "check_behavior")
    kind1, kind2 = get_patch_kind(patch1), get_patch_kind(patch2)

    def _equal(value1, value2) -> bool:
        """A missing value is a wildcard unless the caller is strict."""
        if not strict and (value1 is None or value2 is None):
            return True
        return _values_equal(value1, value2)

    diffs = {x: (kind1[x], kind2[x]) for x in kind1 if not _equal(kind1[x], kind2[x])}
    if not diffs:
        return True
    msg = (
        "Patches are not the same kind; these attributes conflict "
        f"(patch1, patch2): {diffs}. The config option patch_kind_attrs "
        "names the attributes which decide kind."
    )
    warn_or_raise(msg, exception=IncompatiblePatchError, behavior=check_behavior)
    return False


def check_data_units(patch1, patch2, check_behavior: WARN_LEVELS = "raise") -> bool:
    """
    Return True unless the patches hold different data units.

    Splicing or summing data demands one unit, and none is converted for
    the caller: metres beside kilometres, or beside a patch with no units
    at all, must be reconciled first with
    [`Patch.convert_units`](`dascore.Patch.convert_units`) or
    [`Patch.set_units`](`dascore.Patch.set_units`).

    Parameters
    ----------
    patch1
        The first patch.
    patch2
        The second patch.
    check_behavior
        What to do when the units differ: 'raise' (default) raises
        [`IncompatiblePatchError`](`dascore.exceptions.IncompatiblePatchError`),
        'warn' warns and returns False, 'ignore' returns False quietly.
    """
    validate_warn_level(check_behavior, "check_behavior")
    units1 = get_quantity(patch1.attrs.data_units)
    units2 = get_quantity(patch2.attrs.data_units)
    if units1 == units2:
        return True
    msg = (
        f"Patches are not compatible: data units differ ({units1} and {units2}); "
        "convert one with Patch.convert_units first."
    )
    warn_or_raise(msg, exception=IncompatiblePatchError, behavior=check_behavior)
    return False


def check_dims(
    patch1,
    patch2,
    check_behavior: WARN_LEVELS = "raise",
    intersection: bool = False,
) -> bool:
    """
    Return True if dimensions of two patches are equal.

    Parameters
    ----------
    patch1
        first patch
    patch2
        second patch
    check_behavior
        String with 'raise' will raise an error if incompatible,
        'warn' will provide a warning, 'ignore' will do nothing.
    intersection
        If True, allow any intersection of dimensions to pass. This is useful
        when only broadcastability needs to be checked. If false require dims
        to be equal.
    """
    validate_warn_level(check_behavior, "check_behavior")
    dims1, dims2 = patch1.dims, patch2.dims
    if not intersection and patch1.dims == patch2.dims:
        return True
    dset1, dset2 = set(dims1), set(dims2)
    if intersection and (dset1 | dset2):
        return True
    msg = (
        "Patch dimensions are not compatible for merging."
        f" Patch1 dims: {dims1}, Patch2 dims: {dims2}"
    )
    warn_or_raise(msg, exception=IncompatiblePatchError, behavior=check_behavior)
    # The quiet policies skip the patch rather than accept it, as
    # check_coords does; saying True here stacked it anyway.
    return False


def check_coords(
    patch1,
    patch2,
    check_behavior: WARN_LEVELS = "raise",
    dim_to_ignore=None,
    ignore_dim_eq_shape=True,
    riders_vary: bool = False,
) -> bool:
    """
    Return True if the coordinates of two patches are compatible, else False.

    Parameters
    ----------
    patch1
        patch 1
    patch2
        patch 2
    check_behavior
        String with 'raise' will raise an error if incompatible,
        'warn' will provide a warning.
    dim_to_ignore
        None by default (all coordinates must be identical).
        String specifying a dimension that differences in values,
        but not shape, are allowed. A coordinate attached to different
        dimensions in the two patches is never compatible.
    riders_vary
        If True, coordinates riding `dim_to_ignore` are allowed the same
        differences as the dimension, for an operation which joins them
        along it (concatenation); stacking keeps the first patch's.
    ignore_dim_eq_shape
        If True, the ignored dims must be equal shape to pass check.
        If dim_to_ignore is None this has no effect.
    """
    validate_warn_level(check_behavior, "check_behavior")
    cm1 = patch1.coords
    cm2 = patch2.coords
    cset1, cset2 = set(cm1.coord_map), set(cm2.coord_map)
    shared = cset1 & cset2
    not_equal_coords = []
    for coord in shared:
        coord1 = cm1.coord_map[coord]
        coord2 = cm2.coord_map[coord]
        cdims = cm1.dim_map[coord]
        if cdims != cm2.dim_map[coord]:
            not_equal_coords.append(coord)
        elif coord1 == coord2:
            # Straightforward case, coords are identical.
            continue
        elif coord == dim_to_ignore or (riders_vary and dim_to_ignore in cdims):
            # If dimension that's ok to ignore value differences,
            # check whether shape is the same.
            if coord1.shape == coord2.shape:
                continue
            elif ignore_dim_eq_shape:
                not_equal_coords.append(coord)
        else:
            not_equal_coords.append(coord)
    if not_equal_coords and len(shared):
        msg = (
            f"Patches are not compatible. The following shared coordinates "
            f"are not equal: {not_equal_coords}"
        )
        warn_or_raise(msg, exception=IncompatiblePatchError, behavior=check_behavior)
        return False
    return True


def _merge_aligned_coords(cm1, cm2):
    """Merge aligned coordinates removing non coords."""
    assert cm1.dims == cm2.dims, "dimensions are not aligned"
    out = {}
    for name in set(cm1.coord_map) & set(cm2.coord_map):
        coord1 = cm1.coord_map[name]
        coord2 = cm2.coord_map[name]
        dim1, dim2 = cm1.dim_map.get(name), cm2.dim_map.get(name)
        # Coords already equal, just use first.
        if coord1.approx_equal(coord2) and dim1 == dim2:
            out[name] = (dim1, coord1)
        # Deal with Non coords
        non_count = sum([coord1._partial, coord2._partial])
        if non_count == 1:
            out[name] = (dim1, coord1 if coord2._partial else coord2)
        elif non_count == 2:
            out[name] = (dim1, coord1 if coord1.size > coord2.size else coord2)
        assert name in out
    return cm1.update(**out)


def _merge_models(attrs1, attrs2):
    """
    Fold the attrs of two same-kind patches: the first wins, the second adds.

    The caller has already checked kind; nothing here refuses a merge. An
    attribute the first leaves empty takes the second's value, so the
    result's kind is the union of the two. This is the two-operand fold,
    not `combine_patch_attrs`, which combines a collection and so treats
    a missing value as a value rather than as something to fill.
    """
    if attrs1 == attrs2:
        return attrs1
    # keep_first gives the first patch's value for everything, folds the
    # ids, and keeps the history and the attrs subclass; the data units of
    # the output are decided by the operation from each operand's own, so
    # the first's stand here whether or not it has any.
    merged = combine_patch_attrs([attrs1, attrs2], conflict="keep_first")
    # It does not fill, though, which is the one thing two operands do:
    # what the first left empty the second supplies.
    dump1 = attrs1.model_dump(exclude_defaults=True)
    fill = {
        key: value
        for key, value in attrs2.model_dump(exclude_defaults=True).items()
        if key not in _ID_FIELDS  # fold_ids returns {} when ids are disabled
        and key not in ("history", "data_units")
        and not key.startswith("_")
        and not _is_missing(value)
        and _is_missing(dump1.get(key))
    }
    return merged.update(**fill) if fill else merged


def merge_compatible_coords_attrs(
    patch1: PatchType,
    patch2: PatchType,
    *,
    dim_intersection: bool = False,
    validate_coords: bool = True,
) -> tuple[dc.core.CoordManager, dc.PatchAttrs]:
    """
    Merge the coordinates and attributes of patches or raise if incompatible.

    The rules for compatibility are:

    - The patches must be the same kind: no conflicting values for the
      attributes named by the config option `patch_kind_attrs` (see
      [`check_kind`](`dascore.utils.patch.check_kind`)).
    - Patches must share the same dimensions unless dim_intersection == True.
    - All shared dimensional coordinates must be strictly equal
    - If patches share a non-dimensional coordinate they must be equal.

    The remaining attributes never decide compatibility: the first patch's
    values are kept and attributes only the second has are added. Any
    coordinates contained by a single patch will be included in the output.

    Parameters
    ----------
    patch1
        The first patch
    patch2
        The second patch
    dim_intersection
        If True, merge if any dimensions overlap, else raise if all do not
        overlap.
    validate_coords
        If True, ensure the coords are equal, else the responsibility for this
        was handled upstream.
    """

    def _merge_coords(coords1, coords2):
        out = {}
        cmap1, cmap2 = coords1.coord_map, coords2.coord_map
        coord_names = set(cmap1) | set(cmap2)
        # fast path to update identical coordinates
        if coord_names == set(cmap1):
            return coords1
        if coord_names == set(cmap2):
            return coords2
        # otherwise just squish coords from both managers together.
        for name in coord_names:
            coord = coords1 if name in coords1.coord_map else coords2
            dims = coord.dim_map[name]
            out[name] = (dims, coord.coord_map[name])
        # Need to get coordinate that are in output, but preserve order.
        dims = _merge_tuples(coords1.dims, coords2.dims)
        return dc.core.coordmanager.get_coord_manager(out, dims=dims)

    check_kind(patch1, patch2)
    check_dims(patch1, patch2, intersection=dim_intersection)
    if validate_coords:
        check_coords(patch1, patch2)
    coord1, coord2 = patch1.coords, patch2.coords
    attrs1, attrs2 = patch1.attrs, patch2.attrs
    coord_out = _merge_coords(coord1, coord2)
    attrs = _merge_models(attrs1, attrs2)
    return coord_out, attrs


def _spool_up(func):
    """
    Spool the output of a function.

    This is primarily to turn methods that return a list of patches
    into something that can be used as a spool method.
    """

    @functools.wraps(func)
    def _wrapper(self, *args, **kwargs):
        """Wrapper for function."""
        out = func(self, *args, **kwargs)
        return dc.spool(out)

    return _wrapper


@compose_docstring(check_bev=check_behavior_description)
def concatenate_patches(
    patches: Sequence[dc.Patch] | dc.Spool,
    check_behavior: WARN_LEVELS = "warn",
    **kwargs,
) -> Sequence[dc.Patch]:
    """
    Concatenate the patches together.

    Only patches compatible with the first patch are concatenated together:
    the same kind (see [`check_kind`](`dascore.utils.patch.check_kind`);
    compared strictly, so a missing value equals only another missing
    value), the same data units, the same dimensions, and equal
    coordinates other than the concatenated one. The output carries the
    first patch's attributes.

    Parameters
    ----------
    {check_bev}
    **kwargs
        Used to specify the dimension and number of patches to merge
        together. A value of None attempts to concatenate all patches
        into as single patch.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> from dascore.utils.patch import concatenate_patches
    >>>
    >>> # Concatenate patches along time axis
    >>> out = concatenate_patches([patch, patch], time=None)
    >>> assert len(out) == 1
    >>>
    >>> # Concatenate patches along a new dimension.
    >>> # Note: This will only include the first patch if existing
    >>> # dimensions are not identical.
    >>> out = concatenate_patches([patch, patch], wave_rank=None)
    >>> assert "wave_rank" in out[0].dims
    >>>
    >>> # Concatenate patches in groups of 3.
    >>> out = concatenate_patches([patch] * 12, time=3)
    >>> assert len(out) == 4

    Notes
    -----
    - [`Spool.concatenate`](`dascore.Spool.concatenate`) is the spool
      form: it partitions patches which cannot be concatenated together
      into separate outputs instead of skipping them.
    - [`Spool.chunk`](`dascore.Spool.chunk`) performs a similar operation
      but accounts for coordinate values.
    - See also the
      [chunk section of the spool tutorial](`docs/tutorial/spool`#concatenate)
    """

    def _get_dim_and_value(kwargs):
        """Get the dimension name and value"""
        if not len(kwargs) == 1:
            msg = "Exactly one keyword argument must be passed to concatenate."
            raise ParameterError(msg)
        assert len(kwargs) == 1
        [(dim, val)] = kwargs.items()

        return dim, val

    def get_compatible_patches(patches, dim, check_behavior):
        """Get the patches which can be concatenated with the first."""
        # We need to drop private coords for dft concats to work.
        patches = list(x.drop_private_coords() for x in patches)
        first_patch = patches[0]
        compat_patches = []
        first_dims = first_patch.dims
        # Get patches compatible with first. Kind is compared strictly:
        # this combines a collection, so a missing value is a value.
        for p in patches:
            kind_ok = check_kind(first_patch, p, check_behavior, strict=True)
            if kind_ok and p.dims != first_dims:
                # a same-kind patch with other dimensions is never skipped
                msg = "Cannot concatenate patches with different dimensions."
                raise PatchCoordinateError(msg)
            coords_ok = kind_ok and check_coords(
                patch1=first_patch,
                patch2=p,
                check_behavior=check_behavior,
                dim_to_ignore=dim,
                ignore_dim_eq_shape=False,
                riders_vary=True,
            )
            if coords_ok and check_data_units(first_patch, p, check_behavior):
                compat_patches.append(p)
        return compat_patches

    dim, val = _get_dim_and_value(kwargs)
    patches = get_compatible_patches(patches, dim, check_behavior)
    fingerprint = Concatenate.from_kwargs(check_behavior=check_behavior, **kwargs)
    out = []
    for patch_list in yield_sub_sequences(patches, val):
        # The members agree on kind and units, so the first states them.
        attrs = patch_list[0].attrs
        out.append(_concatenate_group(patch_list, dim, attrs, fingerprint.fingerprint))
    return out


def _concatenate_group(
    patches: Sequence[dc.Patch],
    dim: str,
    attrs: dc.PatchAttrs,
    fingerprint: str,
) -> dc.Patch:
    """
    Concatenate patches already known to fit, along `dim`.

    The patches share dimensions and every coordinate but `dim`, which is
    either a dimension they all have (its values are concatenated) or a
    new one (each patch becomes one sample of it). `attrs` are the
    output's, history and ids aside, which this adds: the operation's own
    history entry, and the ids of every member which went in — taking the
    first patch's id would claim the result was only the first source.
    """
    first = patches[0]
    new_dim = dim not in first.dims
    dims = (*first.dims, dim) if new_dim else first.dims
    axis = dims.index(dim)
    arrays = [x.data[..., None] if new_dim else x.data for x in patches]
    data = np.concatenate(arrays, axis=axis)
    if new_dim:
        coords = first.coords.update(**{dim: (dim, len(patches))})
    else:
        members = [x.get_coord(dim) for x in patches]
        units = _lowest_units(members) or next(
            (x.units for x in members if x.units is not None), None
        )
        if units is not None:
            members = [
                x if x.units is None else x.convert_units(units) for x in members
            ]
        values = np.concatenate(_joinable(members, dim), axis=0)
        coords = first.coords.update(
            **{dim: dc.core.coords.get_coord(data=values, units=units)}
        )
        # coordinates riding the dimension which every member states the
        # same way join along it too; resizing the dimension drops them
        riders = {}
        for name, cdims in first.coords.dim_map.items():
            if name == dim or dim not in cdims:
                continue
            if not all(x.coords.dim_map.get(name) == cdims for x in patches):
                # a member lacks it, or attaches it elsewhere: values cannot
                # be invented for it, so it is left out
                continue
            # one spelling, the one the catalog keeps too: that of the
            # member lowest along the rider; a unitless member adopts it,
            # as a unitless operand conflicts with nothing
            members = [x.get_coord(name) for x in patches]
            try:
                units = _lowest_units(members)
                if units is not None:
                    members = [x.convert_units(units) for x in members]
            except (DimensionalityError, UnitError) as err:
                # seconds beside metres: the members cannot be joined at all
                msg = (
                    f"Cannot concatenate along {dim!r}: the coordinate {name!r} "
                    f"is stated in units which do not convert to one another "
                    f"({err}). Pass conflict='drop' to leave it out."
                )
                raise CoordMergeError(msg) from err
            rider_axis = cdims.index(dim)
            # a rider joins the same way its dimension does: a member which
            # states nothing takes the kind of the members which do
            joined = np.concatenate(_joinable(members, dim), axis=rider_axis)
            riders[name] = (cdims, dc.core.coords.get_coord(data=joined, units=units))
        if riders:
            coords = coords.update(**riders)
    warn_if_histories_differ([x.attrs for x in patches], "Concatenating")
    attrs = _maybe_add_history_str(attrs, "concatenate")
    attrs = stamp_combination(attrs, [x.attrs for x in patches], fingerprint)
    return dc.Patch(data=data, attrs=attrs, coords=coords, dims=dims)


def _joinable(coords, dim: str) -> list[np.ndarray]:
    """
    The coordinates' values, the value-less ones taking the others' kind.

    A member with no values along the dimension holds nothing but
    placeholders. Its own dtype cannot be trusted to join with the others'
    — floating NaN will not concatenate with datetimes, and neither will
    NaT with floats — so a blank member is rewritten as the stated
    members' own null. Where the stated kind has no null to write — whole
    numbers, booleans, text — the join is refused rather than inventing
    zeros or empty labels.

    Blankness is a question about values, not about type: a coordinate
    which states no values is blank whether it remembers being made of
    times or has forgotten. A member with no entries at all is blank too,
    but nothing is written into it, so it never forces a refusal.
    """
    arrays = [x.values for x in coords]
    blank = [bool(np.all(pd.isnull(x))) for x in arrays]
    if not any(blank):
        return arrays
    # Only the members which state something choose the kind. When none
    # do, the blanks choose among themselves, the empty ones last.
    stated = [x.dtype for x, b in zip(arrays, blank) if not b]
    if stated:
        target = np.result_type(*stated)
    else:
        voters = [x.dtype for x in arrays if x.size] or [x.dtype for x in arrays]
        try:
            target = np.result_type(*voters)
        except TypeError:
            # NaT beside NaN and nothing stated either way: no kind is the
            # right one, and no values are lost by falling back to floats.
            target = np.dtype("float64")
    written = any(b and x.size for x, b in zip(arrays, blank))
    if written and target.kind not in "fmM":
        msg = (
            f"Cannot concatenate along {dim!r}: a patch states no values "
            f"there, and a {target} coordinate has no missing value to "
            "stand in for them."
        )
        raise CoordMergeError(msg)
    null = _get_nullish(target)
    return [
        np.full(x.shape, null, dtype=target) if b else x for x, b in zip(arrays, blank)
    ]


def _lowest_units(coords):
    """The units of the unitful coordinate whose minimum is lowest, else None."""
    stated = [x for x in coords if x.units is not None]
    if not stated:
        return None
    base = stated[0].units
    lows = np.array([to_float(x.convert_units(base).min()) for x in stated])
    lows = np.where(np.isnan(lows), np.inf, lows)
    return stated[int(np.argmin(lows))].units


def concatenate_planned(
    patches: Sequence[dc.Patch],
    dim: str,
    count: int | None = None,
    conflict: Literal["drop", "raise", "keep_first"] = "raise",
) -> dc.Patch:
    """
    Concatenate the members of one planned output, as the plan decided.

    A concat plan (`dascore.utils.chunk_plan.build_concat_plan`) has
    already decided kind, dimensions, and the dimensions' identities, so
    none of that is asked again. The attrs fold as a merge folds them
    (`combine_patch_attrs`, differing values policed by `conflict`).

    Coordinates are not policed by `conflict`; they are reconciled or
    refused. A coordinate riding `dim` is joined along it, provided every
    member states it the same way; every other coordinate must agree, or
    the output raises rather than dropping metadata the catalog describes.
    """
    patches = [x.drop_private_coords() for x in patches]
    first = patches[0]
    assert all(x.dims == first.dims for x in patches), (
        "a planned output holds one set of dimensions"
    )
    names = {c for x in patches for c in x.coords.coord_map} - {dim}
    unreconciled: set[str] = set()
    for name in sorted(names):
        holders = [x for x in patches if name in x.coords.coord_map]
        cdims = holders[0].coords.dim_map[name]
        if any(x.coords.dim_map[name] != cdims for x in holders):
            # equal values on different dimensions are different coordinates
            unreconciled.add(name)
        elif dim in cdims:
            # a rider follows the dimension member by member: its values are
            # joined, not compared, and none can be invented for a member
            # which does not state it
            if len(holders) != len(patches):
                unreconciled.add(name)
        elif any(
            x.coords.coord_map[name] != holders[0].coords.coord_map[name]
            for x in holders
        ):
            # the members which state it disagree, however many they are
            unreconciled.add(name)
        elif len(holders) != len(patches) and name not in first.coords.coord_map:
            # every holder says the same thing, so it rides along
            update = {name: (cdims, holders[0].coords.coord_map[name])}
            patches[0] = first = first.update(coords=first.coords.update(**update))
    if unreconciled:
        msg = (
            f"Cannot concatenate along {dim!r}: the coordinates "
            f"{sorted(unreconciled)} cannot be reconciled — the patches hold "
            "different values, attach them to different dimensions, or some "
            "patch does not state them. Drop them before concatenating."
        )
        raise CoordMergeError(msg)
    attrs = combine_patch_attrs([x.attrs for x in patches], conflict=conflict)
    # Data units scale the data rather than labelling it, so they are
    # reconciled rather than policed by `conflict` (the plan says the same,
    # see `_carried_columns`): the output speaks the first units any member
    # states and the rest convert to them. Letting a unitless first member
    # stand would splice metre- and kilometre-scaled samples into one array.
    units = (x.attrs.data_units for x in patches)
    stated = [x for x in units if not _is_missing(x)]
    if stated:
        kept = stated[0]
        patches = [
            x
            if _is_missing(x.attrs.data_units) or x.attrs.data_units == kept
            else x.convert_units(kept)
            for x in patches
        ]
        attrs = attrs.update(data_units=kept)
    task = Concatenate(
        arguments=((dim, count),), check_behavior=None, conflict=conflict
    )
    return _concatenate_group(patches, dim, attrs, task.fingerprint)


def stack_patches(
    patches, dim_vary=None, check_behavior: WARN_LEVELS = "warn"
) -> dc.Patch:
    """
    Stack (add) all patches compatible with first patch together.

    Compatible means the same kind (see
    [`check_kind`](`dascore.utils.patch.check_kind`); compared strictly,
    so a missing value equals only another missing value), the same data
    units, the same dimensions, and equal coordinates other than
    `dim_vary`. The output carries the first patch's attributes.

    Parameters
    ----------
    patches
        The patches to stack together.
    dim_vary
        The name of the dimension which can be different in values
        (but not shape) and patches still added together.
        If None, all dimension values must be equal.
    {check_desc}

    Examples
    --------
    >>> import dascore as dc
    >>> # add a spool with equal sized patches but progressing time dim
    >>> spool = dc.get_example_spool()
    >>> stacked_patch = spool.stack(dim_vary='time')
    """
    # check the dims/coords of first patch (considered to be standard for rest)
    init_patch = patches[0]
    stack_arr = np.zeros_like(init_patch.data)

    # ensure dim_vary is in dims
    if dim_vary is not None and dim_vary not in init_patch.dims:
        msg = f"Dimension {dim_vary} is not in first patch."
        raise PatchCoordinateError(msg)

    kept = []
    for p in patches:
        # check kind, then dimensions and coords of patch against init_patch.
        # Kind is compared strictly: a collection is being combined, so a
        # missing value is a value.
        kind_ok = check_kind(init_patch, p, check_behavior, strict=True)
        dims_ok = kind_ok and check_dims(init_patch, p, check_behavior)
        coords_ok = dims_ok and check_coords(init_patch, p, check_behavior, dim_vary)
        # actually do the stacking of data
        if coords_ok and check_data_units(init_patch, p, check_behavior):
            stack_arr = stack_arr + p.data
            kept.append(p.attrs)

    # create attributes for the stack with adjusted history
    stack_attrs = _maybe_add_history_str(init_patch.attrs, "stack")
    # The kept members only: one dropped for being incompatible did not
    # contribute its data, so it is not part of what this data is.
    stack_attrs = stamp_combination(
        stack_attrs,
        kept,
        Stack(dim_vary=dim_vary, check_behavior=check_behavior).fingerprint,
    )

    # create coords array for the stack
    stack_coords = init_patch.coords
    if dim_vary:  # adjust dim_vary to start at 0 for junk dimension indicator
        coord_to_change = stack_coords.coord_map[dim_vary]
        new_dim = coord_to_change.update_limits(min=0)
        stack_coords = stack_coords.update_coords(**{dim_vary: new_dim})
    return dc.Patch(stack_arr, stack_coords, init_patch.dims, stack_attrs)


def swap_kwargs_dim_to_axis(patch, kwargs):
    """
    Convert dimension names to axis indices in kwargs.

    Parameters
    ----------
    patch : Patch
        The patch object containing dimension information.
    kwargs : dict
        Keyword arguments potentially containing 'dim' parameter.

    Returns
    -------
    dict
        The kwargs with 'dim' converted to 'axis' if present.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.utils.patch import swap_kwargs_dim_to_axis
    >>> patch = dc.get_example_patch()
    >>> kwargs = {"dim": "time", "dtype": None}
    >>> new_kwargs = swap_kwargs_dim_to_axis(patch, kwargs)
    >>> # new_kwarg = {'axis': 1, 'dtype': None}
    """
    # Only convert dim to axis if dim is explicitly provided in kwargs
    if "dim" not in kwargs:
        return kwargs

    new_kwargs = dict(kwargs)
    dim = new_kwargs.pop("dim")
    if dim is not None:
        if isinstance(dim, str):
            if dim not in patch.dims:
                msg = f"Dimension '{dim}' not found in patch dimensions {patch.dims}"
                raise ParameterError(msg)
            axis = patch.get_axis(dim)
        else:
            # Handle sequence of dimensions
            axis = []
            for d in dim:
                if d not in patch.dims:
                    msg = f"Dimension '{d}' not found in patch dimensions {patch.dims}"
                    raise ParameterError(msg)
                axis.append(patch.get_axis(d))
        new_kwargs["axis"] = axis

    return new_kwargs
