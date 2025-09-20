"""Utilities for working with the Patch class."""

from __future__ import annotations

import functools
import inspect
import sys
import warnings
from collections import namedtuple
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import pydantic
from pydantic import TypeAdapter

import dascore as dc
from dascore.constants import (
    DEFAULT_ATTRS_TO_IGNORE,
    FLOAT_PRECISION,
    WARN_LEVELS,
    PatchType,
    SpoolType,
    check_behavior_description,
    dascore_styles,
)
from dascore.exceptions import (
    CoordDataError,
    IncompatiblePatchError,
    ParameterError,
    PatchAttributeError,
    PatchCoordinateError,
)
from dascore.units import get_quantity
from dascore.utils.attrs import combine_patch_attrs
from dascore.utils.coordmanager import merge_coord_managers
from dascore.utils.deprecate import deprecate
from dascore.utils.docs import compose_docstring
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import (
    _apply_union_indexers,
    _merge_tuples,
    all_diffs_close_enough,
    get_middle_value,
    iterate,
    to_object_array,
    warn_or_raise,
    yield_sub_sequences,
)
from dascore.utils.time import to_float

attr_type = dict[str, Any] | str | Sequence[str] | None

_DimAxisValue = namedtuple("DimAxisValue", ["dim", "axis", "value"])


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


def check_patch_coords(
    patch: PatchType,
    dims: Sequence[str] | None = None,
    coords: Sequence[str] | None = None,
):
    """
    Check if a patch object has required coordinates, else raise PatchDimError.

    Parameters
    ----------
    patch
        The input patch
    dims
        A tuple of required dimension names.
    coords
        A tuple of required coordinate names.
    """
    missing = set()
    if dims is not None:
        dim_set = set(patch.dims)
        missing |= set(dims) - dim_set
    if coords is not None:
        coord_set = set(patch.coords.coord_map)
        missing |= set(coords) - coord_set
    if missing:
        msg = f"patch is missing required coordinates: {tuple(missing)}"
        raise PatchCoordinateError(msg)
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
    required_coords: tuple[str, ...] | None = None,
    required_attrs: attr_type = None,
    history: Literal["full", "method_name", None] = "full",
    validate_call: bool = False,
):
    """
    Decorator to mark a function as a patch method.

    Parameters
    ----------
    required_dims
        A tuple of dimensions which must be found in the Patch.
    required_coords
        A tuple of coordinates which must be found in the Patch.
    required_attrs
        A dict of attributes which must be found in the Patch and whose
        values must be equal to those provided.
    history
        Specifies how to track history on Patch.
            Full - Records function name and str version of input arguments.
            method_name - Only records method name. Useful if args are long.
            None - Function call is not recorded in history attribute.
    validate_call
        If True, use pydantic to validate the function call. This can save
        quite a lot of code in validation checks, but does have some overhead.
        See [validate_call](https://docs.pydantic.dev/latest/api/validate_call/).

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

    Notes
    -----
    - The original function can still be accessed with the raw_function
      attribute. This may be useful for avoiding calling the patch_func
      machinery multiple times from within another patch function.

    - If using `PatchType` or `SpoolType` type variables from the
      [constants module](`dascore.constants`), make sure dascore is imported
      as dc at the top of the file where the patch function is defined so
      the forward refs can be resolved properly for type checking.
    """

    def _wrapper(func):
        if validate_call:
            config = dict(arbitrary_types_allowed=True)
            func = pydantic.validate_call(func, config=config)

        @functools.wraps(func)
        def _func(patch, *args, **kwargs):
            check_patch_coords(
                patch,
                dims=required_dims,
                coords=required_coords,
            )
            check_patch_attrs(patch, required_attrs)
            out: PatchType = func(patch, *args, **kwargs)
            # attach history string. Need to consider something a bit less hacky.
            if out is not patch and hasattr(out, "attrs"):
                hist_str = _get_history_str(
                    patch, func, *args, _history=history, **kwargs
                )
                if hist_str:
                    hist = list(out.attrs.history)
                    hist.append(hist_str)
                    out = out.update_attrs(history=hist)
            return out

        # Attach original function. Although we want to encourage raw_function
        # for consistency with pydantic, we leave this to not break old code.
        _func.func = getattr(func, "raw_function", func)
        # matches pydantic naming.
        _func.raw_function = getattr(func, "raw_function", func)
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
    # Handle spool case
    if hasattr(patches, "get_contents"):
        df = patches.get_contents()
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
        df["patch"] = to_object_array(patches)
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

    def _get_new_coord(df, merge_dim, coords, drop_conflicting=False):
        """Get new coordinates, also validate anticipated sampling."""
        new_coord = merge_coord_managers(
            coords, dim=merge_dim, drop_conflicting=drop_conflicting
        )
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
    axis = patches[0].get_axis(merge_dim)
    # get data, coords, attrs for merging patch together.
    datas = [x.data for x in patches]
    coords = [x.coords for x in patches]
    attrs = [x.attrs for x in patches]
    new_data = np.concatenate(datas, axis=axis)
    # Determine if conflicting non-dimensional coords should be dropped.
    conf = merge_kwargs.get("conflicts", None)
    drop_conf_coords = True if conf in {"drop", "keep_first"} else False
    new_coord = _get_new_coord(df, merge_dim, coords, drop_conf_coords)
    coord = new_coord.coord_map[merge_dim] if merge_dim in dims else None
    new_attrs = combine_patch_attrs(attrs, merge_dim, coord=coord, **merge_kwargs)
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
    patch_data: pd.DataFrame | dc.Patch | dc.BaseSpool,
    prefix="DAS",
    attrs=("network", "station", "tag"),
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

    The second is when a column called "path" exists. In this case, the
    output will be the file name with the extension removed (if
    strip_extension). The path must use '/' as a delinater.

    See Also
    --------
    - [`Patch.get_patch_name`](`dascore.Patch.get_patch_name`)
    - [`Spool.get_patch_names`](`dascore.BaseSpool.get_patch_names`)

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
            file_names = [x[-1].split(".")[0] for x in split_ser]
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
    if "path" in col_set:
        return _get_filename(df["path"], strip_extension)
    # Determine the requested fields and get the ones that are there.
    coord_fields = zip([f"{x}_min" for x in coords], [f"{x}_max" for x in coords])
    requested_fields = list(attrs) + list(*coord_fields)
    current = set(df.columns)
    fields = [x for x in requested_fields if x in current]
    # Get a sub dataframe and convert any datetime things to strings.
    sub = df[fields].pipe(_format_time_columns).fillna("").astype(str)
    out = f"{prefix}_{sep}" + sub[fields[0]].str.cat(sub[fields[1:]], sep=sep)
    return out


def get_dim_axis_value(
    patch: PatchType,
    *,
    args: tuple = tuple(),
    kwargs: dict = FrozenDict(),
    arg_keys: tuple[str] = ("dim", "coord", "dims", "coords"),
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
    """
    Get window sizes for patch processing operations.

    Parameters
    ----------
    patch
        The input patch.
    kwargs
        Keyword arguments specifying dimension names and their window sizes.
    samples
        If True, kwargs values are in samples; if False, in coordinate units.
    require_odd
        If True, require odd window sizes. When samples=False, even sizes
        are adjusted to be odd. When samples=True, even sizes raise ParameterError.
    warn_above
        If specified, warn when any dimension window size exceeds this value.
    min_samples
        Minimum number of samples required per dimension.
    enforce_lt_coord
        If True, reject windows larger than coordinate length.

    Returns
    -------
    Tuple of window sizes for each dimension of the patch data.

    Raises
    ------
    ParameterError
        If window sizes are too small, or if require_odd=True and samples=True
        but window size is even.
    """
    # Handle empty kwargs case - return all ones
    if not kwargs:
        return tuple([1] * patch.data.ndim)

    aggs = get_dim_axis_value(patch, kwargs=kwargs, allow_multiple=True)
    size = [1] * patch.data.ndim

    for name, axis, val in aggs:
        coord = patch.get_coord(name, require_evenly_sampled=True)
        samps = coord.get_sample_count(
            val, samples=samples, enforce_lt_coord=enforce_lt_coord
        )

        # Check minimum samples requirement
        if samps < min_samples:
            msg = (
                f"Window must have at least {min_samples} samples along each "
                f"dimension. {name} has {samps} samples. Try increasing its value."
            )
            raise ParameterError(msg)

        # Handle odd number requirement
        if require_odd and (samps % 2 != 1):
            if not samples:
                # Adjust even sizes to odd when samples=False
                samps += 1
            else:
                # Raise error when samples=True and size is even
                msg = (
                    f"For clean median calculation, dimension windows must be odd "
                    f"but {name} has a value of {samps} samples."
                )
                raise ParameterError(msg)

        # Issue warning for large window sizes
        if warn_above is not None and samps > warn_above:
            msg = (
                f"Large window size ({samps} samples) in dimension '{name}' "
                f"may result in slow performance. Consider reducing the window size."
            )
            warnings.warn(msg, UserWarning, stacklevel=3)

        size[axis] = samps

    return tuple(size)


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
        axes.append(patch.get_axis(dim_))

    return tuple(out), tuple(axes)


def align_patch_coords(
    patch1: PatchType, patch2: PatchType
) -> tuple[PatchType, PatchType]:
    """
    Align patches so they have compatible attrs and data broadcast together.

    Parameters
    ----------
    patch1
        The first patch.
    patch2
        The second patch.
    """
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
        'warn' will provide a warning, None will do nothing.
    intersection
        If True, allow any intersection of dimensions to pass. This is useful
        when only broad-castablity needs to be checked. If false require dims
        to be equal.
    """
    dims1, dims2 = patch1.dims, patch2.dims
    dims_ok = True
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
    return dims_ok


def check_coords(
    patch1,
    patch2,
    check_behavior: WARN_LEVELS = "raise",
    dim_to_ignore=None,
    ignore_dim_eq_shape=True,
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
        but not shape, are allowed.
    ignore_dim_eq_shape
        If True, the ignored dims must be equal shape to pass check.
        If dim_to_ignore is None this has no effect.
    """
    cm1 = patch1.coords
    cm2 = patch2.coords
    cset1, cset2 = set(cm1.coord_map), set(cm2.coord_map)
    shared = cset1 & cset2
    not_equal_coords = []
    for coord in shared:
        coord1 = cm1.coord_map[coord]
        coord2 = cm2.coord_map[coord]
        if coord1 == coord2:
            # Straightforward case, coords are identical.
            continue
        elif coord == dim_to_ignore:
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
            f"are not equal: {coord}"
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


def _merge_models(attrs1, attrs2, coord=None, attrs_to_ignore=DEFAULT_ATTRS_TO_IGNORE):
    """Ensure models are equal in the right ways, merge together."""
    no_comp_keys = set(attrs_to_ignore)
    if attrs1 == attrs2:
        return attrs1
    dict1, dict2 = dict(attrs1), dict(attrs2)
    if coord is not None:
        new_coords = coord.to_summary_dict()
        dict1["coords"], dict2["coords"] = new_coords, new_coords
    else:
        dict1.pop("coords"), dict2.pop("coords")
    common_keys = set(dict1) & set(dict2)
    ne_attrs = []
    for key in common_keys:
        if key in no_comp_keys:
            continue
        if dict2[key] != dict1[key]:
            ne_attrs.append(key)
    if ne_attrs:
        msg = (
            "Patches are not compatible because the following attributes "
            f"are not equal. {ne_attrs}"
        )
        raise IncompatiblePatchError(msg)
    return combine_patch_attrs([dict1, dict2], conflicts="keep_first")


def merge_compatible_coords_attrs(
    patch1: PatchType,
    patch2: PatchType,
    attrs_to_ignore=DEFAULT_ATTRS_TO_IGNORE,
    dim_intersection: bool = False,
    validate_coords: bool = True,
) -> tuple[dc.core.CoordManager, dc.PatchAttrs]:
    """
    Merge the coordinates and attributes of patches or raise if incompatible.

    The rules for compatibility are:

    - All attrs must be equal other than those specified in attrs_to_ignore.
    - Patches must share the same dimensions unless dim_intersection == True.
    - All shared dimensional coordinates must be strictly equal
    - If patches share a non-dimensional coordinate they must be equal.

    Any coordinates or attributes contained by a single patch will be included
    in the output.

    Parameters
    ----------
    patch1
        The first patch
    patch2
        The second patch
    attrs_to_ignore
        A sequence of attributes to not consider in equality. Only these
        attributes from the first patch are kept in outputs.
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

    check_dims(patch1, patch2, intersection=dim_intersection)
    if validate_coords:
        check_coords(patch1, patch2)
    coord1, coord2 = patch1.coords, patch2.coords
    attrs1, attrs2 = patch1.attrs, patch2.attrs
    coord_out = _merge_coords(coord1, coord2)
    attrs = _merge_models(attrs1, attrs2, coord_out, attrs_to_ignore=attrs_to_ignore)
    return coord_out, attrs


def _add_history_str(attrs, hist_str):
    """Add a single string to the history attribute in attrs."""
    new_history = list(attrs.history)
    new_history.append(hist_str)
    return attrs.update(history=new_history)


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
    patches: Sequence[dc.Patch] | dc.BaseSpool,
    check_behavior: WARN_LEVELS = "warn",
    **kwargs,
) -> Sequence[dc.Patch]:
    """
    Concatenate the patches together.

    Only patches which are compatible with the first patch are concatenated
    together.

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
    >>> # Concatenate patches along time axis
    >>> spool = dc.spool([patch, patch])
    >>> spool_concat = spool.concatenate(time=None)
    >>> assert len(spool_concat) == 1
    >>>
    >>> # Concatenate patches along a new dimension.
    >>> # Note: This will only include the first patch if existing
    >>> # dimensions are not identical.
    >>> spool_concat = spool.concatenate(wave_rank=None)
    >>> assert "wave_rank" in spool_concat[0].dims
    >>>
    >>> # Concatenate patches in groups of 3.
    >>> big_spool = dc.spool([patch] * 12)
    >>> spool_concat = big_spool.concatenate(time=3)
    >>> assert len(spool_concat) == 4

    Notes
    -----
    - [`Spool.chunk`](`dascore.BaseSpool.chunk`) performs a similar operation
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
        """Get the patches which can be concatenated, dim names, and new dim."""
        # We need to drop private coords for dft concats to work.
        patches = list(x.drop_private_coords() for x in patches)
        first_patch = patches[0]
        compat_patches = []
        # Ensure patch dimensions are compatible.
        dim_set = {x.dims for x in patches}
        if not len(dim_set) == 1:
            msg = "Cannot concatenate patches with different dimensions."
            raise PatchCoordinateError(msg)
        # Get dim name and such
        first_dims = next(iter(dim_set))
        new_dim = dim not in first_dims
        dims = tuple([*list(first_dims), dim]) if new_dim else first_dims
        # Get patches compatible with first.
        for p in patches:
            dims_ok = check_dims(first_patch, p, check_behavior)
            coords_ok = check_coords(
                patch1=first_patch,
                patch2=p,
                check_behavior=check_behavior,
                dim_to_ignore=dim,
                ignore_dim_eq_shape=False,
            )
            if dims_ok and coords_ok:
                compat_patches.append(p)
        return compat_patches, dims, new_dim

    def get_output_array(patches, axis, new_dim):
        """Get a list of output arrays."""
        sub_arrays = [x.data[..., None] if new_dim else x.data for x in patches]
        out = np.concatenate(sub_arrays, axis=axis)
        return out

    def _get_new_coords(patch_list, dim, new_dim):
        """Get new coordinates for creating patch."""
        coords = patch_list[0].coords
        if new_dim:
            coords = coords.update(**{dim: (dim, len(patch_list))})
        else:
            array_list = [x.get_array(dim) for x in patch_list]
            array = np.concatenate(array_list, axis=0)
            coords = coords.update(**{dim: array})
        return coords

    dim, val = _get_dim_and_value(kwargs)
    patches, dims, new_dim = get_compatible_patches(patches, dim, check_behavior)
    out = []
    for patch_list in yield_sub_sequences(patches, val):
        ar = get_output_array(patch_list, dims.index(dim), new_dim)
        attrs = _add_history_str(patch_list[0].attrs, "concatenate")
        coords = _get_new_coords(patch_list, dim, new_dim)
        out.append(dc.Patch(data=ar, attrs=attrs, coords=coords, dims=dims))
    return out


@compose_docstring(check_desc=check_behavior_description)
def stack_patches(
    patches, dim_vary=None, check_behavior: WARN_LEVELS = "warn"
) -> PatchType:
    """
    Stack (add) all patches compatible with first patch together.

    Parameters
    ----------
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

    for p in patches:
        # check dimensions of patch compared to init_patch
        dims_ok = check_dims(init_patch, p, check_behavior)
        coords_ok = check_coords(init_patch, p, check_behavior, dim_vary)
        # actually do the stacking of data
        if dims_ok and coords_ok:
            stack_arr = stack_arr + p.data

    # create attributes for the stack with adjusted history
    stack_attrs = _add_history_str(init_patch.attrs, "stack")

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
                from dascore.exceptions import ParameterError

                msg = f"Dimension '{dim}' not found in patch dimensions {patch.dims}"
                raise ParameterError(msg)
            axis = patch.get_axis(dim)
        else:
            # Handle sequence of dimensions
            axis = []
            for d in dim:
                if d not in patch.dims:
                    from dascore.exceptions import ParameterError

                    msg = f"Dimension '{d}' not found in patch dimensions {patch.dims}"
                    raise ParameterError(msg)
                axis.append(patch.get_axis(d))
        new_kwargs["axis"] = axis

    return new_kwargs
