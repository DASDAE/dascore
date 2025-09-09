"""
Utilities for working with patches and arrays.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable

import numpy as np

import dascore as dc
from dascore.compat import array, is_array
from dascore.constants import _AGG_FUNCS, DEFAULT_ATTRS_TO_IGNORE, PatchType
from dascore.exceptions import ParameterError, PatchBroadcastError, UnitError
from dascore.units import DimensionalityError, Quantity, Unit, get_quantity
from dascore.utils.misc import iterate
from dascore.utils.models import ArrayLike
from dascore.utils.patch import (
    _merge_aligned_coords,
    _merge_models,
    align_patch_coords,
    get_dim_axis_value,
    patch_function,
)
from dascore.utils.time import (
    dtype_time_like,
    is_datetime64,
    to_datetime64,
    to_timedelta64,
)


@patch_function()
def apply_ufunc(
    operator: Callable,
    *args: tuple[PatchType | ArrayLike, ...],
    attrs_to_ignore=DEFAULT_ATTRS_TO_IGNORE,
    **kwargs,
) -> PatchType:
    """
    Apply a ufunc-type operator to a patch.

    This is used to implement a patch's operator overloading.

    Parameters
    ----------
    operator
        The operator. Must be numpy ufunc-like.
    *args
        Arguments to pass to the operator, can include arrays, scalars,
        and patches.
    attrs_to_ignore
        Attributes to ignore when considering if patches are compatible.
    **kwargs
        Keyword arguments to pass to the operator.

    Examples
    --------
    >>> from dascore.utils.array import apply_ufunc    >>> import numpy as np
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> # multiply the patch by 10
    >>> new = apply_ufunc(np.multiply, patch, 10)
    >>> assert np.allclose(patch.data * 10, new.data)
    >>> # add a random value to each element of patch data
    >>> noise = np.random.random(patch.shape)
    >>> new = apply_ufunc(np.add, patch, noise)
    >>> assert np.allclose(new.data, patch.data + noise)
    >>> # subtract one patch from another. Coords and attrs must be compatible
    >>> new = apply_ufunc(np.subtract, patch, patch)
    >>> assert np.allclose(new.data, 0)

    Notes
    -----
    See [numpy's ufunc docs](https://numpy.org/doc/stable/reference/ufuncs.html)
    """

    def _get_coords_attrs_from_patches(patch, other):
        """Deal with aligning two patches."""
        # Align patches so their coords are identical and data aligned.
        patch, other_patch = align_patch_coords(patch, other)
        coords = _merge_aligned_coords(patch.coords, other_patch.coords)
        # Get new attributes.
        attrs = _merge_models(
            patch.attrs,
            other_patch.attrs,
            attrs_to_ignore=attrs_to_ignore,
        )
        other = other_patch.data
        if other_units := get_quantity(other_patch.attrs.data_units):
            other = other * other_units
        return patch, other, coords, attrs

    def _ensure_array_compatible(patch, other):
        """Deal with broadcasting a patch and an array."""
        # This handles warning from quantity.
        other = other.magnitude if hasattr(other, "magnitude") else other
        other = np.asanyarray(other)
        if patch.shape == other.shape:
            return patch
        if (patch_ndims := patch.ndim) < (array_ndims := other.ndim):
            msg = f"Cannot broadcast patch/array {patch_ndims=} {array_ndims=}"
            raise PatchBroadcastError(msg)
        patch = patch.make_broadcastable_to(other.shape)
        return patch

    def _apply_op(array1, array2, operator, reversed=False):
        """Simply apply the operator, account for reversal."""
        if reversed:
            array1, array2 = array2, array1
        return operator(array1, array2, *args, **kwargs)

    def _apply_op_units(patch, other, operator, attrs, reversed=False):
        """Apply the operation handling units attached to array."""
        data_units = get_quantity(attrs.data_units)
        data = patch.data if data_units is None else patch.data * data_units
        # other is not numpy array wrapped w/ quantity, convert to quant
        if not hasattr(other, "shape"):
            other = get_quantity(other)
        try:
            new_data_w_units = _apply_op(data, other, operator, reversed=reversed)
        except DimensionalityError:
            msg = f"{operator} failed with units {data_units} and {other.units}"
            raise UnitError(msg)
        attrs = attrs.update(data_units=str(new_data_w_units.units))
        new_data = new_data_w_units.magnitude
        return new_data, attrs

    # Get indices of patches in inputs.
    patch_inds = [num for num, x in enumerate(args) if isinstance(x, dc.Patch)]
    breakpoint()
    # Dealing with multiple patches.
    if len(patch_inds) > 1:
        args, coords, attrs = _get_coords_attrs_from_patches(args, patch_inds)
    else:
        patch = _ensure_array_compatible(patch, other)
        coords, attrs = patch.coords, patch.attrs
    # Apply operation
    if isinstance(other, Quantity | Unit):
        new_data, attrs = _apply_op_units(patch, other, operator, attrs, reversed)
    else:
        new_data = _apply_op(patch.data, other, operator, reversed)

    new = patch.new(data=new_data, coords=coords, attrs=attrs)
    return new


def generate_ufunc(np_ufunc: np.ufunc) -> Callable:
    """
    Create a patch ufunc from a numpy ufunc.

    Parameters
    ----------
    np_ufunc
        Any numpy u function.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.utils.array import generate_ufunc
    >>> # Make a patch function from np add
    >>> ufunc = generate_ufunc(np.add)
    >>> # Now it can be used to operate on two patches
    >>> pat = ufunc(patch, patch)
    >>> # But can also be used to accumulate patches along dimensions.
    >>> # EG cumsum.
    >>> pat2 = ufunc.accumulate(patch, dim="time")
    >>> # Or to reduce dimensions (eg aggregate).
    >>> pat3 = ufunc.reduce(patch, dim="distance")

    Notes
    -----
    - The entire ufunc interface is not yet implemented.
    """


def _get_new_coord(coord, dim_reduce="empty"):
    """Get the new coordinate."""

    def _maybe_handle_datatypes(func, data):
        """Maybe handle the complexity of date times here."""
        try:  # First try function directly
            out = func(data)
        except Exception:  # Fall back to floats and re-packing.
            float_data = dc.to_float(data)
            dfunc = to_datetime64 if is_datetime64(data) else to_timedelta64
            out = dfunc(func(float_data))
        return np.atleast_1d(out)

    if dim_reduce == "empty":
        new_coord = coord.update(shape=(1,), start=None, stop=None, data=None)
    elif dim_reduce == "squeeze":
        return None
    elif (func := _AGG_FUNCS.get(dim_reduce)) or callable(dim_reduce):
        func = dim_reduce if callable(dim_reduce) else func
        coord_data = coord.data
        if dtype_time_like(coord_data):
            result = _maybe_handle_datatypes(func, coord_data)
        else:
            result = func(coord.data)
        new_coord = coord.update(data=result)
    else:
        msg = "dim_reduce must be 'empty', 'squeeze' or valid aggregator."
        raise ParameterError(msg)
    return new_coord


def _apply_aggregator(patch, dim, func, dim_reduce="empty"):
    """Apply an aggregation operator to patch."""
    data = patch.data
    dims = tuple(iterate(patch.dims if dim is None else dim))
    dfo = get_dim_axis_value(patch, args=dims, allow_multiple=True)
    # Iter all specified dimensions.
    for dim, axis, value in dfo:
        new_coord = _get_new_coord(patch.get_coord(dim), dim_reduce=dim_reduce)
        if new_coord is None:
            coords = patch.coords.drop_coords(dim)[0]
            data = func(data, axis=axis)
        else:
            coords = patch.coords.update(**{dim: new_coord})
            data = np.expand_dims(func(data, axis=axis), axis)
        patch = patch.new(data=data, coords=coords)
    return patch


def _find_patches(args, kwargs):
    """Return any patches in args and kwargs."""
    patches1 = [x for x in args if isinstance(x, dc.Patch)]
    patches2 = [x for x in kwargs.values() if isinstance(x, dc.Patch)]
    return patches1 + patches2


def _strip_data_from_patch(patch):
    """Get data from patches."""
    return patch.data if isinstance(patch, dc.Patch) else patch


def _get_dims_and_inds(patch, func, args, kwargs) -> bool:
    """
    Get dimension over which function was applied, and inds to re-example result.
    """

    def _normalize_axes(axis, ndim: int) -> tuple[int, ...]:
        """Normalize axis spec (int | tuple[int] | list[int] | None) to a tuple of positive ints."""
        if axis is None:
            return tuple(range(ndim))  # reduce over all axes
        if isinstance(axis, Iterable) and not isinstance(axis, (str, bytes)):
            axes = tuple(int(a) for a in axis)
        else:
            axes = (int(axis),)
        # normalize negatives and dedupe but preserve order
        seen = set()
        out = []
        for a in axes:
            a = a % ndim
            if a not in seen:
                seen.add(a)
                out.append(a)
        return tuple(out)

    def get_effective_axes(func, args, kwargs, ndim: int) -> tuple[int, ...]:
        """
        Return the axes the function will operate on, normalized to positive indices.
        If the function has no 'axis' parameter, returns an empty tuple (unknown/no-op).
        """
        sig = inspect.signature(func)
        params = sig.parameters
        # bind *partially* so missing, defaulted params are allowed
        bound = sig.bind_partial(*args, **(kwargs or {}))

        if "axis" not in params:
            return ()  # function doesn't expose 'axis'; handle separately

        axis = bound.arguments.get("axis", params["axis"].default)
        if axis is inspect._empty:
            axis = None  # treat as default

        return _normalize_axes(axis, ndim)

    axes = get_effective_axes(func, args, kwargs, patch.ndim)
    # get indexes needed to re-expand array
    inds = tuple(
        None if num in axes else slice(None, None) for num in range(patch.ndim)
    )
    dims = tuple(patch.dims[x] for x in axes)
    return dims, inds


def _reassemble_patch(result, patch, func, args, kwargs):
    """
    Method to put the patch back together.
    """
    # Simples case, data shape hasn't changed.
    if result.shape == patch.shape:
        return patch.new(data=result)
    # Otherwise, we have to do some detective work to figure out what happened
    # to the array so we can adjust the coords accordingly.
    sig = inspect.signature(func)
    if "axis" in sig.parameters:
        # get axis argument regardless of if it was positional or keyword.
        dims, inds = _get_dims_and_inds(patch, func, args, kwargs)
        # re-expand array.
        result = result[inds]
        new_coords = {x: _get_new_coord(patch.get_coord(x)) for x in dims}
        cm = patch.coords.update(**new_coords)
        return patch.new(data=result, coords=cm)


def array_function(self, func, types, args, kwargs):
    """
    Intercept NumPy functions for patch operations.

    Parameters
    ----------
    func : callable
        The NumPy function being called.
    types : tuple
        Types involved in the call.
    args : tuple
        Positional arguments to the function.
    kwargs : dict
        Keyword arguments to the function.
    """
    # Only handle functions involving Patches
    if not any(issubclass(t, dc.Patch) for t in types):
        return NotImplemented

    _args = tuple(_strip_data_from_patch(a) for a in args)
    _kwargs = {k: _strip_data_from_patch(v) for k, v in kwargs.items()}
    patches = _find_patches(args, kwargs)
    assert len(patches) > 0

    # Call the array function
    result = func(*_args, **_kwargs)

    # If we didn't get an array back, try to package it as an array
    if not is_array(result):
        result = array(result)

    # Then we need to put the array back into the patch, but account for
    # dimensions that may have changed.
    patch = _reassemble_patch(result, self, func, args, kwargs)
    return patch


def array_ufunc(self, ufunc, method, *inputs, **kwargs):
    """
    Called when a numpy array is ufunc'ed against a patch.
    """
    if method == "__call__":
        # pull out patch and other thing.
        out = apply_ufunc(ufunc, *inputs, **kwargs)
    elif method == "reduce":
        return array_function(self, ufunc, (type(self),), inputs, kwargs)

    return out
