"""
Utilities for working with patches and arrays.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable

import numpy as np

import dascore as dc
from dascore.compat import array, is_array
from dascore.constants import DEFAULT_ATTRS_TO_IGNORE, PatchType
from dascore.exceptions import ParameterError, PatchBroadcastError, UnitError
from dascore.units import DimensionalityError, Quantity, Unit, get_quantity
from dascore.utils.misc import iterate
from dascore.utils.models import ArrayLike
from dascore.utils.patch import (
    _merge_aligned_coords,
    _merge_models,
    align_patch_coords,
    get_dim_axis_value,
    swap_kwargs_dim_to_axis,
)


def _dummy_accumulate(array, axis=0, dtype=None, out=None):
    """A dummy accumulate function since inspect.signature fails."""


def _dummy_reduce(array, axis=0, dtype=None, out=None, keepdims=False, **kwargs):
    """A dummy reduce function since inspect.signature fails."""


def _raise_on_out(kwargs):
    """DASCore doesn't support the out parameter. Check for it and raise."""
    if "out" in kwargs and kwargs["out"] is not None:
        msg = (
            "Since patches are immutable, the 'out' parameter cannot be "
            "used in patch functions."
        )
        raise ParameterError(msg)


def _clear_units_if_bool_dtype(patch):
    """Clear the units on the patch if it is a boolean."""
    dtype = getattr(patch, "dtype", None)
    if dtype is not None and np.issubdtype(dtype, np.bool_):
        return patch.update_attrs(data_units=None)
    return patch


def _apply_unary_ufunc(operator: np.ufunc, patch, *args, **kwargs):
    """
    Create a patch from a unary ufunc.

    Parameters
    ----------
    operator
        The operator. Must be numpy ufunc-like.
    patch
        The patch instance.
    *args
        Arguments to pass to the operator, can include arrays, scalars,
        and patches.
    **kwargs
        Keyword arguments to pass to the operator.

    Notes
    -----
    We assume the shape of the array won't change.
    """
    out = operator(patch.data, *args, **kwargs)
    return patch.new(data=out)


def _apply_binary_ufunc(
    operator: np.ufunc,
    patch: PatchType | ArrayLike,
    other: PatchType | ArrayLike,
    *args: tuple[PatchType | ArrayLike, ...],
    attrs_to_ignore=DEFAULT_ATTRS_TO_IGNORE,
    **kwargs,
) -> PatchType:
    """
    Apply a binary ufunc-type operator to one or more patches.

    The input should be 2 and the output 1.

    This is used to implement a patch's operator overloading.

    Parameters
    ----------
    operator
        The operator. Must be numpy ufunc-like.
    patch
        The patch instance.
    other
        The other object to apply the operator element-wise. Must be either a
        non-patch which is broadcastable to the shape of the patch's data, or
        a patch which has compatible coordinates. If units are provided they
        must be compatible.
    *args
        Arguments to pass to the operator, can include arrays, scalars,
        and patches.
    attrs_to_ignore
        Attributes to ignore when considering if patches are compatible.
    **kwargs
        Keyword arguments to pass to the operator.

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
        except DimensionalityError as er:
            msg = f"{operator} failed with units {data_units} and {other.units}"
            raise UnitError(msg) from er
        # Check if result has units (comparison operators return plain arrays)
        if hasattr(new_data_w_units, "units"):
            attrs = attrs.update(data_units=str(new_data_w_units.units))
            new_data = new_data_w_units.magnitude
        else:
            # Result is unitless (e.g., from boolean comparison)
            attrs = attrs.update(data_units=None)
            new_data = new_data_w_units
        return new_data, attrs

    # Count patch operands (we only support binary ops on patches).
    patch_is_patch = isinstance(patch, dc.Patch)
    other_is_patch = isinstance(other, dc.Patch)
    patch_count = int(patch_is_patch) + int(other_is_patch)
    # We only support binary ops on patch.
    assert patch_count >= 1, "apply_ufunc requires at least one Patch operand."
    # Make sure patch is a patch. Reverse if needed.
    reversed = False
    if other_is_patch and not patch_is_patch:
        patch, other = other, patch
        reversed = True

    if patch_count > 1:
        patch, other, coords, attrs = _get_coords_attrs_from_patches(patch, other)
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


class _BoundPatchUFunc:
    """A ufunc bound to a specific patch instance."""

    def __init__(self, np_ufunc, patch):
        self.np_ufunc = np_ufunc
        self.patch = patch
        self.__name__ = getattr(np_ufunc, "__name__", "patch_ufunc")
        self.__doc__ = getattr(np_ufunc, "__doc__", None)

    def __call__(self, *args, **kwargs):
        """Call the ufunc with the bound patch."""
        return apply_ufunc(self.np_ufunc, self.patch, *args, **kwargs)

    def reduce(self, dim=None, dtype=None, **kwargs):
        """Apply ufunc reduction along specified dimensions."""
        return apply_ufunc(
            self.np_ufunc.reduce, self.patch, dim=dim, dtype=dtype, **kwargs
        )

    def accumulate(self, dim=None, dtype=None, **kwargs):
        """Apply ufunc accumulation along specified dimensions."""
        return apply_ufunc(
            self.np_ufunc.accumulate, self.patch, dim=dim, dtype=dtype, **kwargs
        )


class PatchUFunc:
    """
    A ufunc wrapper that can be applied to patches with dimension support.

    This class wraps numpy ufuncs to work seamlessly with DASCore patches,
    providing support for dimension-aware operations, coordinate preservation,
    and method binding.

    Parameters
    ----------
    np_ufunc : np.ufunc
        The numpy ufunc to wrap.

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> from dascore.utils.array import PatchUFunc
    >>>
    >>> # Create a patch ufunc from np.add
    >>> add_ufunc = PatchUFunc(np.add)
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Use it to operate on patches
    >>> result = add_ufunc(patch, patch)
    >>>
    >>> # Use accumulate and reduce methods
    >>> cumsum = add_ufunc.accumulate(patch, dim="time")
    >>> total = add_ufunc.reduce(patch, dim="distance")
    >>>
    >>> # Bind to patch instance for cleaner syntax
    >>> bound_ufunc = add_ufunc.__get__(patch, type(patch))
    >>> result2 = bound_ufunc.reduce(dim="time")  # No patch argument needed

    Notes
    -----
    - Automatically handles dimension-to-axis conversion when `dim` parameter is used
    - Preserves patch coordinates and attributes appropriately
    - Supports method binding via the descriptor protocol
    - Methods call `dascore.utils.array.apply_ufunc` under the hood
    """

    def __init__(self, np_ufunc):
        self.np_ufunc = np_ufunc
        self.__name__ = getattr(np_ufunc, "__name__", "patch_ufunc")
        self.__doc__ = getattr(np_ufunc, "__doc__", None)

    def __get__(self, obj, objtype=None):
        """Bind to a patch instance, returning a _BoundPatchUFunc."""
        if obj is None:
            return self
        return _BoundPatchUFunc(self.np_ufunc, obj)

    def __call__(self, patch, *args, **kwargs):
        """Call the ufunc with a patch as first argument."""
        return apply_ufunc(self.np_ufunc, patch, *args, **kwargs)

    def reduce(self, patch, dim=None, dtype=None, **kwargs):
        """Apply ufunc reduction along specified dimensions."""
        return apply_ufunc(self.np_ufunc.reduce, patch, dim=dim, dtype=dtype, **kwargs)

    def accumulate(self, patch, dim=None, dtype=None, **kwargs):
        """Apply ufunc accumulation along specified dimensions."""
        return apply_ufunc(
            self.np_ufunc.accumulate, patch, dim=dim, dtype=dtype, **kwargs
        )


def _apply_aggregator(patch, dim, func, dim_reduce="empty"):
    """Apply an aggregation operator to patch."""
    data = patch.data
    dims = tuple(iterate(patch.dims if dim is None else dim))
    dfo = get_dim_axis_value(patch, args=dims, allow_multiple=True)
    # Iter all specified dimensions.
    for dim, axis, _ in dfo:
        new_coord = patch.get_coord(dim).reduce_coord(dim_reduce=dim_reduce)
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


def _get_dims_and_inds_from_signature(
    patch, sig, args, kwargs
) -> tuple[tuple[str, ...], tuple]:
    """
    Get dimension over which function was applied, and indices to re-expand result.
    """

    def _normalize_axes(axis, ndim: int) -> tuple[int, ...]:
        """
        Normalize axis spec (int | tuple[int] | list[int] | None)
        to a tuple of positive ints.
        """
        if axis is None:
            return tuple(range(ndim))  # reduce over all axes
        if isinstance(axis, Iterable) and not isinstance(axis, str | bytes):
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

    def get_effective_axes(sig, args, kwargs, ndim: int) -> tuple[int, ...]:
        """
        Return the axes the function will operate on, normalized to positive indices.
        If the function has no 'axis' parameter, returns an empty tuple (unknown/no-op).
        """
        params = sig.parameters
        # bind *partially* so missing, defaulted params are allowed
        bound = sig.bind_partial(*args, **(kwargs or {}))
        assert "axis" in params, "axis must be defined in signature"
        axis = bound.arguments.get("axis", params["axis"].default)
        if axis is inspect._empty or axis is None:
            axis = None  # treat as default

        return _normalize_axes(axis, ndim)

    axes = get_effective_axes(sig, args, kwargs, patch.ndim)
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
    # Simple case, data shape hasn't changed.
    if result.shape == patch.shape:
        return patch.new(data=result)
    # Otherwise, we have to do some detective work to figure out what happened
    # to the array so we can adjust the coords accordingly.
    try:
        sig = inspect.signature(func)
    # For some reason, some ufuncs methods like reduce don't have signatures.
    except (ValueError, AttributeError):
        ufunc_method_name = {"reduce": _dummy_reduce, "accumulate": _dummy_accumulate}
        name = getattr(func, "__name__", None)
        assert name in ufunc_method_name
        sig = inspect.signature(ufunc_method_name[name])

    if "axis" in sig.parameters:
        dims, inds = _get_dims_and_inds_from_signature(patch, sig, args, kwargs)
        # re-expand array.
        result = result[inds]
        new_coords = {x: patch.get_coord(x).reduce_coord() for x in dims}
        cm = patch.coords.update(**new_coords)
        return patch.new(data=result, coords=cm)
    else:
        # This case should have already been caught at the start of this function.
        assert result.shape != patch.shape
        func_name = getattr(func, "__name__", None)
        msg = f"Cannot reassemble result of {func_name} without an axis parameter."
        raise ParameterError(msg)


def apply_array_func(func, *args, **kwargs):
    """
    Apply an array function.
    """
    _raise_on_out(kwargs)
    # Only handle functions involving Patches
    patches = _find_patches(args, kwargs)
    assert len(patches), "No patches found in apply_array_func"
    first_patch = patches[0]

    # Convert dim to axis for numpy functions
    converted_kwargs = swap_kwargs_dim_to_axis(first_patch, kwargs)
    converted_args = args

    _args = tuple(_strip_data_from_patch(a) for a in converted_args)
    _kwargs = {k: _strip_data_from_patch(v) for k, v in converted_kwargs.items()}

    # Call the array function
    result = func(*_args, **_kwargs)
    # If we didn't get an array back, try to package it as an array
    if not is_array(result):
        result = array(result)
    # Then we need to put the array back into the patch, but account for
    # dimensions that may have changed.
    patch = _reassemble_patch(
        result, first_patch, func, converted_args, converted_kwargs
    )
    return _clear_units_if_bool_dtype(patch)


# Mapping of ufunc dispatches. Keys are method name or num input/num output.
UFUNC_MAP = {
    (2, 1): _apply_binary_ufunc,
    (1, 1): _apply_unary_ufunc,
    "reduce": apply_array_func,
    "accumulate": apply_array_func,
}


def apply_ufunc(ufunc, *args, **kwargs):
    """
    Apply a ufunc to one or more patches.

    Parameters
    ----------
    ufunc
        The ufunc to use.
    *args
        Additional positional arguments, can contain patches.
    **kwargs
        Keyword arguments, can contain patches.

    Examples
    --------
    >>> from dascore.utils.array import apply_ufunc
    >>> import numpy as np
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Call abs on a patch.
    >>> new1 = apply_ufunc(np.abs, patch)
    >>>
    >>> # Add two patches
    >>> new2 = apply_ufunc(np.add, patch, patch)
    >>>
    >>> # multiply the patch by 10
    >>> new3 = apply_ufunc(np.multiply, patch, 10)

    Notes
    -----
    See [numpy's ufunc docs](https://numpy.org/doc/stable/reference/ufuncs.html)
    """
    _raise_on_out(kwargs)
    is_ufunc = isinstance(ufunc, np.ufunc)
    if is_ufunc:
        key = (ufunc.nin, ufunc.nout)
    else:
        key = ufunc.__name__
    # Handle bad key.
    if (func := UFUNC_MAP.get(key, None)) is None:
        msg = (
            f"ufuncs with input/output numbers or method of {key} are not "
            f"supported for use with Patch. Use the patch.data array directly."
        )
        raise ParameterError(msg)
    out = func(ufunc, *args, **kwargs)
    return _clear_units_if_bool_dtype(out)


def patch_array_ufunc(self, ufunc, method, *inputs, **kwargs):
    """
    Called when a numpy ufunc is applied to a patch (__array_ufunc__).
    """
    method = ufunc if method == "__call__" else getattr(ufunc, method, ufunc)
    return apply_ufunc(method, *inputs, **kwargs)


def patch_array_function(self, func, types, args, kwargs):
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
    assert any(issubclass(t, dc.Patch) for t in types)
    return apply_array_func(func, *args, **kwargs)
