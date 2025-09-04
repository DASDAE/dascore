"""
Utilities for working with/creating universal functions.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

import dascore as dc
from dascore.constants import DEFAULT_ATTRS_TO_IGNORE, PatchType
from dascore.exceptions import PatchBroadcastError, UnitError
from dascore.units import DimensionalityError, Quantity, Unit, get_quantity
from dascore.utils.models import ArrayLike
from dascore.utils.patch import (
    _merge_aligned_coords,
    _merge_models,
    align_patch_coords,
    patch_function,
)


@patch_function()
def apply_ufunc(
    patch: PatchType | ArrayLike,
    other: PatchType | ArrayLike,
    operator: Callable,
    *args,
    attrs_to_ignore=DEFAULT_ATTRS_TO_IGNORE,
    **kwargs,
) -> PatchType:
    """
    Apply a ufunc-type operator to a patch.

    This is used to implement a patch's operator overloading.

    Parameters
    ----------
    patch
        The patch instance.
    other
        The other object to apply the operator element-wise. Must be either a
        non-patch which is broadcastable to the shape of the patch's data, or
        a patch which has compatible coordinates. If units are provided they
        must be compatible.
    operator
        The operator. Must be numpy ufunc-like.
    *args
        Arguments to pass to the operator.
    attrs_to_ignore
        Attributes to ignore when considering if patches are compatible.
    **kwargs
        Keyword arguments to pass to the operator.

    Examples
    --------
    >>> from dascore.proc.ufuncs import apply_ufunc    >>> import numpy as np
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> # multiply the patch by 10
    >>> new = apply_ufunc(patch, 10, np.multiply)
    >>> assert np.allclose(patch.data * 10, new.data)
    >>> # add a random value to each element of patch data
    >>> noise = np.random.random(patch.shape)
    >>> new = apply_ufunc(patch, noise, np.add)
    >>> assert np.allclose(new.data, patch.data + noise)
    >>> # subtract one patch from another. Coords and attrs must be compatible
    >>> new = apply_ufunc(patch, patch, np.subtract)
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

    reversed = False  # flag to indicate we need to reverse data and patch
    if not isinstance(patch, dc.Patch):
        reversed = True
        patch, other = other, patch
    # Align/broadcast patch to input
    if isinstance(other, dc.Patch):
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


# class DASCOr


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
    >>> from dascore.utils.ufuncs import generate_ufunc
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
