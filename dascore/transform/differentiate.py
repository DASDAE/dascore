"""Module for performing differentiation on patches."""
from __future__ import annotations

from operator import truediv

import numpy as np

from dascore.constants import PatchType
from dascore.utils.misc import iterate, optional_import
from dascore.utils.patch import (
    _get_data_units_from_dims,
    _get_dx_or_spacing_and_axes,
    patch_function,
)


def _grad_diff(ar, axes, dxs):
    """Perform differentiation on the data using np.grad."""
    # iteratively apply diff along each axis.
    for ax, dx in zip(axes, dxs):
        ar = np.gradient(ar, dx, axis=ax, edge_order=2)
    return ar


def _findiff_diff(data, axis, order, dx):
    """Use findiff to get differences along specified dimensions."""
    # If more than one axis, need to make tuples.
    findiff = optional_import("findiff")
    if len(axis) > 1:
        orders = [1] * len(axis)  # always first order (first derivative).
        tuples = [tuple(x) for x in zip(axis, dx, orders)]
        func = findiff.FinDiff(*tuples, acc=order)
    else:
        func = findiff.FinDiff(axis[0], dx[0], 1, acc=order)
    out = func(data)
    return out


@patch_function()
def differentiate(
    patch: PatchType,
    dim: str | None,
    order=2,
) -> PatchType:
    """
    Calculate first derivative along dimension(s) using centeral diferences.

    The shape of the output patch is the same as the input patch. Derivative
    along edges are calculated with the same order (accuarcy) as centeral
    points using non-centered stencils.

    Parameters
    ----------
    patch
        The patch to differentiate.
    dim
        The dimension(s) along which to differentiate. If None differentiates
        over all dimensions.
    order
        The order of the differentiation operator. Must be a possitive, even
        integar.

    Notes
    -----
    For order=2 (the default) numpy's gradient function is used. When
    order != the optional package findiff must be installed in which case
    order is interpreted as accuracy ("order" means order of differention
    in that package).

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> # 2nd order stencil for 1st derivative along time dimension
    >>> patch_diff_1 = patch.differentiate(dim='time', order=2)
    >>> # 1st derivative along all dimensions
    >>> patch_diff_2 = patch.differentiate(dim=None)
    """
    dims = iterate(dim if dim is not None else patch.dims)
    dx_or_spacing, axes = _get_dx_or_spacing_and_axes(patch, dims)
    if order == 2:
        new_data = _grad_diff(patch.data, axes, dx_or_spacing)
    else:
        new_data = _findiff_diff(patch.data, axes, order, dx_or_spacing)
    # update units
    data_units = _get_data_units_from_dims(patch, dims, truediv)
    attrs = patch.attrs.update(data_units=data_units)
    return patch.new(data=new_data, attrs=attrs)
