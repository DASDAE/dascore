"""
Module for performing differentiation on patches.
"""
from __future__ import annotations

import numpy as np

from dascore.constants import PatchType
from dascore.units import get_quantity
from dascore.utils.misc import iterate, optional_import
from dascore.utils.patch import patch_function
from dascore.utils.time import to_float


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
        func = findiff.FinDiff(tuples, acc=order)
    else:
        func = findiff.FinDiff(axis[0], dx[0], 1, acc=order)
    out = func(data)
    return out


def _get_data_units(patch, dims):
    """Get new data units."""
    if (data_units := get_quantity(patch.attrs.data_units)) is None:
        return
    dim_units = None
    for dim in dims:
        dim_unit = get_quantity(patch.get_coord(dim).units)
        if dim_unit is None:
            continue
        dim_units = dim_unit if dim_units is None else dim_unit * dim_units
    if dim_units is not None:
        data_units = data_units / dim_units
    return data_units


def _get_dx_or_spacing(
    patch, dim
) -> tuple[tuple[int | np.ndarray, ...], tuple[int, ...]]:
    """
    For each selected coordinates, get dx if evenly sampled or values.

    Also get axes.
    """
    dims = iterate(dim if dim is not None else patch.dims)
    out = []
    axes = []
    for dim_ in dims:
        coord = patch.get_coord(dim_)
        if coord.evenly_sampled:
            val = coord.step
        else:
            val = coord.data
        # need to convert val to float so datetimes work
        out.append(to_float(val))
        axes.append(patch.dims.index(dim_))
    return tuple(out), tuple(axes)


@patch_function()
def differentiate(
    patch: PatchType,
    dim: str | None,
    order=2,
) -> PatchType:
    """
    Differentiate along specified dimension using centeral diferences.

    Parameters
    ----------
    patch
        The patch to differentiate.
    dim
        The dimension along which to differentiate. If None.
    order
        The order of the differentiation operator. Must be a possitive, even
        integar.

    Notes
    -----
    For order=2 (the default) numpy's gradient function is used. When
    order != the optional package findiff must be installed in which case
    order is interpreted as accuracy (as order means order of differention
    in that package).

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> integrated = patch.diff(dim='time')
    """
    dims = iterate(dim if dim is not None else patch.dims)
    dx_or_spacing, axes = _get_dx_or_spacing(patch, dims)
    if order == 2:
        new_data = _grad_diff(patch.data, axes, dx_or_spacing)
    else:
        new_data = _findiff_diff(patch.data, axes, order, dx_or_spacing)
    # update units
    attrs = patch.attrs.update(data_units=_get_data_units(patch, dims))
    return patch.new(data=new_data, attrs=attrs)
