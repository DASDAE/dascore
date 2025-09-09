"""Module for performing differentiation on patches."""

from __future__ import annotations

from collections.abc import Sequence
from operator import truediv

import numpy as np

from dascore.compat import is_array
from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.misc import broadcast_for_index, iterate, optional_import
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


def _get_diff(order, data, axes, dx_or_spacing):
    """Get the diff of an array along specified axes."""
    if order == 2:
        new_data = _grad_diff(data, axes, dx_or_spacing)
    else:
        new_data = _findiff_diff(data, axes, order, dx_or_spacing)
    return new_data


def _strided_diff(order, patch, axes, dx_or_spacing, step):
    """Calculate a strided differentiation along specified axes."""
    if len(axes) > 1:
        msg = "Step in patch.differentiate can only be used along one axis."
        raise ParameterError(msg)
    new_data = np.empty_like(patch.data)
    dx_or_space = dx_or_spacing[0]
    for step_ in range(step):
        current_slice = slice(step_, None, step)
        slicer = broadcast_for_index(patch.ndim, axes[0], current_slice)
        # Need to either double DX or slice the coordinate spacing to
        # account for the fact we are skipping some columns/rows.
        if is_array(dx_or_space):
            _dx_or_space = dx_or_space[current_slice]
        else:
            _dx_or_space = dx_or_space * step
        sub = _get_diff(order, patch.data[slicer], axes, [_dx_or_space])
        new_data[slicer] = sub
    return new_data


@patch_function()
def differentiate(
    patch: PatchType,
    dim: str | Sequence[str] | None,
    order: int = 2,
    step: int = 1,
) -> PatchType:
    r"""
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
    step
        The number of columns/rows to skip for differention.
        eg: an array of [a b c d e] uses b and d to calculate diff of c when
        step = 1 and order = 2. When step = 2, a and e are used to calculate
        diff at c.

    Notes
    -----
    For order=2 (the default) numpy's gradient function is used. When
    order != 2, the optional package findiff must be installed in which case
    order is interpreted as accuracy ("order" means order of differention
    in that package).

    The second order first derivative, for an evenly spaced coordinate,
    is defined as:

    $$
    \hat{f}(x) = \frac{f(x + dx) - f(x - dx)}{2dx} + O({dx}^2)
    $$

    Where $\hat{f}(x)$ is the estiamted derivative of $f$ at $x$, $dx$ is
    the sample spacing, and $O$ is the error term.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Example 1
    >>> # 2nd order stencil for 1st derivative along time dimension
    >>> patch_diff_1 = patch.differentiate(dim='time', order=2)
    >>>
    >>> # Example 2
    >>> # 1st derivative along all dimensions
    >>> patch_diff_2 = patch.differentiate(dim=None)
    >>>
    >>> # Example 3
    >>> # 1st derivative using a step size of 3. This spaces out the columns
    >>> # or rows used for estimating the derivative.
    >>> patch_diff_3 = patch.differentiate(dim="distance", step=3, order=2)
    """
    dims = iterate(dim if dim is not None else patch.dims)
    dx_or_spacing, axes = _get_dx_or_spacing_and_axes(patch, dims)
    if step > 1:
        new_data = _strided_diff(order, patch, axes, dx_or_spacing, step)
    # This avoids an extra copy of the array so probably merits its own case.
    else:
        new_data = _get_diff(order, patch.data, axes, dx_or_spacing)
    # update units
    data_units = _get_data_units_from_dims(patch, dims, truediv)
    attrs = patch.attrs.update(data_units=data_units)
    return patch.new(data=new_data, attrs=attrs)
