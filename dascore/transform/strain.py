"""Transformations to strain rates."""

from __future__ import annotations

import warnings

import dascore as dc
from dascore.constants import PatchType
from dascore.transform.differentiate import differentiate
from dascore.utils.patch import patch_function


@patch_function(
    required_dims=("distance",),
    required_attrs={"data_type": "velocity"},
)
def velocity_to_strain_rate(
    patch: PatchType,
    step_multiple: int = 1,
    gauge_multiple: None = None,
    order: int = 2,
) -> PatchType:
    r"""
    Convert velocity DAS data to strain rate using central differences.

    This is primarily used with Terra15 data and simply uses
    [patch.differentiate](`dascore.transform.differentiate.differentiate`)
    under the hood which allows higher order differentiating.
    It doesn't change the shape of the array since edge derivatives are
    estimated with forward or backward differences of the specified order.

    When order=2 and step_multiple=1 the derivative for non-edge values
    is estimated by:

    $$
    f'(x) = \frac{f(x + dx) - f(x - dx)}{2 dx}
    $$

    Parameters
    ----------
    patch
        A patch object containing DAS data. Note: attrs['data_type'] should be
        velocity.
    gauge_multiple
        The multiples of spatial sampling to make the simulated gauge length.
    order
        The order for the finite difference 1st derivative stencil (accuracy).

    See Also
    --------
    - [staggered](`dascore.Patch.staggered_velocity_to_strain_rate_fd`) version
      of this function.
    """
    if gauge_multiple is not None:
        msg = "gauge_multiple has been renamed to step_multiple"
        warnings.warn(msg, DeprecationWarning)
        step_multiple = gauge_multiple

    coord = patch.get_coord("distance", require_evenly_sampled=True)
    step = coord.step
    patch = differentiate.func(patch, dim="distance", order=order, step=step_multiple)
    new_attrs = patch.attrs.update(
        data_type="strain_rate", gauge_length=2 * step * step_multiple
    )
    return patch.update(attrs=new_attrs)


@patch_function(
    required_dims=("distance",),
    required_attrs={"data_type": "velocity"},
)
def staggered_velocity_to_strain_rate(
    patch: PatchType,
    step_multiple: int = 1,
) -> PatchType:
    r"""
    Convert velocity data to strain rate using staggered central differences.

    When step_multiple=1 the derivative for non-edge values is estimated by:

    $$
    f'(x) = \frac{f(x + dx/2) - f(x - dx/2)}{dx}
    $$

    The means the returned distance coordinates are different than the input
    coordinates because the strain is estimated *between* existing points.

    Parameters
    ----------
    patch
        A patch object containing DAS data. Note: attrs['data_type'] should be
        velocity.
    step_multiple : int, optional
        The multiples of spatial sampling to make the simulated gauge length.

    See Also
    --------
    - [velocity_to_strain_rate](`dascore.Patch.velocity_to_strain_rate`)
    """
    assert isinstance(step_multiple, int), "gauge_multiple must be an integer."
    coord = patch.get_coord("distance", require_evenly_sampled=True)
    distance_step = coord.step
    gauge_length = step_multiple * distance_step

    data_1 = patch.select(distance=(step_multiple, None), samples=True).data
    data_2 = patch.select(distance=(None, -step_multiple), samples=True).data
    strain_rate = (data_1 - data_2) / gauge_length

    # Need to get distance values between current ones.
    dists = patch.get_array("distance")
    new_dist = (dists[step_multiple:] - dists[:-step_multiple]) / 2
    new_coords = patch.coords.update(distance=new_dist)

    new_data_units = None
    if dist_unit := dc.get_quantity(patch.get_coord("distance").units):
        new_data_units = dc.get_quantity(patch.attrs["data_units"]) / dist_unit

    new_attrs = patch.attrs.update(
        data_type="strain_rate",
        gauge_length=distance_step * step_multiple,
        data_units=new_data_units,
    )

    return dc.Patch(data=strain_rate, coords=new_coords, attrs=new_attrs)
