"""Transformations related to strain."""

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
    step_multiple
        The multiples of spatial sampling to make the simulated gauge length.
    gauge_multiple
        Deprecated name for step_multiple. Use that instead.
    order
        The order for the finite difference 1st derivative stencil (accuracy).
        It must be a multiple of 2

    Notes
    -----
    This is primarily used with Terra15 data and simply uses
    [patch.differentiate](`dascore.transform.differentiate.differentiate`)
    under the hood to calculate spatial derivatives.

    The output gauge length is step_multiple x 2, although the concept of
    gauge_length is more complex with higher oder filters. See
    @yang2022filtering for more details.

    This function doesn't change the shape of the array since edge derivatives
    are estimated with forward or backward differences of the specified order.

    See Also
    --------
    - [staggered](`dascore.Patch.staggered_velocity_to_strain_rate`) version
      of this function.
    - [patch.differentiate](`dascore.transform.differentiate.differentiate`)
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
) -> PatchType:
    r"""
    Estimate strain-rate between spatial points using central differences.

    This function estimates the strain by taking a staggered central
    difference according to:

    $$
    f'(x) = \frac{f(x + dx/2) - f(x - dx/2)}{dx}
    $$

    The returned distance coordinates are different from the input
    coordinates because the strain is estimated *between* existing points.

    See [velocity_to_strain_rate](`dascore.Patch.velocity_to_strain_rate`)
    for a similar function that supports more gauge lengths and higher order
    differentiation stencils.

    Parameters
    ----------
    patch
        A patch object containing DAS data. Note: attrs['data_type'] should be
        velocity.
    """
    # See #399 for a discussion of why the step_multiple is fixed.
    step_multiple: int = 1
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

    # Handle unit conversions.
    new_data_units = None
    data_units = dc.get_quantity(patch.attrs.data_units)
    dist_units = dc.get_quantity(patch.get_coord("distance").units)
    if data_units and dist_units:
        new_data_units = data_units / dist_units

    new_attrs = patch.attrs.update(
        data_type="strain_rate",
        gauge_length=distance_step * step_multiple,
        data_units=new_data_units,
    )

    return dc.Patch(data=strain_rate, coords=new_coords, attrs=new_attrs)
