"""Transformations related to strain."""

from __future__ import annotations

import warnings

import dascore as dc
from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.transform.differentiate import differentiate
from dascore.utils.patch import patch_function


@patch_function(
    required_dims=("distance",),
    required_attrs={"data_type": "velocity"},
)
def velocity_to_strain_rate(
    patch: PatchType,
    step_multiple: int = 2,
    gauge_multiple: None | int = None,
    order: int = 2,
) -> PatchType:
    r"""
    Convert velocity DAS data to strain rate using central differences.

    When order=2 and step_multiple=2 the derivative for non-edge values
    is estimated by:

    $$
    \hat{f}(x) = \frac{f(x + (n/2)dx) - f(x - (n/2)dx)}{n dx}
    $$

    Where $dx$ is the distance step and $n$ is the step_multiple. Values for
    edges are estimate with the appropriate forward/backward stencils so that
    the shape of the output data match the input data. The equation
    becomes more complicated for higher order stencils.

    Parameters
    ----------
    patch
        A patch object containing DAS data. Note: attrs['data_type'] should be
        velocity.
    step_multiple
        The multiples of spatial sampling for the central averaging stencil.
        Must be even as odd values result in a staggered grid.
    gauge_multiple
        Deprecated name for step_multiple. Use that instead.
    order
        The order for the finite difference 1st derivative stencil (accuracy).
        It must be a multiple of 2

    Examples
    --------
    >>> from contextlib import suppress
    >>> import dascore as dc
    >>> from dascore.exceptions import MissingOptionalDependencyError
    >>> patch = dc.get_example_patch("deformation_rate_event_1")
    >>>
    >>> # Example 1
    >>> # Estimate the strain rate with a gauge length twice the distance step.
    >>> patch_strain = patch.velocity_to_strain_rate(step_multiple=2)
    >>>
    >>> # Example 2
    >>> # Estimate the strain rate with a 10th order filter. This will raise
    >>> # an exception if the package findiff is not installed.
    >>> with suppress(MissingOptionalDependencyError):
    ...     patch_strain = patch.velocity_to_strain_rate(order=10)
    >>>
    >>> # Example 3
    >>> # Estimate strain rate with a 4th order filter and gauge length 4 times
    >>> # the distance step.
    >>> with suppress(MissingOptionalDependencyError):
    ...     patch_strain = patch.velocity_to_strain_rate(step_multiple=4, order=4)

    Notes
    -----
    This is primarily used with Terra15 data and simply uses
    [patch.differentiate](`dascore.transform.differentiate.differentiate`)
    under the hood to calculate spatial derivatives.

    The output gauge length is equal to the step_multiple multuplied by the
    spacing along the distance coordinate, although the concept of
    gauge_length is more complex with higher oder filters. See
    @yang2022filtering for more info.

    See the [`velocity_to_strain_rate` note](docs/notes/velocity_to_strain_rate.qmd)
    for more details on step_multiple and order effects.

    The [edgeless](`dascore.Patch.velocity_to_strain_rate_edgeless`) version
    of this function removes potential edge effects and supports even and odd
    `step_multiple` values.
    """
    if gauge_multiple is not None:
        msg = "gauge_multiple will be removed in the future. Use step_multiple."
        warnings.warn(msg, DeprecationWarning)
        step_multiple = gauge_multiple * 2

    if step_multiple % 2 != 0:
        msg = (
            "Step_multiple must be even. Use velocity_to_strain_rate_edgeless "
            "if odd step multiples are required."
        )
        raise ParameterError(msg)

    coord = patch.get_coord("distance", require_evenly_sampled=True)
    step = coord.step
    patch = differentiate.func(
        patch, dim="distance", order=order, step=step_multiple // 2
    )
    new_attrs = patch.attrs.update(
        data_type="strain_rate", gauge_length=step * step_multiple
    )
    return patch.update(attrs=new_attrs)


@patch_function(
    required_dims=("distance",),
    required_attrs={"data_type": "velocity"},
)
def velocity_to_strain_rate_edgeless(
    patch: PatchType,
    step_multiple: int = 1,
) -> PatchType:
    r"""
    Estimate strain-rate using central differences.

    For odd step_multiple values this function estimates strain by taking a
    staggered central difference according to:

    $$
    \hat{f} = \frac{f(x + n * dx/2) - f(x - n * dx/2)}{dx}
    $$

    Where $dx$ is the spatial sampling and $n$ is the step_multiple. As a result
    the strain-rate between existing samples is estimated when $n$ is odd. Edges
    (points where full central differences are not possible) are discarded in
    the output.

    Parameters
    ----------
    patch
        A patch object containing DAS data. Note: attrs['data_type'] should be
        velocity.
    step_multiple
        The number of spatial sampling steps to use in the central averaging.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch("deformation_rate_event_1")
    >>>
    >>> # Example 1
    >>> # Estimate strain rate with a gauge length equal to distance step.
    >>> patch_strain = patch.velocity_to_strain_rate_edgeless(step_multiple=1)
    >>>
    >>> # Example 2
    >>> # Estimate strain rate with a gauge length 5 times the distance step.
    >>> patch_strain = patch.velocity_to_strain_rate_edgeless(step_multiple=5)

    Notes
    -----
    See [velocity_to_strain_rate](`dascore.Patch.velocity_to_strain_rate`)
    for a similar function which does not change the shape of the patch.

    The resulting gauge length is equal to the step_multiple multiplied by
    the sampling along the distance dimension.

    See the
    [`velocity_to_strain_rate` note](docs/notes/velocity_to_strain_rate.qmd)
    for more details on step_multiple and order effects.
    """
    coord = patch.get_coord("distance", require_evenly_sampled=True)
    distance_step = coord.step
    gauge_length = step_multiple * distance_step

    data_1 = patch.select(distance=(step_multiple, None), samples=True).data
    data_2 = patch.select(distance=(None, -step_multiple), samples=True).data
    strain_rate = (data_1 - data_2) / gauge_length

    # Need to get distance values between current ones.
    dists = patch.get_array("distance")
    new_dist = (dists[step_multiple:] + dists[:-step_multiple]) / 2
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
