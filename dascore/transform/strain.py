"""Transformations related to strain."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

import dascore as dc
from dascore.constants import PatchType
from dascore.exceptions import ParameterError, UnitError
from dascore.transform.differentiate import differentiate
from dascore.units import convert_units, get_factor_and_unit, get_unit, invert_quantity
from dascore.utils.array import get_tapered_outer_coord_division_and_mask
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

    if step_multiple <= 0:
        msg = "step_multiple must be positive."
        raise ParameterError(msg)

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
    if step_multiple <= 0:
        msg = "step_multiple must be positive."
        raise ParameterError(msg)

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


@patch_function()
def radians_to_strain(
    patch: PatchType,
    gauge_length=None,
    wave_length: float = 1550.0 * 10 ** (-9),
    stress_constant: float = 0.79,
    refractive_index: float = 1.445,
):
    r"""
    Convert data in radians to strain (rate).

    This applies the simple formula found in @lindsey2020broadband to convert
    data whose units have a radians component to strain.

    Parameters
    ----------
    gauge_length ($L_g$)
        The gauge length in meters.
    wave_length ($\lambda$)
        The laser wavelength in m.
    stress_constant ($\zeta$)
        The stress constant.
    refractive_index ($n$)
        The refractive index of the cable.

    Notes
    -----
    Equation 3 of @lindsey2020broadband:
    $$
    \epsilon_{xx}(t, x_j) = \frac{\lambda}{4 \pi n L_{g} \zeta} \Delta \Phi
    $$
    """
    # First get gauge length, using gl passed into function or attached to attrs.
    gl = getattr(patch.attrs, "gauge_length", None)
    gauge = convert_units(gauge_length if gauge_length is not None else gl, "m")
    if gauge is None or gauge <= 0:
        msg = (
            "Gauge length must be non-zero positive and provided "
            "or defined in patch attrs."
        )
        raise ParameterError(msg)
    # If units doesn't contain radians just return so function is idempotent
    quant = dc.get_quantity(patch.attrs.data_units)
    if str(dc.get_unit("radians")) not in str(quant):
        msg = (
            f"Patch {patch} has no radians in its data_units, "
            f"skipping strain conversion."
        )
        warnings.warn(msg)
        return patch
    # Get constant to multiply with data array.
    const = wave_length / (4 * np.pi * refractive_index * gauge * stress_constant)
    # Handle unit conversions.
    data_units = patch.attrs.get("data_units", None)
    d_factor, d_units = get_factor_and_unit(data_units, simplify=True)
    new_units = get_unit(d_units) * get_unit("strain/radians")
    # Radians wasn't eliminated from the output units. Something went wrong.
    if str(dc.get_unit("radians")) in str(new_units):
        msg = f"radians to strain failed to convert {data_units} to strain."
        raise UnitError(msg)
    # Build output patch
    new_attrs = patch.attrs.update(data_units=new_units)
    new_data = patch.data * const * d_factor
    return patch.update(data=new_data, attrs=new_attrs)


@patch_function(required_coords=("time", "distance"))
def strain_to_particle_fk(
    patch: PatchType,
    filt: Sequence[float] | Sequence[dc.Quantity] | None = None,
    water_level: float = 1e-6,
):
    """
    Scale the strain/strain_rate data to a particle velocity/acceleration.

    This is done in the FK domain by simply dividing by the velocity.

    Parameters
    ----------
    patch
        The patch object.
    filt
        A length 4 array of the form [va, vb, vc, vd].
    water_level
        If not None, the water level of the wave number. Can also be a percent,
        in which case it is a percent of the maximum.

    Notes
    -----
    - If neither filt nor water_level are specified the results can be wildely
    unstable due to small numbers in the wavenumber.

    - [DASPy](https://github.com/HMZ-03/DASPy) has several more methods for
      converting to particle velocity.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch("example_event_2")
    >>>
    >>> # Basic usage with filter
    >>> filt = [100, 400, 5000, 6000]
    >>> out = patch.strain_to_particle_fk(filt=filt)
    >>>
    >>> # Water level as a float
    >>> out = patch.strain_to_particle_fk(water_level=1e-3)
    >>>
    >>> # Water level as a percent
    >>> out = patch.strain_to_particle_fk(water_level=dc.get_quantity("10%"))
    """

    def _update_data_units(fk_patch, fk_dims):
        """Update data units by velocity units if dims have units."""
        data_units = dc.get_quantity(fk_patch.attrs.data_units)
        coord_unit_1 = dc.get_quantity(fk_patch.get_coord(fk_dims[-1]).units)
        coord_unit_2 = dc.get_quantity(fk_patch.get_coord(fk_dims[-2]).units)
        if data_units is None:
            return fk_patch.attrs
        coord_unit_1 = coord_unit_1 or dc.get_quantity("1")
        coord_unit_2 = coord_unit_2 or dc.get_quantity("1")
        slope_units = coord_unit_1 / coord_unit_2
        vel_units = invert_quantity(slope_units)
        return fk_patch.attrs.update(data_units=data_units * vel_units)

    if filt is None and water_level is None:
        msg = (
            "Warning: using strain_to_particle_fk without the filt or "
            "water_level parameters usually results in unstable results!"
        )
        warnings.warn(msg)
    fk_dims = ("ft_time", "ft_distance")
    fk_patch = patch.dft.func(patch, dim=("time", "distance"), real="time")
    transformed = patch is not fk_patch

    slope, mask = get_tapered_outer_coord_division_and_mask(
        fk_patch,
        fk_dims,
        filt=filt,
        invert=False,
        directional=True,
        water_level=water_level,
    )
    new_attrs = _update_data_units(fk_patch, fk_dims)
    velocities = slope * mask
    new_data = np.zeros_like(fk_patch.data)
    np.divide(fk_patch.data, velocities, out=new_data, where=velocities != 0)
    new = fk_patch.new(data=new_data, attrs=new_attrs)
    if transformed:
        new = new.idft()
    return new
