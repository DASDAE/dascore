"""Transformations to strain rates."""
from __future__ import annotations

import numpy as np

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
    gauge_multiple: int = 1,
    order: int = 2,
) -> PatchType:
    """
    Convert velocity DAS data to strain rate.

    This is primarily used with Terra15 data and simply uses
    [patch.differentiate](`dascore.transform.differentiate.differentiate`)
    under the hood which allows higher order differenciating.
    It doesn't change the shape of the array.

    Parameters
    ----------
    patch
        A patch object containing DAS data. Note: attrs['data_type'] should be
        velocity.
    gauge_multiple
        The multiples of spatial sampling to make the simulated gauge length.
        Must be equal to 1 for now.
    order
        The order for the finite difference 1st derivative stencil (accuracy).

    Notes
    -----
    See also [velocity_to_strain_rate_fd]\
    (`dascore.transform.strain.velocity_to_strain_rate_fd`)
    """
    assert gauge_multiple == 1, "only supporting 1 for now."
    coord = patch.get_coord("distance", require_evenly_sampled=True)
    distance_step = coord.step
    patch = differentiate.func(patch, dim="distance", order=order)
    data = patch.data / distance_step
    new_attrs = patch.attrs.update(
        data_type="strain_rate", gauge_length=distance_step * gauge_multiple
    )
    return patch.update(data=data, attrs=new_attrs)


@patch_function(
    required_dims=("distance",),
    required_attrs={"data_type": "velocity"},
)
def velocity_to_strain_rate_fd(
    patch: PatchType,
    gauge_multiple: int = 1,
) -> PatchType:
    """
    Convert velocity DAS data to strain rate.

    This is primarily used with Terra15 data and forward differences the data.
    It does change the shape of the array.

    Parameters
    ----------
    patch
        A patch object containing DAS data. Note: attrs['data_type'] should be
        velocity.
    gauge_multiple : int, optional
        The multiples of spatial sampling to make the simulated gauge length.

    Notes
    -----
    See also [velocity_to_strain_rate]\
    (`dascore.transform.strain.velocity_to_strain_rate`)
    """
    assert isinstance(gauge_multiple, int), "gauge_multiple must be an integer."
    coord = patch.get_coord("distance", require_evenly_sampled=True)
    distance_step = coord.step
    gauge_length = gauge_multiple * distance_step

    data_1 = patch.select(distance=(gauge_multiple, None), samples=True).data
    data_2 = patch.select(distance=(None, -gauge_multiple), samples=True).data
    strain_rate = (data_1 - data_2) / gauge_length

    dist_1 = int(np.floor(gauge_multiple / 2))
    dist_2 = -int(np.ceil(gauge_multiple / 2))
    new_coords, _ = patch.coords.select(distance=(dist_1, dist_2), samples=True)

    dist_unit = dc.get_quantity(patch.get_coord("distance").units)
    new_data_units = dc.get_quantity(patch.attrs["data_units"]) / dist_unit
    new_attrs = patch.attrs.update(
        data_type="strain_rate",
        gauge_length=distance_step * gauge_multiple,
        data_units=new_data_units,
    )

    return dc.Patch(data=strain_rate, coords=new_coords, attrs=new_attrs)
