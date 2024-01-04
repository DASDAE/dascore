"""Transformations to strain rates."""
from __future__ import annotations

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
    Convert velocity das data to strain rate.

    This is primarily used with Terra15 data and simply uses
    [patch.differentiate](`dascore.transform.differentiate.differentiate`)
    under the hood.

    Parameters
    ----------
    patch
        A patch object containing das data. Note: attrs['data_type'] should be
        velocity.
    gauge_multiple
        The multiples of spatial sampling to make the simulated gauge length.
    order
        The order for the finite difference 1st derivative stencil (accuracy).
    """
    assert gauge_multiple == 1, "only supporting 1 for now."
    coord = patch.get_coord("distance", require_evenly_sampled=True)
    step = coord.step
    patch = differentiate.func(patch, dim="distance", order=order)
    new_attrs = patch.attrs.update(
        data_type="strain_rate", gauge_length=step * gauge_multiple
    )
    return patch.update(attrs=new_attrs)
