"""
Transformations to strain rates.
"""
from typing import Union

import findiff

from dascore.constants import PatchType
from dascore.utils.patch import patch_function


@patch_function(required_dims=("time",), required_attrs={"data_type": "velocity"})
def velocity_to_strain_rate(
    patch: PatchType,
    gauge_multiple: Union[int] = 1,
    order=1,
) -> PatchType:
    """
    Convert velocity das data to strain rate.

    This is primarily used with Terra15 data.

    Parameters
    ----------
    patch
        A patch object containing das data. Note: attrs['data_type'] should be
        velocity.
    gauge_multiple
        The multiples of spatial sampling to make the simulated gauge length.
    order
        The order for the derivative operator. Second order is default.
    """
    assert gauge_multiple == 1, "only supporting 1 for now."
    axis = patch.dims.index("distance")
    d_distance = patch.attrs["d_distance"]
    differ = findiff.FinDiff(axis, d_distance, order)
    new = differ(patch.data)
    attrs = dict(patch.attrs)
    attrs["data_type"] = "strain_rate"
    attrs["gauge_length"] = d_distance * gauge_multiple
    return patch.new(data=new, attrs=attrs)
