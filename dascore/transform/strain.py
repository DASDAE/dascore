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


#
# def convert_velocity_to_strain_rate(
#         patch: PatchType, gauge_length, dx_dec, start_index2=None, end_index2=None
# ) -> PatchType:
#     """The nasty func copied from terra15 note."""
#     sliced_inp = get_slices_by_range(inp, start_index2, end_index2)
#     n_t, n_d = sliced_inp[0, :, :].shape  # Convert gauge length to spatial
#     samplesg = np.int32(round(gauge_length / dx_dec))
#     inv_gauge_length = np.float32(1.0 / gauge_length)
#     strain_rates = (
#         sliced_inp[:, :, g:n_d] - sliced_inp[:, :, 0 : n_d - g]
#     ) * inv_gauge_length
#     return strain_rates
