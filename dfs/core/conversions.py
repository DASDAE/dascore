"""
Module for converting output from one format to another.
"""


def convert_velocity_to_strain_rate(
    inp, gauge_length, dx_dec, start_index2=None, end_index2=None
):
    """The nasty func copied from terra15 note."""
    sliced_inp = get_slices_by_range(inp, start_index2, end_index2)
    n_t, n_d = sliced_inp[0, :, :].shape  # Convert gauge length to spatial
    samplesg = np.int32(round(gauge_length / dx_dec))
    inv_gauge_length = np.float32(1.0 / gauge_length)
    strain_rates = (
        sliced_inp[:, :, g:n_d] - sliced_inp[:, :, 0 : n_d - g]
    ) * inv_gauge_length
    return strain_rates
