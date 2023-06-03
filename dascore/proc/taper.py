"""
Processing for applying a taper.
"""

import numpy as np
import pandas as pd
from scipy.signal import windows  # the best operating system?

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import broadcast_slice
from dascore.utils.patch import get_dim_value_from_kwargs, patch_function

TAPER_FUNCTIONS = dict(
    barthann=windows.barthann,
    bartlett=windows.bartlett,
    blackman=windows.blackman,
    blackmanharris=windows.blackmanharris,
    bohman=windows.bohman,
    boxcar=windows.boxcar,
    flattop=windows.flattop,
    hamming=windows.hamming,
    hann=windows.hann,
    nuttall=windows.nuttall,
    parzen=windows.parzen,
    triang=windows.triang,
)


def _get_taper_slices(patch, kwargs):
    """Get slice for start/end of patch."""
    dim, axis, value = get_dim_value_from_kwargs(patch, kwargs)
    d_len = patch.shape[axis]
    # TODO add unit support once patch_refactor lands
    start, stop = np.broadcast_to(np.array(value), (2,)) * d_len
    slice_1 = slice(None, int(start)) if not pd.isnull(start) else slice(None)
    slice_2 = slice(d_len - int(stop), None) if not pd.isnull(stop) else slice(None)
    return axis, (int(start), int(stop)), slice_1, slice_2


@patch_function()
@compose_docstring(taper_type=sorted(TAPER_FUNCTIONS))
def taper(
    patch: PatchType,
    type: str = "hann",
    **kwargs,
) -> PatchType:
    """
    Taper the ends of the signal.

    Parameters
    ----------
    patch
        The patch instance.
    type
        The type of taper to use. Options are:
            {taper_type}.
    **kwargs
        Used to specify the dimension along which to taper and the percentage
        of total length of the dimension. If a single value is passed, the
        taper will be applied to both ends. A length two tuple can specify
        different values for each end, or no taper on one end.

    Returns
    -------
    The tapered patch.

    Examples
    --------
    """
    # get taper function or raise if it isn't known.
    if type not in TAPER_FUNCTIONS:
        msg = (
            f"{type} is not a known taper function. "
            f"Options are: {sorted(TAPER_FUNCTIONS)}"
        )
        raise ParameterError(msg)
    func = TAPER_FUNCTIONS[type]
    # get taper values in samples.
    out = np.array(patch.data)
    n_dims = len(out.shape)
    axis, samps, start_slice, end_slice = _get_taper_slices(patch, kwargs)
    # we can't taper more than the length of the patch.
    assert np.sum(samps) <= out.shape[axis]
    if start_slice is not None:
        window = func(2 * int(samps[0]))[: samps[0]]
        # get indices window (which will broadcast) and data
        data_inds = broadcast_slice(n_dims, axis, start_slice)
        window_inds = broadcast_slice(n_dims, axis, slice(None), fill_none=True)
        out[data_inds] = out[data_inds] * window[window_inds]
    if end_slice is not None:
        window = func(2 * int(samps[0]))[samps[1] :]
        data_inds = broadcast_slice(n_dims, axis, end_slice)
        window_inds = broadcast_slice(n_dims, axis, slice(None), fill_none=True)
        out[data_inds] = out[data_inds] * window[window_inds]
    return patch.new(data=out)
