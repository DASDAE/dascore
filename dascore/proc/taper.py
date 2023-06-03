"""
Processing for applying a taper.
"""

import numpy as np
import pandas as pd
from scipy.signal import windows  # the best operating system?

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import broadcast_for_index
from dascore.utils.patch import get_dim_value_from_kwargs, patch_function

TAPER_FUNCTIONS = dict(
    barthann=windows.barthann,
    bartlett=windows.bartlett,
    blackman=windows.blackman,
    blackmanharris=windows.blackmanharris,
    bohman=windows.bohman,
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
    # TODO add unit support once patch_refactor branch lands
    start, stop = np.broadcast_to(np.array(value), (2,))
    slice1, slice2 = slice(None), slice(None)
    if not pd.isnull(start):
        start = int(start * d_len)
        slice1 = slice(None, start)
    if not pd.isnull(stop):
        stop = int(d_len - stop * d_len)
        slice2 = slice(stop, None)
    return axis, (start, stop), slice1, slice2


def _get_window_function(window_type):
    """Get the window function to use for taper."""
    # get taper function or raise if it isn't known.
    if window_type not in TAPER_FUNCTIONS:
        msg = (
            f"'{window_type}' is not a known window type. "
            f"Options are: {sorted(TAPER_FUNCTIONS)}"
        )
        raise ParameterError(msg)
    func = TAPER_FUNCTIONS[window_type]
    return func


def _validate_windows(samps, shape, axis):
    """Validate the the windows don't overlap or exceed dim len."""
    samps = np.array([x for x in samps if not pd.isnull(x)])
    if len(samps) > 1 and samps[0] > samps[1]:
        msg = "Taper windows cannot overlap"
        raise ParameterError(msg)
    elif np.any(samps < 0) or np.any(samps > shape[axis]):
        msg = "Total taper lengths exceed total dim length"
        raise ParameterError(msg)


@patch_function()
@compose_docstring(taper_type=sorted(TAPER_FUNCTIONS))
def taper(
    patch: PatchType,
    window_type: str = "hann",
    **kwargs,
) -> PatchType:
    """
    Taper the ends of the signal.

    Parameters
    ----------
    patch
        The patch instance.
    window_type
        The type of window to use For tapering. Supported Options are:
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
    >>> import dascore as dc
    >>> patch = dc.get_example_patch() # generate example patch
    >>> # Apply an Hanning taper to 5% of each end for time dimension.
    >>> patch_taper1 = patch.taper(time=0.05, window_type="hann")
    >>> # Apply a triangular taper to 10% of the start of the distance dimension.
    >>> patch_taper2 = patch.taper(distance=(0.10, None), window_type='triang')
    """
    func = _get_window_function(window_type)
    # get taper values in samples.
    out = np.array(patch.data)
    shape = out.shape
    n_dims = len(out.shape)
    axis, samps, start_slice, end_slice = _get_taper_slices(patch, kwargs)
    _validate_windows(samps, shape, axis)
    if samps[0] is not None:
        window = func(2 * int(samps[0]))[: samps[0]]
        # get indices window (which will broadcast) and data
        data_inds = broadcast_for_index(n_dims, axis, start_slice)
        window_inds = broadcast_for_index(n_dims, axis, slice(None), fill_none=True)
        out[data_inds] = out[data_inds] * window[window_inds]
    if samps[1] is not None:
        diff = shape[axis] - samps[1]
        window = func(2 * diff)[diff:]
        data_inds = broadcast_for_index(n_dims, axis, end_slice)
        window_inds = broadcast_for_index(n_dims, axis, slice(None), fill_none=True)
        out[data_inds] = out[data_inds] * window[window_inds]
    return patch.new(data=out)
