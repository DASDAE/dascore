"""Processing for applying a taper."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import windows  # the best operating system?

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.units import Quantity
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
    coord = patch.coords.coord_map[dim]
    if isinstance(value, Sequence | np.ndarray):
        assert len(value) == 2, "Length 2 sequence required."
        start, stop = value[0], value[1]
    else:
        start, stop = value, value
    dur = coord.max() - coord.min()
    # either let units pass through or multiply by d_len
    start = start if isinstance(start, Quantity) or start is None else start * dur
    stop = stop if isinstance(stop, Quantity) or stop is None else stop * dur
    stop = -stop if stop is not None else stop
    _, inds_1 = coord.select((None, start), relative=True)
    _, inds_2 = coord.select((stop, None), relative=True)
    return axis, (start, stop), inds_1, inds_2


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


def _validate_windows(samps, start_slice, end_slice, shape, axis):
    """Validate the the windows don't overlap or exceed dim len."""
    max_len = shape[axis]
    start_ind = start_slice.stop
    end_ind = end_slice.start

    bad_start = samps[0] is not None and (start_ind is None or start_ind < 0)
    bad_end = samps[1] is not None and (end_ind is None or end_ind > max_len)

    if bad_start or bad_end:
        msg = "Total taper lengths exceed total dim length"
        raise ParameterError(msg)

    if start_ind is None or end_ind is None:
        return
    if start_ind > end_ind:
        msg = "Taper windows cannot overlap"
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
        of total length of the dimension or abolsute units. If a single value
        is passed, the taper will be applied to both ends. A length two tuple
        can specify different values for each end, or no taper on one end.

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
    >>> # Apply taper on first and last 15 m along distance axis.
    >>> from dascore.units import m
    >>> patch_taper3 = patch.taper(distance=15 * m)
    """
    func = _get_window_function(window_type)
    # get taper values in samples.
    out = np.array(patch.data)
    shape = out.shape
    n_dims = len(out.shape)
    axis, samps, start_slice, end_slice = _get_taper_slices(patch, kwargs)
    _validate_windows(samps, start_slice, end_slice, shape, axis)
    if samps[0] is not None:
        val = start_slice.stop
        window = func(2 * val)[:val]
        # get indices window (which will broadcast) and data
        data_inds = broadcast_for_index(n_dims, axis, start_slice)
        window_inds = broadcast_for_index(n_dims, axis, slice(None), fill=None)
        out[data_inds] = out[data_inds] * window[window_inds]
    if samps[1] is not None:
        val = shape[axis] - end_slice.start
        window = func(2 * val)[val:]
        data_inds = broadcast_for_index(n_dims, axis, end_slice)
        window_inds = broadcast_for_index(n_dims, axis, slice(None), fill=None)
        out[data_inds] = out[data_inds] * window[window_inds]
    return patch.new(data=out)
