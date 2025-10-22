"""Processing for applying a taper."""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from operator import add

import numpy as np

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.units import Quantity
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import broadcast_for_index
from dascore.utils.patch import get_dim_axis_value, patch_function
from dascore.utils.signal import WINDOW_FUNCTIONS, _get_window_function
from dascore.utils.time import to_float


def _get_taper_slices(patch, kwargs):
    """Get slice for start/end of patch."""
    dim, axis, value = get_dim_axis_value(patch, kwargs=kwargs)[0]
    coord = patch.coords.coord_map[dim]
    if isinstance(value, Sequence | np.ndarray):
        assert len(value) == 2, "Length 2 sequence required."
        start, stop = value[0], value[1]
    else:
        start, stop = value, value
    dur = coord.coord_range(extend=False)
    # either let units pass through or multiply by d_len
    clses = (Quantity, np.timedelta64)
    start = start if isinstance(start, clses) or start is None else start * dur
    stop = stop if isinstance(stop, clses) or stop is None else stop * dur
    stop = -stop if stop is not None else stop
    _, inds_1 = coord.select((None, start), relative=True)
    _, inds_2 = coord.select((stop, None), relative=True)
    return axis, (start, stop), inds_1, inds_2


def _validate_windows(samps, start_slice, end_slice, shape, axis):
    """Validate the windows don't overlap or exceed dim len."""
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
@compose_docstring(taper_type=sorted(WINDOW_FUNCTIONS))
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
        of total length of the dimension (if a decimal or percente, see examples),
        or absolute units. If a single value is passed, the taper will be applied
        to both ends. A length two tuple can specify different values for each
        end, or no taper on one end.

    Returns
    -------
    The tapered patch.

    See Also
    --------
    [Patch.taper_range](`dascore.Patch.taper_range`)

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch() # generate example patch
    >>>
    >>> # Apply an Hanning taper to 5% of each end for time dimension.
    >>> patch_taper1 = patch.taper(time=0.05, window_type="hann")
    >>>
    >>> # Apply a triangular taper to 10% of the start of the distance dimension.
    >>> patch_taper2 = patch.taper(distance=(0.10, None), window_type='triang')
    >>>
    >>> # Apply taper to first 20 percent and last 12 percent of time dimension.
    >>> from dascore.units import percent
    >>> patch_taper3 = patch.taper(time=(20 * percent, 12 * percent))
    >>>
    >>> # Apply taper on first and last 15 m along distance axis.
    >>> from dascore.units import m
    >>> patch_taper4 = patch.taper(distance=15 * m)
    """
    func = _get_window_function(window_type)
    # get taper values in samples.
    out = np.array(patch.data)  # Need to make a copy here.
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


def _get_taper_coord_inds(coord, values, relative, samples):
    """Get the index of the referenced coord inds."""
    error_msg = "A len 2 or 4 sequence is required for taper values"
    if not isinstance(values, (Sequence | np.ndarray)) or not len(values):
        raise ParameterError(error_msg)
    # More than 1 sequence was passed, recurse to flatten out.
    elif isinstance(values[0], (Sequence | np.ndarray)):
        out = [_get_taper_coord_inds(coord, x, relative, samples) for x in values]
        return reduce(add, out)
    elif len(values) not in {2, 4}:
        raise ParameterError(error_msg)
    # Ok inputs, convert to index along coordinate.
    out = [None] * len(values)
    for num, val in enumerate(values):
        if val is None or val == ...:
            if len(values) == 2:
                msg = "Cannot use ... or None when only two values provided"
                raise ParameterError(msg)
            # None or ... means min_val in first half of list else max_val
            out[num] = 0 if (num / len(out)) < 0.5 else len(coord) - 1
        else:
            out[num] = coord.get_next_index(val, samples=samples, relative=relative)
    # Always need a len 4 sequence
    if len(out) == 2:
        out = [0, *out, len(coord)]
    return [out]  # return a list of len4 sequences.


def _get_taper_curve(coord, ind_1, ind_2, window_type, reverse=False):
    """Get the taper curve between index1 and index2."""
    func = _get_window_function(window_type)
    samps = ind_2 - ind_1
    taper = func(samps * 2 + 1)[:samps]
    if reverse:
        taper = taper[::-1]
    # Need to extrapolate to get correct values for non evenly sampled coords.
    if not coord.evenly_sampled:
        # The current window represents the snapped (evenly sampled) coords
        old_coord = coord.select((ind_1, ind_2), samples=True)[0]
        new_coord = old_coord.snap().change_length(len(old_coord))
        old_x, new_x = to_float(old_coord.values), to_float(new_coord.values)
        assert len(old_x) == len(new_x) == len(taper)
        taper = np.interp(new_x, old_x, taper)
    return taper


def _get_range_envelope(coord, inds, window_type, invert):
    """Create a broadcast envelope for taper."""
    out = np.zeros(len(coord))
    for ind_set in inds:
        assert len(ind_set) == 4
        i1, i2, i3, i4 = ind_set
        left_taper = _get_taper_curve(coord, i1, i2, window_type)
        right_taper = _get_taper_curve(coord, i3, i4, window_type, reverse=True)
        out[i1:i2] += left_taper
        out[i3:i4] += right_taper
        out[i2:i3] += 1
    if invert:
        out = np.abs(out - np.max(out))

    return out


@patch_function()
@compose_docstring(taper_type=sorted(WINDOW_FUNCTIONS))
def taper_range(
    patch: PatchType,
    window_type: str = "hann",
    invert=False,
    relative=False,
    samples=False,
    **kwargs,
) -> PatchType:
    """
    Taper a range inside the patch.

    Parameters
    ----------
    patch
        A patch instance.
    window_type
        The type of window to use For tapering. Supported Options are:
            {taper_type}.
    invert
        If True, the values inside the specified range are set to zero
        and gradually tapered to 1.
    samples
        If True, the values specified by the kwargs indicate samples
        rather than values along the indicated dimension.
    relative
        If True, the values specified in kwargs are relateive to the
        start (if positive) or end (if negative) of the indicated
        dimension.
    **kwargs
        Used to specify the dimension along which to taper. Values can be
        either a length 2 sequence or a length 4 sequence. If len == 2
        then the left taper starts at [0] and ends at the start of
        the coordinate. The left taper starts at [1] and ends at
        the end of the coordinate. If len == 4, values between [1] and [2]
        are left alone, values between [0] and [1] as well as values between
        [2] and [3] are gradually tapered. Values outside of this range are
        set to 0.

    Returns
    -------
    The tapered patch.

    See Also
    --------
    [Patch.taper](`dascore.Patch.taper`)

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>>
    >>> epatch = dc.get_example_patch()
    >>> patch = epatch.new(data=np.ones_like(epatch.data))
    >>>
    >>> # Taper values outside of specified times to zero
    >>> t1 = dc.to_datetime64("2017-09-18T00:00:04")
    >>> t2 = dc.to_datetime64("2017-09-18T00:00:07")
    >>> patch_tapered_1 = patch.taper_range(time=(t1, t2))
    >>>
    >>> # Taper values inside specified range to 0
    >>> patch_tapered_2 = patch.taper_range(time=(t1, t2), invert=True)
    >>>
    >>> # Specify taper range (4 values) such that values outside
    >>> # that range are 0, between [0] and [1] as well as
    >>> # [2] and [3] are tapered and values inside [1] and [2] are
    >>> # not effected.
    >>> patch_tapered_3 = patch.taper_range(
    ...     time=(1, 2, 5, 5),
    ...     relative=True,
    ... )
    >>>
    >>> # Use samples rather than absolute time values.
    >>> patch_tapered_4 = patch.taper_range(
    ...     distance=(10, 80),
    ...     samples=True
    ... )
    >>>
    >>> # Apply two non-overlapping tapers
    >>> taper_range = ((25,50,100,125), (150,175,200,225))
    >>> patch_tapered_5 = patch.taper_range(distance=taper_range)

    """
    dim, ax, values = get_dim_axis_value(patch, kwargs=kwargs)[0]
    coord = patch.get_coord(dim, require_sorted=True)
    inds = _get_taper_coord_inds(coord, values, relative, samples)
    env = _get_range_envelope(coord, inds, window_type, invert)
    # Ensure envelope broadcasts to index
    indexer = broadcast_for_index(patch.ndim, ax, value=slice(None), fill=None)
    env_broadcastable = env[indexer]
    return patch.new(data=patch.data * env_broadcastable)
