"""Module for calculating cross-correlation over time or distance."""
from __future__ import annotations

import numpy as np
from scipy.fftpack import next_fast_len

import dascore as dc
from dascore.constants import PatchType
from dascore.utils.misc import broadcast_for_index
from dascore.utils.time import to_float
from dascore.utils.patch import (
    get_dim_value_from_kwargs,
    patch_function,
)


def _get_correlated_coord(old_coord, data_shape):
    """Get the new coordinate which corresponds to correlated values."""
    step = old_coord.step
    one_sided_len = data_shape // 2
    new = dc.core.get_coord(
        start=-one_sided_len * step,
        stop=(one_sided_len + 1) * step,
        step=step,
    )
    return new.change_length(data_shape)


def _shift(data, cx_len, axis):
    """
    Re-assemble fft data so zero lag is in center.

    Also accounts for padding for fast fft.
    """
    ndims = len(data.shape)
    start_slice = slice(-np.floor(cx_len / 2).astype(int), None)
    start_ind = broadcast_for_index(ndims, axis, start_slice)
    stop_slice = slice(None, np.ceil(cx_len / 2).astype(int))
    stop_ind = broadcast_for_index(ndims, axis, stop_slice)
    data1 = data[start_ind]
    data2 = data[stop_ind]
    data = np.concatenate([data1, data2], axis=axis)
    return data


def _correlate_fft(patch, fft_axis, fft_dim):
    """Perform a padded FFT for correlations."""
    is_real = not np.issubdtype(patch.data.dtype, np.complexfloating)
    # perform FFT (since the input is not already in the frequency domain)
    fft_func = np.fft.rfft if is_real else np.fft.fft
    # get the closest fast length. Some padding is applied in the fft to avoid
    # inefficient lengths. Note: This is not always a power of 2.
    cx_len = patch.shape[fft_axis] * 2 - 1
    fast_len = next_fast_len(cx_len)
    # perform ffts and get source array (a sub-slice of larger fft)
    fft = fft_func(patch.data, axis=fft_axis, n=fast_len)
    # ensure coordinate is evenly spaced
    fft_coord = patch.get_coord(fft_dim)
    return fft, fft_coord, cx_len, fast_len, is_real


def _get_fft_array(patch, fft_axis, fft_dim):
    """Get the array info from pre-transformed patch."""
    fft = patch.data
    fft_coord = patch.get_coord(fft_dim)
    cx_len = patch.shape[fft_axis]
    fast_len = cx_len
    # If fft axis is strictly positive a real dft was used.
    is_real = fft_coord.min() >= 0
    return fft, fft_coord, cx_len, fast_len, is_real


def _get_source_fft(patch, dim, source, source_axis, samples):
    """
    Get an array of coordinate sources.

    This function will place the new sources in a third dimension so
    they broadcast with the original fft matrix.
    """
    ndim = patch.ndim
    # Extract an array containing just the sources
    coord_source = patch.get_coord(dim)
    index_source = coord_source.get_next_index(source, samples=samples)
    selecter = [slice(None), slice(None), None]
    selecter[source_axis] = np.atleast_1d(index_source)
    source = patch.data[tuple(selecter)]
    # Now transpose source so source dim is list. Essentially we just
    # need to swap the source axis with the last axis.
    dims_now = list(range(len(source.shape)))
    dims_now[-1] = source_axis
    dims_now[source_axis] = len(dims_now) - 1
    out = np.transpose(source, tuple(dims_now))
    return out


@patch_function()
def correlate_shift(patch, dim, undo_weighting=True):
    """
    Apply a shift to the patch data to undo correlation in frequency domain.

    Also adds the appropriate coordinate prefixed with "lag" and has a datatype
    of float.

    Parameters
    ----------
    patch
        The input patch
    dim
        The dimension name that was correlated in the freq. domain.
    undo_weighting
        If True, also undo the weighting artifact caused by DASCore's dft
        weighting. This is done by simply dividing by the coordinate step.
        See [dft note](`notes/dft_notes.qmd`) for more details.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Example 1
    >>> # An auto-correlation of the example patch
    >>> dft = patch.dft("time", real=True)
    >>> dft_sq = dft * dft.conj()
    >>> idft = dft_sq.idft()
    >>> auto_patch = idft.correlate_shift(dim="time")
    """
    coord = patch.get_coord(dim, require_evenly_sampled=True)
    axis = patch.dims.index(dim)
    data = np.fft.fftshift(patch.data, axes=axis)
    if undo_weighting:
        data = data / to_float(coord.step)
    # so it appears (from testing) there is one lest sample on the positive
    # side.
    step = to_float(coord.step)
    new_start = -np.ceil((len(coord) - 1) / 2) * step
    new_end = np.ceil((len(coord) - 1) / 2) * step
    new_coord = dc.get_coord(start=new_start, stop=new_end, step=step)
    assert len(new_coord) == len(coord)
    cm = patch.coords
    new_cm = cm.update(**{dim: new_coord}).rename_coord(**{dim: f"lag_{dim}"})
    out = patch.update(data=data, coords=new_cm)
    return out


@patch_function()
def correlate(
    patch: PatchType,
    samples=False,
    **kwargs,
) -> PatchType:
    """
    Correlate source row/columns in a 2D patch with all other row/columns.

    Correlations are done in the frequency domain. This function can accept a
    patch whose target dimension has already been transformed with the
    [`Patch.dft`](`dascore.proc.fourier.dft`) method, otherwise the dft
    will be performed. If the input has already been transformed,
    [`Patch.correlation_shift`](`dascore.proc.correlation.correlation_shift`)
    is useful to undo dft artefacts after the idft is applied.

    While a 2D patch is required for input, a 3D patch is returned where the
    3rd dimesion cooresponds to the source rows/columns. For the case of a
    single source, the [`Patch.squeeze`](`dascore.Patch.squeeze`) method
    can be helpful to remove length 1 dimensions.

    Parameters
    ----------
    patch : PatchType
        The input data patch to be cross-correlated. Must be 2-dimensional.
        The patch can be in time or frequency domains.
    samples : bool, optional (default = False)
        If True, the argument specified in kwargs refers to the *sample* not
        value along that axis. See examples for details.
    **kwargs
        Specifies correlation dimension and the master source(s), to which
        we want to cross-correlate all other channels/time samples.If the
        master source is an array, the function will compute cc for all
        the posible pairs.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.units import m, s

    >>> # Get a patch composed of sin waves whose correlation results
    >>> # can easily be checked.
    >>> patch = dc.get_example_patch(
    >>>     "sin_wav",
    >>>     sample_rate=100,
    >>>     frequency=range(10, 20),
    >>>     duration=5,
    >>>     channel_count=10,
    >>> ).taper(time=0.5)

    >>> # Example 1
    >>> # Calculate cc for all channels as receivers and
    >>> # the 10 m channel as the master channel. Squeeze the output
    >>> # so the returned patch is 2D.
    >>> cc_patch = patch.correlate(distance = 10 * m).squeeze()

    >>> # Example 2
    >>> # Get cc within (-2,2) sec of lag for all channels as receivers
    >>> # and the 10 m channel as the master channel. The new patch has dimensions
    >>> # (lag_time, distance, source_distance)
    >>> cc_patch = (
    >>>     patch.correlate(distance = 10 * m)
    >>>     .select(lag_time=(-2, 2))
    >>> )
    >>>

    >>> # Example 3
    >>> # Use 2nd channel (python is 0 indexed) along distance as master channel
    >>> cc_patch = patch.correlate(distance=1, samples=True)

    >>> # Example 4
    >>> # Correlate along time dimension
    >>> cc_patch = patch.correlate(time=100, samples=True)

    >>> # Example 5
    >>> # An example pipeline of frequency domain correlation.
    >>> padded_patch = patch.pad(time="correlate")  # pad to at least 2n + 1
    >>> dft_patch = patch.dft("time", real=True)
    >>> # Any other pre-processing steps go here...
    >>> ...
    >>> # Perform the correlation with 3 source channels
    >>> cc_patch = dft_patch.correlate(distance=[1, 3, 7], samples=True)
    >>> # Perform any post-processing here
    >>> ...
    >>> # Convert back to time domain, apply `correlate shift` to undo
    >>> # fft related shifting and scaling.
    >>> cc_out = cc_patch.idft().correlate_shift("time")

    Notes
    -----
    1- The cross-correlation is performed in the frequency domain.

    2- The output dimension is opposite of the one specified in kwargs, has
    the units of float, and the string "lag_" prepended. For example, "lag_time".

    3- If the patch is in the frequency domain, it needs to be zero-padded in
    the time domain first. If not, first apply the
    [idft](`dascore.transform.fourier.idft`) function to the patch, and then
    use the correlate function, which automatically handles the zero padding.
    """
    assert len(patch.dims) == 2, "must be a 2D patch."
    dim, source_axis, source = get_dim_value_from_kwargs(patch, kwargs)
    # Get the axis and coord over which fft should be calculated.
    fft_axis = next(iter(set(range(len(patch.dims))) - {source_axis}))
    fft_dim = patch.dims[fft_axis]
    # Determine if the input patch has already been transformed.
    input_dft = fft_dim.startswith("ft_")
    is_real = not np.issubdtype(patch.data.dtype, np.complexfloating)
    if not input_dft:  # Standard dft workflow for correlation
        # Note: we use .func here to avoid getting these added to the history.
        padded = patch.pad.func(patch, **{fft_dim: "correlate"})
        patch = padded.dft.func(padded, fft_dim, real=fft_dim if is_real else None)
    # Get the sources.
    source = patch.get_coord(dim).values if source is None else source
    source_fft = _get_source_fft(patch, dim, source, source_axis, samples)
    # Need to insert new axis so the arrays broadcast correctly.
    fft_patch_array = patch.data[..., None]
    fft_prod = fft_patch_array * np.conj(source_fft)
    # Create frequency domain patch with results
    new_coord = dc.get_coord(values=np.atleast_1d(source))
    dim_name = f"source_{dim}"
    cm = patch.coords.update(**{dim_name: (dim_name, new_coord)})
    out = patch.update(data=fft_prod, coords=cm)
    # Undo fft if this function did one, shift, and update coord.
    if not input_dft:
        idft = out.idft.func(out)
        out = idft.correlate_shift.func(idft, fft_dim)
    return out

