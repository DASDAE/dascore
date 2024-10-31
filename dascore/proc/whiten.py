"""Spectral whitening."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d

from dascore.exceptions import ParameterError
from dascore.utils.patch import patch_function
from dascore.utils.transformatter import FourierTransformatter


def _get_dim_freq_range_from_kwargs(patch, kwargs):
    """Get the dimension and frequency range."""
    dim_set = set(patch.dims)
    # Handles the default case when no kwargs passed.
    if not kwargs:
        expected = {"time", "ft_time"} & dim_set
        if not expected:
            msg = "No dim name provided in kwargs and patch has no time dimension."
            raise ParameterError(msg)
        dim = "time"
        freq_range = None
    # A single kwarg was passed.
    elif len(kwargs) == 1:
        dim, freq_range = next(iter(kwargs.items()))
        fft_dim = FourierTransformatter().rename_dims(dim)[0]
        if dim not in dim_set and fft_dim not in dim_set:
            msg = f"passed dim of {dim} to whiten but it is not in patch dimensions."
            raise ParameterError(msg)
    else:  # Something when wrong.
        msg = "Whiten kwargs must specify a single patch dimension."
        raise ParameterError(msg)

    return dim, freq_range


def _get_amp_envelope(fft_patch, axis, window_len, water_level):
    """Get a smoothed amplitude envelope."""
    conv_window = np.ones(int(window_len)) / float(window_len)
    # convolve original spectrum with smoothing window
    amp = np.abs(fft_patch.data)
    conv = convolve1d(amp, conv_window, axis=axis, mode="wrap")
    # Enforce water level to avoid instability in dividing small numbers
    conv[conv < water_level * conv.max()] = water_level * conv.max()
    # Then smooth once more to take the edge off the values that were flattened
    # by the water-level clip.
    conv = convolve1d(conv, conv_window, axis=axis, mode="wrap")
    smoothed_amp = amp / conv
    return smoothed_amp


def _check_smooth(fft_coord, smooth_size):
    """Check the smooth size."""
    if smooth_size <= 0:
        msg = "Frequency smoothing size must be positive"
        raise ParameterError(msg)
    if smooth_size >= fft_coord.max():
        msg = "Frequency smoothing size is larger than Nyquist"
        raise ParameterError(msg)


def _check_freq_range(fft_coord, freq_range):
    """Check the frequency range."""
    # Note: the freq_range can be a len 2 or 4 sequence
    frange = np.asarray(freq_range)
    diffs = frange[1:] - frange[:-1]
    min_size = fft_coord.step * 2

    if np.any(diffs < min_size):
        msg = "Frequency range is too narrow"
        raise ParameterError(msg)


@patch_function()
def whiten(patch, smooth_size=None, samples=False, water_level=0.05, **kwargs):
    """
    Spectral whitening of a signal.

    The whitened signal is returned in the same domain (eq frequency or
    time domain) as the input signal.


    Parameters
    ----------
    patch
        Patch to transform. Has to have dimensions of time and distance.
    smooth_size
        Size in transformed domain units (eg Hz) or samples of moving average
        window, used to compute the spectrum before whitening.
        If None, don't smooth signal which results in a uniform amplitude.
    samples
        If True, the `smooth_size` parameter is in samples not coordinate
        units.
    water_level
        Water level for stability in the smoothing.
    **kwargs
        Used to specify the dimension range in transformed units (e.g, Hz)
        of the smoothing. Can either be a sequence of two values or
        four values to specify a taper range. Simply uses
        [Patch.taper_range](`dascore.Patch.taper_range`) under the hood.
        If kwargs are provided, try to smooth `time` or `ft_time` coords.

    Notes
    -----
    1) The FFT result is divided by the smoothed spectrum before inverting
       back to time-domain signal. The phase is not changed.

    2) Amplitude is NOT preserved

    Example
    -------
    >>>

    """
    dim, freq_range = _get_dim_freq_range_from_kwargs(patch, kwargs)
    fft_dim = FourierTransformatter().rename_dims(dim)[0]
    # Get frequency domain patch
    fft_patch = patch.dft(dim, real=np.isrealobj(patch.data))
    input_patch_fft = fft_patch is patch  # if input patch had fft
    # Get amplitude spectra if smoothing, otherwise use ones.
    fft_coord = fft_patch.get_coord(fft_dim)
    if smooth_size is None:
        amp = np.ones_like(fft_patch.data)
    else:
        _check_smooth(fft_coord, smooth_size)
        axis = fft_patch.dims.index(fft_dim)
        window_len = fft_coord.get_sample_count(
            smooth_size, samples=samples, enforce_lt_coord=True
        )
        amp = _get_amp_envelope(fft_patch, axis, window_len, water_level)
    # Init new output using only phase data.
    out = fft_patch.new(data=amp * np.exp(1j * np.angle(fft_patch.data)))
    # Apply band-limited taper to remove some frequencies.
    if freq_range:
        _check_freq_range(fft_coord, freq_range)
        out = out.taper_range.func(out, **{fft_dim: freq_range})
    # Convert back to time domain if input was in time-domain.
    if not input_patch_fft:
        out = out.idft()
    return out
