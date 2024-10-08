"""Spectral whitening."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from scipy.signal.windows import tukey

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.misc import (
    broadcast_for_index,
    check_filter_range,
    check_filter_sequence,
)
from dascore.utils.time import to_float
from dascore.utils.transformatter import FourierTransformatter


def _check_whiten_inputs(patch, smooth_size, tukey_alpha, dim, freq_range):
    """Ensure inputs to whiten function are ok."""
    coord = patch.get_coord(dim, require_evenly_sampled=True)
    step = to_float(coord.step)
    nyquist = 1 / (2 * step)
    fft_step = nyquist / len(coord)

    if tukey_alpha < 0 or tukey_alpha > 1:
        msg = "Tukey alpha needs to be between 0 and 1"
        raise ParameterError(msg)

    if freq_range is None:
        return nyquist

    check_filter_sequence(freq_range)
    range_min, range_max = freq_range

    low = None if pd.isnull(range_min) else range_min / nyquist
    high = None if pd.isnull(range_max) else range_max / nyquist
    check_filter_range(nyquist, low, high, range_min, range_max)

    if smooth_size is None:
        smooth_size = range_max - range_min
    elif smooth_size <= 0:
        msg = "Frequency smoothing size must be positive"
        raise ParameterError(msg)
    elif smooth_size >= nyquist:
        msg = "Frequency smoothing size is larger than Nyquist"
        raise ParameterError(msg)

    if ((range_max - range_min) / fft_step) < 2:
        msg = "Frequency range is too narrow"
        raise ParameterError(msg)

    # We need at least 5 smoothing points. Before, we could increase fft resolution
    # so that smooth size could be smaller, but now that whiten can accept fft
    # patches this is no longer practical. The user will have to resample to
    # higher resolutions before whitening.
    if smooth_size < (fft_step * 5):
        msg = "Frequency smoothing size is smaller than 5x frequency resolution"
        raise ParameterError(msg)

    return smooth_size


def _get_amplitude_envelope(
    fft_patch,
    dim,
    smooth_size,
    freq_range,
):
    """Calculate a normalization envelope for the spectra."""
    fft_coord = fft_patch.get_coord(dim)
    index = fft_patch.dims.index(dim)
    freq_step = fft_coord.step
    mean_window = math.floor(smooth_size / freq_step)
    if freq_range and mean_window > len(fft_coord.select(freq_range)[0]):
        msg = "Frequency smoothing size is larger than frequency range"
        raise ParameterError(msg)

    conv_window = np.arange(mean_window) / mean_window

    # convolve original spectrum with smoothing window
    amp = np.abs(fft_patch.data)
    conv = convolve1d(amp, conv_window, axis=index, mode="constant")
    sm_amp = np.divide(
        1,
        conv,
        out=np.zeros_like(conv),
        where=~np.isclose(conv, 0),
    )
    return sm_amp


def _filter_array(envelope, fft_patch, dim, freq_range, tukey_alpha):
    """Apply Tukey window to filter array for frequencies of interest."""
    fft_coord = fft_patch.get_coord(dim)
    fft_ind = fft_patch.dims.index(dim)
    assert isinstance(freq_range, Sequence) and len(freq_range) == 2

    freq_ind1 = fft_coord.get_next_index(freq_range[0])
    freq_ind2 = fft_coord.get_next_index(freq_range[1])
    freq_win_size = freq_ind2 - freq_ind1

    # Get indexer to make 1D array broadcastable along fft
    tuk_bcast_inds = broadcast_for_index(
        fft_patch.ndim, fft_ind, value=Ellipsis, fill=None
    )
    env_bcast_inds = broadcast_for_index(
        fft_patch.ndim, fft_ind, value=slice(freq_ind1, freq_ind2), fill=slice(0)
    )
    tuk = tukey(freq_win_size, alpha=tukey_alpha)[tuk_bcast_inds]
    envelope[env_bcast_inds] *= tuk
    return envelope


def _get_dim_freq_range_from_kwargs(patch, kwargs):
    """Get the dimension and frequency range."""
    # No kwargs provided, look for time / ft_time.
    dim_set = set(patch.dims)
    # Handles the default case when no kwargs passed.
    if not kwargs:
        expected = {"time", "ft_time"} & dim_set
        if not expected:
            msg = "No dim name provided in kwargs and patch has no time dimension."
            raise ParameterError(msg)
        dim = next(iter(expected))
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


def whiten(
    patch: PatchType,
    smooth_size: float | None = None,
    tukey_alpha: float = 0.1,
    idft: bool = True,
    **kwargs,
) -> PatchType:
    """
    Band-limited signal whitening.

    The whitened signal is returned in either frequency or time domains.

    Parameters
    ----------
    patch
        Patch to transform. Has to have dimensions of time and distance.
    smooth_size
        Size (in Hz) of moving average window, used to compute the spectrum
        before whitening. If no value is inputted, smoothing is over
        the entire spectrum.
    tukey_alpha
        Alpha parameter for Tukey window applied as windowing to the
        smoothed spectrum within the required frequency range. By default
        its value is 0.1.
        See more details at https://docs.scipy.org/doc/scipy/reference
        /generated/scipy.signal.windows.tukey.html
    idft
        Deprecated, output domain is now the same as input domain.
    **kwargs
        Used to specify the dimension and frequency, wavelength, or equivalent
        limits. Can also be None which defaults to [0, Nyquist]. If no input is
        provided, whitening will be applied to a dimension called "time" or
        "ft_time" if one exists, otherwise a ParameterError is raised.

    Notes
    -----
    1) The FFT result is divided by the smoothed spectrum before inverting
       back to time-domain signal. The phase is not changed.

    2) A tukey window (fixed) is applied to window the smoothed spectrum
       within the frequency range of interest. Be aware of its effect and
       consider enlarging the frequency range according to the tukey_alpha
       parameter.

    3) Amplitude is NOT preserved

    4) If idft = False, since for the purely real input data the negative
       frequency terms are just the complex conjugates of the corresponding
       positive-frequency terms, the output does not include the negative
       frequency terms, and therefore the length of the transformed axis
       of the output is n//2 + 1. Refer to the
       [dft patch function](`dascore.transform.fourier.dft`) and its `real` flag.

    Example
    -------
    ```{python}
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy.fft as fft

    import dascore as dc
    from dascore.units import Hz

    def plot_spectrum(x, T, ax, phase=False):
        fftphase = np.angle(fft.fft(x))
        fftsig = np.abs(fft.fft(x))
        fftlen = fftsig.size
        fftsig = fftsig[0 : int(fftlen / 2) + 1]
        fftphase = fftphase[0 : int(fftlen / 2) + 1]
        freqvec = np.linspace(0, 0.5 / T, fftsig.size)
        if not phase:
            ax.plot(freqvec, fftsig)
            ax.set_xlabel("frequency [Hz]")
            ax.set_ylabel("Amplitude (|H(w)|)")
        else:
            ax.plot(freqvec, fftphase)
            ax.set_xlabel("frequency [Hz]")
            ax.set_ylabel("Phase (radians)")


    patch = dc.get_example_patch("dispersion_event")
    patch = patch.resample(time=(200 * Hz))

    white_patch = patch.whiten(smooth_size=3, time = (10,50))

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 7))
    ax1.plot(patch.data[50, :])
    ax1.set_title("Original data, distance = 50 m")
    ax2.plot(white_patch.data[50, :])
    ax2.set_title("Whitened data, distance = 50 m")

    plot_spectrum(patch.data[50, :], 1 / 200, ax3)
    ax3.set_title("Original data, distance = 50 m")
    plot_spectrum(white_patch.data[50, :], 1 / 200, ax4)
    ax4.set_title("Whitened data, distance = 50 m")

    plot_spectrum(patch.data[50, :], 1 / 200, ax5, phase=True)
    ax5.set_title("Original data, distance = 50 m")
    plot_spectrum(white_patch.data[50, :], 1 / 200, ax6, phase=True)
    ax6.set_title("Whitened data, distance = 50 m")
    plt.tight_layout()
    plt.show()
    ```
    """
    # Set default kwargs if not set, check inputs.
    dim, freq_range = _get_dim_freq_range_from_kwargs(patch, kwargs)
    smooth_size = _check_whiten_inputs(patch, smooth_size, tukey_alpha, dim, freq_range)
    # Get patch in dft form.
    fft_patch = patch.dft(dim, real=np.isrealobj(patch.data))
    input_patch_fft = fft_patch is patch  # if input patch had fft
    fft_dim = FourierTransformatter().rename_dims(dim)[0]

    # Get an envelope which can be used to normalize the spectra so each freq
    # has roughly the same amplitude (but phase remains unchanged).
    envelope = _get_amplitude_envelope(fft_patch, fft_dim, smooth_size, freq_range)

    # If a frequency range was specified, we can simply modify the envelope
    # array so that only frequencies outside of it are attenuated.
    if freq_range is not None:
        envelope = _filter_array(envelope, fft_patch, fft_dim, freq_range, tukey_alpha)

    # Now use calculated envelope to normalize spectra
    array = fft_patch.data * envelope

    # Package up output and return.
    out = fft_patch.new(data=array)
    if not input_patch_fft:
        out = out.idft()
    return out
