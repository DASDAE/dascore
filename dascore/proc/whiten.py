"""Spectral whitening."""
from __future__ import annotations

import math

import numpy as np
import numpy.fft as nft
import pandas as pd
from scipy.ndimage import convolve1d
from scipy.signal.windows import tukey

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.misc import check_filter_kwargs, check_filter_range
from dascore.utils.patch import get_dim_sampling_rate


def whiten(
    patch: PatchType,
    smooth_size: float | None = None,
    tukey_alpha: float = 0.1,
    **kwargs,
) -> PatchType:
    """
    Band-limited signal whitening.

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
    **kwargs
        Used to specify the dimension and frequency, wavelength, or equivalent
        limits. If no input is provided, whitening is also the last axis
        with frequency band of [0,Nyquist]

    Notes
    -----
    1) The FFT result is divided by the smoothed spectrum before inverting
       back to time-domain signal. The phase is not changed.

    2) A tukey window (fixed) is applied to window the smoothed spectrum
       within the frequency range of interest. Be aware of its effect and
       consider enlarging the frequency range according to the tukey_alpha
       parameter.

    3) Amplitude is NOT preserved


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
    if kwargs:
        dim, (rang_min, rang_max) = check_filter_kwargs(kwargs)
        dim_ind = patch.dims.index(dim)
        dsamp = 1.0 / get_dim_sampling_rate(patch, dim)
        nyquist = 0.5 / dsamp
    else:
        dim_ind = -1
        dsamp = 1.0 / get_dim_sampling_rate(patch, patch.dims[dim_ind])
        nyquist = 0.5 / dsamp

        rang_min = 0
        rang_max = nyquist

    low = None if pd.isnull(rang_min) else rang_min / nyquist
    high = None if pd.isnull(rang_max) else rang_max / nyquist
    check_filter_range(nyquist, low, high, rang_min, rang_max)

    if smooth_size is None:
        smooth_size = rang_max - rang_min
    else:
        if smooth_size <= 0:
            msg = "Frequency smoothing size must be positive"
            raise ParameterError(msg)
        if smooth_size >= nyquist:
            msg = "Frequency smoothing size is larger than Nyquist"
            raise ParameterError(msg)

    if tukey_alpha < 0 or tukey_alpha > 1:
        msg = "Tukey alpha needs to be between 0 and 1"
        raise ParameterError(msg)

    nsamp = patch.data.shape[dim_ind]
    temp = nft.rfftfreq(nsamp, d=dsamp)  # Compute default frequency resolution
    default_df = temp[1] - temp[0]
    # RFFT does not compute nyquist itself, so we stop one discrete frequency earlier
    if rang_max == nyquist:
        rang_max -= default_df

    # Virtually increase computation resolution so that the smoothing window
    # contains 5 discrete frequencies. However, if the input value is smaller
    # than the default resolution, raise a parameter error because the
    # computational cost would go up
    if smooth_size < default_df:
        msg = "Frequency smoothing size is smaller than default frequency resolution"
        raise ParameterError(msg)

    if math.floor(smooth_size / default_df) < 5:
        comp_nsamp = math.floor(nsamp * 5 / (smooth_size / default_df))
    else:
        comp_nsamp = nsamp

    freqs = nft.rfftfreq(comp_nsamp, d=dsamp)
    df = freqs[1] - freqs[0]
    nf = np.size(freqs)
    fft_size = np.asarray(np.shape(patch.data))
    fft_size[dim_ind] = comp_nsamp

    fft_d = np.zeros(shape=fft_size, dtype=complex)
    whitened_data = np.zeros(np.shape(patch.data))

    first_freq_ind = np.argmax((np.abs(freqs) - rang_min) >= 0.0)
    last_freq_ind = np.argmax((np.abs(freqs) - rang_max) >= 0.0)
    freq_win_size = last_freq_ind - first_freq_ind

    mean_window = math.floor(smooth_size / df)

    if freq_win_size < 2:
        msg = "Frequency range is too narrow"
        raise ParameterError(msg)
    if mean_window > last_freq_ind - first_freq_ind + 1:
        msg = "Frequency smoothing size is larger than frequency range"
        raise ParameterError(msg)

    assert np.all(np.isreal(patch.data)), "Input data needs to be real"

    fft_d = nft.rfft(patch.data, n=comp_nsamp, axis=dim_ind)
    amp = np.abs(fft_d)  # Original spectrum
    phase = np.angle(fft_d)  # Original phase

    conv_window = np.arange(mean_window) / mean_window

    # Define tapering window
    taper_win = np.zeros(nf)
    taper_win[first_freq_ind:last_freq_ind] = tukey(freq_win_size, alpha=tukey_alpha)

    # convolve original spectrum with smoothing window
    sm_amp = convolve1d(amp, conv_window, axis=dim_ind, mode="constant")

    # divide original spectrum by smoothed spectrum
    norm_amp = np.divide(
        amp, np.abs(sm_amp), out=np.zeros_like(sm_amp), where=abs(sm_amp) != 0
    )

    # Generate N-D tapering window and multiply normalize spectrum to get desired
    # frequency range.
    exp_dims = np.asarray(norm_amp.shape)
    exp_dims[dim_ind] = 1
    tiled_win = taper_win

    for d in exp_dims:
        if d > 1:
            tiled_win = np.tile(np.expand_dims(tiled_win, axis=-1), (1, d))

    tiled_win = np.transpose(tiled_win, np.roll(np.arange(len(exp_dims)), dim_ind))

    norm_amp *= tiled_win

    # Revert back to time-domain, using the phase of the original signal
    whitened_data = np.real(
        nft.irfft(norm_amp * np.exp(1j * phase), n=comp_nsamp, axis=dim_ind)
    )
    whitened_data = np.take(whitened_data, np.arange(nsamp), axis=dim_ind)

    return patch.new(data=whitened_data)
