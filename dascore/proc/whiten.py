"""Spectral whitening."""
from __future__ import annotations

import math

import numpy as np
import numpy.fft as nft
from scipy.ndimage import convolve1d
from scipy.signal.windows import tukey

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.patch import get_dim_sampling_rate, patch_function


@patch_function(required_dims=("time", "distance"))
def whiten(
    patch: PatchType,
    freq_range: tuple[float, float] | None = None,
    freq_smooth_size: float | None = None,
    tukey_alpha: float | None = None,
) -> PatchType:
    """
    Band-limited signal whitening.

    Parameters
    ----------
    patch
        Patch to transform. Has to have dimensions of time and distance.
    freq_range
        Frequency range for which the whitening is applied, in Hz.
        If no value is inputted, the entire spectrum is used.
    freq_smooth_size
        Size (in Hz) of moving average window, used to compute the spectrum
        before whitening. If no value is inputted, smoothing is over
        the entire spectrum.
    tukey_alpha
        Alpha parameter for Tukey window applied as windowing to the
        smoothed spectrum within the required frequency range. By default
        its value is 0.05.
        See more details at https://docs.scipy.org/doc/scipy/reference
        /generated/scipy.signal.windows.tukey.html

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

    filt_patch = patch.pass_filter(time=(5, 60))
    white_patch = filt_patch.whiten(freq_range=[10, 50], freq_smooth_size=3)

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 7))
    ax1.plot(filt_patch.data[50, :])
    ax1.set_title("Original data, distance = 50 m")
    ax2.plot(white_patch.data[50, :])
    ax2.set_title("Whitened data, distance = 50 m")

    plot_spectrum(filt_patch.data[50, :], 1 / 200, ax3)
    ax3.set_title("Original data, distance = 50 m")
    plot_spectrum(white_patch.data[50, :], 1 / 200, ax4)
    ax4.set_title("Whitened data, distance = 50 m")

    plot_spectrum(filt_patch.data[50, :], 1 / 200, ax5, phase=True)
    ax5.set_title("Original data, distance = 50 m")
    plot_spectrum(white_patch.data[50, :], 1 / 200, ax6, phase=True)
    ax6.set_title("Whitened data, distance = 50 m")
    plt.tight_layout()
    plt.show()

    """
    patch_cop = patch.convert_units(distance="m").transpose("distance", "time")
    dt = 1.0 / get_dim_sampling_rate(patch_cop, "time")

    if freq_range is None:
        freq_range = [0, 1 / 2 / dt]
    else:
        if not np.size(freq_range) == 2:
            msg = "Frequency range must include two values"
            raise ParameterError(msg)

        if freq_range[0] < 0 or freq_range[1] < 0:
            msg = "Minimal and maximal frequencies have to be non-negative"
            raise ParameterError(msg)

        if freq_range[1] >= 0.5 / dt:
            msg = "Frequency range exceeds Nyquist frequency"
            raise ParameterError(msg)

        if freq_range[0] >= freq_range[1]:
            msg = "Frequency range must be increasing"
            raise ParameterError(msg)

    if freq_smooth_size is None:
        freq_smooth_size = freq_range[1] - freq_range[0]
    else:
        if freq_smooth_size <= 0:
            msg = "Frequency smoothing size must be positive"
            raise ParameterError(msg)
    if tukey_alpha is None:
        tukey_alpha = 0.1
    else:
        if tukey_alpha < 0 or tukey_alpha > 1:
            msg = "Tukey alpha needs to be between 0 and 1"
            raise ParameterError(msg)

    (nchan, nt) = patch_cop.data.shape

    temp = nft.rfftfreq(nt, d=dt)  # Compute default frequency resolution
    default_df = temp[1] - temp[0]

    # Virtually increase computation resolution so that the smoothing window
    # contains 5 discrete frequencies. However, if the input value is smaller
    # than the default resolution, raise a parameter error because the
    # computational cost would go up

    if freq_smooth_size < default_df:
        msg = "Frequency smoothing size is smaller than default frequency resolution"
        raise ParameterError(msg)

    if math.floor(freq_smooth_size / default_df) < 5:
        comp_nt = math.floor(nt * 5 / (freq_smooth_size / default_df))
    else:
        comp_nt = nt

    freqs = nft.rfftfreq(comp_nt, d=dt)
    nf = np.size(freqs)
    fft_d = np.zeros([nchan, nf], dtype=complex)
    whitened_data = np.zeros([nchan, nt])

    first_freq_ind = np.argmax((np.abs(freqs) - freq_range[0]) >= 0.0)
    last_freq_ind = np.argmax((np.abs(freqs) - freq_range[1]) >= 0.0)
    freq_win_size = last_freq_ind - first_freq_ind

    mean_window = math.floor(freq_smooth_size / (freqs[1] - freqs[0]))

    if freq_win_size < 2:
        msg = "Frequency range is too narrow"
        raise ParameterError(msg)
    if mean_window > last_freq_ind - first_freq_ind + 1:
        msg = "Frequency smoothing size is larger than frequency range"
        raise ParameterError(msg)

    assert np.all(np.isreal(patch_cop.data)), "Input data needs to be real"

    fft_d = nft.rfft(patch_cop.data, n=comp_nt, axis=-1)
    amp = np.abs(fft_d)  # Original spectrum
    phase = np.angle(fft_d)  # Original phase

    conv_window = np.arange(mean_window) / mean_window

    # Define tapering window
    taper_win = np.zeros(nf)
    taper_win[first_freq_ind:last_freq_ind] = tukey(freq_win_size, alpha=tukey_alpha)

    # convolve original spectrum with smoothing window
    sm_amp = convolve1d(amp, conv_window, axis=-1, mode="constant")

    # divide original spectrum by smoothed spectrum
    norm_amp = np.divide(
        amp, np.abs(sm_amp), out=np.zeros_like(sm_amp), where=abs(sm_amp) != 0
    )

    # multiply result by tapering window to retrieve desired frequency range
    norm_amp *= np.tile(taper_win, (nchan, 1))

    # Revert back to time-domain, using the phase of the original signal
    whitened_data = np.real(nft.irfft(norm_amp * np.exp(1j * phase), n=comp_nt))[
        :, 0:nt
    ]

    return patch_cop.new(data=whitened_data)
