"""Signal whitening for time-domain applications."""
from __future__ import annotations

import math

import numpy as np
import numpy.fft as nft

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.patch import patch_function


@patch_function(required_dims=("time", "distance"))
def whitening(
    patch: PatchType,
    freq_range: [None, None] | float = None,
    freq_smooth_size: None | float = None,
) -> PatchType:
    """
    Band-limited signal whitening.

    Parameters
    ----------
    patch
        Patch to transform. Has to have dimensions of time and distance.
    freq_range
        Frequency range for which the whitening is applied, in Hz
    freq_smooth_size
        Size (in Hz) of moving average window, used to compute the spectrum
        before whitening

    Notes
    -----
    1) The FFT result is divided by the smoothed spectrum before inverting
       back to time-domain signal. The phase is not changed.

    2) Amplitude is NOT preserved

    3) The output is also bandpass filtered to the desired frequency range,
    but the smoothing window includes frequencies outside that range

    Example
    -------

    import dascore as dc
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.signal as spec
    from dascore.units import Hz
    import numpy.fft as fft

    def plot_spectrum(x,T,ax):
        fftsig=np.abs(fft.fft(x))
        fftlen=fftsig.size
        fftsig=fftsig[0:int(fftlen/2)+1]
        freqvec=np.linspace(0,0.5/T,fftsig.size)
        ax.plot(freqvec, fftsig)
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('Amplitude (|H(w)|)')

    patch = (
        dc.get_example_patch('dispersion_event')
    )
    patch = patch.resample(time=(200 * Hz))

    filt_patch = patch.pass_filter(time=(5, 60))
    white_patch = filt_patch.whitening(freq_range=[10,50],freq_smooth_size=10)

    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,7))
    ax1.plot(filt_patch.data[50,:])
    ax1.set_title('Original data, distance = 50 m')
    ax2.plot(white_patch.data[50,:])
    ax2.set_title('Whitened data, distance = 50 m')

    plot_spectrum(filt_patch.data[50,:],1/200,ax3)
    ax3.set_title('Original data, distance = 50 m')
    plot_spectrum(white_patch.data[50,:],1/200,ax4)
    ax4.set_title('Whitened data, distance = 50 m')
    plt.show()
    """
    patch_cop = patch.convert_units(distance="m").transpose("distance", "time")
    time = patch_cop.coords.get_array("time")
    dt = (time[1] - time[0]) / np.timedelta64(1, "s")

    if freq_range is None:
        freq_range = [0, 1 / 2 / dt]
    else:
        if freq_range[0] < 0 or freq_range[1] < 0:
            msg = "Minimal and maximal frequencies have to be non-negative"
            raise ParameterError(msg)

        if not np.size(freq_range) == 2:
            msg = "Frequency range must include two values"
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

    # dt = (time[1] - time[0]) / np.timedelta64(1, "s")
    (nchan, nt) = patch_cop.data.shape
    nf = 2 ** (math.ceil(math.log(nt, 2)))
    h_nf = int(nf / 2)
    fft_d = np.zeros([nchan, nf], dtype=complex)
    whitened_data = np.zeros([nchan, nt])

    freqs = nft.fftfreq(nf, d=dt)
    first_freq_ind = np.argmax((np.abs(freqs) - freq_range[0]) >= 0.0)
    last_freq_ind = np.argmax((np.abs(freqs) - freq_range[1]) >= 0.0)
    mean_window = math.floor(freq_smooth_size / (freqs[1] - freqs[0]))

    if mean_window < 1:
        msg = "Frequency smoothing size yields a smoothing window of size 0"
        raise ParameterError(msg)
    if last_freq_ind - first_freq_ind < 2:
        msg = "Frequency range is too narrow"
        raise ParameterError(msg)

    if mean_window > last_freq_ind - first_freq_ind + 1:
        msg = "Frequency smoothing size is larger than frequency range"
        raise ParameterError(msg)

    for i in range(nchan):
        fft_d[i, :] = nft.fft(patch_cop.data[i, :], n=nf)

    amp = np.abs(fft_d)
    phase = np.angle(fft_d)

    freqs[first_freq_ind:last_freq_ind]
    conv_window = np.arange(mean_window) / mean_window

    for i in range(nchan):
        amp[i, first_freq_ind:last_freq_ind] = np.convolve(
            amp[i, first_freq_ind:last_freq_ind], conv_window, mode="same"
        )
        sm_amp = amp[i, first_freq_ind:last_freq_ind]
        amp[i, first_freq_ind:last_freq_ind] = np.divide(
            sm_amp, np.abs(sm_amp), out=np.zeros_like(sm_amp), where=abs(sm_amp) != 0
        )
        amp[i, h_nf + 1 : nf] = np.flip(np.squeeze(amp[i, 1:h_nf]))
        whitened_data[i, :] = np.real(nft.ifft(amp[i, :] * np.exp(1j * phase[i, :])))[
            0:nt
        ]

    new_patch = patch.new(data=whitened_data)
    if freq_range[0] > 0 and freq_range[1] < 1 / 2 / dt:
        return new_patch.pass_filter(time=freq_range)
    elif freq_range[0] == 0 and freq_range[1] == 1 / 2 / dt:
        return new_patch
    elif freq_range[0] == 0:
        return new_patch.pass_filter(time=[..., freq_range[1]])
    else:
        return new_patch.pass_filter(time=[freq_range[0], ...])
