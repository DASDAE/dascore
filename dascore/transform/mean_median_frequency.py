"""Rolling mean and median frequency transforms for DASCore patches"""

from __future__ import annotations

from functools import partial

import numpy as np
from scipy import fftpack
from scipy.signal import get_window

from dascore.constants import PatchType
from dascore.utils.patch import patch_function


def _welch(xx, win, fs=1.0, nperseg=256):
    """
    Estimate power spectral density using Welch's method.
    Specialized WELCH function to speed up processing in mean-frequency

    Welch's method computes an estimate of the power spectral density by
    dividing the data into overlapping segments, computing a modified
    periodogram for each segment and averaging the periodograms.

    This welch method is from the scipy library, to cut down on processing
    cost this is a reduced algorithm.

    Assumed_values:
        nfft: None
        detrend: 'constant'
        returns_onsided: True
        scaling: 'density'
        noverlap: None
        axis: -1

    Args:
        x: Time series of measurement data
        win: Desired window to use in the type of an array
        fs: Sampling frequency of the time series
        nperseg: Length of each segment

    Returns
    -------
        f (array of sampling frequencies), pxx (Power spectral density)
    """
    if xx.shape[-1] < nperseg:
        nperseg = xx.shape[-1]
        win = get_window("hann", nperseg)

    # if scaling == 'density':
    scale = 1.0 / (fs * (win * win).sum())

    # if noverlap is None:
    noverlap = nperseg // 2

    # if nfft is None:
    nfft = nperseg

    step = nperseg - noverlap
    indices = np.arange(0, xx.shape[2] - nperseg + 1, step)
    psd = np.zeros((xx.shape[0], xx.shape[1], nfft // 2 + 1), float)

    # Although this seems inefficient, tests show that a loop is the fastest solution
    for i in range(xx.shape[1]):
        x = xx[:, i, :]
        outshape = list(x.shape)
        if nfft % 2 == 0:  # even
            outshape[-1] = nfft // 2 + 1
            pxx = np.empty(outshape, x.dtype)

            ind = indices[0]
            x_dt = x[:, ind : ind + nperseg] - np.mean(
                x[:, ind : ind + nperseg], 1, keepdims=True
            )
            xft = fftpack.rfft(x_dt * win, nfft)
            pxx[:, (0, -1)] = xft[:, (0, -1)] ** 2
            pxx[:, 1:-1] = xft[:, 1:-1:2] ** 2 + xft[:, 2::2] ** 2
            for k_i, ind in enumerate(indices[1:]):
                x_dt = x[:, ind : ind + nperseg] - np.mean(
                    x[:, ind : ind + nperseg], 1, keepdims=True
                )
                xft = fftpack.rfft(x_dt * win, nfft)
                # fftpack.rfft returns the positive frequency part of the fft
                # as real values, packed r r i r i r i ...
                # this indexing is to extract the matching real and imaginary
                # parts, while also handling the pure real zero and nyquist
                # frequencies.
                k = k_i + 1
                pxx *= k / (k + 1.0)
                pxx[:, (0, -1)] += xft[:, (0, -1)] ** 2 / (k + 1.0)
                pxx[:, 1:-1] += (xft[:, 1:-1:2] ** 2 + xft[:, 2::2] ** 2) / (k + 1.0)

        else:  # odd
            outshape[-1] = (nfft + 1) // 2
            pxx = np.empty(outshape, x.dtype)

            ind = indices[0]
            x_dt = x[:, ind : ind + nperseg] - np.mean(
                x[:, ind : ind + nperseg], 1, keepdims=True
            )
            xft = fftpack.rfft(x_dt * win, nfft)
            pxx[:, 0] = xft[:, 0] ** 2
            pxx[:, 1:] = xft[:, 1::2] ** 2 + xft[:, 2::2] ** 2
            for k_i, ind in enumerate(indices[1:]):
                x_dt = x[:, ind : ind + nperseg] - np.mean(
                    x[:, ind : ind + nperseg], 1, keepdims=True
                )
                xft = fftpack.rfft(x_dt * win, nfft)
                k = k_i + 1
                pxx *= k / (k + 1.0)
                pxx[:, 0] += xft[:, 0] ** 2 / (k + 1)
                pxx[:, 1:] += (xft[:, 1::2] ** 2 + xft[:, 2::2] ** 2) / (k + 1.0)

        if nfft % 2 == 0:
            # even: DC and Nyquist not doubled
            pxx[:, 1:-1] *= 2 * scale
            pxx[:, (0, -1)] *= scale
        else:
            # odd: only DC excluded from doubling
            pxx[:, 1:] *= 2 * scale
            pxx[:, 0] *= scale

        psd[:, i, :] = pxx
    f = np.arange(pxx.shape[-1]) * (fs / nfft)

    return f, psd


# %%
def _fft_psd(xx, fs=1.0):
    """
    Simple one-sided PSD from FFT.

    Parameters
    ----------
    xx : ndarray
        Input array with shape (..., time)
    fs : float
        Sampling frequency

    Returns
    -------
    f : ndarray
        Frequency axis
    pxx : ndarray
        One-sided power spectral density
    """
    n = xx.shape[-1]

    # remove mean along time axis (like detrend='constant')
    x = xx - np.mean(xx, axis=-1, keepdims=True)

    # FFT
    xft = np.fft.rfft(x, axis=-1)

    # Power spectral density
    pxx = (np.abs(xft) ** 2) / (fs * n)

    # One-sided correction
    if n % 2 == 0:
        if pxx.shape[-1] > 2:
            pxx[..., 1:-1] *= 2.0
    else:
        if pxx.shape[-1] > 1:
            pxx[..., 1:] *= 2.0

    f = np.fft.rfftfreq(n, d=1.0 / fs)
    return f, pxx


# %% define a wrapper function to call
def _get_psd_in_window(data, method="WELCH", dt=1, fmin=None, fmax=None, nperseg=None):
    allowed_methods = ["WELCH", "FFT"]
    if method.upper() not in allowed_methods:
        raise ValueError(
            f"Invalid method: '{method}'. Must be one of {allowed_methods}"
        )

    if fmin is not None and fmax is not None:
        if fmin >= fmax:
            raise ValueError(f"fmin ({fmin}) must be smaller than fmax ({fmax})")

    if nperseg is None:
        nperseg = data.shape[-1]
    win = get_window("hann", nperseg)

    if method.upper() == "WELCH":
        frq, pxx = _welch(data, win, (1 / dt), nperseg=nperseg)

    elif method.upper() == "FFT":
        frq, pxx = _fft_psd(data, fs=(1 / dt))

    # use only desired part of spectrum
    inf1 = 0 if fmin is None else np.searchsorted(frq, fmin)
    inf2 = -1 if fmax is None else np.searchsorted(frq, fmax)

    pxx = pxx[:, :, inf1:inf2]
    frq = frq[inf1:inf2]
    return (
        frq,
        pxx,
    )


# %%
@patch_function(required_dims=("time",), history="full")
def mean_frequency(
    patch: PatchType,
    winlen: float,
    step: float | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    method: str = "welch",
    nperseg: int = 256,
) -> PatchType:
    """
    Compute rolling mean-frequency along a time axis. This represents "center of
    gravity" of the signal's power spectrum.

    See for more detail:
        - @Phinyomark12
        - [Online version of the Phinyomark publication](
            https://www.intechopen.com/chapters/40123)
        - [Matlab's meanfreq](https://se.mathworks.com/help/signal/ref/meanfreq.html)

    Note: The output  replaces the original `time` coordinate with window-centered
    timestamps.



    Parameters
    ----------
    patch
        Input DASCore patch.
    winlen
        Window length in seconds.
    step
        Step between windows in seconds. Defaults to original sampling rate.
    fmin
        Optional lower frequency bound in Hz, applied on calculated spectra
    fmax
        Optional upper frequency bound in Hz, applied on calculated spectra
    method
        Method for calculating spectrum.
        Can be any of "welch" (default), or "fft"

    Returns
    -------
        The Patch instance with the mean-frequency as data.

    Example
    -------
    >>> import dascore as dc
    >>> import matplotlib.pyplot as plt
    >>>
    >>> patch = dc.examples.get_example_patch('example_event_2')
    >>>
    >>> fig, axs = plt.subplots(1,2, layout='constrained', figsize=(12,4))
    >>> ax = patch.viz.waterfall(cmap='seismic', ax=axs[0])
    >>>
    >>> para = {'cmap':'turbo', 'scale':(50,300), 'scale_type':'absolute' }
    >>> mea = patch.mean_frequency(winlen=.010, step=.001, fmin=50, fmax=300)
    >>> ax = mea.viz.waterfall(**para, ax=axs[1])

    """
    dt = patch.get_coord("time").step
    if isinstance(dt, np.timedelta64):
        dt = dt / np.timedelta64(1, "s")

    if step is None:
        step = dt

    def _get_mean_freq_in_window(
        data, axis=None, dt=dt, method=method, fmin=fmin, fmax=fmax, nperseg=nperseg
    ):
        frq, pxx = _get_psd_in_window(
            data, dt=dt, method=method, fmin=fmin, fmax=fmax, nperseg=nperseg
        )
        # calculate mean frequency
        """
        mfr = np.sum(frq * pxx, axis=-1) / np.sum(pxx, axis=-1)
        """
        numerator = np.sum(frq * pxx, axis=-1)
        denominator = np.nansum(pxx, axis=-1)
        mfr = np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
        )

        return mfr

    # create a "partial" function to allow arguments being parsed to the "apply"
    # method of the rolling function
    fun = partial(
        _get_mean_freq_in_window,
        dt=dt,
        method=method,
        fmin=fmin,
        fmax=fmax,
        nperseg=nperseg,
    )

    # now apply the rolling function; make sure time axis is last dimension
    dim = "time"
    mfr_patch = (
        patch.transpose(..., dim)
        .rolling(time=winlen, step=step, samples=False)
        .apply(fun)
    )
    if patch.get_axis("distance") != mfr_patch.get_axis("distance"):
        mfr_patch = mfr_patch.transpose()

    return mfr_patch.update(attrs={"data_type": "Mean Frequency", "data_units": "Hz"})


# %%
@patch_function(required_dims=("time",), history="full")
def median_frequency(
    patch: PatchType,
    winlen: float,
    step: float | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    nperseg: int = 256,
    method: str = "welch",
) -> PatchType:
    """
    Compute rolling median-frequency along a time axis. This measure divides a signal's
    power spectrum into two regions of equal total power. It identifies the exact
    frequency point where 50% of the signal's energy is above, and 50% is below.

    See for more detail:
        - @Phinyomark12
        - [Online version of the Phinyomark publication](
            https://www.intechopen.com/chapters/40123)
        - [Matlab's medfreq](https://se.mathworks.com/help/signal/ref/medfreq.html)

    Note: The output  replaces the original `time` coordinate with window-centered
    timestamps.


    Parameters
    ----------
    patch
        Input DASCore patch.
    winlen
        Window length in seconds.
    step
        Step between windows in seconds. Defaults to original sampling rate.
    fmin
        Optional lower frequency bound in Hz, applied on calculated spectra
    fmax
        Optional upper frequency bound in Hz, applied on calculated spectra
    method
        Method for calculating spectrum.
        Can be any of "welch" (default), or "fft"

    Returns
    -------
        The Patch instance with the median-frequency as data.


    Example
    -------
    >>> import dascore as dc
    >>> import matplotlib.pyplot as plt
    >>>
    >>> patch = dc.examples.get_example_patch('example_event_2')
    >>>
    >>> fig, axs = plt.subplots(1,2, layout='constrained', figsize=(12,4))
    >>> ax = patch.viz.waterfall(cmap='seismic', ax=axs[0])
    >>>
    >>> para = {'cmap':'turbo', 'scale':(50,300), 'scale_type':'absolute' }
    >>> med = patch.median_frequency(winlen=.01, step=.001, fmin=50, fmax=300)
    >>> ax = med.viz.waterfall(**para, ax=axs[1] )
    """
    dt = patch.get_coord("time").step
    if isinstance(dt, np.timedelta64):
        dt = dt / np.timedelta64(1, "s")

    if step is None:
        step = dt

    def _get_median_freq_in_window(
        data, axis=None, dt=dt, method=method, fmin=fmin, fmax=fmax, nperseg=nperseg
    ):
        frq, pxx = _get_psd_in_window(
            data, dt=dt, method=method, fmin=fmin, fmax=fmax, nperseg=nperseg
        )

        # calculate median frequency
        cumulative_psd = np.cumsum(pxx, axis=-1)
        total_power = cumulative_psd[:, :, -1]
        half = total_power[..., None] / 2
        median_idx = np.argmax(cumulative_psd >= half, axis=-1)

        return frq[median_idx]

    # create a "partial" function to allow arguments being parsed to the "apply"
    # method of the rolling function
    fun = partial(
        _get_median_freq_in_window,
        dt=dt,
        method=method,
        fmin=fmin,
        fmax=fmax,
        nperseg=nperseg,
    )

    # now apply the rolling function
    dim = "time"
    med_patch = (
        patch.transpose(..., dim)
        .rolling(time=winlen, step=step, samples=False)
        .apply(fun)
    )
    if patch.get_axis("distance") != med_patch.get_axis("distance"):
        med_patch = med_patch.transpose()

    return med_patch.update(attrs={"data_type": "Median Frequency", "data_units": "Hz"})
