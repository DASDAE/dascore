"""Mean- and median-frequency transforms for DASCore patches"""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import PatchType
from dascore.units import Quantity
from dascore.utils.patch import patch_function


def _get_overlap(patch, dim, winlen, step):
    """Calculate overlap from winlen and step"""
    if step is None:
        step = patch.get_coord(dim).step
    overlap = winlen - step
    if overlap < 0:
        raise ValueError("Step must be less than or equal to winlen.")
    return overlap


def _get_stft_power(
    patch: PatchType,
    winlen,
    overlap=None,
    dim: str = "time",
    fmin: float | None = None,
    fmax: float | None = None,
) -> tuple[PatchType, np.ndarray, np.ndarray, str]:
    """
    Compute STFT power and apply optional hard frequency limits.

    Returns
    -------
    spec
        STFT patch.
    freqs
        Frequency coordinate after masking.
    power
        Power spectrum after masking.
    freq_dim
        Name of frequency dimension.
    """
    spec = patch.stft(**{dim: winlen}, overlap=overlap, taper_window="boxcar")

    freq_dim = f"ft_{dim}"
    freq_axis = spec.dims.index(freq_dim)
    freqs = np.asarray(spec.get_array(freq_dim), dtype=float)

    power = np.abs(spec.data) ** 2

    mask = np.ones(freqs.shape, dtype=bool)
    if fmin is not None:
        mask &= freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax

    if not np.any(mask):
        raise ValueError("Frequency limits exclude all STFT frequency bins.")

    freqs = freqs[mask]
    power = np.take(power, np.flatnonzero(mask), axis=freq_axis)

    return spec, freqs, power, freq_dim


def _prepare_output(
    spec: PatchType,
    data: np.ndarray,
    freq_dim: str,
    data_type: str,
    data_units: str | Quantity | None,
) -> PatchType:
    """Return a patch with the STFT frequency dimension removed."""
    dims = tuple(dim for dim in spec.dims if dim != freq_dim)
    coords = {dim: spec.get_array(dim) for dim in dims}
    attrs = {"data_type": data_type, "data_units": data_units}
    return dc.Patch(data=data, dims=dims, coords=coords, attrs=attrs)


def _broadcast_freqs(
    freqs: np.ndarray,
    ndim: int,
    freq_axis: int,
) -> np.ndarray:
    """Broadcast frequency vector against an STFT power array."""
    shape = [1] * ndim
    shape[freq_axis] = freqs.size
    return freqs.reshape(shape)


@patch_function()
def mean_frequency(
    patch: PatchType,
    winlen: float,
    dim: str = "time",
    step: float | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
) -> PatchType:
    """
    Compute rolling mean-frequency along a time axis. This represents "center of
    gravity" of the signal's power spectrum, and is sometimes also called the
    "spectral centroid".

    See for more detail:
        - @Phinyomark12
        - [Online version of the Phinyomark publication](
            https://www.intechopen.com/chapters/40123)
        - [Matlab's meanfreq](https://se.mathworks.com/help/signal/ref/meanfreq.html)

    Parameters
    ----------
    patch
        Input DASCore patch.
    winlen
        Window length to calculate spectrum.
    step
        The window is evaluated at every step result, equivalent to slicing
        at every step. Setting ``step`` to larger than the original sampling interval
        significantly speeds up processing. If the ``step`` argument is not None, the
        result will have a different shape than the input.  Default None.
    fmin
        Optional lower frequency bound, applied on calculated spectra
    fmax
        Optional upper frequency bound, applied on calculated spectra
    dim
        Dimension along which to compute the mean frequency.


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
    >>> mea = patch.mean_frequency(winlen=.02, step=.001, fmin=50, fmax=300)
    >>> ax = mea.viz.waterfall(cmap='turbo', ax=axs[1])

    """
    overlap = _get_overlap(patch, dim, winlen, step)
    spec, freqs, power, freq_dim = _get_stft_power(
        patch,
        winlen=winlen,
        overlap=overlap,
        dim=dim,
        fmin=fmin,
        fmax=fmax,
    )

    freq_axis = spec.dims.index(freq_dim)
    freqs_b = _broadcast_freqs(freqs, power.ndim, freq_axis)

    numerator = np.sum(freqs_b * power, axis=freq_axis)
    denominator = np.sum(power, axis=freq_axis)

    out = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )
    data_units = spec.get_coord(freq_dim).units
    data_type = "Mean Frequency"
    return _prepare_output(spec, out, freq_dim, data_type, data_units)


@patch_function()
def median_frequency(
    patch: PatchType,
    winlen: float,
    dim: str = "time",
    step: float | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
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

    Parameters
    ----------
    patch
        Input DASCore patch.
    winlen
        Window length to calculate spectrum.
    step
        The window is evaluated at every step result, equivalent to slicing
        at every step. Setting ``step`` to larger than the original sampling interval
        significantly speeds up processing. If the ``step`` argument is not None, the
        result will have a different shape than the input.  Default None.
    fmin
        Optional lower frequency bound, applied on calculated spectra
    fmax
        Optional upper frequency bound, applied on calculated spectra
    dim
        Dimension along which to compute the median frequency.

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
    >>> med = patch.median_frequency(winlen=.02, step=.001, fmin=50, fmax=300)
    >>> ax = med.viz.waterfall(cmap='turbo', ax=axs[1], scale=[0,1])
    """
    overlap = _get_overlap(patch, dim, winlen, step)
    spec, freqs, power, freq_dim = _get_stft_power(
        patch,
        winlen=winlen,
        overlap=overlap,
        dim=dim,
        fmin=fmin,
        fmax=fmax,
    )

    freq_axis = spec.dims.index(freq_dim)

    # Move frequency axis to front for easier cumulative-power calculation.
    power_f = np.moveaxis(power, freq_axis, 0)

    cumulative_power = np.cumsum(power_f, axis=0)
    total_power = cumulative_power[-1, ...]
    half_power = 0.5 * total_power

    # First frequency bin where cumulative power >= half total power.
    idx = np.argmax(cumulative_power >= half_power[None, ...], axis=0)

    out = freqs[idx]

    # No valid power -> NaN
    out = np.where(total_power > 0, out, np.nan)

    data_units = spec.get_coord(freq_dim).units
    data_type = "Median Frequency"
    return _prepare_output(spec, out, freq_dim, data_type, data_units)


@patch_function()
def max_frequency(
    patch: PatchType,
    winlen: float,
    dim: str = "time",
    step: float | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
) -> PatchType:
    """
    Compute the dominant (maximum-power) frequency from an STFT power spectrum.

    Parameters
    ----------
    patch
        Input DASCore patch.
    winlen
        Window length to calculate spectrum.
    step
        The window is evaluated at every step result, equivalent to slicing
        at every step. Setting ``step`` to larger than the original sampling interval
        significantly speeds up processing. If the ``step`` argument is not None, the
        result will have a different shape than the input.  Default None.
    dim
        Dimension along which to compute the spectrum.
    fmin
        Optional lower frequency limit.
    fmax
        Optional upper frequency limit.

    Returns
    -------
    PatchType
        Patch containing the frequency corresponding to the maximum
        spectral power.
    """
    overlap = _get_overlap(patch, dim, winlen, step)
    spec, freqs, power, freq_dim = _get_stft_power(
        patch, winlen, overlap, dim, fmin, fmax
    )

    freq_axis = spec.dims.index(freq_dim)

    power_f = np.moveaxis(power, freq_axis, 0)

    idx = np.argmax(power_f, axis=0)

    out = freqs[idx]

    data_units = spec.get_coord(freq_dim).units
    data_type = "Frequency at Maximum"
    return _prepare_output(spec, out, freq_dim, data_type, data_units)


@patch_function()
def spectral_entropy(
    patch: PatchType,
    winlen: float,
    dim: str = "time",
    step: float | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    normalize: bool = True,
) -> PatchType:
    """
    Compute spectral entropy from an STFT power spectrum.

    Spectral entropy measures the disorder of the spectral power
    distribution. Low values indicate concentrated spectral energy,
    while high values indicate broadband or noisy spectra.

    Parameters
    ----------
    patch
        Input DASCore patch.
    winlen
        Window length to calculate spectrum.
    step
        The window is evaluated at every step result, equivalent to slicing
        at every step. Setting ``step`` to larger than the original sampling interval
        significantly speeds up processing. If the ``step`` argument is not None, the
        result will have a different shape than the input.  Default None.
    dim
        Dimension along which to compute the spectrum.
    fmin
        Optional lower frequency limit.
    fmax
        Optional upper frequency limit.
    normalize
        If True, normalize entropy to [0, 1]. Defaults to True.

    Returns
    -------
    PatchType
        Patch containing spectral entropy.
    """
    overlap = _get_overlap(patch, dim, winlen, step)
    spec, freqs, power, freq_dim = _get_stft_power(
        patch, winlen, overlap, dim, fmin, fmax
    )

    freq_axis = spec.dims.index(freq_dim)

    total_power = np.sum(power, axis=freq_axis, keepdims=True)

    p = np.divide(
        power,
        total_power,
        out=np.zeros_like(power),
        where=total_power > 0,
    )

    entropy = -np.sum(
        p * np.log2(p, out=np.zeros_like(p), where=p > 0),
        axis=freq_axis,
    )

    if normalize:
        entropy /= np.log2(freqs.size)

    data_units = None
    data_type = "Spectral Entropy"
    return _prepare_output(spec, entropy, freq_dim, data_type, data_units)


@patch_function()
def spectral_kurtosis(
    patch: PatchType,
    winlen: float,
    dim: str = "time",
    step: float | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
) -> PatchType:
    """
    Compute spectral kurtosis from an STFT power spectrum.

    Spectral kurtosis measures the peakedness of the spectral power
    distribution.

    Large values indicate highly concentrated spectral energy.
    Lower values indicate broader spectral distributions.

    Parameters
    ----------
    patch
        Input DASCore patch.
    winlen
        Window length to calculate spectrum.
    step
        The window is evaluated at every step result, equivalent to slicing
        at every step. Setting ``step`` to larger than the original sampling interval
        significantly speeds up processing. If the ``step`` argument is not None, the
        result will have a different shape than the input.  Default None.
    dim
        Dimension along which to compute the spectrum.
    fmin
        Optional lower frequency limit.
    fmax
        Optional upper frequency limit.

    Returns
    -------
    PatchType
        Patch containing spectral kurtosis.
    """
    overlap = _get_overlap(patch, dim, winlen, step)
    spec, freqs, power, freq_dim = _get_stft_power(
        patch, winlen, overlap, dim, fmin, fmax
    )

    freq_axis = spec.dims.index(freq_dim)

    total_power = np.sum(power, axis=freq_axis, keepdims=True)

    p = np.divide(
        power,
        total_power,
        out=np.zeros_like(power),
        where=total_power > 0,
    )

    shape = [1] * power.ndim
    shape[freq_axis] = freqs.size

    f = freqs.reshape(shape)

    mean_f = np.sum(f * p, axis=freq_axis, keepdims=True)

    var_f = np.sum(
        ((f - mean_f) ** 2) * p,
        axis=freq_axis,
        keepdims=True,
    )

    kurt = np.divide(
        np.sum(
            ((f - mean_f) ** 4) * p,
            axis=freq_axis,
        ),
        np.squeeze(var_f, axis=freq_axis) ** 2,
        out=np.full_like(
            np.squeeze(var_f, axis=freq_axis),
            np.nan,
        ),
        where=np.squeeze(var_f, axis=freq_axis) > 0,
    )

    data_units = None
    data_type = "Spectral Kurtosis"
    return _prepare_output(spec, kurt, freq_dim, data_type, data_units)


@patch_function()
def spectral_flatness(
    patch: PatchType,
    winlen: float,
    dim: str = "time",
    step: float | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
) -> PatchType:
    """
    Compute spectral flatness from an STFT power spectrum.

    Spectral flatness is the ratio between the geometric mean and
    arithmetic mean of the power spectrum.

    Values near 1 indicate white-noise-like spectra.
    Values near 0 indicate tonal or peaked spectra.

    Parameters
    ----------
    patch
        Input DASCore patch.
    winlen
        Window length to calculate spectrum.
    step
        The window is evaluated at every step result, equivalent to slicing
        at every step. Setting ``step`` to larger than the original sampling interval
        significantly speeds up processing. If the ``step`` argument is not None, the
        result will have a different shape than the input.  Default None.
    dim
        Dimension along which to compute the spectrum.
    fmin
        Optional lower frequency limit.
    fmax
        Optional upper frequency limit.

    Returns
    -------
    PatchType
        Patch containing spectral flatness.
    """
    overlap = _get_overlap(patch, dim, winlen, step)
    spec, freqs, power, freq_dim = _get_stft_power(
        patch, winlen, overlap, dim, fmin, fmax
    )

    freq_axis = spec.dims.index(freq_dim)

    eps = np.finfo(float).eps

    geo_mean = np.exp(np.mean(np.log(power + eps), axis=freq_axis))

    arith_mean = np.mean(power, axis=freq_axis)

    flatness = np.divide(
        geo_mean,
        arith_mean,
        out=np.full_like(geo_mean, np.nan),
        where=arith_mean > 0,
    )

    data_units = None
    data_type = "Spectral Flatness"
    return _prepare_output(spec, flatness, freq_dim, data_type, data_units)
