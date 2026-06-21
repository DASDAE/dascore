"""Spectral descriptor transforms for DASCore patches."""

from __future__ import annotations

from typing import Literal

import numpy as np

import dascore as dc
from dascore.constants import PatchType
from dascore.units import Quantity
from dascore.utils.namespace import PatchNameSpace
from dascore.utils.patch import patch_function

SpectralFormat = Literal["auto", "fft", "amplitude", "power", "density"]
NegativeFrequencies = Literal["auto", "drop", "raise", "keep"]

_SPECTRAL_FORMAT_ALIASES = {
    "auto": "auto",
    "fft": "fft",
    "fourier": "fft",
    "fourier transform": "fft",
    "as": "amplitude",
    "amplitude": "amplitude",
    "amplitude spectrum": "amplitude",
    "ps": "power",
    "power": "power",
    "power spectrum": "power",
    "psd": "density",
    "density": "density",
    "spectral density": "density",
}
_DFT_OUTPUT_TO_FORMAT = {
    "FFT": "fft",
    "AS": "amplitude",
    "PS": "power",
    "PSD": "density",
}


def _get_frequency_dim(patch: PatchType, dim: str | None) -> str:
    """Return the Fourier frequency dimension to use."""
    ft_dims = tuple(x for x in patch.dims if x.startswith("ft_"))
    stft_dim = patch.attrs.get("_stft_frequency_dimension")
    if dim is None:
        if stft_dim in patch.dims:
            return stft_dim
        if len(ft_dims) == 1:
            return ft_dims[0]
        if not ft_dims:
            msg = (
                "Spectral descriptors require Fourier-domain input from "
                "Patch.dft or Patch.stft."
            )
            raise ValueError(msg)
        msg = (
            "Multiple Fourier dimensions found. Pass dim= with the original "
            "dimension name, such as 'time', or the Fourier dimension name, "
            "such as 'ft_time'."
        )
        raise ValueError(msg)

    freq_dim = dim if dim.startswith("ft_") else f"ft_{dim}"
    if freq_dim not in patch.dims:
        msg = f"Fourier dimension {freq_dim!r} was not found in patch dims."
        raise ValueError(msg)
    return freq_dim


def _normalize_spectral_format(
    patch: PatchType,
    spectral_format: SpectralFormat,
) -> str:
    """Determine how patch data should be converted to spectral power."""
    format_key = str(spectral_format).lower()
    if format_key not in _SPECTRAL_FORMAT_ALIASES:
        msg = (
            f"Unknown spectral_format={spectral_format!r}. Expected one of "
            "'auto', 'fft', 'amplitude', 'power', or 'density'."
        )
        raise ValueError(msg)
    out = _SPECTRAL_FORMAT_ALIASES[format_key]
    if out != "auto":
        return out

    dft_output = patch.attrs.get("_dft_output")
    if dft_output in _DFT_OUTPUT_TO_FORMAT:
        return _DFT_OUTPUT_TO_FORMAT[dft_output]
    if patch.attrs.get("_stft_performed", False):
        return "fft"

    data_type = patch.attrs.get("data_type")
    type_key = "" if data_type is None else str(data_type).lower()
    if type_key in _SPECTRAL_FORMAT_ALIASES:
        return _SPECTRAL_FORMAT_ALIASES[type_key]
    if np.iscomplexobj(patch.data):
        return "fft"

    msg = (
        "Could not infer spectral data representation from patch metadata. "
        "Pass spectral_format='amplitude', 'power', or 'density'."
    )
    raise ValueError(msg)


def _ensure_not_db_scaled(patch: PatchType) -> None:
    """Raise if the spectrum appears to be log-scaled."""
    units = str(patch.attrs.get("data_units", ""))
    if "dB" in units:
        msg = (
            "Spectral descriptors require linear spectra. Decibel-scaled DFT "
            "outputs cannot be converted back to power."
        )
        raise ValueError(msg)


def _get_power(patch: PatchType, spectral_format: SpectralFormat) -> np.ndarray:
    """Convert known Fourier representations to spectral power."""
    fmt = _normalize_spectral_format(patch, spectral_format)
    data = np.asarray(patch.data)
    _ensure_not_db_scaled(patch)

    if fmt == "fft":
        if not np.iscomplexobj(data):
            msg = (
                "spectral_format='fft' requires complex Fourier coefficients. "
                "For real-valued spectra, pass spectral_format='amplitude', "
                "'power', or 'density'."
            )
            raise ValueError(msg)
        return np.abs(data) ** 2

    if np.iscomplexobj(data):
        msg = f"spectral_format={fmt!r} requires real-valued spectral data."
        raise ValueError(msg)

    data = data.astype(float, copy=False)
    if fmt == "amplitude":
        if np.any(data < 0):
            msg = "Amplitude spectra must be non-negative."
            raise ValueError(msg)
        return data**2
    if fmt in {"power", "density"}:
        if np.any(data < 0):
            msg = "Power spectra and spectral densities must be non-negative."
            raise ValueError(msg)
        return data

    raise AssertionError(f"Unhandled spectral format {fmt!r}.")


def _power_has_symmetric_frequencies(
    freqs: np.ndarray,
    power: np.ndarray,
    freq_axis: int,
) -> bool:
    """Return True if negative-frequency power mirrors positive power."""
    neg_inds = np.flatnonzero(freqs < 0)
    if not len(neg_inds):
        return True
    spacing = np.min(np.abs(np.diff(freqs))) if len(freqs) > 1 else 0
    atol = max(float(spacing) * 1e-6, np.finfo(float).eps)

    for neg_ind in neg_inds:
        pos_inds = np.flatnonzero(np.isclose(freqs, -freqs[neg_ind], atol=atol))
        if not len(pos_inds):
            continue
        neg_power = np.take(power, neg_ind, axis=freq_axis)
        pos_power = np.take(power, pos_inds[0], axis=freq_axis)
        if not np.allclose(neg_power, pos_power):
            return False
    return True


def _get_spectral_power(
    patch: PatchType,
    dim: str | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    spectral_format: SpectralFormat = "auto",
    negative_frequencies: NegativeFrequencies = "auto",
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Return frequency coordinates and spectral power from DFT or STFT input.

    The returned power is always linear, non-negative, and trimmed according
    to ``negative_frequencies``, ``fmin``, and ``fmax``.
    """
    if negative_frequencies not in {"auto", "drop", "raise", "keep"}:
        msg = (
            "negative_frequencies must be one of 'auto', 'drop', 'raise', " "or 'keep'."
        )
        raise ValueError(msg)

    freq_dim = _get_frequency_dim(patch, dim)
    freq_axis = patch.dims.index(freq_dim)
    freqs = np.asarray(patch.get_array(freq_dim), dtype=float)
    power = _get_power(patch, spectral_format)

    mask = np.ones(freqs.shape, dtype=bool)
    has_negative = np.any(freqs < 0)
    if negative_frequencies == "raise" and has_negative:
        msg = (
            "Fourier coordinate contains negative frequencies. Use "
            "negative_frequencies='auto', 'drop', or 'keep' to choose how to "
            "handle them."
        )
        raise ValueError(msg)
    if negative_frequencies == "auto" and has_negative:
        if not _power_has_symmetric_frequencies(freqs, power, freq_axis):
            msg = (
                "Fourier coordinate contains negative frequencies with "
                "non-symmetric power. Pass negative_frequencies='drop' to use "
                "only non-negative bins, or 'keep' to include all bins."
            )
            raise ValueError(msg)
        mask &= freqs >= 0
    if negative_frequencies == "drop":
        mask &= freqs >= 0
    if fmin is not None:
        mask &= freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax

    if not np.any(mask):
        raise ValueError("Frequency limits exclude all Fourier frequency bins.")

    freqs = freqs[mask]
    power = np.take(power, np.flatnonzero(mask), axis=freq_axis)

    return freqs, power, freq_dim


def _prepare_output(
    patch: PatchType,
    data: np.ndarray,
    freq_dim: str,
    data_type: str,
    data_units: str | Quantity | None,
) -> PatchType:
    """Return a patch with the frequency dimension removed."""
    dims = tuple(dim for dim in patch.dims if dim != freq_dim)
    coords = {dim: patch.get_array(dim) for dim in dims}
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
def spectral_centroid(
    patch: PatchType,
    dim: str | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    spectral_format: SpectralFormat = "auto",
    negative_frequencies: NegativeFrequencies = "auto",
) -> PatchType:
    """
    Compute the spectral centroid of a Fourier-domain patch.

    This represents the center of gravity of the signal's power spectrum, and
    is sometimes called the mean frequency. The input patch must already be
    transformed with [Patch.dft](`dascore.Patch.dft`) or
    [Patch.stft](`dascore.Patch.stft`). STFT inputs produce rolling descriptors;
    DFT inputs produce descriptors over the remaining non-frequency dimensions.

    See for more detail:
        - @Phinyomark12
        - [Online version of the Phinyomark publication](
            https://www.intechopen.com/chapters/40123)
        - [Matlab's meanfreq](https://se.mathworks.com/help/signal/ref/meanfreq.html)

    Parameters
    ----------
    patch
        Fourier-domain DASCore patch from ``dft`` or ``stft``.
    dim
        Frequency dimension over which to compute the descriptor. This can be
        either the original dimension name, such as ``"time"``, or the Fourier
        dimension name, such as ``"ft_time"``. If omitted, a single Fourier
        dimension is inferred.
    fmin
        Optional lower frequency bound.
    fmax
        Optional upper frequency bound.
    spectral_format
        Representation of the spectral data. ``"auto"`` uses DASCore DFT/STFT
        metadata when available. Other options are ``"fft"`` for complex Fourier
        coefficients, ``"amplitude"`` for amplitude spectra, ``"power"`` for
        power spectra, and ``"density"`` for power spectral densities.
    negative_frequencies
        How to handle negative frequency bins. ``"auto"`` drops negative bins
        only when power is symmetric, ``"drop"`` always uses non-negative
        frequencies, ``"raise"`` rejects spectra with negative bins, and
        ``"keep"`` includes them in the calculation.

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
    >>> spec = patch.stft(time=.02, overlap=.019, taper_window="boxcar")
    >>> centroid = spec.spectral.spectral_centroid(fmin=50, fmax=300)
    >>> ax = centroid.viz.waterfall(cmap='turbo', ax=axs[1])

    """
    freqs, power, freq_dim = _get_spectral_power(
        patch,
        dim=dim,
        fmin=fmin,
        fmax=fmax,
        spectral_format=spectral_format,
        negative_frequencies=negative_frequencies,
    )

    freq_axis = patch.dims.index(freq_dim)
    freqs_b = _broadcast_freqs(freqs, power.ndim, freq_axis)

    numerator = np.sum(freqs_b * power, axis=freq_axis)
    denominator = np.sum(power, axis=freq_axis)

    out = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )
    data_units = patch.get_coord(freq_dim).units
    data_type = "Spectral Centroid"
    return _prepare_output(patch, out, freq_dim, data_type, data_units)


@patch_function()
def median_frequency(
    patch: PatchType,
    dim: str | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    spectral_format: SpectralFormat = "auto",
    negative_frequencies: NegativeFrequencies = "auto",
) -> PatchType:
    """
    Compute the median frequency of a Fourier-domain patch.

    This measure divides a signal's power spectrum into two regions of equal
    total power. The input patch must already be transformed with
    [Patch.dft](`dascore.Patch.dft`) or [Patch.stft](`dascore.Patch.stft`).

    See for more detail:
        - @Phinyomark12
        - [Online version of the Phinyomark publication](
            https://www.intechopen.com/chapters/40123)
        - [Matlab's medfreq](https://se.mathworks.com/help/signal/ref/medfreq.html)

    Parameters
    ----------
    patch
        Fourier-domain DASCore patch from ``dft`` or ``stft``.
    dim
        Frequency dimension over which to compute the descriptor. This can be
        either the original dimension name or the Fourier dimension name.
    fmin
        Optional lower frequency bound.
    fmax
        Optional upper frequency bound.
    spectral_format
        Representation of the spectral data: ``"auto"``, ``"fft"``,
        ``"amplitude"``, ``"power"``, or ``"density"``.
    negative_frequencies
        How to handle negative frequency bins: ``"auto"``, ``"drop"``,
        ``"raise"``, or ``"keep"``.

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
    >>> spec = patch.stft(time=.02, overlap=.019, taper_window="boxcar")
    >>> med = spec.spectral.median_frequency(fmin=50, fmax=300)
    >>> ax = med.viz.waterfall(cmap='turbo', ax=axs[1], scale=[0,1])
    """
    freqs, power, freq_dim = _get_spectral_power(
        patch,
        dim=dim,
        fmin=fmin,
        fmax=fmax,
        spectral_format=spectral_format,
        negative_frequencies=negative_frequencies,
    )

    freq_axis = patch.dims.index(freq_dim)

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

    data_units = patch.get_coord(freq_dim).units
    data_type = "Median Frequency"
    return _prepare_output(patch, out, freq_dim, data_type, data_units)


@patch_function()
def max_frequency(
    patch: PatchType,
    dim: str | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    spectral_format: SpectralFormat = "auto",
    negative_frequencies: NegativeFrequencies = "auto",
) -> PatchType:
    """
    Compute the dominant frequency of a Fourier-domain patch.

    The dominant frequency is the frequency bin with maximum spectral power.
    The input patch must already be transformed with
    [Patch.dft](`dascore.Patch.dft`) or [Patch.stft](`dascore.Patch.stft`).

    Parameters
    ----------
    patch
        Fourier-domain DASCore patch from ``dft`` or ``stft``.
    dim
        Frequency dimension over which to compute the descriptor. This can be
        either the original dimension name or the Fourier dimension name.
    fmin
        Optional lower frequency limit.
    fmax
        Optional upper frequency limit.
    spectral_format
        Representation of the spectral data: ``"auto"``, ``"fft"``,
        ``"amplitude"``, ``"power"``, or ``"density"``.
    negative_frequencies
        How to handle negative frequency bins: ``"auto"``, ``"drop"``,
        ``"raise"``, or ``"keep"``.

    Returns
    -------
    PatchType
        Patch containing the frequency corresponding to the maximum
        spectral power.
    """
    freqs, power, freq_dim = _get_spectral_power(
        patch, dim, fmin, fmax, spectral_format, negative_frequencies
    )

    freq_axis = patch.dims.index(freq_dim)

    power_f = np.moveaxis(power, freq_axis, 0)

    idx = np.argmax(power_f, axis=0)

    out = freqs[idx]

    data_units = patch.get_coord(freq_dim).units
    data_type = "Frequency at Maximum"
    return _prepare_output(patch, out, freq_dim, data_type, data_units)


@patch_function()
def spectral_max(
    patch: PatchType,
    dim: str | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    spectral_format: SpectralFormat = "auto",
    negative_frequencies: NegativeFrequencies = "auto",
) -> PatchType:
    """
    Compute the maximum spectral amplitude of a Fourier-domain patch.

    The input patch must already be transformed with
    [Patch.dft](`dascore.Patch.dft`) or [Patch.stft](`dascore.Patch.stft`).
    The descriptor returns the largest linear spectral amplitude along the
    requested Fourier dimension.

    Parameters
    ----------
    patch
        Fourier-domain DASCore patch from ``dft`` or ``stft``.
    dim
        Frequency dimension over which to compute the descriptor. This can be
        either the original dimension name or the Fourier dimension name.
    fmin
        Optional lower frequency limit.
    fmax
        Optional upper frequency limit.
    spectral_format
        Representation of the spectral data: ``"auto"``, ``"fft"``,
        ``"amplitude"``, ``"power"``, or ``"density"``.
    negative_frequencies
        How to handle negative frequency bins: ``"auto"``, ``"drop"``,
        ``"raise"``, or ``"keep"``.

    Returns
    -------
    PatchType
        Patch containing the maximum spectral amplitude.
    """
    _freqs, power, freq_dim = _get_spectral_power(
        patch, dim, fmin, fmax, spectral_format, negative_frequencies
    )

    freq_axis = patch.dims.index(freq_dim)
    amplitude = np.sqrt(power)
    out = np.max(amplitude, axis=freq_axis)

    data_units = patch.attrs.get("data_units")
    data_type = "Maximum Spectral Amplitude"
    return _prepare_output(patch, out, freq_dim, data_type, data_units)


@patch_function()
def spectral_entropy(
    patch: PatchType,
    dim: str | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    normalize: bool = True,
    spectral_format: SpectralFormat = "auto",
    negative_frequencies: NegativeFrequencies = "auto",
) -> PatchType:
    """
    Compute spectral entropy from a Fourier-domain patch.

    Spectral entropy measures the disorder of the spectral power
    distribution. Low values indicate concentrated spectral energy,
    while high values indicate broadband or noisy spectra. The input patch
    must already be transformed with [Patch.dft](`dascore.Patch.dft`) or
    [Patch.stft](`dascore.Patch.stft`).

    Parameters
    ----------
    patch
        Fourier-domain DASCore patch from ``dft`` or ``stft``.
    dim
        Frequency dimension over which to compute the descriptor. This can be
        either the original dimension name or the Fourier dimension name.
    fmin
        Optional lower frequency limit.
    fmax
        Optional upper frequency limit.
    normalize
        If True, normalize entropy to [0, 1]. Defaults to True.
    spectral_format
        Representation of the spectral data: ``"auto"``, ``"fft"``,
        ``"amplitude"``, ``"power"``, or ``"density"``.
    negative_frequencies
        How to handle negative frequency bins: ``"auto"``, ``"drop"``,
        ``"raise"``, or ``"keep"``.

    Returns
    -------
    PatchType
        Patch containing spectral entropy.
    """
    freqs, power, freq_dim = _get_spectral_power(
        patch, dim, fmin, fmax, spectral_format, negative_frequencies
    )

    freq_axis = patch.dims.index(freq_dim)

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
    return _prepare_output(patch, entropy, freq_dim, data_type, data_units)


@patch_function()
def spectral_kurtosis(
    patch: PatchType,
    dim: str | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    spectral_format: SpectralFormat = "auto",
    negative_frequencies: NegativeFrequencies = "auto",
) -> PatchType:
    """
    Compute spectral kurtosis from a Fourier-domain patch.

    Spectral kurtosis measures the peakedness of the spectral power
    distribution.

    Large values indicate highly concentrated spectral energy.
    Lower values indicate broader spectral distributions. The input patch
    must already be transformed with [Patch.dft](`dascore.Patch.dft`) or
    [Patch.stft](`dascore.Patch.stft`).

    Parameters
    ----------
    patch
        Fourier-domain DASCore patch from ``dft`` or ``stft``.
    dim
        Frequency dimension over which to compute the descriptor. This can be
        either the original dimension name or the Fourier dimension name.
    fmin
        Optional lower frequency limit.
    fmax
        Optional upper frequency limit.
    spectral_format
        Representation of the spectral data: ``"auto"``, ``"fft"``,
        ``"amplitude"``, ``"power"``, or ``"density"``.
    negative_frequencies
        How to handle negative frequency bins: ``"auto"``, ``"drop"``,
        ``"raise"``, or ``"keep"``.

    Returns
    -------
    PatchType
        Patch containing spectral kurtosis.
    """
    freqs, power, freq_dim = _get_spectral_power(
        patch, dim, fmin, fmax, spectral_format, negative_frequencies
    )

    freq_axis = patch.dims.index(freq_dim)

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
    return _prepare_output(patch, kurt, freq_dim, data_type, data_units)


@patch_function()
def spectral_flatness(
    patch: PatchType,
    dim: str | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    spectral_format: SpectralFormat = "auto",
    negative_frequencies: NegativeFrequencies = "auto",
) -> PatchType:
    """
    Compute spectral flatness from a Fourier-domain patch.

    Spectral flatness is the ratio between the geometric mean and
    arithmetic mean of the power spectrum.

    Values near 1 indicate white-noise-like spectra.
    Values near 0 indicate tonal or peaked spectra. The input patch must
    already be transformed with [Patch.dft](`dascore.Patch.dft`) or
    [Patch.stft](`dascore.Patch.stft`).

    Parameters
    ----------
    patch
        Fourier-domain DASCore patch from ``dft`` or ``stft``.
    dim
        Frequency dimension over which to compute the descriptor. This can be
        either the original dimension name or the Fourier dimension name.
    fmin
        Optional lower frequency limit.
    fmax
        Optional upper frequency limit.
    spectral_format
        Representation of the spectral data: ``"auto"``, ``"fft"``,
        ``"amplitude"``, ``"power"``, or ``"density"``.
    negative_frequencies
        How to handle negative frequency bins: ``"auto"``, ``"drop"``,
        ``"raise"``, or ``"keep"``.

    Returns
    -------
    PatchType
        Patch containing spectral flatness.
    """
    _freqs, power, freq_dim = _get_spectral_power(
        patch, dim, fmin, fmax, spectral_format, negative_frequencies
    )

    freq_axis = patch.dims.index(freq_dim)

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
    return _prepare_output(patch, flatness, freq_dim, data_type, data_units)


class SpectralPatchNameSpace(PatchNameSpace):
    """A class for storing spectral descriptor methods."""

    name = "spectral"

    spectral_centroid = spectral_centroid
    median_frequency = median_frequency
    max_frequency = max_frequency
    spectral_max = spectral_max
    spectral_entropy = spectral_entropy
    spectral_kurtosis = spectral_kurtosis
    spectral_flatness = spectral_flatness
