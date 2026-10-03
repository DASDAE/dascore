"""
Spectral descriptor transforms for DASCore patches.

Outputs preserve input metadata and processing history, replace data type and
units, and remove obsolete DFT/STFT attributes and frequency-dependent coordinates.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np

import dascore as dc
from dascore.constants import PatchType
from dascore.core.processor import PatchProcessor
from dascore.transform.fourier import DFT_OUTPUT_DATA_TYPE_MAP, _is_complex
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import broadcast_for_index

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
# What a plan says the data are, by position: a plan holds no strings.
_FORMATS = ("fft", "amplitude", "power", "density")
_DFT_OUTPUT_TO_FORMAT = {
    "FFT": "fft",
    "AS": "amplitude",
    "PS": "power",
    "PSD": "density",
}
# The data_type values dft assigns, eg "power_spectral_density".
_DATA_TYPE_TO_FORMAT = {
    name: _DFT_OUTPUT_TO_FORMAT[output]
    for output, name in DFT_OUTPUT_DATA_TYPE_MAP.items()
}


_SPECTRAL_PARAMETER_DOCS = {
    "dim": """
    dim
        Frequency dimension over which to compute the descriptor. This can be
        either the original dimension name, such as ``"time"``, or the Fourier
        dimension name, such as ``"ft_time"``. If omitted, a single Fourier
        dimension is inferred.
    """,
    "fmin": """
    fmin
        Optional lower frequency bound.
    """,
    "fmax": """
    fmax
        Optional upper frequency bound.
    """,
    "spectral_format": """
    spectral_format
        Representation of the spectral data. ``"auto"`` uses DASCore DFT/STFT
        metadata when available. Other options are ``"fft"`` for complex Fourier
        coefficients, ``"amplitude"`` for amplitude spectra, ``"power"`` for
        power spectra, and ``"density"`` for power spectral densities.
    """,
    "negative_frequencies": """
    negative_frequencies
        How to handle negative frequency bins. ``"auto"`` drops negative bins
        when power is symmetric and raises otherwise, folding a verified
        negative Nyquist bin to positive frequency without changing its
        power. ``"drop"`` always uses
        non-negative frequencies, ``"raise"`` rejects spectra with negative bins, and
        ``"keep"`` includes them in the calculation.
    """,
}


def _get_frequency_dim(patch: PatchType, dim: str | None) -> str:
    """Return the Fourier frequency dimension to use."""
    ft_dims = tuple(x for x in patch.dims if x.startswith("ft_"))
    if dim is None:
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
    if "_stft_real" in dict(patch.attrs):
        return "fft"

    data_type = patch.attrs.get("data_type")
    if data_type in _DATA_TYPE_TO_FORMAT:
        return _DATA_TYPE_TO_FORMAT[data_type]
    type_key = "" if data_type is None else str(data_type).lower()
    if type_key in _SPECTRAL_FORMAT_ALIASES:
        return _SPECTRAL_FORMAT_ALIASES[type_key]
    if _is_complex(patch.dtype):
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


def _check_spectral_dtype(fmt: str, dtype) -> None:
    """Raise if the data are not the kind of numbers a format holds."""
    complex_data = _is_complex(dtype)
    if fmt == "fft" and not complex_data:
        msg = (
            "spectral_format='fft' requires complex Fourier coefficients. "
            "For real-valued spectra, pass spectral_format='amplitude', "
            "'power', or 'density'."
        )
        raise ValueError(msg)
    if fmt != "fft" and complex_data:
        msg = f"spectral_format={fmt!r} requires real-valued spectral data."
        raise ValueError(msg)


def _get_power(data: np.ndarray, fmt: str) -> np.ndarray:
    """Convert known Fourier representations to spectral power."""
    data = np.asarray(data)
    if fmt == "fft":
        return np.abs(data) ** 2

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


def _select_frequencies(
    patch: PatchType,
    freq_dim: str,
    fmin: float | None,
    fmax: float | None,
    negative_frequencies: NegativeFrequencies,
) -> dict[str, Any]:
    """
    Return which frequency bins a descriptor reads, from metadata alone.

    The bins kept according to ``negative_frequencies``, ``fmin``, and
    ``fmax``: their frequencies, the order to read the bins in, which of
    them to keep, and, where negative bins are dropped only if the power
    mirrors, the frequencies to check that against.
    """
    freqs = np.asarray(patch.coords.get_array(freq_dim), dtype=float)
    plan: dict[str, Any] = {"symmetric": None, "order": None}
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
        plan["symmetric"] = freqs
        # A full even-length DFT stores its unpaired Nyquist bin at -Nyquist.
        # Retain its existing weight, matching DASCore's real DFT outputs.
        original_dim = freq_dim.removeprefix("ft_")
        original_coord = patch.coords.coord_map.get(original_dim)
        size = (
            len(original_coord)
            if patch.attrs.get("_dft_output") and original_coord is not None
            else patch.attrs.get(f"_stft_mfft_{original_dim}", 0)
        )
        if size >= 2 and size % 2 == 0 and len(freqs) == size:
            spacing = freqs[1] - freqs[0]
            expected = np.arange(-size // 2, size // 2) * spacing
            if np.allclose(freqs, expected, rtol=1e-7, atol=abs(spacing) * 1e-7):
                freqs = freqs.copy()
                freqs[0] = -freqs[0]
                plan["order"] = np.argsort(freqs)
                freqs = freqs[plan["order"]]
        mask &= freqs >= 0
    if negative_frequencies == "drop":
        mask &= freqs >= 0
    if fmin is not None:
        mask &= freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax

    if not np.any(mask):
        raise ValueError("Frequency limits exclude all Fourier frequency bins.")
    return plan | {"keep": np.flatnonzero(mask), "freqs": freqs[mask]}


def _broadcast_freqs(
    freqs: np.ndarray,
    ndim: int,
    freq_axis: int,
) -> np.ndarray:
    """Broadcast a frequency vector against a spectral power array."""
    return freqs[broadcast_for_index(ndim, freq_axis, slice(None), fill=None)]


class _SpectralDescriptor(PatchProcessor):
    """
    What the spectral descriptors share: the power along a Fourier dimension.

    `get_metadata` settles the frequency bins and the output's metadata;
    the kernel turns the data into linear, non-negative power over those
    bins, and `describe` reduces it along the frequency axis.
    """

    name = None

    dim: Any = None
    fmin: Any = None
    fmax: Any = None

    # The output's data_type.
    label: ClassVar[str] = ""

    def get_metadata(self, meta):
        """Return the reduced metadata, and the bins and format to read."""
        # Declared by each subclass, after any option of its own.
        options = self.kwargs
        negative_frequencies = options["negative_frequencies"]
        if negative_frequencies not in {"auto", "drop", "raise", "keep"}:
            msg = (
                "negative_frequencies must be one of 'auto', 'drop', 'raise', "
                "or 'keep'."
            )
            raise ValueError(msg)
        freq_dim = _get_frequency_dim(meta, self.dim)
        fmt = _normalize_spectral_format(meta, options["spectral_format"])
        _ensure_not_db_scaled(meta)
        _check_spectral_dtype(fmt, meta.dtype)
        plan = _select_frequencies(
            meta, freq_dim, self.fmin, self.fmax, negative_frequencies
        )
        # Descriptor data is already reduced, so only drop coordinates here.
        coords, _ = meta.coords.drop_coords(freq_dim)
        # Reduction no longer represents Fourier coefficients, even when other
        # Fourier dimensions remain. Preserve unrelated (including private) attrs.
        obsolete = tuple(
            key
            for key in dict(meta.attrs)
            if key.startswith(("_dft_", "_stft_", "_pre_dft_", "_pre_stft_"))
        )
        units = self.units(meta, freq_dim, fmt)
        attrs = meta.attrs.drop(*obsolete).update(
            data_type=self.label, data_units=units
        )
        out = meta.new(coords=coords, attrs=attrs)
        axis = meta.dims.index(freq_dim)
        return out, plan | {"axis": axis, "fmt": _FORMATS.index(fmt)}

    def units(self, meta, freq_dim: str, fmt: str):
        """Return the output's data units: the frequencies' by default."""
        return meta.get_coord(freq_dim).units

    def numpy_kernel(self, data, *, axis, fmt, symmetric, order, keep, freqs):
        """Return the descriptor of the power in the bins kept."""
        power = _get_power(data, _FORMATS[fmt])
        if symmetric is not None and not _power_has_symmetric_frequencies(
            symmetric, power, axis
        ):
            msg = (
                "Fourier coordinate contains negative frequencies with "
                "non-symmetric power. Pass negative_frequencies='drop' to use "
                "only non-negative bins, or 'keep' to include all bins."
            )
            raise ValueError(msg)
        if order is not None:
            power = np.take(power, order, axis=axis)
        power = np.take(power, keep, axis=axis)
        return self.describe(power, freqs, axis)

    def describe(self, power: np.ndarray, freqs: np.ndarray, axis: int) -> Any:
        """Return the descriptor of linear power along an axis; each class's own."""


class _SharedFields(_SpectralDescriptor):
    """The descriptors whose options follow the frequency limits directly."""

    name = None

    spectral_format: Any = "auto"
    negative_frequencies: Any = "auto"


@compose_docstring(**_SPECTRAL_PARAMETER_DOCS)
class SpectralCentroid(_SharedFields):
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
    {dim}
    {fmin}
    {fmax}
    {spectral_format}
    {negative_frequencies}

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
    >>> centroid = spec.spectral_centroid(fmin=50, fmax=300)
    >>> ax = centroid.viz.waterfall(cmap='turbo', ax=axs[1])

    """

    label = "Spectral Centroid"

    def describe(self, power, freqs, axis):
        """Return the power-weighted mean frequency."""
        freqs_b = _broadcast_freqs(freqs, power.ndim, axis)

        numerator = np.sum(freqs_b * power, axis=axis)
        denominator = np.sum(power, axis=axis)

        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=float),
            where=denominator > 0,
        )


@compose_docstring(**_SPECTRAL_PARAMETER_DOCS)
class MedianFrequency(_SharedFields):
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
    {dim}
    {fmin}
    {fmax}
    {spectral_format}
    {negative_frequencies}

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
    >>> med = spec.median_frequency(fmin=50, fmax=300)
    >>> ax = med.viz.waterfall(cmap='turbo', ax=axs[1], scale=[0,1])
    """

    label = "Median Frequency"

    def describe(self, power, freqs, axis):
        """Return the frequency which halves the power."""
        # Move frequency axis to front for easier cumulative-power calculation.
        power_f = np.moveaxis(power, axis, 0)

        cumulative_power = np.cumsum(power_f, axis=0)
        total_power = cumulative_power[-1, ...]
        half_power = 0.5 * total_power

        # First frequency bin where cumulative power >= half total power.
        idx = np.argmax(cumulative_power >= half_power[None, ...], axis=0)

        out = freqs[idx]

        # No valid power -> NaN
        return np.where(total_power > 0, out, np.nan)


@compose_docstring(**_SPECTRAL_PARAMETER_DOCS)
class SpectralPeakFrequency(_SharedFields):
    """
    Compute the peak frequency of a Fourier-domain patch.

    The peak frequency is the frequency bin with maximum spectral power
    within the selected frequency range. This is the global maximum, not
    a search for multiple local peaks. Ties select the first bin in coordinate
    order.
    The input patch must already be transformed with
    [Patch.dft](`dascore.Patch.dft`) or [Patch.stft](`dascore.Patch.stft`).

    Parameters
    ----------
    {dim}
    {fmin}
    {fmax}
    {spectral_format}
    {negative_frequencies}

    Returns
    -------
    PatchType
        Patch containing the frequency corresponding to the maximum
        spectral power.
    """

    label = "Frequency at Maximum"

    def describe(self, power, freqs, axis):
        """Return the frequency of the largest power."""
        power_f = np.moveaxis(power, axis, 0)

        idx = np.argmax(power_f, axis=0)

        out = freqs[idx]
        return np.where(np.sum(power_f, axis=0) > 0, out, np.nan)


@compose_docstring(**_SPECTRAL_PARAMETER_DOCS)
class SpectralPeakAmplitude(_SharedFields):
    """
    Compute the peak spectral amplitude of a Fourier-domain patch.

    The input patch must already be transformed with
    [Patch.dft](`dascore.Patch.dft`) or [Patch.stft](`dascore.Patch.stft`).
    The descriptor returns the largest linear spectral amplitude along the
    requested Fourier dimension within the selected frequency range. This is
    the global maximum, not a search for multiple local peaks.

    Parameters
    ----------
    {dim}
    {fmin}
    {fmax}
    {spectral_format}
    {negative_frequencies}

    Returns
    -------
    PatchType
        Patch containing the maximum spectral amplitude.
    """

    label = "Maximum Spectral Amplitude"

    def units(self, meta, freq_dim, fmt):
        """Return the amplitude's units: a power's square root."""
        data_units = meta.attrs.get("data_units")
        quantity = dc.get_quantity(data_units)
        if fmt in {"power", "density"} and quantity is not None:
            data_units = quantity**0.5
        return data_units

    def describe(self, power, freqs, axis):
        """Return the largest amplitude."""
        amplitude = np.sqrt(power)
        return np.max(amplitude, axis=axis)


@compose_docstring(**_SPECTRAL_PARAMETER_DOCS)
class SpectralEntropy(_SpectralDescriptor):
    """
    Compute spectral entropy from a Fourier-domain patch.

    Spectral entropy measures the disorder of the spectral power
    distribution. Low values indicate concentrated spectral energy,
    while high values indicate broadband or noisy spectra. The input patch
    must already be transformed with [Patch.dft](`dascore.Patch.dft`) or
    [Patch.stft](`dascore.Patch.stft`).

    Parameters
    ----------
    {dim}
    {fmin}
    {fmax}
    normalize
        If True, normalize entropy to [0, 1]. Defaults to True.
    {spectral_format}
    {negative_frequencies}

    Returns
    -------
    PatchType
        Patch containing spectral entropy.
    """

    normalize: Any = True
    spectral_format: Any = "auto"
    negative_frequencies: Any = "auto"

    label = "Spectral Entropy"

    def units(self, meta, freq_dim, fmt):
        """Return no units: entropy is a number of bits."""
        return None

    def describe(self, power, freqs, axis):
        """Return the entropy of the power distribution."""
        total_power = np.sum(power, axis=axis, keepdims=True)

        p = np.divide(
            power,
            total_power,
            out=np.zeros_like(power),
            where=total_power > 0,
        )

        entropy = -np.sum(
            p * np.log2(p, out=np.zeros_like(p), where=p > 0),
            axis=axis,
        )

        if self.normalize:
            entropy = (
                np.zeros_like(entropy)
                if freqs.size == 1
                else entropy / np.log2(freqs.size)
            )
        return entropy


@compose_docstring(**_SPECTRAL_PARAMETER_DOCS)
class SpectralKurtosis(_SharedFields):
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
    {dim}
    {fmin}
    {fmax}
    {spectral_format}
    {negative_frequencies}

    Returns
    -------
    PatchType
        Patch containing spectral kurtosis.
    """

    label = "Spectral Kurtosis"

    def units(self, meta, freq_dim, fmt):
        """Return no units: kurtosis is a ratio."""
        return None

    def describe(self, power, freqs, axis):
        """Return the kurtosis of the power distribution over frequency."""
        total_power = np.sum(power, axis=axis, keepdims=True)

        p = np.divide(
            power,
            total_power,
            out=np.zeros_like(power),
            where=total_power > 0,
        )

        f = _broadcast_freqs(freqs, power.ndim, axis)

        mean_f = np.sum(f * p, axis=axis, keepdims=True)

        var_f = np.sum(
            ((f - mean_f) ** 2) * p,
            axis=axis,
            keepdims=True,
        )

        return np.divide(
            np.sum(
                ((f - mean_f) ** 4) * p,
                axis=axis,
            ),
            np.squeeze(var_f, axis=axis) ** 2,
            out=np.full_like(
                np.squeeze(var_f, axis=axis),
                np.nan,
            ),
            where=np.squeeze(var_f, axis=axis) > 0,
        )


@compose_docstring(**_SPECTRAL_PARAMETER_DOCS)
class SpectralFlatness(_SharedFields):
    """
    Compute spectral flatness from a Fourier-domain patch.

    Spectral flatness is the ratio between the geometric mean and
    arithmetic mean of the power spectrum. It is invariant to positive
    scaling of the power and lies in [0, 1] for finite, nonzero spectra.
    A zero-power bin gives flatness 0; an entirely zero spectrum gives NaN.

    Values near 1 indicate white-noise-like spectra.
    Values near 0 indicate tonal or peaked spectra. The input patch must
    already be transformed with [Patch.dft](`dascore.Patch.dft`) or
    [Patch.stft](`dascore.Patch.stft`).

    Parameters
    ----------
    {dim}
    {fmin}
    {fmax}
    {spectral_format}
    {negative_frequencies}

    Returns
    -------
    PatchType
        Patch containing spectral flatness.
    """

    label = "Spectral Flatness"

    def units(self, meta, freq_dim, fmt):
        """Return no units: flatness is a ratio."""
        return None

    def describe(self, power, freqs, axis):
        """Return the geometric over the arithmetic mean of the power."""
        # Normalize each slice to preserve spectral shape at any power scale.
        peak_power = np.max(power, axis=axis, keepdims=True)
        power = np.divide(
            power, peak_power, out=np.zeros_like(power), where=peak_power > 0
        )
        log_power = np.log(power, out=np.full_like(power, -np.inf), where=power > 0)
        geo_mean = np.exp(np.mean(log_power, axis=axis))

        arith_mean = np.mean(power, axis=axis)

        flatness = np.divide(
            geo_mean,
            arith_mean,
            out=np.full_like(geo_mean, np.nan),
            where=arith_mean > 0,
        )
        return np.clip(flatness, 0, 1)
