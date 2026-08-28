"""
Adaptive spectral filtering for DASCore patches.

The filter walks a patch in overlapping windows along one or two dimensions.
Each window is transformed to the spectral domain, every coefficient is
weighted by a power of its own magnitude, and the window is transformed back
and added into the output under a tapered overlap-add. Energy which is
coherent within a window concentrates in a few large coefficients, which the
weighting keeps; energy spread across the spectrum is suppressed.

Over two dimensions this is the adaptive frequency-wavenumber filter of
@isken2022denoising as implemented by Pyrocko
[Lightguide](https://github.com/pyrocko/lightguide), which the SciPy engine
here matches to floating-point precision. Over one dimension it is the same
weighting applied to each trace on its own.

The public patch function converts dimension names and coordinate units to
axes and sample counts and batches over every dimension not selected. The
engines work on raw one- or two-dimensional arrays; the optional
Numba/rocket-fft engine in `_adaptive_spectral_filter_numba` handles the
two-dimensional case and shares this module's validation, padding, and taper
so the two produce the same output.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import numpy as np
from pydantic import ConfigDict
from scipy import fft as sp_fft

from dascore.constants import PatchType
from dascore.exceptions import MissingOptionalDependencyError, ParameterError
from dascore.utils.patch import patch_function
from dascore.utils.signal import get_taper
from dascore.utils.tiles import TilePlan, get_tile_plan
from dascore.utils.window import Window, resolve_window
from dascore.workflow.meta import PatchMeta
from dascore.workflow.processor import PatchProcessor, register_implementation

_AdaptiveSpectralEngine = Literal["auto", "numba", "scipy"]
__all__ = ("AdaptiveSpectralFilter", "adaptive_spectral_filter")


def _check_window(window: Any, overlap: Any, label: str) -> None:
    """Raise ValueError unless a window and its overlap can tile an axis."""
    if not isinstance(window, int | np.integer):
        msg = f"window for {label} must be an integer; got {window!r}."
        raise ValueError(msg)
    if not isinstance(overlap, int | np.integer):
        msg = f"overlap for {label} must be an integer; got {overlap!r}."
        raise ValueError(msg)
    if window < 2:
        msg = f"window for {label} must be at least 2 samples; got {window!r}."
        raise ValueError(msg)
    if overlap < 0:
        msg = f"overlap for {label} must be non-negative; got {overlap!r}."
        raise ValueError(msg)
    if overlap >= window / 2:
        msg = f"overlap for {label} is too large; maximum is {window // 2 - 1} samples."
        raise ValueError(msg)


def _check_exponent(exponent: float) -> None:
    """Raise ValueError unless the exponent is finite and non-negative."""
    # Negative: a silent coefficient would be raised to a negative power,
    # and zero times infinity is the NaN every sample of the tile becomes.
    if not np.isfinite(exponent) or exponent < 0:
        msg = f"exponent must be finite and non-negative; got {exponent!r}."
        raise ValueError(msg)


def _validate_filter_inputs(
    data: np.ndarray,
    *,
    window_size: tuple[int, ...],
    overlap: tuple[int, ...],
    exponent: float,
) -> None:
    """Validate direct array-filter inputs before entering FFT kernels."""
    if data.ndim not in {1, 2}:
        msg = (
            f"adaptive spectral array filters require 1D or 2D input; got {data.ndim}D."
        )
        raise ValueError(msg)
    if len(window_size) != data.ndim or len(overlap) != data.ndim:
        msg = "window_size and overlap must match the input dimensionality."
        raise ValueError(msg)
    _check_exponent(exponent)
    for axis, (window, axis_overlap) in enumerate(zip(window_size, overlap)):
        _check_window(window, axis_overlap, f"axis {axis}")


def _validate_window_and_overlap(
    dims: tuple[str, ...],
    windows: tuple[int, ...],
    overlaps: tuple[int, ...],
    exponent: float,
) -> None:
    """Validate the patch-level settings, naming dimensions rather than axes."""
    try:
        _check_exponent(exponent)
        for dim, window, overlap in zip(dims, windows, overlaps):
            _check_window(window, overlap, repr(dim))
    except ValueError as exc:
        raise ParameterError(str(exc)) from exc


def _restore_dtype(out: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Return float32 output in the input's dtype, if that is a wide enough float.

    The output grows as the input to the power of one plus the exponent,
    which overflows float16 at the default exponent for inputs of a few
    hundred; those, and integers, come back as float32.
    """
    if np.issubdtype(dtype, np.floating) and np.dtype(dtype).itemsize >= 4:
        return out.astype(dtype, copy=False)
    return out


def _plan_and_taper(
    data: np.ndarray, window_size: tuple[int, ...], overlap: tuple[int, ...]
) -> tuple[TilePlan, np.ndarray]:
    """Validate the filter's inputs and return where its tiles sit and their taper."""
    stride = tuple(win - over for win, over in zip(window_size, overlap))
    plan = get_tile_plan(data.shape, window_size, stride)
    return plan, get_taper("triang", window_size, overlap)


def _weight_spectra(
    tiles: np.ndarray, *, exponent: float, normalize_power: bool
) -> np.ndarray:
    """
    Return every tile with its spectrum weighted by a power of its magnitude.

    The filter itself, on a stack of tiles: one FFT over the stack, one
    weighting, one inverse.
    """
    axes = tuple(range(1, tiles.ndim))
    size = tiles.shape[1:]
    spec = sp_fft.rfftn(tiles, s=size, axes=axes, workers=-1)
    if exponent != 0.0:
        power = np.abs(spec).astype(np.float32, copy=False)
        if normalize_power:
            max_power = power.max(axis=axes, keepdims=True)
            power = np.divide(
                power, max_power, out=np.zeros_like(power), where=max_power != 0
            )
        spec *= power**exponent
    return sp_fft.irfftn(spec, s=size, axes=axes, workers=-1).astype(
        np.float32, copy=False
    )


def _adaptive_spectral_filter_scipy(
    data: np.ndarray,
    *,
    window_size: tuple[int, ...],
    overlap: tuple[int, ...],
    exponent: float = 0.8,
    normalize_power: bool = False,
) -> np.ndarray:
    """
    Filter a 1D or 2D array with the SciPy adaptive spectral implementation.

    Parameters
    ----------
    data
        One- or two-dimensional input array. The filter computes in ``float32``.
    window_size
        Window lengths in samples, one per array axis, of at least 2.
    overlap
        Number of samples each neighboring window overlaps on each axis. Values
        must be non-negative and smaller than half the matching window.
    exponent
        Spectral magnitude exponent used as the adaptive weighting power. ``0``
        leaves the spectrum unweighted before overlap-add reconstruction.
    normalize_power
        If ``True``, normalize each tile's spectral magnitudes by that tile's
        maximum magnitude before applying ``exponent``.

    Returns
    -------
    numpy.ndarray
        The filtered array with the same shape as ``data``. Floating input
        dtypes are restored; non-floating inputs return ``float32`` output.

    Raises
    ------
    ValueError
        If ``data`` is not one- or two-dimensional, ``exponent`` is not finite,
        ``window_size`` and ``overlap`` do not match ``data.ndim``, any window
        size is under 2 samples, or any overlap is negative or
        at least half the matching window size.
    """
    data = np.asarray(data)
    _validate_filter_inputs(
        data, window_size=window_size, overlap=overlap, exponent=float(exponent)
    )
    plan, taper = _plan_and_taper(data, window_size, overlap)
    weight = partial(
        _weight_spectra, exponent=float(exponent), normalize_power=normalize_power
    )
    # The filter works in float32 whatever it is given.
    out = plan.apply(np.asarray(data, dtype=np.float32), weight, taper)
    return _restore_dtype(out, data.dtype)


def _get_engine(engine: str, selected_ndim: int) -> Callable:
    """Return the requested adaptive spectral array filter implementation."""
    if engine == "scipy" or (engine == "auto" and selected_ndim == 1):
        return _adaptive_spectral_filter_scipy
    if engine not in {"auto", "numba"}:
        msg = "engine must be one of 'auto', 'numba', or 'scipy'."
        raise ParameterError(msg)
    if selected_ndim != 2:
        msg = "engine='numba' currently supports exactly two selected dimensions."
        raise ParameterError(msg)
    # Deferred: the numba engine is optional, and importing it eagerly
    # would compile it whenever dascore is imported.
    from dascore.proc._adaptive_spectral_filter_numba import (  # noqa: PLC0415
        _NUMBA_ENGINE_AVAILABLE,
        _adaptive_spectral_filter_numba,
    )

    if _NUMBA_ENGINE_AVAILABLE:
        return _adaptive_spectral_filter_numba
    if engine == "numba":
        msg = (
            "engine='numba' requires optional dependencies numba and "
            "rocket-fft to be installed."
        )
        raise MissingOptionalDependencyError(msg)
    return _adaptive_spectral_filter_scipy


@patch_function()
def adaptive_spectral_filter(
    patch: PatchType,
    *,
    overlap: Any = None,
    exponent: float = 0.8,
    normalize_power: bool = False,
    samples: bool = False,
    engine: _AdaptiveSpectralEngine = "auto",
    **kwargs: Any,
) -> PatchType:
    """
    Apply adaptive spectral filtering over one or two patch dimensions.

    Parameters
    ----------
    patch
        DASCore patch whose data should be filtered.
    overlap
        Window overlap in samples when ``samples=True`` or in coordinate units
        otherwise. A single value applies to all selected dimensions; a mapping
        can specify dimensions independently. When omitted, each dimension
        defaults to ``window // 2 - 1`` samples, the largest overlap allowed.
    exponent
        Spectral magnitude exponent used as the adaptive weighting power.
        Larger values suppress incoherent energy harder; ``0`` leaves the
        spectrum unweighted, and values above 1 begin to remove weak coherent
        arrivals along with the noise. Must be non-negative.
    normalize_power
        If ``True``, normalize each tile's spectral magnitudes by that tile's
        maximum magnitude before applying ``exponent``. This keeps the
        amplitude of every window near its input level, at the cost of
        suppressing much less noise in windows which hold no signal.
    samples
        If ``True``, dimension kwargs and overlap values are interpreted as
        sample counts. If ``False``, values are converted through evenly sampled
        patch coordinates.
    engine
        ``"auto"`` uses SciPy for one selected dimension and the optional
        Numba/rocket-fft implementation for two selected dimensions when
        available. ``"numba"`` requires two selected dimensions and the optional
        fast engine. ``"scipy"`` always uses the SciPy FFT implementation.
    **kwargs
        One or two dimension names and their window sizes, such as ``time=32``
        or ``time=32, distance=32``.

    Returns
    -------
    Patch
        A new patch with filtered data and original dimensions and coordinates.
        The data are ``float32``, or ``float64`` for ``float64`` input.

    Raises
    ------
    ParameterError
        If one or two dimensions are not selected, if selected window or overlap
        values are invalid, if ``exponent`` is not finite and non-negative, or
        if an invalid engine name is requested.
    MissingOptionalDependencyError
        If ``engine="numba"`` is requested for two selected dimensions but the
        optional fast-engine dependencies are not installed.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch("example_event_2").pass_filter(time=(1, 300))
    >>> # Suppress energy which is not coherent across both time and distance,
    >>> # in windows of 16 samples along each.
    >>> filtered = patch.adaptive_spectral_filter(
    ...     time=16, distance=16, samples=True
    ... )
    >>> # Or weight each trace's spectrum on its own.
    >>> per_trace = patch.adaptive_spectral_filter(time=32, samples=True)

    Notes
    -----
    - With two selected dimensions this is the adaptive frequency-wavenumber
      (AFK) filter of @isken2022denoising, and matches Pyrocko
      [Lightguide](https://github.com/pyrocko/lightguide)'s `afk_filter`,
      whose defaults are ``window_size=16, overlap=7, exponent=0.8``.
    - The filter is not amplitude preserving. Each coefficient is scaled by
      its own magnitude to the power of ``exponent``, so the output's units
      are not the input's and its amplitudes grow with the input's; compare
      arrivals within one output rather than across inputs.
    - A window should hold a few cycles of the arrivals to keep and be short
      against the distance over which their moveout changes. Any length of
      at least 2 samples serves; a power of two transforms fastest.
    """
    return AdaptiveSpectralFilter(
        overlap=overlap,
        exponent=exponent,
        normalize_power=normalize_power,
        samples=samples,
        engine=engine,
        **kwargs,
    )._apply(patch)


class AdaptiveSpectralFilter(PatchProcessor):
    """
    Weight every window's spectrum by a power of its own magnitude.

    The dimensions to filter arrive as extras carrying their window sizes,
    as the patch function takes them. Windows and overlaps may be given in
    coordinate units, so they are only sample counts once the coordinates
    are known: `geometry` is that conversion, and where a kernel for any
    backend starts.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    overlap: Any = None
    exponent: float = 0.8
    normalize_power: bool = False
    samples: bool = False
    # A str rather than the Literal, so a wrong name is refused by
    # `_get_engine` as a ParameterError like every other bad argument.
    engine: str = "auto"

    def geometry(self, meta: PatchMeta) -> Window:
        """Return the window in samples: the selected axes, sizes, and overlaps."""
        selected = self.model_extra or {}
        if len(selected) not in {1, 2}:
            msg = (
                "adaptive_spectral_filter requires one or two dimension window "
                "kwargs, e.g. patch.adaptive_spectral_filter(time=32, samples=True)."
            )
            raise ParameterError(msg)
        window = resolve_window(
            meta,
            selected,
            samples=self.samples,
            overlap=self.overlap,
            # Sample counts are read as given, whatever the coordinate is.
            require_evenly_sampled=False,
            min_samples=2,
            # The most the window allows, which is what Lightguide uses. A
            # default is a sample count whatever `samples` says.
            default_overlap=lambda size: size // 2 - 1,
        )
        # A default was given, so every dimension has an overlap.
        assert window.overlap is not None
        _validate_window_and_overlap(
            window.dims, window.size, window.overlap, float(self.exponent)
        )
        return window

    def kernel(self, data, meta, out_meta):
        """Filter every batch over the selected axes and stack the results."""
        window = self.geometry(meta)
        axes, windows, overlaps = window.axes, window.size, window.overlap
        engine = _get_engine(self.engine, len(axes))
        data = np.asarray(data)
        tail = tuple(range(-len(axes), 0))
        moved = np.moveaxis(data, axes, tail)
        working = moved.reshape((-1, *moved.shape[-len(axes) :]))
        filtered = np.empty_like(working, dtype=np.float32)
        for ind, array in enumerate(working):
            filtered[ind] = engine(
                array,
                window_size=windows,
                overlap=overlaps,
                exponent=float(self.exponent),
                normalize_power=bool(self.normalize_power),
            )
        filtered = np.moveaxis(filtered.reshape(moved.shape), tail, axes)
        return _restore_dtype(filtered, data.dtype)


register_implementation("adaptive_spectral_filter", AdaptiveSpectralFilter)
