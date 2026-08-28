"""
The adaptive spectral (AFK) filter: every window's spectrum weighted by a
power of its own magnitude, so coherent energy is kept and the rest suppressed.

Over two dimensions this is the adaptive frequency-wavenumber filter of
@isken2022denoising as implemented by Pyrocko
[Lightguide](https://github.com/pyrocko/lightguide), which the SciPy engine
matches to floating-point precision. The engines take one- or
two-dimensional arrays; the patch function resolves windows and batches over
every other dimension. The optional numba engine in
`_adaptive_spectral_filter_numba` shares this module's validation and taper.
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
        msg = (
            f"overlap for {label} is too large; maximum is {(window - 1) // 2} samples."
        )
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
    Filter a 1D or 2D array with the SciPy engine.

    Parameters
    ----------
    data
        A one- or two-dimensional array; the filter computes in ``float32``.
    window_size
        Window lengths in samples, one per axis, of at least 2.
    overlap
        Samples each window shares with the next along each axis: at least 0
        and under half the window.
    exponent
        The weighting power; ``0`` leaves the spectrum unweighted.
    normalize_power
        If True, scale each window's magnitudes by their maximum first.

    Returns
    -------
    The filtered array, in the input's floating dtype or ``float32``.

    Raises
    ------
    ValueError
        For an array of another dimensionality, or a window, overlap, or
        exponent outside the ranges above.
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
    Suppress energy which is not coherent within a window: the AFK filter.

    Parameters
    ----------
    patch
        The patch to filter.
    overlap
        How far each window reaches into the next, in coordinate units or,
        with `samples`, in samples; a mapping gives each dimension its own.
        Default is ``(window - 1) // 2`` samples, the most allowed.
    exponent
        The weighting power. ``0`` leaves the spectrum unweighted; above 1
        weak coherent arrivals go with the noise. Non-negative.
    normalize_power
        If True, scale each window's magnitudes by their maximum before the
        exponent, which keeps every window near its input amplitude at the
        cost of suppressing much less noise where there is no signal.
    samples
        If True, windows and overlaps are sample counts.
    engine
        ``"scipy"``, ``"numba"`` (two dimensions, needs numba and rocket-fft),
        or ``"auto"``, which is numba when it can be.
    **kwargs
        One or two dimensions and their windows, such as ``time=32`` or
        ``time=32, distance=32``.

    Returns
    -------
    A patch with the input's shape and coordinates, in ``float32`` unless the
    input is ``float64``.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch("example_event_2").pass_filter(time=(1, 300))
    >>> # Keep what is coherent across both time and distance.
    >>> filtered = patch.adaptive_spectral_filter(
    ...     time=16, distance=16, samples=True
    ... )
    >>> # Or weight each trace's spectrum on its own.
    >>> per_trace = patch.adaptive_spectral_filter(time=32, samples=True)

    Notes
    -----
    - Over two dimensions this is the adaptive frequency-wavenumber filter of
      @isken2022denoising, matching Pyrocko
      [Lightguide](https://github.com/pyrocko/lightguide)'s `afk_filter` and
      its defaults, ``window_size=16, overlap=7, exponent=0.8``.
    - Not amplitude preserving: each coefficient is scaled by its own
      magnitude to the power of ``exponent``, so compare arrivals within one
      output rather than across inputs.
    - A window should hold a few cycles of the arrivals to keep and be short
      against the distance over which their moveout changes; a power of two
      transforms fastest.
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
            default_overlap=lambda size: (size - 1) // 2,
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
