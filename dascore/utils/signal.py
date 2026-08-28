"""
Windows, ramps, and tile tapers, built one way.

A *window* is a whole array of weights, the thing a spectral transform
multiplies a tile by. A *ramp* is the rising edge of one, which an edge
taper multiplies the first samples of a patch by. A *taper* is a tile of
ones with ramps down every edge, which overlapping tiles are blended with.
All three come from one place, so a name means the same shape wherever it
is used, and a ramp can be made complementary -- adjacent ramps summing to
one -- when a blend has to invert exactly.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np

from dascore.exceptions import ParameterError
from dascore.utils.imports import lazy_import

_scipy_get_window = lazy_import("scipy.signal", "get_window")

# DASCore's own names for windows, including two which scipy does not
# spell: `cos` for hann and `ramp` for triang. Anything not here is handed
# to scipy, which knows the rest and takes `(name, parameter)` tuples.
WINDOW_FUNCTIONS = dict(
    barthann=lazy_import("scipy.signal.windows", "barthann"),
    bartlett=lazy_import("scipy.signal.windows", "bartlett"),
    blackman=lazy_import("scipy.signal.windows", "blackman"),
    blackmanharris=lazy_import("scipy.signal.windows", "blackmanharris"),
    bohman=lazy_import("scipy.signal.windows", "bohman"),
    hamming=lazy_import("scipy.signal.windows", "hamming"),
    hann=lazy_import("scipy.signal.windows", "hann"),
    cos=lazy_import("scipy.signal.windows", "hann"),
    nuttall=lazy_import("scipy.signal.windows", "nuttall"),
    parzen=lazy_import("scipy.signal.windows", "parzen"),
    triang=lazy_import("scipy.signal.windows", "triang"),
    ramp=lazy_import("scipy.signal.windows", "triang"),
    boxcar=lazy_import("scipy.signal.windows", "boxcar"),
)


def get_window(window: Any, size: int, *, fftbins: bool = False) -> np.ndarray:
    """
    Return a window of `size` samples.

    Parameters
    ----------
    window
        A name from `WINDOW_FUNCTIONS`, any name or ``(name, parameter)``
        tuple `scipy.signal.get_window` accepts, or an array, which is
        returned as it is if it has `size` samples.
    size
        How many samples the window has.
    fftbins
        If True, the window is periodic, as for a spectral estimate; the
        default is symmetric, as for a taper.

    Raises
    ------
    ParameterError
        For a name nothing knows, or an array of the wrong length.
    """
    if isinstance(window, np.ndarray):
        if len(window) != size:
            msg = f"The window has {len(window)} samples, not {size}."
            raise ParameterError(msg)
        return window
    if isinstance(window, str) and window in WINDOW_FUNCTIONS:
        return WINDOW_FUNCTIONS[window](size, sym=not fftbins)
    try:
        return _scipy_get_window(window, size, fftbins=fftbins)
    except ValueError as exc:
        # scipy says "Unknown window type" for a name it lacks; its other
        # complaints -- a parameter out of range -- are its own to make.
        if "Unknown window type" not in str(exc):
            raise
        msg = (
            f"'{window}' is not a known window type. Options are: "
            f"{sorted(WINDOW_FUNCTIONS)}, or any name scipy.signal.get_window takes."
        )
        raise ParameterError(msg) from exc


def get_ramp(window: Any, length: int, *, complementary: bool = False) -> np.ndarray:
    """
    Return the rising edge of a window: `length` samples climbing towards one.

    The ramp is the first `length` samples of a symmetric window of
    ``2 * length + 1``, so its peak, which the window holds at the middle,
    is the sample just past the ramp.

    Parameters
    ----------
    window
        The window whose edge this is; see `get_window`.
    length
        How many samples the ramp has.
    complementary
        If True, scale the ramp so that it and its reverse sum to one at
        every sample. Two tiles whose ramps overlap then blend to exactly
        the signal between them, whatever the window's shape.
    """
    ramp = get_window(window, 2 * length + 1)[:length]
    if complementary and length:
        # Weights, whatever the window was given as.
        ramp = np.asarray(ramp, dtype=np.float64)
        total = ramp + ramp[::-1]
        ramp = np.divide(ramp, total, out=np.full_like(ramp, 0.5), where=total > 0)
        # A window such as blackman touches zero from below by rounding; a
        # weight is never negative.
        ramp = np.clip(ramp, 0.0, 1.0)
    return ramp


def _build_taper(
    window: Any, size: tuple[int, ...], overlap: tuple[int, ...]
) -> np.ndarray:
    """Build a taper; see `get_taper`."""
    if len(size) != len(overlap):
        msg = "size and overlap must have the same length."
        raise ParameterError(msg)
    if any(over < 0 for over in overlap):
        msg = "overlap must be non-negative."
        raise ParameterError(msg)
    if any(2 * over > length for length, over in zip(size, overlap)):
        msg = "overlap cannot exceed half the tile: the ramps would cross."
        raise ParameterError(msg)
    edges = []
    for length, over in zip(size, overlap):
        # float32 throughout, edges and product alike, so the taper is the
        # one the filter has always used to the last bit.
        edge = np.ones(length, dtype=np.float32)
        # Built even when the overlap is zero, so a window nothing knows is
        # refused rather than never asked for.
        ramp = get_ramp(window, over, complementary=True).astype(np.float32)
        if over:
            edge[:over] = ramp
            edge[length - over :] = ramp[::-1]
        edges.append(edge)
    # Separable: the taper is the outer product of its edges.
    taper = math.prod(np.ix_(*edges)) if len(edges) > 1 else edges[0]
    return np.asarray(taper, dtype=np.float32)


@lru_cache(maxsize=64)
def _cached_taper(
    window: Any, size: tuple[int, ...], overlap: tuple[int, ...]
) -> np.ndarray:
    """The tapers named by something hashable, built once each."""
    return _build_taper(window, size, overlap)


def _hashable(value: Any) -> bool:
    """Whether a window can key the cache: a name, or a tuple of such."""
    if isinstance(value, tuple):
        return all(_hashable(item) for item in value)
    return isinstance(value, str | int | float)


def get_taper(
    window: Any, size: tuple[int, ...], overlap: tuple[int, ...]
) -> np.ndarray:
    """
    Return a tile of ones with complementary ramps of `overlap` down every edge.

    Tiles of this `size` laid at a stride of ``size - overlap`` and
    multiplied by this taper sum to exactly one everywhere they cover, so
    an overlap-add of them returns the signal they were cut from.

    Parameters
    ----------
    window
        The window whose edge the ramps take; see `get_window`.
    size
        The tile's shape.
    overlap
        How many samples each edge ramps over, per axis. Zero is a boxcar
        along that axis.

    Returns
    -------
    numpy.ndarray
        A ``float32`` array of shape `size`; a copy, so it may be written to.

    Examples
    --------
    >>> from dascore.utils.signal import get_taper
    >>> taper = get_taper("triang", (8, 8), (3, 3))
    >>> taper.shape
    (8, 8)
    """
    size, overlap = tuple(size), tuple(overlap)
    if _hashable(window):
        return _cached_taper(window, size, overlap).copy()
    # An array, or a tuple carrying a list: built each time, uncached.
    return _build_taper(window, size, overlap)
