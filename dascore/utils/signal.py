"""
Utilities for signal processing.
"""

from functools import lru_cache

import numpy as np
from scipy.signal import windows

from dascore.exceptions import ParameterError

WINDOW_FUNCTIONS = dict(
    barthann=windows.barthann,
    bartlett=windows.bartlett,
    blackman=windows.blackman,
    blackmanharris=windows.blackmanharris,
    bohman=windows.bohman,
    hamming=windows.hamming,
    hann=windows.hann,
    cos=windows.hann,
    nuttall=windows.nuttall,
    parzen=windows.parzen,
    triang=windows.triang,
    ramp=windows.triang,
    boxcar=windows.boxcar,
)


def _get_window_function(window_type):
    """Get the window function to use for taper."""
    # get taper function or raise if it isn't known.
    if window_type not in WINDOW_FUNCTIONS:
        msg = (
            f"'{window_type}' is not a known window type. "
            f"Options are: {sorted(WINDOW_FUNCTIONS)}"
        )
        raise ParameterError(msg)
    func = WINDOW_FUNCTIONS[window_type]
    return func


def _triangular_taper_1d(size: int, plateau: int) -> np.ndarray:
    """Return a one-dimensional triangular plateau taper."""
    ramp_size = (size - plateau) // 2
    taper = np.ones(size, dtype=np.float32)
    if ramp_size:
        ramp = _get_window_function("triang")(ramp_size * 2 + 1)[:ramp_size]
        taper[:ramp_size] = ramp
        taper[size - ramp_size :] = ramp[::-1]
    return taper


@lru_cache(maxsize=64)
def _cached_triangular_taper(
    window_size: tuple[int, ...], plateau: tuple[int, ...]
) -> np.ndarray:
    """Build the cached taper array used by :func:`_triangular_taper`."""
    if len(window_size) != len(plateau):
        msg = "window_size and plateau must have the same length."
        raise ValueError(msg)
    if len(window_size) not in {1, 2}:
        msg = "Only one- and two-dimensional tapers are supported."
        raise ValueError(msg)
    if any(plat > win for win, plat in zip(window_size, plateau)):
        msg = "Plateau cannot be larger than window size."
        raise ValueError(msg)
    if any(plat < 0 for plat in plateau):
        msg = "Plateau sizes must be non-negative."
        raise ValueError(msg)
    if any(win % 2 for win in window_size):
        msg = "Window sizes must be even."
        raise ValueError(msg)
    tapers = [
        _triangular_taper_1d(win, plat) for win, plat in zip(window_size, plateau)
    ]
    if len(tapers) == 1:
        return tapers[0].astype(np.float32)
    return (tapers[0][:, None] * tapers[1][None, :]).astype(np.float32)


def _triangular_taper(
    window_size: tuple[int, ...], plateau: tuple[int, ...]
) -> np.ndarray:
    """
    Return a one- or two-dimensional triangular plateau taper.

    Parameters
    ----------
    window_size
        Number of samples in the window dimensions. Values must be even.
    plateau
        Number of central samples with unit weight in each dimension. Values
        must be non-negative and no larger than the corresponding window size.

    Returns
    -------
    numpy.ndarray
        A ``float32`` array with shape ``window_size``. The returned array is a
        copy, so callers can mutate it without corrupting the internal cache.

    Raises
    ------
    ValueError
        If the inputs have different lengths, if the taper dimensionality is
        not one or two, if any plateau is negative, if any plateau is greater
        than the corresponding window size, or if any window size is odd.

    Notes
    -----
    The taper is separable in 2D. Each axis contains a central unit-weight
    plateau and triangular ramps on both sides generated from DASCore's
    registered ``"triang"`` window function. When plateau equals window size,
    the result is all ones along that axis.

    Examples
    --------
    >>> from dascore.utils.signal import _triangular_taper
    >>> taper = _triangular_taper((8, 8), (2, 2))
    >>> taper.shape
    (8, 8)
    """
    return _cached_triangular_taper(window_size, plateau).copy()
