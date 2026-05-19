"""
Patch function for 'short-term average' to 'long-term average' ratio transform
"""

from __future__ import annotations

from dascore.constants import PatchType
from dascore.utils.patch import patch_function
from dascore.utils.time import to_float


@patch_function()
def stalta(
    patch: PatchType,
    short_window: float | None = None,
    long_window: float | None = None,
    absolute: bool = True,
    dim: str = "time",
) -> PatchType:
    """
    Compute the short-term / long-term average (STA/LTA) ratio along a patch dimension.

    Parameters
    ----------
    patch :
        The input DASCore patch.
    short_window : float
        Short window length in seconds. If None, it defaults to 20 samples.
    long_window : float
        Long window length in seconds. If None it defaults to 5*short.
    absolute : boolean
        Uses the absolute value of the signal (default=True)
    dim
        Dimension along which to compute the STA/LTA ratio. Defaults to ``"time"``.

    Returns
    -------
    PatchType
        A new patch containing the STA/LTA ratio.

    Example
    --------
    >>> import dascore as dc
    >>>
    >>> p = dc.examples.get_example_patch('example_event_2')
    >>>
    >>> r = p.stalta(short_window = 0.001, long_window = 0.01, dim = 'time')
    >>> ax = r.viz.waterfall(cmap = 'RdGy_r', scale = [0, 2], scale_type = 'absolute')

    """
    # these default values are heuristically derived and may not work in all cases
    if short_window is None:
        step = to_float(patch.get_coord(dim).step)
        short_window = 20 * step
    if long_window is None:
        long_window = 5 * short_window

    if absolute:
        patch = patch.abs()

    sta = patch.rolling(**{dim: short_window}).mean()
    lta = patch.rolling(**{dim: long_window}).mean()

    return (sta / lta).update(attrs={"data_type": "STALTA", "data_units": ""})
