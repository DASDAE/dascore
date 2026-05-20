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
    sta: float | None = None,
    lta: float | None = None,
    absolute: bool = True,
    dim: str = "time",
) -> PatchType:
    """
    Compute the short-term / long-term average (STA/LTA) ratio along a patch dimension.

    Parameters
    ----------
    patch :
        The input DASCore patch.
    sta : float
        Short window length in seconds. If None, it defaults to 20 samples.
    lta : float
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
    >>> s = p.stalta(sta = 0.001, lta = 0.01, dim = 'time')
    >>> ax = s.viz.waterfall(cmap = 'RdGy_r', scale = [0, 2], scale_type = 'absolute')

    """
    # these default values are heuristically derived and may not work in all cases
    if sta is None:
        step = to_float(patch.get_coord(dim).step)
        sta = 20 * step
    if lta is None:
        lta = 5 * sta

    if absolute:
        patch = patch.abs()

    sta_data = patch.rolling(**{dim: sta}).mean()
    lta_data = patch.rolling(**{dim: lta}).mean()

    return (sta_data / lta_data).update(attrs={"data_type": "STALTA", "data_units": ""})
