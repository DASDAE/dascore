"""
Patch function for 'short-term average' to 'long-term average' ratio transform
"""

from __future__ import annotations

from dascore.constants import PatchType
from dascore.utils.misc import check_filter_kwargs
from dascore.utils.patch import patch_function


@patch_function()
def stalta(
    patch: PatchType,
    **kwargs,
) -> PatchType:
    """
    Compute the short-term / long-term average (STA/LTA) ratio along a patch dimension.

    Parameters
    ----------
    patch :
        The input DASCore patch.
    **kwargs
        Used to pass one dimension name and the short/long-term window lengths.
        For example, `time=(0.1, 0.5)` uses windows of 0.1 and 0.5 seconds
        along the time axis, and `distance=(5, 25)` uses windows of 5 and 25
        distance units along the distance axis.
        Note that a good first guess is to choose the long-term window 5x the length
        of the short-term window.


    Returns
    -------
    PatchType
        A new patch containing the STA/LTA ratio.

    Example
    --------
    >>> import dascore as dc
    >>>
    >>> p = dc.examples.example_event_2()
    >>>
    >>> s = p.envelope(dim='time').stalta(time=(0.002, 0.01))
    >>> ax = s.viz.waterfall(cmap = 'RdGy_r', scale = [0, 2], scale_type = 'absolute')
    """
    dim, (sta, lta) = check_filter_kwargs(kwargs)

    sta_data = patch.rolling(**{dim: sta}).mean()
    lta_data = patch.rolling(**{dim: lta}).mean()

    return (sta_data / lta_data).update(attrs={"data_type": "stalta", "data_units": ""})
