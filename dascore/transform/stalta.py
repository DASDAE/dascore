"""
Patch function for 'short-term average' to 'long-term average' ratio transform
"""

from __future__ import annotations

from dascore.constants import PatchType
from dascore.utils.misc import check_filter_kwargs
from dascore.utils.patch import patch_function


@patch_function(data_type="stalta")
def stalta(
    patch: PatchType,
    *,
    samples: bool = False,
    **kwargs,
) -> PatchType:
    """
    Compute the short-term / long-term average (STA/LTA) ratio along a patch dimension.

    Parameters
    ----------
    patch :
        The input DASCore patch.
    samples
        If True, values specified by kwargs are in samples not coordinate units.
    **kwargs
        Used to pass one dimension name and the short/long-term window lengths.

    Returns
    -------
    PatchType
        A new patch containing the STA/LTA ratio.

    Notes
    -----
    A good first guess is to choose the long-term window 5x the length of the
    short-term window.

    Examples
    --------
    >>> import dascore as dc
    >>>
    >>> p = dc.examples.example_event_2()
    >>>
    >>> s = p.envelope(dim="time").stalta(time=(0.002, 0.01))
    >>> ax = s.viz.waterfall(cmap="RdGy_r", scale=[0, 2], scale_type="absolute")
    >>> s_samples = p.envelope(dim="time").stalta(time=(2, 10), samples=True)
    """
    dim, (sta, lta) = check_filter_kwargs(kwargs)

    sta_data = patch.rolling(**{dim: sta}, samples=samples).mean()
    lta_data = patch.rolling(**{dim: lta}, samples=samples).mean()

    return sta_data / lta_data
