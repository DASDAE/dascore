"""
Patch function for 'short-term average' to 'long-term average' ratio transform
"""

from __future__ import annotations

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
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
        For example `time=(0.1, 0.5)` uses windows of 0.1 and 0.5 seconds along
        the time axis.

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
    >>> s.viz.waterfall(  # doctest: +SKIP
    ...     cmap="RdGy_r", scale=[0, 2], scale_type="absolute"
    ... );
    """
    dim, (sta, lta) = check_filter_kwargs(kwargs)
    if lta <= sta:
        msg = f"The long-term window must exceed the short-term window, got {lta}."
        raise ParameterError(msg)

    sta_data = patch.rolling(**{dim: sta}, samples=samples).mean()
    lta_data = patch.rolling(**{dim: lta}, samples=samples).mean()

    return sta_data / lta_data
