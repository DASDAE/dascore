"""Module for detrending."""

from __future__ import annotations

from typing import Literal

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.imports import lazy_import
from dascore.utils.patch import patch_function

scipy_detrend = lazy_import("scipy.signal", "detrend")


@patch_function()
def detrend(
    patch: PatchType, dim: str, type: Literal["linear", "constant"] = "linear"
) -> PatchType:
    """
    Perform detrending along a given dimension (distance or time) of a patch.

    Parameters
    ----------
    patch
        The patch to detrend.
    dim
        The dimension ("distance" or "time") along where detrending is applied.
    type
        Specifies least-squares fit type for detrend,
        with "linear" (default) or "constant" as options.

    Returns
    -------
    The Patch instance after applying the detrend function.

    See Also
    --------
    [Patch.taper](`dascore.Patch.taper`)
        Taper the ends of a signal, often used with detrending before transforms.
    [Patch.pass_filter](`dascore.Patch.pass_filter`)
        Apply a Butterworth highpass, lowpass, or bandpass filter.
    [Patch.savgol_filter](`dascore.Patch.savgol_filter`)
        Smooth data using a Savitzky-Golay filter while preserving local trends.

    Examples
    --------
    >>> import dascore # import dascore library
    >>> pa = dascore.get_example_patch() # generate example patch
    >>> out = pa.detrend("time") # detrend along the time dimension
    """
    if dim not in patch.dims:
        msg = f"dim '{dim}' is not in patch dimensions {patch.dims}"
        raise ParameterError(msg)
    axis = patch.get_axis(dim)
    out = scipy_detrend(patch.data, axis=axis, type=type)
    return patch.new(data=out)
