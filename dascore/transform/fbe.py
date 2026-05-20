"""
Patch function for 'Frequency-Band Energy' transform
"""

from __future__ import annotations

from dascore.constants import PatchType
from dascore.utils.patch import patch_function
from dascore.utils.time import to_float


@patch_function()
def fbe(
    patch: PatchType,
    freqmin: float | None = None,
    freqmax: float | None = None,
    time: float | None = None,
    step: float | None = None,
    db: bool = True,
) -> PatchType:
    """
    Compute the rolling Root-Mean-Squared (RMS) of the Energy in a
    Frequency Band (i.e  the Frequency-Band Energy, or FBE).
    This is commonly called 'the waterfall plot' in DAS-processing.
    This implementation is a wrapper to DAScore functionality:
        1) Apply a 'pass_filter' to the patch
        2) Apply rolling-funtction of window length "time"
        3) Calculate RMS value for each channel


    Parameters
    ----------
    patch
        Input DASCore patch.
    freqmin
        Lower-end of the frequency band to calculate energy in.
        If set to None a low-pass filter is applied
    freqmax
        Upper-end of the frequency band to calculate energy in.
        If set to None a high-pass filter is applied
    time
        window length in which to caluclate engergy.
    step
        time-step for rolling window. Defaults to sampling rate. This can be
        used for coarser sampling of the resulting patch
    db
        Return patch data in decibel [dB] instead of orginal units

    Returns
    -------
    PatchType
        A new patch containing FBE-RMS traces.

    Example
    --------
    >>> import dascore as dc
    >>>
    >>> p = dc.examples.get_example_patch('example_event_2')
    >>>
    >>> fbe = p.fbe(corners = (100, 200), time = 0.002)
    >>> ax = fbe.viz.waterfall(cmap = 'Spectral_r')
    """
    if step is None:
        step = to_float(patch.get_coord("time").step)

    if time is None:
        time = 20 * step

    corners = (freqmin, freqmax)
    if corners != (None, None):
        if freqmin is not None and freqmax is not None and freqmax <= freqmin:
            raise ValueError("freqmax must be larger than freqmin.")

        patch = patch.pass_filter(time=corners)

    fbe = ((patch**2).rolling(time=time, step=step).mean() ** 0.5).update(
        attrs={"data_type": "Frequency-Band Energy"}
    )

    if db:
        fbe = (10 * fbe.log10()).update(
            attrs={"data_type": "Frequency-Band Energy", "data_units": "dB"}
        )

    return fbe
