"""
Patch function for 'Frequency-Band Energy' transform
"""

from __future__ import annotations

from dascore.constants import PatchType
from dascore.units import get_filter_units
from dascore.utils.misc import check_filter_kwargs, check_filter_range
from dascore.utils.patch import get_dim_sampling_rate, patch_function
from dascore.utils.time import to_float


@patch_function()
def fbe(
    patch: PatchType,
    window: float,
    step: float | None = None,
    db: bool = True,
    **kwargs,
) -> PatchType:
    """
    Compute the rolling Frequency Band Energy in a window.
    This is the Root-Mean-Squared (RMS) of the Energy in a Frequency Band (the FBE),
    and commonly called 'the waterfall plot' in DAS-processing.
    This implementation is a wrapper to DAScore functionality:
        1) Apply a 'pass_filter' to the patch
        2) Apply rolling-funtction of a window along a coordinate
        3) Calculate RMS value


    Parameters
    ----------
    patch
        Input DASCore patch.

    window
        window length in which to caluclate engergy (in units of the sampling rate)
    step
        time-step for rolling window. Defaults to original sampling rate, but this
        can be used for downsampling of the resulting patch.
        See also [rolling](`dascore.Patch.rolling`)
    db
        Return patch data in decibel [dB] instead of orginal units
    **kwargs
        Used to specify the dimension and asociated frequency, wavelength, or
        equivalent limits. For example time=(1, 100) applies a time-dimension bandpass
        of 1-100 Hz. See [pass_filter](`dascore.Patch.pass_filter`) for more details.

    Returns
    -------
    PatchType
        A new patch containing FBE-RMS traces.

    Example
    --------
    >>> import dascore as dc
    >>>
    >>> p = dc.examples.example_event_2()
    >>>
    >>> fbe_patch = p.fbe(time=(100,200), window = 0.002)
    >>> ax = fbe_patch.viz.waterfall(cmap = 'Spectral_r')
    >>> _ = ax.set_title('FBE along time-axis')


    Or along the distance-axis:
    >>> import dascore as dc
    >>>
    >>> p = dc.examples.get_example_patch('example_event_2')
    >>>
    >>> fbe_patch = p.fbe(distance=(.01,.05), window = 5)
    >>> ax = fbe_patch.viz.waterfall(cmap = 'Spectral_r')
    >>> _ = ax.set_title('FBE along distance-axis')
    """
    dim, (arg1, arg2) = check_filter_kwargs(kwargs)
    coord_units = patch.coords.coord_map[dim].units
    filt_min, filt_max = get_filter_units(arg1, arg2, to_unit=coord_units, dim=dim)
    sample_rate = get_dim_sampling_rate(patch, dim)

    nyquist = 0.5 * sample_rate
    low = None if filt_min is None else filt_min / nyquist
    high = None if filt_max is None else filt_max / nyquist
    check_filter_range(nyquist, low, high, filt_min, filt_max)

    if step is None:
        step = to_float(1 / sample_rate)

    patch = patch.pass_filter(**kwargs)

    fbe = ((patch**2).rolling(**{dim: window, "step": step}).mean() ** 0.5).update(
        attrs={"data_type": "Frequency-Band Energy"}
    )

    if db:
        fbe = (10 * fbe.log10()).update(
            attrs={"data_type": "Frequency-Band Energy", "data_units": "dB"}
        )

    return fbe
