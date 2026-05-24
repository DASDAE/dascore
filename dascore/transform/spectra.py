"""Patch function for calculating spectra as a matrix"""

from __future__ import annotations

import numpy as np

from dascore import units
from dascore.constants import PatchType
from dascore.utils.patch import _get_dx_or_spacing_and_axes, patch_function


@patch_function(required_dims=("time",), history="full")
def spectra(
    patch: PatchType,
    dim: str = "time",
    kind: str = "PSD",
    db: bool = False,
) -> PatchType:
    """
    Compute the amplitude spectrum, power spectrum, or power spectral density.

    This function applies a real-valued discrete Fourier transform along the
    specified dimension and returns the requested spectral representation as a
    DASCore Patch. The transformed dimension is replaced by its corresponding
    frequency-domain coordinate.

    Parameters
    ----------
    patch
        Input patch containing the data to transform.
    dim
        Dimension along which to compute the spectrum. Defaults to ``"time"``.
    kind
        Spectral quantity to return. Options are:

        - ``"AS"``: amplitude spectrum, ``abs(DFT(x))``.
        - ``"PS"``: power spectrum, ``abs(DFT(x)) ** 2``.
        - ``"PSD"``: power spectral density, normalized by
          ``n_samples * sampling_rate``.

        The comparison is case-insensitive. Defaults to ``"PSD"``.
    db
        If ``True``, return the spectrum in decibels. Amplitude spectra use
        ``20 * log10(...)``; power spectra and PSDs use ``10 * log10(...)``.

    Returns
    -------
    PatchType
        Patch containing the requested spectral representation.


    Example
    -------

    Example 2: dispersion event
    >>> import dascore as dc
    >>> import matplotlib.pyplot as plt
    >>>
    >>> patch = dc.examples.dispersion_event().T.set_units(distance='m')
    >>>
    >>> fig, axs = plt.subplots(1,3,figsize=(16,4), layout='constrained')
    >>> patch.viz.waterfall(cmap='RdBu_r', ax=axs[0], show=False);
    >>>
    >>> for a, dim in enumerate(['time', 'distance']):
    >>>     spec  = patch.spectra(dim=dim, kind='PSD',  db=True)
    >>>     ax = spec.viz.spectraplot(log=True, ax=axs[a+1], show=False, scale=[0.7, 1])
    >>>     ax.set_title('Fourier-Transform along ' + dim.capitalize())
    >>>     if dim=='time':
    >>>         ax.set_ylim((5,200))


    Example 2: seimic event
    >>> import dascore as dc
    >>> import matplotlib.pyplot as plt
    >>>
    >>> patch = dc.examples.example_event_2().T.decimate(time=10)
    >>> new = dc.to_datetime64(patch.coords.get_array('time') + 1704400020)
    >>> patch = patch.update_coords(time=new)
    >>>
    >>> fig, axs = plt.subplots(1,3,figsize=(18,5), layout='constrained')
    >>> patch.viz.waterfall(cmap='RdBu_r', ax=axs[0]);
    >>>
    >>> for a, dim in enumerate(['time', 'distance']):
    >>>     spec  = patch.spectra(dim=dim, kind='PSD',  db=True)
    >>>     ax = spec.viz.spectraplot(log=True, ax=axs[a+1], show=False, scale=[0.7, 1])
    >>>     ax.set_title('Fourier-Transform along ' + dim.capitalize())
    """
    spec = patch.dft(dim=dim, real=True, pad=True).abs()
    if db:
        # add a small number to prevent division-by zero when taking the log10
        spec += np.finfo(spec.data.dtype).eps

    _, axis = _get_dx_or_spacing_and_axes(patch, dim)
    n = patch.data.shape[axis[0]]

    if kind.upper() == "AS":
        out = spec
        if db:
            out = 20 * out.log10()

        out = out.update(attrs={"data_type": "Amplitude Spectrum"})

    elif kind.upper() == "PS":
        out = spec * spec
        if db:
            out = 10 * out.log10()

        out = out.update(
            attrs={
                "data_type": "Power Spectrum",
            }
        )

    elif kind.upper() == "PSD":
        fsamp = 1 / (
            spec.get_coord("ft_" + dim).step * spec.get_coord("ft_" + dim).units
        )
        out = spec * spec / (n * fsamp)
        if db:
            out = 10 * out.log10()

        out = out.update(attrs={"data_type": "Power Spectral Density"})

    else:
        raise ValueError("ERROR: Unknown option: kind=", kind)

    if db:
        out = out.set_units(units.dB)

    return out
