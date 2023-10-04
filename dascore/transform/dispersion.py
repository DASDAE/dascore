"""Dispersion computation using the phase-shift (Park et al., 1999) method."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.fft as nft

from dascore.constants import PatchType
from dascore.exceptions import DispersionParameterError
from dascore.utils.patch import patch_function


@patch_function
def phase_shift(
    patch: PatchType, vels: Sequence[float], direction: str, approxdf: float = 0.0
) -> PatchType:
    """
    Compute dispersion images using the phase-shift method (Park et al., 1999)
    Inspired by https://geophydog.cool/post/masw_phase_shift/.

    Parameters
    ----------
    patch
        Patch to transform. Has to have dimensions of time and distance
    vels
        NumPY array of positive velocities, monotonically increasing, for
        which the dispersion will be computed
    direction
        Indicating whether the event is down-dip ('ltr') or up-dip ('rtl')
    approxdf
        Approximated frequency (Hz) resolution for the output. If left empty,
        the frequency resolution is dictated by the number of samples.

    Notes
    -----
    - Dims/Units of the output are forced to be 'frequency' ('Hz')
      and 'velocity' ('m/s')

    Example
    --------
    import dascore as dc
    import numpy as np

    patch = (
        dc.get_example_patch('example_event_1')
        .set_units("mm/(m*s)", distance='m', time='s')
        .taper(time=0.05)
        .pass_filter(time=(None, 300))
    )

    disp_patch = patch.phase_shift(np.arange(1500,6001,10),direction='rtl',approxdf=5.0)
    ax = disp_patch.viz.waterfall(show=False, scale=0.5,cmap=None)
    ax.set_xlim(10, 300)
    ax.set_ylim(6000, 1500)
    disp_patch.viz.waterfall(show=True, scale=0.5, ax=ax)

    """
    patch_cop = patch.new()

    if not np.all(np.diff(np.abs(vels)) > 0):
        raise DispersionParameterError(
            "Velocities for dispersion computation\
         must be monotonically increasing"
        )

    if set(patch_cop.dims) != set(["distance", "time"]):
        raise DispersionParameterError(
            "Dispersion can only be computed\
         for distance-time patches"
        )

    if np.amin(np.abs(vels)) <= 0:
        raise DispersionParameterError(
            "Velocity has to be positive.\
         Control the direction parameter if needed."
        )

    if direction == "ltr":
        dirflag = 1
    elif direction == "rtl":
        dirflag = -1
    else:
        raise DispersionParameterError("Direction can only be ltr or rtl")

    if patch.dims[0] == "time":
        patch_cop.transpose("distance", "time")

    patch_cop.convert_units(distance="m")
    dist = patch_cop.coords.get_array("distance")
    time = patch_cop.coords.get_array("time")

    dt = (time[1] - time[0]) / np.timedelta64(1, "s")  # There has to be an easier way.
    nchan = dist.size
    nt = time.size
    m, n = patch_cop.data.shape
    assert m == nchan
    assert n == nt

    fs = 1 / dt
    if approxdf > 0.0:
        approxnf = int(nt * (fs / (nt)) / approxdf)
        f = np.arange(approxnf) * fs / (approxnf - 1)
    else:
        f = np.arange(nt) * fs / (nt - 1)
    f[1] - f[0]

    nf = np.size(f)

    nv = np.size(vels)
    w = 2 * np.pi * f
    fft_d = np.zeros((nchan, nf), dtype=complex)
    for i in range(nchan):
        fft_d[i] = nft.fft(patch_cop.data[i, :], n=nf)

    fft_d = np.divide(
        fft_d, abs(fft_d), out=np.zeros_like(fft_d), where=abs(fft_d) != 0
    )

    fft_d[np.isnan(fft_d)] = 0

    hnf = int(nf / 2)
    fc = np.zeros(shape=(nv, hnf))
    preamb = 1j * dirflag
    for ci in range(nv):
        for fi in range(hnf):
            fc[ci, fi] = abs(
                sum(np.exp(preamb * w[fi] / vels[ci] * dist) * fft_d[:, fi])
            )

    attrs = dict(category="Dispersion")
    coords = dict(velocity=vels, frequency=f[0:hnf])

    disp_patch = patch.new(
        data=fc, coords=coords, attrs=attrs, dims=["velocity", "frequency"]
    )
    disp_patch.set_units(velocity="m/s", frequency="Hz")

    return disp_patch
