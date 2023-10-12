"""Dispersion computation using the phase-shift (Park et al., 1999) method."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.fft as nft

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.patch import patch_function


@patch_function(required_dims=("time", "distance"))
def dispersion_phase_shift(
    patch: PatchType, phase_velocities: Sequence[float], approx_resolution: float
) -> PatchType:
    """
    Compute dispersion images using the phase-shift method.

    Parameters
    ----------
    patch
        Patch to transform. Has to have dimensions of time and distance
    phase_velocities
        NumPY array of positive velocities, monotonically increasing, for
        which the dispersion will be computed
    direction
        Indicating whether the event is down-dip ('ltr') or up-dip ('rtl')
    approx_resolution
        Approximated frequency (Hz) resolution for the output. If left empty,
        the frequency resolution is dictated by the number of samples.

    Notes
    -----
    - See also @park1998imaging

    - Inspired by https://geophydog.cool/post/masw_phase_shift/.

    - Dims/Units of the output are forced to be 'frequency' ('Hz')
      and 'velocity' ('m/s')

    - The patch's distance coordinates are assumed to be ordered by
    distance from the source, and not "fiber distance". In other
    words, data are effectively mapped along a 2-D line.

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

    disp_patch = patch.dispersion_phase_shift(np.arange(1500,6001,10),
                    approx_resolution=5.0)
    ax = disp_patch.viz.waterfall(show=False, scale=0.5,cmap=None)
    ax.set_xlim(10, 300)
    ax.set_ylim(6000, 1500)
    disp_patch.viz.waterfall(show=True, scale=0.5, ax=ax)

    """
    patch_cop = patch.convert_units(distance="m").transpose("distance", "time")

    if not np.all(np.diff(phase_velocities)) > 0:
        raise ParameterError(
            "Velocities for dispersion must be monotonically increasing"
        )

    if np.amin(phase_velocities) <= 0:
        raise ParameterError("Velocities must to be positive.")

    if approx_resolution <= 0:
        raise ParameterError("Frequency resolution has to be positive")

    dist = patch_cop.coords.get_array("distance")
    time = patch_cop.coords.get_array("time")

    dt = (time[1] - time[0]) / np.timedelta64(1, "s")
    nchan = dist.size
    nt = time.size
    assert (nchan, nt) == patch_cop.data.shape

    fs = 1 / dt
    if approx_resolution:
        approxnf = int(nt * (fs / (nt)) / approx_resolution)
        f = np.arange(approxnf) * fs / (approxnf - 1)
    else:
        f = np.arange(nt) * fs / (nt - 1)

    nf = np.size(f)

    nv = np.size(phase_velocities)
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

    # New loop
    fft_d = fft_d[:, 0:hnf]
    w = w[0:hnf]
    preamb = 1j * np.outer(dist, w)
    for ci in range(nv):
        fc[ci, :] = abs(sum(np.exp(preamb / phase_velocities[ci]) * fft_d))

    attrs = patch.attrs.update(category="dispersion")
    coords = dict(velocity=phase_velocities, frequency=f[0:hnf])

    disp_patch = patch.new(
        data=fc / nchan, coords=coords, attrs=attrs, dims=["velocity", "frequency"]
    )
    return disp_patch.set_units(velocity="m/s", frequency="Hz")
