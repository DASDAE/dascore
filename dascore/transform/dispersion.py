"""Dispersion computation using the phase-shift (Park et al., 1999) method."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.fft as nft

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.patch import patch_function


@patch_function(
    required_dims=("time", "distance"), data_type="dispersion", version="1.1"
)
def dispersion_phase_shift(
    patch: PatchType,
    phase_velocities: Sequence[float],
    approx_resolution: float | None = None,
    approx_freq: tuple[float, float] | None = None,
) -> PatchType:
    """
    Compute dispersion images using the phase-shift method.

    Parameters
    ----------
    patch
        Patch to transform. Has to have dimensions of time and distance.
        It also needs to be right-sided (see notes below).
    phase_velocities
        NumPy array of positive velocities, monotonically increasing, for
        which the dispersion will be computed.
    approx_resolution
        Approximated frequency (Hz) resolution for the output. If left empty,
        the frequency resolution is dictated by the number of samples.
    approx_freq
        Minimum and maximum frequency to compute dispersion for, in Hz
        If left empty, minimum is 0 Hz, and maximum is Nyquist

    Notes
    -----
    - See also @park1998imaging

    - Inspired by https://geophydog.cool/post/masw_phase_shift/.

    - Dims/Units of the output are forced to be 'frequency' ('Hz')
      and 'velocity' ('m/s').

    - Each channel's distance is read as its position along the wave's
    path, so data are effectively mapped along a 2-D line; the coordinate
    need not be sorted.

    - The image depends only on distances relative to each other, so any
    origin works for a one-sided gather, but the wave must travel toward
    increasing distance. For a gather whose wave travels toward lower
    distance, negate the coordinate; for a two-sided gather, use the offset
    from the source (`abs(distance - source_distance)`). The new values carry
    no units, so convert to metres first:
    `p = patch.convert_units(distance="m")` then
    `p.update_coords(distance=-p.get_array("distance"))`. Reversing the
    patch with `flip` changes nothing, since it reverses the data and the
    coordinate together.

    Examples
    --------
    ```{python}
    import dascore as dc
    import numpy as np

    # Example 1 - Right-sided wavefield
    patch = (
        dc.get_example_patch('dispersion_event')
    )

    disp_patch = patch.dispersion_phase_shift(np.arange(100,1500,1),
                approx_resolution=0.1,approx_freq=[5,70])
    ax = disp_patch.viz.waterfall(show=False, cbar=False)
    ax.set_xlim(5, 70)
    ax.set_ylim(1500, 100)
    disp_patch.viz.waterfall(show=True, ax=ax)

    ```
    """
    patch_cop = patch.convert_units(distance="m").transpose("distance", "time")
    dist = patch_cop.coords.get_array("distance")
    time = patch_cop.coords.get_array("time")

    dt = (time[1] - time[0]) / np.timedelta64(1, "s")

    if not np.all(np.diff(phase_velocities) > 0):
        raise ParameterError(
            "Velocities for dispersion must be monotonically increasing"
        )

    if np.amin(phase_velocities) <= 0:
        raise ParameterError("Velocities must be positive.")

    if approx_resolution is not None and approx_resolution <= 0:
        raise ParameterError("Frequency resolution has to be positive")

    if not approx_freq:
        approx_min_freq = 0
        approx_max_freq = 0.5 / dt
    else:
        approx_min_freq = approx_freq[0]
        approx_max_freq = approx_freq[1]
        if approx_min_freq <= 0 or approx_max_freq <= 0:
            msg = "Minimal and maximal frequencies have to be positive"
            raise ParameterError(msg)

        if approx_min_freq >= approx_max_freq:
            msg = "Maximal frequency needs to be larger than minimal frequency"
            raise ParameterError(msg)

        if approx_min_freq >= 0.5 / dt or approx_max_freq >= 0.5 / dt:
            msg = "Frequency range cannot exceed Nyquist"
            raise ParameterError(msg)

    nchan = dist.size
    nt = time.size
    assert (nchan, nt) == patch_cop.data.shape

    fs = 1 / dt
    nf = int(fs / approx_resolution) if approx_resolution else nt
    f = np.arange(nf) * fs / nf

    nv = np.size(phase_velocities)
    w = 2 * np.pi * f
    # Pad to a multiple of nf and decimate the bins so no sample is dropped.
    step = -(-nt // nf)
    fft_d = np.zeros((nchan, nf), dtype=complex)
    for i in range(nchan):
        fft_d[i] = nft.fft(patch_cop.data[i, :], n=nf * step)[::step]

    fft_d = np.divide(
        fft_d, abs(fft_d), out=np.zeros_like(fft_d), where=abs(fft_d) != 0
    )

    fft_d[np.isnan(fft_d)] = 0

    first_live_f = np.argmax(w >= 2 * np.pi * approx_min_freq)
    last_live_f = np.argmax(w >= 2 * np.pi * approx_max_freq)
    w = w[first_live_f:last_live_f]
    fft_d = fft_d[:, first_live_f:last_live_f]
    nlivef = last_live_f - first_live_f

    if nlivef < 1:
        msg = "Combination of frequency resolution and range is not an array"
        raise ParameterError(msg)

    fc = np.zeros(shape=(nv, nlivef))
    preamb = 1j * np.outer(dist, w)
    for ci in range(nv):
        fc[ci, :] = abs(sum(np.exp(preamb / phase_velocities[ci]) * fft_d))

    attrs = patch.attrs.update(category="dispersion")
    coords = dict(velocity=phase_velocities, frequency=w / (2 * np.pi))

    disp_patch = patch.new(
        data=fc / nchan, coords=coords, attrs=attrs, dims=["velocity", "frequency"]
    )
    return disp_patch.set_units(velocity="m/s", frequency="Hz")
