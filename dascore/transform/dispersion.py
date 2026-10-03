"""Dispersion computation using the phase-shift (Park et al., 1999) method."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.fft as nft

from dascore.core.processor import PatchProcessor
from dascore.exceptions import ParameterError
from dascore.proc.coords import Transpose
from dascore.proc.units import ConvertUnits, SetUnits


class DispersionPhaseShift(PatchProcessor):
    """
    Compute dispersion images using the phase-shift method.

    Parameters
    ----------
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

    __version__ = "1.1"
    phase_velocities: Any
    approx_resolution: Any = None
    approx_freq: Any = None

    required_dims = ("time", "distance")
    data_type = "dispersion"

    def get_metadata(self, meta):
        """Return the image's metadata, and the frequencies the kernel keeps."""
        phase_velocities = self.phase_velocities
        approx_resolution, approx_freq = self.approx_resolution, self.approx_freq
        meta_cop, _ = ConvertUnits(distance="m").get_metadata(meta)
        meta_cop, transpose = Transpose(dims=("distance", "time")).get_metadata(
            meta_cop
        )
        dist = meta_cop.coords.get_array("distance")
        time = meta_cop.coords.get_array("time")
        dt = (time[1] - time[0]) / np.timedelta64(1, "s")
        approx_min_freq, approx_max_freq = _frequency_range(
            phase_velocities, approx_resolution, approx_freq, dt
        )
        nt = time.size
        fs = 1 / dt
        nf = int(fs / approx_resolution) if approx_resolution else nt
        w = 2 * np.pi * (np.arange(nf) * fs / nf)
        first_live_f = np.argmax(w >= 2 * np.pi * approx_min_freq)
        last_live_f = np.argmax(w >= 2 * np.pi * approx_max_freq)
        w = w[first_live_f:last_live_f]
        nlivef = last_live_f - first_live_f
        if nlivef < 1:
            msg = "Combination of frequency resolution and range is not an array"
            raise ParameterError(msg)
        attrs = meta.attrs.update(category="dispersion")
        coords = dict(velocity=phase_velocities, frequency=w / (2 * np.pi))
        dims = ["velocity", "frequency"]
        out = meta.new(coords=coords, attrs=attrs, dims=dims, dtype=np.float64)
        out, _ = SetUnits(velocity="m/s", frequency="Hz").get_metadata(out)
        return out, {
            "axes": transpose.get("axes"),
            "nf": nf,
            # Pad to a multiple of nf and decimate the bins so no sample is dropped.
            "step": -(-nt // nf),
            "live": slice(int(first_live_f), int(last_live_f)),
            "dist": dist,
            "w": w,
        }

    def numpy_kernel(self, data, *, axes, nf, step, live, dist, w):
        """Return the normalized phase-shift stack of each velocity and frequency."""
        data = data if axes is None else np.permute_dims(data, axes)
        nchan = dist.size
        fft_d = np.zeros((nchan, nf), dtype=complex)
        for i in range(nchan):
            fft_d[i] = nft.fft(data[i, :], n=nf * step)[::step]
        fft_d = np.divide(
            fft_d, abs(fft_d), out=np.zeros_like(fft_d), where=abs(fft_d) != 0
        )
        fft_d[np.isnan(fft_d)] = 0
        fft_d = fft_d[:, live]
        phase_velocities = self.phase_velocities
        fc = np.zeros(shape=(np.size(phase_velocities), w.size))
        preamb = 1j * np.outer(dist, w)
        for ci in range(np.size(phase_velocities)):
            fc[ci, :] = abs(sum(np.exp(preamb / phase_velocities[ci]) * fft_d))
        return fc / nchan


def _frequency_range(phase_velocities, approx_resolution, approx_freq, dt):
    """Check the arguments, and return the frequency range to image."""
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
    return approx_min_freq, approx_max_freq
