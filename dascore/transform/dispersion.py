"""Dispersion computation using the phase-shift (Park et al., 1999) method."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.fft as nft

from dascore.core.processor import PatchProcessor
from dascore.exceptions import ParameterError, PatchCoordinateError
from dascore.proc.coords import Transpose
from dascore.proc.units import ConvertUnits, SetUnits
from dascore.units import convert_units
from dascore.utils.patch import check_patch_coords


class DispersionPhaseShift(PatchProcessor):
    """
    Compute dispersion images using the phase-shift method.

    The patch must have time-like and distance-like dimensions (see notes).
    Any other dimensions are imaged independently and lead the output's
    dimensions, which end with velocity and frequency.

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
    time_dim
        The time-like dimension. If None, `time`, else `lag_time`, so
        the output of [`Patch.correlate`](`dascore.Patch.correlate`) chains.
    distance_dim
        The distance-like dimension.
    source_distance
        The source's position along `distance_dim`, a float in the
        coordinate's units or a quantity. If given, each channel's offset
        `abs(distance - source_distance)` is used, which images a gather
        with channels on both sides of the source.
    normalize
        "phase" divides each channel's spectrum by its magnitude, keeping
        only its phase, and the output is dimensionless. None stacks the
        spectra as they are (a slant stack), and the output keeps the
        patch's data units. Either way, the stack is divided by the number
        of channels.

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
    from the source with `source_distance`. The new values carry
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

    # Example 2 - Image the correlation of every channel with the first
    cc = patch.correlate(distance=0, samples=True).squeeze()
    cc_disp = cc.select(lag_time=(0, None)).dispersion_phase_shift(
        np.arange(100, 1500, 5), approx_freq=[5, 70]
    )
    ```
    """

    __version__ = "1.2"
    phase_velocities: Any
    approx_resolution: Any = None
    approx_freq: Any = None
    time_dim: str | None = None
    distance_dim: str = "distance"
    source_distance: Any = None
    normalize: Literal["phase"] | None = "phase"

    data_type = "dispersion"

    def get_metadata(self, meta):
        """Return the image's metadata, and the frequencies the kernel keeps."""
        phase_velocities = self.phase_velocities
        approx_resolution, approx_freq = self.approx_resolution, self.approx_freq
        dist_dim, time_dim = self.distance_dim, self.time_dim
        if time_dim is None:
            names = [x for x in ("time", "lag_time") if x in meta.dims]
            if not names:
                msg = "patch is missing required coordinates: ('time' or 'lag_time')"
                raise PatchCoordinateError(msg)
            time_dim = names[0]
        check_patch_coords(meta, dims=(time_dim, dist_dim))
        extras = [x for x in meta.dims if x not in (time_dim, dist_dim)]
        if clash := sorted({"velocity", "frequency"} & set(extras)):
            msg = (
                f"dispersion_phase_shift creates {clash}, which the patch "
                "already has as dimensions; rename them with rename_coords."
            )
            raise ParameterError(msg)
        meta_cop, _ = ConvertUnits(**{dist_dim: "m"}).get_metadata(meta)
        meta_cop, transpose = Transpose(
            dims=(*extras, dist_dim, time_dim)
        ).get_metadata(meta_cop)
        dist = meta_cop.coords.get_array(dist_dim)
        if self.source_distance is not None:
            units = meta.get_coord(dist_dim).units
            dist = np.abs(dist - convert_units(self.source_distance, "m", units))
        time = meta_cop.coords.get_array(time_dim)
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
        # Neither dimension survives, so nothing riding them can.
        for dim in (time_dim, dist_dim):
            meta.coords.drop_associated(dim, "Transforming")
        # Keep the coordinates which lie only along the surviving dimensions.
        coords = {
            name: (dims, meta.coords.coord_map[name])
            for name, dims in meta.coords.dim_map.items()
            if dims and set(dims) <= set(extras)
        }
        coords |= dict(velocity=phase_velocities, frequency=w / (2 * np.pi))
        dims = [*extras, "velocity", "frequency"]
        out = meta.new(coords=coords, attrs=attrs, dims=dims, dtype=np.float64)
        data_units = (
            meta.attrs.data_units if self.normalize is None else "dimensionless"
        )
        units = SetUnits(data_units=data_units, velocity="m/s", frequency="Hz")
        out, _ = units.get_metadata(out)
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
        # Each channel's kept bins, channels first: (channel, *extras, frequency).
        # astype copies, so no channel's full spectrum outlives its turn.
        fft_d = np.stack(
            [
                nft.fft(data[..., i, :], n=nf * step)[..., ::step].astype(complex)
                for i in range(nchan)
            ]
        )
        if self.normalize == "phase":
            fft_d = np.divide(
                fft_d, abs(fft_d), out=np.zeros_like(fft_d), where=abs(fft_d) != 0
            )
            fft_d[np.isnan(fft_d)] = 0
        fft_d = fft_d[..., live]
        preamb = (1j * np.outer(dist, w)).reshape(nchan, *[1] * (fft_d.ndim - 2), -1)
        stacks = [abs(sum(np.exp(preamb / v) * fft_d)) for v in self.phase_velocities]
        return np.stack(stacks, axis=-2) / nchan


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
