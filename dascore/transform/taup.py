"""Tau-p Patch transforms."""

from __future__ import annotations

from typing import Any

import numpy as np

from dascore.core.processor import PatchProcessor
from dascore.exceptions import ParameterError
from dascore.proc.coords import Transpose
from dascore.proc.units import ConvertUnits, SetUnits
from dascore.units import convert_units
from dascore.utils.time import to_float


class TauP(PatchProcessor):
    """
    Compute linear tau-p transform.

    The patch must have time and distance dimensions.

    Parameters
    ----------
    velocities
        NumPy array of velocities, in m/s if units are not attached,
        for which to compute slowness (p).

    Notes
    -----
    - Output will always be double the size of vels, with negative velocities
      (right-to-left) first, followed by positive velocities (left-to-right).

    - Uses linear interpolation in time

    Example
    -------
    ```{python}
    >>> import dascore as dc
    >>> import numpy as np
    >>>
    >>> patch = (
    ...    dc.get_example_patch('example_event_1')
    ... )

    >>> taup_patch = (
    ...     patch.taper(time=0.1)
    ...     .pass_filter(time=(..., 300))
    ...     .tau_p(np.arange(1000,6000,10))
    ...     .transpose('time','slowness')
    ...     .sort_coords('slowness')
    ... )
    >>> ax = taup_patch.viz.waterfall(show=False, cbar=False)
    >>> _ = taup_patch.viz.waterfall(ax=ax)
    """

    velocities: Any

    required_dims = ("time", "distance")
    data_type = "tau_p"

    def get_metadata(self, meta):
        """Return the tau-p metadata, and the gather's axes, spacing and slownesses."""
        meta_cop, _ = ConvertUnits(distance="m", time="s").get_metadata(meta)
        meta_cop, transpose = Transpose(dims=("distance", "time")).get_metadata(
            meta_cop
        )
        dist = meta_cop.get_coord("distance")
        time = meta_cop.get_coord("time", require_evenly_sampled=True)
        velocities = self.velocities
        if np.any(velocities <= 0):
            msg = "Input velocities must be positive."
            raise ParameterError(msg)
        if not np.all(np.diff(velocities) > 0):
            raise ParameterError("Input velocities must be monotonically increasing.")
        # Handle unit conversions if needed.
        p_vals = 1.0 / convert_units(velocities, to_units="m/s")
        slowness = np.concatenate((-np.flip(p_vals), p_vals))
        attrs = meta.attrs.update(category="taup")
        coords = dict(slowness=slowness, time=time)
        dims = ["slowness", "time"]
        out = meta.new(coords=coords, attrs=attrs, dims=dims, dtype=np.float64)
        out, _ = SetUnits(slowness="s/m", time="s").get_metadata(out)
        # Chooses code version based on whether distance between channels
        # is uniform or not
        uniform = dist.evenly_sampled
        return out, {
            "axes": transpose.get("axes"),
            "uniform": uniform,
            "distance": dist.step if uniform else dist.values,
            "dt": to_float(time.step),
            "p_vals": p_vals,
        }

    def numpy_kernel(self, data, *, axes, uniform, distance, dt, p_vals):
        """Return the transform of the gather, distance first."""
        # Imported here (not at module scope) to keep numba out of `import dascore`.
        from dascore.transform._taup_kernels import (  # noqa: PLC0415
            _jit_taup_general,
            _jit_taup_uniform,
        )

        data = data if axes is None else np.permute_dims(data, axes)
        func = _jit_taup_uniform if uniform else _jit_taup_general
        return func(data, distance, dt, p_vals)[1]
