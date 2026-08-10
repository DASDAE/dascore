"""Tau-p Patch transforms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.units import convert_units
from dascore.utils.patch import patch_function
from dascore.utils.time import to_float


@patch_function(required_dims=("time", "distance"), data_type="tau_p")
def tau_p(
    patch: PatchType,
    velocities: NDArray[np.floating],
) -> PatchType:
    """
    Compute linear tau-p transform.

    Parameters
    ----------
    patch
        Patch to transform. Has to have dimensions of time and distance.
    velocities
        NumPY array of velocities, in m/s if units are not attached,
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
    ... )
    >>> ax = taup_patch.viz.waterfall(show=False, cbar=False)
    >>> _ = taup_patch.viz.waterfall(ax=ax)
    """
    patch_cop = patch.convert_units(distance="m", time="s").transpose(
        "distance", "time"
    )
    dist = patch_cop.get_coord("distance")
    time = patch_cop.get_coord("time", require_evenly_sampled=True)
    dt = to_float(time.step)

    if np.any(velocities <= 0):
        msg = "Input velocities must be positive."
        raise ParameterError(msg)

    if not np.all(np.diff(velocities) > 0):
        raise ParameterError("Input velocities must be monotonically increasing.")

    # Handle unit conversions if needed.
    velocities = convert_units(velocities, to_units="m/s")

    # Imported here (not at module scope) to keep numba out of `import dascore`.
    from dascore.transform._taup_kernels import (  # noqa: PLC0415
        _jit_taup_general,
        _jit_taup_uniform,
    )

    # Chooses code version based on whether distance between channels
    # is uniform or not
    if dist.evenly_sampled:
        func = _jit_taup_uniform
        dist_val = dist.step
    else:
        func = _jit_taup_general
        dist_val = dist.values

    slowness, tau_p_data = func(patch.data, dist_val, dt, 1.0 / velocities)

    attrs = patch.attrs.update(category="taup")
    coords = dict(slowness=slowness, time=time)

    tau_p_patch = patch.new(
        data=tau_p_data, coords=coords, attrs=attrs, dims=["slowness", "time"]
    )
    return tau_p_patch.set_units(slowness="s/m", time="s")
