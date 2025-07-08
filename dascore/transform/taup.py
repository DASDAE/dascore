"""Tau-p Patch transforms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.units import convert_units
from dascore.utils.jit import maybe_numba_jit
from dascore.utils.patch import patch_function
from dascore.utils.time import to_float


@maybe_numba_jit(nopython=True)
def _jit_taup_uniform(data, dx, dt, p_vals):
    """
    A numba version of uniform Tau-P transform.

    See [`tau_p`](`dascore.transform.taup.tau_p`) for details.
    """
    (nx, nt) = data.shape
    n_slo = p_vals.size
    two_sided_p_vals = np.concatenate((-np.flip(p_vals), p_vals))
    taup = np.zeros(shape=(2 * n_slo, nt))

    for ip in numba.prange(n_slo):  # noqa
        pdx = p_vals[ip] * dx
        for tau in range(nt):
            for ix in range(nx):
                samp_val = tau + pdx * ix / dt
                ind = int(samp_val)
                if ind + 1 < nt:
                    taup[ip + n_slo, tau] += (1.0 + ind - samp_val) * data[ix, ind] + (
                        samp_val - ind
                    ) * data[ix, ind + 1]
                    taup[n_slo - 1 - ip, tau] += (1.0 + ind - samp_val) * data[
                        nx - 1 - ix, ind
                    ] + (samp_val - ind) * data[nx - 1 - ix, ind + 1]
    return two_sided_p_vals, taup


@maybe_numba_jit(nopython=True)
def _jit_taup_general(data, distance, dt, p_vals):
    """
    A numba version of general Tau-P transform.

    See [`tau_p`](`dascore.transform.taup.tau_p`) for details.
    """
    (nx, nt) = data.shape
    n_slo = p_vals.size
    two_sided_p_vals = np.concatenate((-np.flip(p_vals), p_vals))
    taup = np.zeros(shape=(2 * n_slo, nt))
    mod_distance = distance - distance[0]

    for ip in numba.prange(n_slo):  # noqa
        p = p_vals[ip]
        for tau in range(nt):
            for ix in range(nx):
                samp_val_pos = tau + p * mod_distance[ix] / dt
                ind_pos = int(samp_val_pos)
                samp_val_neg = (
                    tau + p * (mod_distance[-1] - mod_distance[nx - 1 - ix]) / dt
                )
                ind_neg = int(samp_val_pos)
                if ind_pos + 1 < nt:
                    taup[ip + n_slo, tau] += (1.0 + ind_pos - samp_val_pos) * data[
                        ix, ind_pos
                    ] + (samp_val_pos - ind_pos) * data[ix, ind_pos + 1]
                if ind_neg + 1 < nt:
                    taup[n_slo - 1 - ip, tau] += (1.0 + ind_neg - samp_val_neg) * data[
                        nx - 1 - ix, ind_neg
                    ] + (samp_val_neg - ind_neg) * data[nx - 1 - ix, ind_neg + 1]
    return two_sided_p_vals, taup


@patch_function(required_dims=("time", "distance"))
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
    >>> ax = taup_patch.viz.waterfall(show=False, cmap=None)
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
