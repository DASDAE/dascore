"""Tau-p transformations"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

import numpy as np

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.misc import optional_import
from dascore.utils.patch import patch_function


@cache
def _make_jit_taup_uniform():
    """Build a numba version of tau-p, see documentation for tau-p"""
    numba = optional_import("numba")

    @numba.jit(nopython=True)
    def jit_taup_uniform(data, dx, dt, p_vals):
        (nx, nt) = data.shape
        n_slo = p_vals.size
        two_sided_p_vals = np.concatenate((-np.flip(p_vals), p_vals))
        taup = np.zeros(shape=(2 * n_slo, nt))

        for ip in numba.prange(n_slo):
            pdx = p_vals[ip] * dx
            for tau in range(nt):
                for ix in range(nx):
                    samp_val = tau + pdx * ix / dt
                    ind = int(samp_val)
                    if ind + 1 < nt:
                        taup[ip + n_slo, tau] += (1.0 + ind - samp_val) * data[
                            ix, ind
                        ] + (samp_val - ind) * data[ix, ind + 1]
                        taup[n_slo - 1 - ip, tau] += (1.0 + ind - samp_val) * data[
                            nx - 1 - ix, ind
                        ] + (samp_val - ind) * data[nx - 1 - ix, ind + 1]
        return two_sided_p_vals, taup

    return jit_taup_uniform


@cache
def _make_jit_taup_general():
    numba = optional_import("numba")

    @numba.jit(nopython=True)
    def jit_taup_general(data, distance, dt, p_vals):
        (nx, nt) = data.shape
        n_slo = p_vals.size
        two_sided_p_vals = np.concatenate((-np.flip(p_vals), p_vals))
        taup = np.zeros(shape=(2 * n_slo, nt))
        mod_distance = distance - distance[0]

        for ip in numba.prange(n_slo):
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
                        taup[n_slo - 1 - ip, tau] += (
                            1.0 + ind_neg - samp_val_neg
                        ) * data[nx - 1 - ix, ind_neg] + (
                            samp_val_neg - ind_neg
                        ) * data[nx - 1 - ix, ind_neg + 1]
        return two_sided_p_vals, taup

    return jit_taup_general


@patch_function(required_dims=("time", "distance"))
def tau_p(
    patch: PatchType,
    vels: Sequence[float],
) -> PatchType:
    """
    Compute linear tau-p transform.

    Parameters
    ----------
    patch
        Patch to transform. Has to have dimensions of time and distance.
    vels
        NumPY array of velocities, in m/s, for which to compute slowness (p).

    Notes
    -----
    - Output will always be double the size of vels, with negative velocities
      (right-to-left) first, followed by positive velocities (left-to-right)

    - Uses linear interpolation in time

    Example
    --------
    ```{python}
    import dascore as dc
    import numpy as np

    patch = (
        dc.get_example_patch('example_event_1')
    )

    taup_patch = patch.taper(time=0.1).pass_filter(time=(..., 300)).
        tau_p(np.arange(1000,6000,10)).transpose('time','slowness')
    ax = taup_patch.viz.waterfall(show=False,cmap=None)
    taup_patch.viz.waterfall(show=True, ax=ax)
    ```
    """
    patch_cop = patch.convert_units(distance="m", time="s").transpose(
        "distance", "time"
    )
    dist = patch_cop.coords.get_array("distance")
    time = patch_cop.coords.get_array("time")

    ## This is causing me problem with get_example_2, but I don't
    ## know how to fix. I used the same code in dispersion.py which
    ## is also a problem... would be useful to have a function that
    ## returns dt in seconds.
    dt = (time[1] - time[0]) / np.timedelta64(1, "s")

    if np.any(vels <= 0):
        msg = "Input velocities must be positive."
        raise ParameterError(msg)

    if not np.all(np.diff(vels) > 0):
        raise ParameterError("Input velocities must be monotonically increasing.")

    # Chooses code version based on whether distance between channels
    # is uniform or not

    dist_bet_chans = np.diff(dist)
    if np.all(dist_bet_chans[0] == dist_bet_chans):
        numba_taup = _make_jit_taup_uniform()
        slowness, tau_p_data = numba_taup(patch.data, dist_bet_chans[0], dt, 1.0 / vels)
    else:
        numba_taup = _make_jit_taup_general()
        slowness, tau_p_data = numba_taup(patch.data, dist, dt, 1.0 / vels)

    attrs = patch.attrs.update(category="taup")
    coords = dict(slowness=slowness, time=time)

    tau_p_patch = patch.new(
        data=tau_p_data, coords=coords, attrs=attrs, dims=["slowness", "time"]
    )
    return tau_p_patch.set_units(slowness="s/m", time="s")
