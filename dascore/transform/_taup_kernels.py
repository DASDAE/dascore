"""
Numba kernels for the tau-p transform.

These live apart from `dascore.transform.taup` because decorating with
`maybe_numba_jit` imports numba, which is slow. `tau_p` imports this module
when it is called so `import dascore` stays fast.
"""

from __future__ import annotations

import numpy as np

from dascore.utils.jit import maybe_numba_jit


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

    for ip in numba.prange(n_slo):  # noqa  # ty: ignore[unresolved-reference]
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

    for ip in numba.prange(n_slo):  # noqa  # ty: ignore[unresolved-reference]
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
