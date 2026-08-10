"""
Numba kernels for the kurtosis transform.

These live apart from `dascore.transform.kurtosis` because decorating with
`maybe_numba_jit` imports numba, which is slow. `kurtosis` imports this module
when it is called so `import dascore` stays fast.
"""

from __future__ import annotations

import numpy as np

from dascore.utils.jit import maybe_numba_jit


@maybe_numba_jit
def _moving_sum(x: np.ndarray, nwin: int):
    """Moving sum along axis 0 using clipped centered windows."""
    npts = x.shape[0]
    left = nwin // 2
    right = nwin - left

    out = np.empty_like(x, dtype=np.float64)
    counts = np.empty(npts, dtype=np.float64)

    for i in range(npts):
        start = max(i - left, 0)
        stop = min(i + right, npts)
        counts[i] = stop - start

        for j in range(x.shape[1]):
            out[i, j] = np.sum(x[start:stop, j])

    return out, counts


@maybe_numba_jit
def _windowed_kurtosis(data: np.ndarray, nwin: int) -> np.ndarray:
    """Compute Pearson kurtosis in moving windows along axis 0."""
    s1, counts = _moving_sum(data, nwin)
    s2, _ = _moving_sum(data**2, nwin)
    s3, _ = _moving_sum(data**3, nwin)
    s4, _ = _moving_sum(data**4, nwin)

    out = np.empty_like(data, dtype=np.float64)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            count = counts[i]

            m1 = s1[i, j] / count
            m2 = s2[i, j] / count
            m3 = s3[i, j] / count
            m4 = s4[i, j] / count

            mu2 = m2 - m1**2
            mu4 = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4

            if mu2 > 0:
                out[i, j] = mu4 / mu2**2
            else:
                out[i, j] = np.nan

    return out


@maybe_numba_jit
def _recursive_kurtosis(
    data: np.ndarray, step: float, winlen: float, varx: np.ndarray
) -> np.ndarray:
    """Recursive pseudo-kurtosis after Langet et al.-style formulation."""
    c = 1.0 - step / winlen
    npts = data.shape[0]
    nchans = data.shape[1]

    out = np.empty_like(data, dtype=np.float64)
    mean_value = np.zeros(nchans, dtype=np.float64)
    var_value = np.zeros(nchans, dtype=np.float64)
    kurt_value = np.zeros(nchans, dtype=np.float64)

    for i in range(npts):
        for j in range(nchans):
            xi = data[i, j]

            mean_value[j] = c * mean_value[j] + (1.0 - c) * xi
            var_value[j] = c * var_value[j] + (1.0 - c) * (xi - mean_value[j]) ** 2

            if var_value[j] > varx[j]:
                norm_factor = var_value[j] ** 2
            else:
                norm_factor = varx[j] ** 2

            if norm_factor > 0:
                kurt_value[j] = (
                    c * kurt_value[j]
                    + (1.0 - c) * (xi - mean_value[j]) ** 4 / norm_factor
                )
            else:
                kurt_value[j] = np.nan
            out[i, j] = kurt_value[j]

    return out
