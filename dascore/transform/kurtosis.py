"""
Patch function for kurtosis transform
"""

from __future__ import annotations

import numpy as np

from dascore.constants import PatchType
from dascore.utils.jit import maybe_numba_jit
from dascore.utils.patch import patch_function
from dascore.utils.time import to_float


def _validate_window(winlen: float, step: float) -> int:
    """Convert window length in seconds to samples and validate."""
    if winlen <= 0:
        raise ValueError("winlen must be positive.")
    nwin = round(winlen / step)
    if nwin < 2:
        raise ValueError("winlen is too small for the sampling interval.")
    return nwin


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


@patch_function()
def kurtosis(
    patch: PatchType,
    winlen: float,
    dim: str = "time",
    recursive: bool = True,
) -> PatchType:
    """
    Compute kurtosis along a patch dimension.
    Background seismic noise is approximately Gaussian. A seismic arrival
    (especially a P-wave onset) produces a transient, impulsive signal with
    a sharply peaked amplitude distribution. Kurtosis — the normalized 4th
    statistical moment — becomes strongly positive during such impulsive
    arrivals.

    Here, kurtosis is determined in a window of lenght "winlen". We
    then determine kurtosis of the amplitude distribution in that window.
    Higher kurtosis thus indicates high amplitude outliers. This in turn
    can be interpreted as a signal arrival.


    Parameters
    ----------
    patch
        Input DASCore patch.
    winlen
        Window length used to calculate the kurtosis in.
    dim
        Dimension along which to compute kurtosis. Defaults to ``"time"``.
    recursive
        If True, use recursive pseudo-kurtosis: Instead of computing kurtosis
        in a sliding window (computationally expensive for continuous data),
        @langet2014 propose a recursive formulation. This acts like an
        exponentially weighted moving estimator, so the algorithm updates continuously
        without storing long windows of data.
        If False, the common kurtosis calculation is used

    Returns
    -------
    PatchType
        A new patch with kurtosis traces.

    Examples
    --------
    1) Kurtosis of example event
    >>> import dascore as dc
    >>>
    >>> p = dc.examples.get_example_patch('example_event_2')
    >>>
    >>> k = p.kurtosis(winlen = 0.002,dim = 'time')
    >>> ax = k.viz.waterfall(cmap = 'inferno')

    2) To better understand how kutosis works, we replace the data with
    normal-distributed random data. We then amplify a block of those
    data. The modified data has a broader tail, since more high-amplitude
    values are in the dataset. The kurtosis picks the onset accurately.

    >>> import dascore as dc
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>>
    >>> p = dc.examples.get_example_patch('example_event_2')
    >>>
    >>> # replace event data with normal-distributed random values
    >>> rng = np.random.default_rng()
    >>> data = rng.normal(loc=0, scale=1, size=p.data.shape)
    >>> data0 = data.copy() # original
    >>> data[:,300:450] = data[:, 300:450]*3 #modified
    >>>
    >>> orig = p.update(data=data0)
    >>> modi = p.update(data=data)
    >>>
    >>> # calculate kurtosis on modified data
    >>> k = modi.kurtosis(winlen = 0.002, dim = 'time')
    >>>
    >>> fix, axs = plt.subplots(2,2, figsize=(10,6), layout='constrained')
    >>> ax = orig.viz.waterfall(cmap = 'RdBu', ax=axs[0,0])
    >>> _ = ax.set_title('Original')
    >>>
    >>> ax = modi.viz.waterfall(cmap = 'RdBu', ax=axs[0,1])
    >>> _ = ax.set_title('Modified')
    >>>
    >>> ax = k.viz.waterfall(cmap = 'inferno_r', scale=[0, .4], ax=axs[1,1])
    >>> _ = ax.set_title('Kurtosis')
    >>>
    >>> # plot histograms of both datasets. Note the modified has broader tail!
    >>> _ = axs[1,0].hist(data.ravel(),  100, alpha=0.5, label='Modified', density=True)
    >>> _ = axs[1,0].hist(data0.ravel(), 100, alpha=0.5, label='Original', density=True)
    >>> _ = axs[1,0].legend(loc='upper right')
    >>> _ = axs[1,0].grid('on')
    >>> _ = axs[1,0].set_title('Amplitude Distributions')
    >>> _ = axs[1,0].set_xlabel('Amplitude')
    >>> _ = axs[1,0].set_ylabel('Probability of occurrence')
    """
    orig_dims = patch.dims
    patch_t = patch.transpose(dim, ...)

    data = np.asarray(patch_t.data, dtype=float)
    orig_shape = data.shape

    data_2d = data.reshape(orig_shape[0], -1)

    step = abs(to_float(patch_t.get_coord(dim, require_evenly_sampled=True).step))

    nwin = _validate_window(winlen, step)

    if recursive:
        varx = np.var(data_2d, axis=0)
        out = _recursive_kurtosis(data_2d, step=step, winlen=winlen, varx=varx)
    else:
        out = _windowed_kurtosis(data_2d, nwin=nwin)

    out = out.reshape(orig_shape)

    return (
        patch_t.new(data=out)
        .transpose(*orig_dims)
        .update(attrs={"data_type": "kurtosis", "data_units": ""})
    )
