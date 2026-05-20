"""
Patch function for kurtosis transform
"""

from __future__ import annotations

import numpy as np

from dascore.constants import PatchType
from dascore.utils.patch import patch_function
from dascore.utils.time import to_float


@patch_function()
def kurtosis(
    patch: PatchType,
    winlen: float,
    dim: str = "time",
    recursive: bool = True,
) -> PatchType:
    """
    Compute kurtosis along a patch dimension. Note that for best results
    normalize data, or convert to nano-strain(rate)


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
        Window length in seconds.
    dim
        Dimension along which to compute kurtosis. Defaults to ``"time"``.
    recursive
        If True, use recursive pseudo-kurtosis: Instead of computing kurtosis
        in a sliding window (computationally expensive for continuous data),
        the @langet2014 propose a recursive formulation. This acts like an
        exponentially weighted moving estimator, so the algorithm updates
        continuously without storing long windows of data.

    Returns
    -------
    PatchType
        A new patch with kurtosis traces.

    Example
    --------
    >>> import dascore as dc
    >>>
    >>> p = dc.examples.get_example_patch('example_event_2')
    >>>
    >>> k = p.kurtosis(winlen = 0.002,dim = 'time')
    >>> ax = k.viz.waterfall(cmap = 'magma')

    """

    def _validate_window(winlen: float, dt: float) -> int:
        """Convert window length in seconds to samples and validate."""
        if winlen <= 0:
            raise ValueError("winlen must be positive.")

        nwin = int(round(winlen / dt))
        if nwin < 2:
            raise ValueError("winlen is too small for the sampling interval.")

        return nwin

    def _moving_sum(x: np.ndarray, nwin: int) -> np.ndarray:
        """
        Moving sum along axis 0 using cumulative sums.

        Returns sums over centered windows with clipped boundaries, matching the
        original script's edge behavior approximately.
        """
        npts = x.shape[0]
        left = nwin // 2
        right = nwin - left

        starts = np.arange(npts) - left
        stops = np.arange(npts) + right

        starts = np.clip(starts, 0, npts)
        stops = np.clip(stops, 0, npts)

        csum = np.cumsum(x, axis=0)
        csum = np.concatenate(
            [np.zeros((1, *x.shape[1:]), dtype=x.dtype), csum], axis=0
        )

        return csum[stops, ...] - csum[starts, ...], (stops - starts)

    def _windowed_kurtosis(data: np.ndarray, nwin: int) -> np.ndarray:
        """
        Compute Pearson kurtosis in moving windows along axis 0.

        Uses raw moments from cumulative sums for speed.
        """
        x1 = data
        x2 = data**2
        x3 = data**3
        x4 = data**4

        s1, counts = _moving_sum(x1, nwin)
        s2, _ = _moving_sum(x2, nwin)
        s3, _ = _moving_sum(x3, nwin)
        s4, _ = _moving_sum(x4, nwin)

        counts = counts.reshape((-1,) + (1,) * (data.ndim - 1))

        m1 = s1 / counts
        m2 = s2 / counts
        m3 = s3 / counts
        m4 = s4 / counts

        # Central moments
        mu2 = m2 - m1**2
        mu4 = m4 - 4 * m1 * m3 + 6 * (m1**2) * m2 - 3 * m1**4

        out = np.divide(
            mu4,
            mu2**2,
            out=np.full_like(mu4, np.nan, dtype=float),
            where=mu2 > 0,
        )
        return out

    def _recursive_kurtosis(data: np.ndarray, dt: float, winlen: float) -> np.ndarray:
        """
        Recursive pseudo-kurtosis after Langet et al.-style formulation.

        Acts along axis 0.
        """
        fsamp = 1.0 / dt
        c = 1.0 - 1.0 / (fsamp * winlen)

        npts = data.shape[0]
        out = np.empty_like(data, dtype=float)

        # Per-channel initialization
        varx = np.std(data, axis=0)
        mean_value = np.zeros(data.shape[1:], dtype=float)
        var_value = np.zeros(data.shape[1:], dtype=float)
        kurt_value = np.zeros(data.shape[1:], dtype=float)

        varx2 = varx**2

        for i in range(npts):
            xi = data[i, ...]

            mean_value = c * mean_value + (1.0 - c) * xi
            var_value = c * var_value + (1.0 - c) * (xi - mean_value) ** 2

            norm_factor = np.where(var_value > varx, var_value**2, varx2)
            kurt_value = (
                c * kurt_value + (1.0 - c) * (xi - mean_value) ** 4 / norm_factor
            )
            out[i, ...] = kurt_value

        return out

    patch_t = patch.transpose(dim, ...)
    data = np.asarray(patch_t.data, dtype=float)

    coord = patch_t.get_coord(dim)
    dt = to_float(coord.step)
    if dt is None or dt <= 0:
        raise ValueError(
            f"Coordinate step for dim={dim!r} must be defined and positive."
        )

    if recursive:
        out = _recursive_kurtosis(data, dt=dt, winlen=winlen)
    else:
        nwin = _validate_window(winlen, dt)
        out = _windowed_kurtosis(data, nwin=nwin)

    return (
        patch_t.new(data=out)
        .transpose(*patch.dims)
        .update(attrs={"data_type": "Kurtosis", "data_units": ""})
    )
