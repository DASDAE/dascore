"""
Patch function for kurtosis transform
"""

from __future__ import annotations

import numpy as np

from dascore.constants import PatchType, samples_arg_description
from dascore.exceptions import ParameterError
from dascore.utils.docs import compose_docstring
from dascore.utils.patch import get_dim_axis_value, patch_function
from dascore.utils.time import to_float


def _get_window_samples(coord, dim, value, samples: bool) -> int:
    """Convert a window length along a coordinate to a validated sample count."""
    if coord.reverse_sorted:
        coord, _ = coord.sort()
    nwin = coord.get_sample_count(value, samples=samples)
    if nwin < 2:
        msg = (
            f"kurtosis window of {value} along dimension '{dim}' spans "
            f"{nwin} sample(s) but must span at least 2. The coordinate "
            f"step is {coord.step}."
        )
        raise ParameterError(msg)
    return nwin


@patch_function()
@compose_docstring(sample_explanation=samples_arg_description)
def kurtosis(
    patch: PatchType,
    samples: bool = False,
    recursive: bool = True,
    **kwargs,
) -> PatchType:
    """
    Compute kurtosis along a patch dimension.
    Background seismic noise is approximately Gaussian. A seismic arrival
    (especially a P-wave onset) produces a transient, impulsive signal with
    a sharply peaked amplitude distribution. Kurtosis — the normalized 4th
    statistical moment — becomes strongly positive during such impulsive
    arrivals.

    Here, kurtosis is determined in a window whose length is given as a
    dimension keyword argument (e.g. ``time=0.5``). We then determine
    kurtosis of the amplitude distribution in that window.
    Higher kurtosis thus indicates high amplitude outliers. This in turn
    can be interpreted as a signal arrival.


    Parameters
    ----------
    patch
        Input DASCore patch.
    samples
        {sample_explanation}
    recursive
        If True, use recursive pseudo-kurtosis: Instead of computing kurtosis
        in a sliding window (computationally expensive for continuous data),
        @langet2014 propose a recursive formulation. This acts like an
        exponentially weighted moving estimator, so the algorithm updates continuously
        without storing long windows of data.
        If False, the common kurtosis calculation is used
    **kwargs
        Used to specify the dimension and window length, e.g. ``time=0.5``
        computes kurtosis in 0.5 second windows along the time dimension.
        Units are also supported, e.g. ``distance=10 * dascore.units.get_unit('m')``.

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
    >>> k = p.kurtosis(time=0.002)
    >>> ax = k.viz.waterfall(cmap = 'inferno')

    2) To better understand how kurtosis works, we replace the data with
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
    >>> k = modi.kurtosis(time=0.002)
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
    # Imported here (not at module scope) to keep numba out of `import dascore`.
    from dascore.transform._kurtosis_kernels import (  # noqa: PLC0415
        _recursive_kurtosis,
        _windowed_kurtosis,
    )

    dim, _, winlen = get_dim_axis_value(patch, kwargs=kwargs)[0]
    orig_dims = patch.dims
    patch_t = patch.transpose(dim, ...)

    data = np.asarray(patch_t.data, dtype=float)
    orig_shape = data.shape

    data_2d = data.reshape(orig_shape[0], -1)

    coord = patch_t.get_coord(dim, require_evenly_sampled=True)
    step = abs(to_float(coord.step))
    nwin = _get_window_samples(coord, dim, winlen, samples)

    if recursive:
        varx = np.var(data_2d, axis=0)
        out = _recursive_kurtosis(data_2d, step=step, winlen=nwin * step, varx=varx)
    else:
        out = _windowed_kurtosis(data_2d, nwin=nwin)

    out = out.reshape(orig_shape)

    return (
        patch_t.new(data=out)
        .transpose(*orig_dims)
        .update(attrs={"data_type": "kurtosis", "data_units": ""})
    )
