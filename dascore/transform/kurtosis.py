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

    Seismic arrivals are more impulsive than approximately Gaussian background
    noise, so their windowed amplitude distribution has higher kurtosis. This
    makes kurtosis useful for detecting arrivals, especially P-wave onsets.

    Parameters
    ----------
    patch
        Input DASCore patch.
    samples
        {sample_explanation}
    recursive
        Use the recursive pseudo-kurtosis of @langet2014, an exponentially
        weighted estimator that avoids storing sliding windows. False computes
        ordinary windowed kurtosis.
    **kwargs
        Dimension and window length, such as ``time=0.5`` or
        ``distance=10 * dascore.units.m``.

    Returns
    -------
    PatchType
        A new patch with kurtosis traces.

    Examples
    --------
    >>> import dascore as dc
    >>> import numpy as np
    >>> patch = dc.get_example_patch("example_event_2")
    >>> kurtosis = patch.kurtosis(time=0.002)
    >>> _ = kurtosis.viz.waterfall(cmap="inferno")
    >>>
    >>> # Amplify a block of Gaussian noise to create an impulsive onset.
    >>> rng = np.random.default_rng()
    >>> data = rng.normal(size=patch.shape)
    >>> data[:, 300:450] *= 3
    >>> synthetic = patch.update(data=data)
    >>> onset = synthetic.kurtosis(time=0.002)
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
