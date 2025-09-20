"""
Functionality for Hampel despiking.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.patch import get_dim_axis_value, patch_function


def _get_hampel_window_size(patch, kwargs, samples):
    """Get the size of the hampel operator."""
    aggs = get_dim_axis_value(patch, kwargs=kwargs, allow_multiple=True)
    size = [1] * len(aggs)
    for name, axis, val in aggs:
        coord = patch.get_coord(name, require_evenly_sampled=True)
        samps = coord.get_sample_count(val, samples=samples)
        if samps < 3:
            msg = (
                f"hampel must have at least 3 samples along each dimension. "
                f"{name} has {samps} samples. Try increasing its value."
            )
            raise ParameterError(msg)
        # Just to make sure the median is a nice value, we need an odd number
        # of samples. Just bump if samples wasn't specified, else raise.
        if (samps % 2 == 1) and not samples:
            samps += 1
        if (samps % 2 == 1) and samples:
            msg = (
                "For clean median calculation in hampel function, dimension "
                f"windows must be odd but {name} has a value of {samps} samples."
            )
            raise ParameterError(msg)
    return tuple(size)


@patch_function()
def hampel_filter(
    patch: PatchType,
    *,
    mad_threshold: float = 3.5,
    samples=False,
    **kwargs,
):
    """
    A Hampel filter implementation useful for removing spikes in data.

    Parameters
    ----------
    patch
        Input patch.
    mad_threshold
        Outlier threshold in MAD units.
    samples
        If True, values specified by kwargs are in samples not coordinate units.
    **kwargs
        Used to specify the lengths of the filter in each dimension. Each
        selected dim must be evenly sampled.

    Returns
    -------
    Patch with outliers replaced by local median.
    """
    # First build axis windows
    data = patch.data
    # For now we just hardcode mode as it is probably the only one that
    # makes sense in a DAS data context.
    mode = "reflect"
    size = _get_hampel_window_size(patch, kwargs, samples)
    # Local median and MAD via median filters.
    med = median_filter(data, size=size, mode=mode)
    abs_med_diff = np.abs(data - med)
    mad = median_filter(abs_med_diff, size=size, mode=mode)
    # Handle mad values of 0 so denominator doesn't blow up.
    mad = np.where(mad == 0.0, np.finfo(float).eps, mad)
    # Hampel test and replacement.
    thresholded = abs_med_diff / mad
    out = np.where(thresholded > mad_threshold, med, data)
    return patch.update(data=out)
