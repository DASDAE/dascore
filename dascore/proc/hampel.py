"""
Functionality for Hampel despiking.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.patch import get_patch_window_size, patch_function


def _separable_median(data, size, mode):
    """Calculate the median along each dimension sequentially."""
    med = data

    # Apply 1D median filters along each dimension with size > 1
    for axis, window_size in enumerate(size):
        if window_size > 1:
            # Create size tuple for this dimension
            axis_size = [1] * len(size)
            axis_size[axis] = window_size
            med = median_filter(med, size=tuple(axis_size), mode=mode)

    return med


def _calculate_standard_median_and_mad(data, size, mode):
    """Calculate median and MAD using standard 2D median filters."""
    med = median_filter(data, size=size, mode=mode)
    abs_med_diff = np.abs(data - med)
    mad = median_filter(abs_med_diff, size=size, mode=mode)
    return med, abs_med_diff, mad


@patch_function()
def hampel_filter(
    patch: PatchType,
    *,
    threshold: float,
    samples=False,
    separable=False,
    **kwargs,
):
    """
    A Hampel filter implementation useful for removing spikes in data.

    Parameters
    ----------
    patch
        Input patch.
    threshold
        Outlier threshold in MAD units.
    samples
        If True, values specified by kwargs are in samples not coordinate units.
    separable
        If True, apply 1D median filters sequentially along each dimension
        instead of a true 2D median filter. This is much faster (3-4x speedup)
        but provides an approximation of the true 2D median. The approximation
        is usually good enough for spike removal purposes.
    **kwargs
        Used to specify the lengths of the filter in each dimension. Each
        selected dim must be evenly sampled and should represent a window
        with an odd number of samples.

    Warning
    -------
    Selecting windows with many samples can be *very* slow. It is recommended
    window size in each dimension be <10 samples. For larger windows, consider
    using `separable=True` for significantly faster processing (3-4x speedup)
    at the cost of a slight approximation.

    Returns
    -------
    Patch with outliers replaced by local median.

    Notes
    -----
    When samples=False, even window lengths are bumped to the next odd
    value to ensure a clean median calculation. When samples=True, an
    even sample count raises a ParameterError.

    See Also
    --------
    - Despiking recipe: docs/recipes/despiking.qmd

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> # Get an example patch and add artificial spikes
    >>> patch = dc.get_example_patch()
    >>> data = patch.data.copy()
    >>> data[10, 5] = 10  # Add a large spike
    >>> patch = patch.update(data=data)
    >>>
    >>> # Apply hampel filter along time dimension with 1.0 unit window
    >>> filtered = patch.hampel_filter(time=0.2, threshold=3.5)
    >>> assert filtered.data.shape == patch.data.shape
    >>> # The spike should be reduced
    >>> assert abs(filtered.data[10, 5]) < abs(patch.data[10, 5])
    >>>
    >>> # Apply filter along multiple dimensions:
    >>> filtered_2d = patch.hampel_filter(time=1.0, distance=5.0, threshold=3.5)
    >>> assert filtered_2d.data.shape == patch.data.shape
    >>>
    >>> # Specify hampel window in samples not coord units.
    >>> filtered_samps = patch.hampel_filter(
    ...     time=3, distance=3, samples=True, threshold=3.5
    ... )
    >>>
    """
    if threshold <= 0 or not np.isfinite(threshold):
        msg = "hampel_filter threshold must be finite and greater than zero"
        raise ParameterError(msg)
    # First build axis windows
    data = patch.data
    # For now we just hardcode mode as it is probably the only one that
    # makes sense in a DAS data context.
    mode = "reflect"
    size = get_patch_window_size(
        patch, kwargs, samples, require_odd=True, warn_above=10, min_samples=3
    )

    # Convert to float64 to avoid integer overflow/precision loss
    dataf = data.astype(np.float64, copy=False)

    # Local median and MAD via median filters.
    if separable and sum(s > 1 for s in size) >= 2:
        # Use separable filtering for multi-dimensional windows
        # This is faster but provides an approximation
        med = _separable_median(dataf, size, mode)
        abs_med_diff = np.abs(dataf - med)
        mad = _separable_median(abs_med_diff, size, mode)
    else:
        # Use standard 2D median filter (more accurate but slower)
        med, abs_med_diff, mad = _calculate_standard_median_and_mad(dataf, size, mode)
    # Handle mad values of 0 so denominator doesn't blow up.
    mad_safe = np.where(mad == 0.0, np.finfo(float).eps, mad)
    # Hampel test and replacement.
    thresholded = abs_med_diff / mad_safe
    out = np.where(thresholded > threshold, med, dataf)
    # Cast back to original dtype (round if original was integer)
    if np.issubdtype(data.dtype, np.integer):
        out = np.rint(out)
    out = out.astype(data.dtype, copy=False)
    return patch.update(data=out)
