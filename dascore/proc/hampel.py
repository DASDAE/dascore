"""
Functionality for Hampel despiking.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.moving import move_median
from dascore.utils.patch import get_patch_window_size, patch_function


def _separable_median(data, size, mode, out):
    """
    Calculate the median along each dimension sequentially using bottleneck
    (optimized for speed).
    """
    # Start by copying input data to output buffer
    np.copyto(out, data)
    current = out

    # Apply 1D median filters along each dimension with size > 1
    for axis, window_size in enumerate(size):
        if window_size > 1:
            # Use unified moving window interface with automatic engine selection
            filtered = move_median(
                current, window_size, axis=axis, engine="auto", mode=mode
            )
            np.copyto(current, filtered)

    return current


def _calculate_standard_median_and_mad(data, size, mode):
    """Calculate median and MAD using standard 2D median filters."""
    med = median_filter(data, size=size, mode=mode)
    abs_med_diff = np.abs(data - med)
    mad = median_filter(abs_med_diff, size=size, mode=mode)
    return med, abs_med_diff, mad


def _hampel_separable(dataf, size, mode, threshold):
    """Apply hampel filter using separable median filtering (faster approximation)."""
    # Pre-allocate buffers to reduce memory allocations
    med = np.empty_like(dataf)
    abs_med_diff = np.empty_like(dataf)
    mad = np.empty_like(dataf)

    # Calculate median using pre-allocated buffer
    _separable_median(dataf, size, mode, out=med)

    # Calculate absolute difference using out parameter for efficiency
    np.subtract(dataf, med, out=abs_med_diff)
    np.abs(abs_med_diff, out=abs_med_diff)

    # Calculate MAD using pre-allocated buffer
    _separable_median(abs_med_diff, size, mode, out=mad)

    # Handle zero MAD values in-place to avoid creating mad_safe array
    mad[mad == 0.0] = np.finfo(dataf.dtype).eps

    # Reuse abs_med_diff buffer for thresholded calculation
    np.true_divide(abs_med_diff, mad, out=abs_med_diff)

    # Final comparison using in-place assignment (abs_med_diff now contains
    # thresholded values).
    outlier_mask = abs_med_diff > threshold
    dataf[outlier_mask] = med[outlier_mask]

    return dataf


def _hampel_non_separable(dataf, size, mode, threshold):
    """
    Apply hampel filter using standard 2D median filtering
    (more accurate but slower).
    """
    # Use standard 2D median filter
    med, abs_med_diff, mad = _calculate_standard_median_and_mad(dataf, size, mode)

    # Handle mad values of 0 so denominator doesn't blow up
    mad_safe = np.where(mad == 0.0, np.finfo(dataf.dtype).eps, mad)

    # Hampel test and replacement
    thresholded = abs_med_diff / mad_safe

    # Use in-place assignment for consistency
    outlier_mask = thresholded > threshold
    dataf[outlier_mask] = med[outlier_mask]

    return dataf


@patch_function()
def hampel_filter(
    patch: PatchType,
    *,
    threshold: float = 10.0,
    samples=False,
    approximate=True,
    **kwargs,
):
    """
    A Hampel filter implementation useful for removing spikes in data.

    Parameters
    ----------
    patch
        Input patch.
    threshold
        Outlier threshold in MAD units. Default is 10.0.
    samples
        If True, values specified by kwargs are in samples not coordinate units.
    approximate
        If True, use fast approximation algorithms for improved performance.
        This applies 1D median filters sequentially along each dimension
        instead of a true 2D median filter, providing a 3-4x speedup.
        The approximation is usually good enough for spike removal purposes.
    **kwargs
        Used to specify the lengths of the filter in each dimension. Each
        selected dim must be evenly sampled and should represent a window
        with an odd number of samples.

    Warning
    -------
    Selecting windows with many samples can be *very* slow. It is recommended
    window size in each dimension be <10 samples.

    Returns
    -------
    Patch with outliers replaced by local median.

    Notes
    -----
    When samples=False, even window lengths are bumped to the next odd
    value to ensure a clean median calculation. When samples=True, an
    even sample count raises a ParameterError.

    **Edge Handling:**
    - Edge effects may differ slightly between modes due to different padding
      strategies based on the patch's dimensionality and use of `approximate`
      parameter.

    **Performance:**
    - `approximate=True` provides 3-4x speedup over exact calculations
    - Installing `bottleneck` package can further improve performance (~50%)
      which applies to both approximate and exact modes.

    See Also
    --------
    - [Despiking recipe](`docs/recipes/despiking.qmd`)

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
    >>> # Apply hampel filter along time dimension with 0.2 unit window
    >>> filtered = patch.hampel_filter(time=0.2, threshold=3.5)
    >>> assert filtered.data.shape == patch.data.shape
    >>> # The spike should be reduced
    >>> assert abs(filtered.data[10, 5]) < abs(patch.data[10, 5])
    >>>
    >>> # Apply filter along multiple dimensions using samples and
    >>> # default threshold.
    >>> filtered_2d = patch.hampel_filter(time=5, distance=5, samples=True)
    >>> assert filtered_2d.data.shape == patch.data.shape
    >>>
    >>> # Use exact median calculations (slower, more accurate)
    >>> filtered_exact = patch.hampel_filter(
    ...     time=5, distance=5, samples=True, approximate=False
    ... )
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
    # Need to convert ints to float for calculations to avoid roundoff error.
    # There were issues using np.issubdtype not working so this uses kind.
    is_int = data.dtype.kind in {"i", "u"}
    dataf = data.copy() if not is_int else data.astype(np.float32)
    # Apply hampel filtering using appropriate method
    if approximate:
        dataf = _hampel_separable(dataf, size, mode, threshold)
    else:
        # Use standard 2D median filter (more accurate but slower)
        dataf = _hampel_non_separable(dataf, size, mode, threshold)
    # Cast back to original dtype (round if original was integer)
    if np.issubdtype(data.dtype, np.integer):
        dataf = np.rint(dataf)
    out = dataf.astype(data.dtype, copy=False)
    return patch.update(data=out)
