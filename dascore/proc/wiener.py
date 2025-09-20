"""
Wiener filtering functionality for noise reduction.
"""

from __future__ import annotations

from scipy.signal import wiener

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.patch import get_dim_axis_value, patch_function


def _get_wiener_window_size(patch, kwargs, samples):
    """Get the size of the wiener filter window."""
    aggs = get_dim_axis_value(patch, kwargs=kwargs, allow_multiple=True)
    size = [1] * patch.data.ndim
    for name, axis, val in aggs:
        coord = patch.get_coord(name, require_evenly_sampled=True)
        samps = coord.get_sample_count(val, samples=samples)
        if samps < 1:
            msg = (
                f"wiener filter must have at least 1 sample along each dimension. "
                f"{name} has {samps} samples. Try increasing its value."
            )
            raise ParameterError(msg)
        # Wiener filter requires odd window sizes for best results
        if (samps % 2 != 1) and not samples:
            samps += 1
        if (samps % 2 != 1) and samples:
            msg = (
                "For optimal Wiener filtering, dimension windows should be odd "
                f"but {name} has a value of {samps} samples. Consider using an "
                "odd number."
            )
            import warnings

            warnings.warn(msg, UserWarning)

        size[axis] = samps
    return tuple(size)


@patch_function()
def wiener_filter(
    patch: PatchType,
    *,
    noise=None,
    samples=False,
    **kwargs,
):
    """
    Apply a Wiener filter to reduce noise in the patch data.

    The Wiener filter is an adaptive filter that reduces noise while preserving
    signal features. It estimates the local mean and variance within a sliding
    window and uses these statistics to determine the optimal filtering.

    Parameters
    ----------
    patch
        Input patch.
    noise
        The noise-power to use. If None, noise is estimated as the average
        of the local variance of the input.
    samples
        If True, values specified by kwargs are in samples not coordinate units.
    **kwargs
        Used to specify the window sizes for each dimension. Each selected
        dimension must be evenly sampled.

    Returns
    -------
    Patch with noise-reduced data.

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> # Get an example patch and add noise
    >>> patch = dc.get_example_patch()
    >>> noisy_data = patch.data + np.random.normal(0, 0.1, patch.data.shape)
    >>> noisy_patch = patch.update(data=noisy_data)
    >>>
    >>> # Apply Wiener filter along time dimension with 5-sample window
    >>> filtered = noisy_patch.wiener_filter(time=5, samples=True)
    >>> assert filtered.data.shape == patch.data.shape
    >>>
    >>> # Apply filter with custom noise parameter
    >>> filtered_custom = noisy_patch.wiener_filter(time=5, samples=True, noise=0.01)
    >>> assert isinstance(filtered_custom, dc.Patch)
    >>>
    >>> # Apply filter along multiple dimensions
    >>> filtered_2d = noisy_patch.wiener_filter(time=5, distance=3, samples=True)
    >>> assert filtered_2d.data.shape == patch.data.shape

    Notes
    -----
    This implementation uses scipy.signal.wiener which performs adaptive
    noise reduction based on local statistics within the specified window.
    """
    # Use kwargs to specify per-dimension window sizes
    if not kwargs:
        msg = (
            "Must specify dimension-specific window sizes via kwargs "
            "(e.g., time=5, distance=3)"
        )
        raise ParameterError(msg)

    size = _get_wiener_window_size(patch, kwargs, samples)
    # Convert size tuple to format expected by scipy.signal.wiener
    # scipy.signal.wiener expects None for single value or a list of ints
    # (no None values)
    if all(s == 1 for s in size):
        # All dimensions have size 1, use default behavior
        filtered_data = wiener(patch.data, mysize=None, noise=noise)
    else:
        # Use the size tuple directly (no None values)
        filtered_data = wiener(patch.data, mysize=size, noise=noise)

    return patch.update(data=filtered_data)
