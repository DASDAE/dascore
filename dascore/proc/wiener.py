"""
Wiener filtering functionality for noise reduction.
"""

from __future__ import annotations

from scipy.signal import wiener

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.patch import get_patch_window_size, patch_function


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
        dimension must be evenly sampled. It works best when the window samples
        are odd.

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
    if not kwargs:
        msg = (
            "To use wiener_filter you must specify dimension-specific window "
            "sizes via kwargs (e.g., time=5, distance=3)"
        )
        raise ParameterError(msg)

    size = get_patch_window_size(patch, kwargs, samples, min_samples=1)
    filtered_data = wiener(patch.data, mysize=size, noise=noise)
    return patch.update(data=filtered_data)
