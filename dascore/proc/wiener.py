"""
Wiener filtering functionality for noise reduction.
"""

from __future__ import annotations

from typing import Any, ClassVar

from dascore.exceptions import ParameterError
from dascore.proc.filter import _WindowFilter
from dascore.utils.imports import lazy_import

wiener = lazy_import("scipy.signal", "wiener")


class WienerFilter(_WindowFilter):
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

    noise: Any = None
    samples: Any = False

    _positional_fields = ()
    _window_rules: ClassVar[dict[str, Any]] = {}

    def get_metadata(self, meta):
        """Return the window, once one is known to be given."""
        if not self.model_extra:
            msg = (
                "To use wiener_filter you must specify dimension-specific window "
                "sizes via kwargs (e.g., time=5, distance=3)"
            )
            raise ParameterError(msg)
        return super().get_metadata(meta)

    def numpy_kernel(self, data, *, size, axes):
        """Return the Wiener-filtered data."""
        return wiener(data, mysize=size, noise=self.noise)
