"""Processing for applying roller operations."""

import numpy as np
from dascore.utils.patch import get_dim_value_from_kwargs


class PatchRoller:
    """A class to apply roller operations to patches"""

    def __init__(self, patch, *, window=None, step=None, center=False, **kwargs):
        self.patch = patch
        self.window = window
        self.step = step
        self.center = center
        self.kwargs = kwargs

    def mean(self):
        patch = self.patch
        window_size = self.window
        step_size = self.step
        center = self.center
        axis = self.kwargs

        total_samples = patch.data.shape[0]
        mean_values = np.empty((int(total_samples / step_size), patch.data.shape[1]))

        if center:
            for j, k in enumerate(range(0, total_samples, step_size)):
                mean_values[j, :] = np.mean(
                    patch.data[k - int(window_size / 2) : k + int(window_size / 2), :],
                    axis=axis,
                )
                # need to take care of edges?
        else:
            for j, k in enumerate(range(0, total_samples, step_size)):
                mean_values[j, :] = np.mean(
                    patch.data[k : k + window_size, :], axis=axis
                )

        return mean_values


def rolling(patch, *, window=None, step=None, center=False, **kwargs):
    """
    nice docs
    """
    dim, axis, value = get_dim_value_from_kwargs(patch, kwargs)
    # probably other things I am missing.
    return PatchRoller(patch, window=window, step=step, center=center, axis=axis)
