"""Processing for applying roller operations."""

import numpy as np
from dascore.utils.patch import get_dim_value_from_kwargs


class PatchRoller:
    """A class to apply roller operations to patches."""

    def __init__(self, patch, *, window=None, step=None, center=False, **kwargs):
        self.patch = patch

        self.center = center
        self.kwargs = kwargs
        dim, axis, value = get_dim_value_from_kwargs(patch, self.kwargs)
        self.axis = axis
        coord = patch.get_coord(dim)
        if dim == "time":
            window = coord.get_sample_count(value)
            step = 1 if step is None else coord.get_sample_count(step)
        else:
            window = value
            step = 1 if step is None else step
        self.window = window
        self.step = step

        assert (
            window < patch.data.shape[axis] or step < patch.data.shape[axis]
        ), "Window or step size is larger than total number \
        of samples in the specified coordinate."

    def apply(self, function):
        patch = self.patch
        window_size = self.window
        step_size = self.step
        center = self.center
        axis = self.axis

        time_samples = len(patch.coords["time"])
        distance_samples = len(patch.coords["distance"])

        if axis == 0:
            if patch.dims[0] == "time":
                iter_range = range(0, time_samples, step_size)
                shape = (int(time_samples / step_size), int(distance_samples))
                out = np.empty(shape)
                for j, k in enumerate(iter_range):
                    if k + window_size > time_samples:
                        out[j, :] = np.full((1, distance_samples), np.nan)
                    else:
                        out[j, :] = function(
                            patch.data[k : k + window_size, :], axis=axis
                        )
            elif patch.dims[0] == "distance":
                iter_range = range(0, distance_samples, step_size)
                shape = (int(distance_samples / step_size), int(time_samples))
                out = np.empty(shape)
                for j, k in enumerate(iter_range):
                    if k + window_size > distance_samples:
                        out[j, :] = np.full((1, time_samples), np.nan)
                    else:
                        out[j, :] = function(
                            patch.data[k : k + window_size, :], axis=axis
                        )
        elif axis == 1:
            if patch.dims[0] == "time":
                iter_range = range(0, distance_samples, step_size)
                shape = (time_samples, int(distance_samples / step_size))
                out = np.empty(shape)
                for j, k in enumerate(iter_range):
                    if k + window_size > distance_samples:
                        out[:, j] = np.full((time_samples, 1), np.nan)
                    else:
                        out[:, j] = function(
                            patch.data[:, k : k + window_size], axis=axis
                        )
            elif patch.dims[0] == "distance":
                iter_range = range(0, time_samples, step_size)
                shape = (int(distance_samples), int(time_samples / step_size))
                out = np.empty(shape)
                for j, k in enumerate(iter_range):
                    if k + window_size > time_samples:
                        out[:, j] = np.full((distance_samples), np.nan)
                    else:
                        out[:, j] = function(
                            patch.data[:, k : k + window_size], axis=axis
                        )

        if center:
            out = np.roll(out, int(window_size / 2), axis=axis)

        return out

    def mean(self):
        return self.apply(np.mean)

    def median(self):
        return self.apply(np.median)

    def min(self):
        return self.apply(np.min)

    def max(self):
        return self.apply(np.max)


def rolling(patch, step=None, center=False, **kwargs):
    """
    Apply a rolling function along a specified dimension
    with a specified factor as the window size.

    Parameters
    ----------
    step
        The window is evaluated at every step result,
        equivalent to slicing at every step.
        If the step argument is not None or 1,
        the result will have a different shape than the input.
    center
        If False, set the window labels as the right edge of the window index.
        If True, set the window labels as the center of the window index.
    **kwargs
        Used to pass dimension and factor.
        For example `time=10` represents window size of
        10*(default unit) along the time axis.

    Examples
    --------
    # Simple example for rolling mean function
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> mean_values = patch.rolling(time=10, step=10)
    """
    return PatchRoller(patch, step=step, center=center, **kwargs)
