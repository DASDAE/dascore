"""Processing for applying roller operations."""

import numpy as np

from dascore.utils.patch import get_dim_value_from_kwargs
from dascore.exceptions import ParameterError


class PatchRoller:
    """
    A class to apply roller operations to patches.

    Parameters
    ----------
    patch
        The patch to apply rolling function(s) to.
    step
        The step between rolling windows (aka stride). Units are also supported.
        Defaults to 1.
    center
        If True, center the moving window else the label value occurs at the end
        of the window.
    **kwargs
        Used to specify the coordinate.
    """

    def __init__(self, patch, *, window=None, step=None, center=False, **kwargs):
        self.patch = patch

        self.center = center
        self.kwargs = kwargs
        dim, axis, value = get_dim_value_from_kwargs(patch, self.kwargs)
        self.axis = axis
        self.dim = dim
        self.coord = patch.get_coord(dim)
        if dim == "time":
            window = self.coord.get_sample_count(value)
            step = 1 if step is None else self.coord.get_sample_count(step)
        else:
            window = self.coord.get_sample_count(value)
            step = 1 if step is None else self.coord.get_sample_count(step)
        self.window = window
        self.step = step

        if self.window > len(self.coord) or self.step > len(self.coord):
            msg = (
                "Window or step size is larger than total number of samples in "
                "the specified dimension."
            )
            raise ParameterError(msg)

        if self.window == 0 or self.step == 0:
            msg = "Window or step size can't be zero."
            raise ParameterError(msg)

        self._roll_hist = f"rolling({dim}={value}, step={step}, center={center})"

    def apply(self, function):
        patch = self.patch
        window_size = self.window
        step_size = self.step
        center = self.center
        axis = self.axis

        time_samples = len(patch.coords["time"])
        distance_samples = len(patch.coords["distance"])
        sampling_interval = patch.attrs["time_step"]
        sampling_rate = 1 / (sampling_interval / np.timedelta64(1, "s"))
        channel_spacing = patch.attrs["distance_step"]

        if axis == 0:
            if patch.dims[0] == "time":
                iter_range = range(0, time_samples, step_size)
                shape = (
                    np.ceil(time_samples / step_size).astype(int),
                    int(distance_samples),
                )
                out = np.empty(shape)
                for j, k in enumerate(iter_range):
                    if k < window_size - 1:
                        out[j, :] = np.full((1, distance_samples), np.nan)
                    else:
                        out[j, :] = function(
                            patch.data[k - window_size + 1 : k + 1, :], axis=axis
                        )
                    new_attrs = dict(patch.attrs)
                    samples = np.array(patch.coords["time"])[::step_size]
                    new_coords = {x: patch.coords[x] for x in patch.dims}
                    new_coords["time"] = samples
                    new_attrs["time_step"] = step_size / sampling_rate
                    new_attrs["min_time"] = np.min(samples)
                    new_attrs["max_time"] = np.max(samples)
            elif patch.dims[0] == "distance":
                iter_range = range(0, distance_samples, step_size)
                shape = (
                    np.ceil(distance_samples / step_size).astype(int),
                    int(time_samples),
                )
                out = np.empty(shape)
                for j, k in enumerate(iter_range):
                    if k < window_size - 1:
                        out[j, :] = np.full((1, time_samples), np.nan)
                    else:
                        out[j, :] = function(
                            patch.data[k - window_size + 1 : k + 1, :], axis=axis
                        )
                    new_attrs = dict(patch.attrs)
                    samples = np.array(patch.coords["distance"])[::step_size]
                    new_coords = {x: patch.coords[x] for x in patch.dims}
                    new_coords["distance"] = samples
                    new_attrs["distance_step"] = step_size / channel_spacing
                    new_attrs["min_distance"] = np.min(samples)
                    new_attrs["max_distance"] = np.max(samples)
        elif axis == 1:
            if patch.dims[0] == "time":
                iter_range = range(0, distance_samples, step_size)
                shape = (
                    time_samples,
                    np.ceil(distance_samples / step_size).astype(int),
                )
                out = np.empty(shape)
                for j, k in enumerate(iter_range):
                    if k < window_size - 1:
                        out[:, j] = np.full(time_samples, np.nan)
                    else:
                        out[:, j] = function(
                            patch.data[:, k - window_size + 1 : k + 1], axis=axis
                        )
                    new_attrs = dict(patch.attrs)
                    samples = np.array(patch.coords["distance"])[::step_size]
                    new_coords = {x: patch.coords[x] for x in patch.dims}
                    new_coords["distance"] = samples
                    new_attrs["distance_step"] = step_size / channel_spacing
                    new_attrs["min_distance"] = np.min(samples)
                    new_attrs["max_distance"] = np.max(samples)
            elif patch.dims[0] == "distance":
                iter_range = range(0, time_samples, step_size)
                shape = (
                    int(distance_samples),
                    np.ceil(time_samples / step_size).astype(int),
                )
                out = np.empty(shape)
                for j, k in enumerate(iter_range):
                    if k < window_size - 1:
                        out[:, j] = np.full(distance_samples, np.nan)
                    else:
                        out[:, j] = function(
                            patch.data[:, k - window_size + 1 : k + 1], axis=axis
                        )
                    new_attrs = dict(patch.attrs)
                    samples = np.array(patch.coords["time"])[::step_size]
                    new_coords = {x: patch.coords[x] for x in patch.dims}
                    new_coords["time"] = samples
                    new_attrs["time_step"] = step_size / sampling_rate
                    new_attrs["min_time"] = np.min(samples)
                    new_attrs["max_time"] = np.max(samples)
        if center:
            out = np.roll(out, int(window_size / 2), axis=axis)

        rolling_patch = patch.new(
            data=out, attrs=new_attrs, dims=patch.dims, coords=new_coords
        )

        return rolling_patch

    def mean(self):
        """Apply mean to moving window."""
        return self.apply(np.mean)

    def median(self):
        """Apply median to moving window."""
        return self.apply(np.median)

    def min(self):
        """Apply min to moving window."""
        return self.apply(np.min)

    def max(self):
        """Apply max to moving window."""
        return self.apply(np.max)

    def std(self):
        """Apply standard deviation to moving window."""
        return self.apply(np.std)

    def sum(self):
        """Apply sum to moving window."""
        return self.apply(np.sum)


def rolling(patch, step=None, center=False, **kwargs):
    """
    Apply a rolling function along a specified dimension
    with a specified factor as the window size and return
    a new patch with updated attributes.

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
        For example `time=10*s` represents window size of
        10 seconds along the time axis.

    Examples
    --------
    # Simple example for rolling mean function
    >>> import numpy as np

    >>> import dascore as dc
    >>> from dascore.units import s


    >>> patch = dc.get_example_patch()
    >>> patch_mean = patch.rolling(time=1*s, step=0.5*s).mean()

    >>> # We can drop the nan values at the beginning of array
    >>> rolling_mean_values = patch_mean.data
    >>> valid_data = ~np.isnan(rolling_mean_values).any(axis=0)
    >>> rolling_mean_values_no_nan = rolling_mean_values[:, valid_data]

    >>> new_attrs = dict(patch_mean.attrs)
    >>> samples = np.array(patch_mean.coords["time"])[valid_data]
    >>> new_coords = {x: patch_mean.coords[x] for x in patch.dims}
    >>> new_coords["time"] = samples
    >>> new_attrs["min_time"] = np.min(samples)
    >>> new_attrs["max_time"] = np.max(samples)

    >>> patch_mean_no_nan = patch.new(
    ...     data=rolling_mean_values_no_nan,
    ...     attrs=new_attrs,
    ...     dims=patch.dims,
    ...     coords=new_coords
    ... )
    """
    return PatchRoller(patch, step=step, center=center, **kwargs)
