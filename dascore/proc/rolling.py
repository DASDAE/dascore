"""Processing for applying roller operations."""
from __future__ import annotations

from functools import cache

import numpy as np

import dascore as dc
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

    def __init__(self, patch: dc.Patch, *, step=None, center=False, **kwargs):
        self.patch = patch
        self.center = center
        # get window sizes in samples
        dim, axis, value = get_dim_value_from_kwargs(patch, kwargs)
        self.axis = axis
        self.dim = dim
        self.coord = patch.get_coord(dim)
        self.window = self.coord.get_sample_count(value)
        self.step = 1 if step is None else self.coord.get_sample_count(step)
        if self.window > len(self.coord) or self.step > len(self.coord):
            msg = (
                "Window or step size is larger than total number of samples in "
                "the specified dimension."
            )
            raise ParameterError(msg)
        self._roll_hist = f"rolling({dim}={value}, step={step}, center={center})"

    @cache
    def get_coords(self):
        """
        Get the new coordinates for "rolled" patch.

        Accounts for centered or non-centered coordinates. If the window
        length is even, the first half value is used.
        """
        coord = self.coord
        if self.step > 1:
            coord = self.coord[:: self.step]
        return self.patch.coords.update_coords(**{self.dim: coord})

    def _get_attrs_with_apply_history(self, func):
        """Get new attrs that has history from apply attached."""
        new_history = list(self.patch.attrs.history)
        func_name = getattr(func, "__name__", "")
        new_history.append(f"{self._roll_hist}.apply({func_name})")
        attrs = self.patch.attrs.update(history=new_history, coords={})
        return attrs

    def _pad_roll_array(self, data):
        """Pad"""
        num_nans = 1 + (self.window - 2) // self.step
        pad_width = [(0, 0)] * len(data.shape)
        pad_width[self.axis] = (num_nans, 0)
        padded = np.pad(data, pad_width, constant_values=np.NaN)
        if self.step == 1:
            assert padded.shape == self.patch.data.shape
        if self.center:
            # roll array along axis to center
            padded = np.roll(padded, self.window // 2, axis=self.axis)
        return padded

    def apply(self, function):
        """
        Apply a function over the specified moving window.

        Parameters
        ----------
        function
            The function which is applied. This must accept a numpy array
            with the same number of dimensions as input patch, then return an
            array with the same shape except axis is removed.
        """
        # TODO look at replacing this with a call to `as_strided` that
        # accounts for strides.
        array = np.lib.stride_tricks.sliding_window_view(
            self.patch.data,
            self.window,
            self.axis,
        )
        # get slice to account for step (stride)
        step_slice = [slice(None, None)] * len(self.patch.data.shape)
        step_slice.append(slice(None, None))
        # this accounts for NaNs that pad the start of the array.
        start = (self.step - ((self.window - 2) % self.step)) % self.step
        # start = (self.window - 1) % self.step
        step_slice[self.axis] = slice(start, None, self.step)
        # apply function, then pad with zeros and roll
        raw = function(array[tuple(step_slice)], axis=-1).astype(np.float_)
        out = self._pad_roll_array(raw)
        new_coords = self.get_coords()
        attrs = self._get_attrs_with_apply_history(function)
        return self.patch.new(data=out, coords=new_coords, attrs=attrs)

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


def rolling(patch: dc.Patch, step=None, center=False, **kwargs) -> PatchRoller:
    """
    Apply a rolling function along a specified dimension.

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
        Used to pass dimension and window size.
        For example `time=10` represents window size of
        10*(default unit) along the time axis.

    Examples
    --------
    >>> # Simple example for rolling mean function
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> mean_patch = patch.rolling(time=10, step=5).mean()
    >>> # note this will contain


    Notes
    -----
    This class behaves like pandas.rolling
    (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html)
    which has some important implications.

    First, when step is not defined or set to 1, the output patch will have the
    same shape as the input patch. The consequence of this is that NaN values
    will appear at the start of the dimension. You can use
    [`patch.dropna`](`dascore.Patch.dropna`) to remove the NaN values.

    Second, the step parameter is equivalent applying striding along the
    specified dimension.

    To understand how this works, consider a patch with a simple 1D array:
        a = [0, 1, 2, 3, 4, 5]
    If window = 2 the output is
        [NaN, 0.5, 1.5, 2.5, 3.5, 4.5]
    If window = 3 the output is
        [NaN, NaN, 1.0, 2.0, 3.0, 4.0]
    if window = 3 and step = 2
        [NaN, 1.0, 3.0]
    if window = 3 and step = 3
        [NaN, 2.0]
    """
    return PatchRoller(patch, step=step, center=center, **kwargs)
