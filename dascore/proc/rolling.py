"""Processing for applying roller operations."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

import dascore as dc
from dascore.constants import samples_arg_description
from dascore.exceptions import ParameterError
from dascore.utils.docs import compose_docstring
from dascore.utils.models import DascoreBaseModel
from dascore.utils.patch import get_dim_axis_value
from dascore.utils.pd import rolling_df


class _PatchRollerInfo(DascoreBaseModel):
    """
    A dataclass for storing info on rolling operation.

    Should be subclassed to implement rolling methods.
    """

    patch: Any  # cant set to patch due to circular import
    window: int
    step: int
    dim: str
    axis: int
    center: bool
    roll_hist: str = ""
    func_kwargs: dict = Field(default_factory=dict)

    def get_coords(self):
        """
        Get the new coordinates for "rolled" patch.

        Accounts for centered or non-centered coordinates. If the window
        length is even, the first half value is used.
        """
        coord = self.patch.get_coord(self.dim)
        if self.step > 1:
            coord = coord[:: self.step]
        return self.patch.coords.update(**{self.dim: coord})

    def _get_attrs_with_apply_history(self, func_or_str):
        """Get new attrs that has history from apply attached."""
        new_history = list(self.patch.attrs.history)
        if callable(func_or_str):
            func_name = getattr(func_or_str, "__name__", "")
            hist_str = f"{self.roll_hist}.apply({func_name})"
        else:
            hist_str = f"{self.roll_hist}.{func_or_str}()"
        new_history.append(hist_str)
        attrs = self.patch.attrs.update(history=new_history, coords={})
        return attrs


class _NumpyPatchRoller(_PatchRollerInfo):
    """A class to apply roller operations to patches."""

    def get_start_index(self):
        """
        Get the start index to account for non-zero step size.

        This only applies for numpy engine.
        """
        wsize = self.window - 1
        out = np.ceil(wsize / self.step) * self.step - wsize
        return int(out)

    def _pad_roll_array(self, data):
        """Pad."""
        num_nans = 1 + (self.window - 2) // self.step
        pad_width = [(0, 0)] * len(data.shape)
        pad_width[self.axis] = (num_nans, 0)
        padded = np.pad(data, pad_width, constant_values=np.nan)
        if self.step == 1:
            assert padded.shape == self.patch.data.shape
        if self.center:
            # roll array along axis to center
            padded = np.roll(padded, -(num_nans // 2), axis=self.axis)
        return padded

    def apply(self, function):
        """
        Apply a function over the specified moving window.

        Parameters
        ----------
        function
            The function which is applied. Must accept an axis argument.
        """
        # TODO look at replacing this with a call to `as_strided` that
        # accounts for strides.
        slide_view = np.lib.stride_tricks.sliding_window_view(
            self.patch.data,
            self.window,
            self.axis,
        )
        # get slice to account for step (stride)
        step_slice = [slice(None, None)] * len(self.patch.data.shape)
        step_slice.append(slice(None, None))
        # this accounts for NaNs that pad the start of the array.
        start = self.get_start_index()
        step_slice[self.axis] = slice(start, None, self.step)
        # apply function, then pad with NaNs and roll
        kwargs = self.func_kwargs
        trimmed_slide_view = slide_view[tuple(step_slice)]
        raw = function(trimmed_slide_view, axis=-1, **kwargs).astype(np.float64)
        out = self._pad_roll_array(raw)
        new_coords = self.get_coords()
        attrs = self._get_attrs_with_apply_history(function)
        return self.patch.update(data=out, coords=new_coords, attrs=attrs)

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


class _PandasPatchRoller(_PatchRollerInfo):
    """A class to apply pandas rolling operations."""

    def _get_df(self) -> pd.DataFrame:
        """Get the dataframe from patch data."""
        if len(self.patch.dims) > 2:
            msg = "Cannot use Pandas engine on patches with more than 2 dims."
            raise ParameterError(msg)
        df = pd.DataFrame(self.patch.data)
        return df

    def _get_rolling(self):
        """Get rolling."""
        df = self._get_df()
        roll = rolling_df(
            df=df,
            window=self.window,
            step=self.step,
            axis=self.axis,
            center=self.center,
        )
        return roll

    def _repack_patch(self, df, attrs=None):
        """Repack patch into dataframe."""
        data = df.values if not self.axis else df.T.values
        # get rid of extra dims if original data doesn't have them.
        if len(data.shape) != len(self.patch.data.shape):
            data = np.squeeze(data)
        coords = self.get_coords()
        return self.patch.update(data=data, coords=coords, attrs=attrs)

    def _call_rolling_func(self, name, *args, **kwargs):
        """Helper function for calling a rolling function."""
        rolling = self._get_rolling()
        df = getattr(rolling, name)(*args, **kwargs)
        attrs = self._get_attrs_with_apply_history(name)
        return self._repack_patch(df, attrs=attrs)

    def apply(self, func):
        df = self._get_rolling().apply(func, **self.func_kwargs)
        attrs = self._get_attrs_with_apply_history(func)
        return self._repack_patch(df, attrs=attrs)

    def mean(self):
        """Apply mean."""
        return self._call_rolling_func(name="mean")

    def median(self):
        """Apply median to moving window."""
        return self._call_rolling_func(name="median")

    def min(self):
        """Apply min to moving window."""
        return self._call_rolling_func(name="min")

    def max(self):
        """Apply max to moving window."""
        return self._call_rolling_func(name="max")

    def std(self):
        """Apply standard deviation to moving window."""
        return self._call_rolling_func(name="std")

    def sum(self):
        """Apply sum to moving window."""
        return self._call_rolling_func(name="sum")


@compose_docstring(sample_explanation=samples_arg_description)
def rolling(
    patch: dc.Patch,
    step=None,
    center=False,
    engine: Literal["numpy", "pandas", None] = None,
    samples=False,
    **kwargs,
) -> _NumpyPatchRoller | _PandasPatchRoller:
    """
    Apply a rolling function along a specified dimension.

    See also the
    [rolling section of the processing tutorial](/tutorial/processing.qmd#rolling)
    and the [smoothing recipe](/recipes/smoothing.qmd).

    Parameters
    ----------
    step
        The window is evaluated at every step result, equivalent to slicing
        at every step. If the step argument is not None, the result will
        have a different shape than the input. Default None.
    center
        If False, set the window labels as the right edge of the window index.
        If True, set the window labels as the center of the window index. Default False.
    engine
        Determines how the rolling operations are applied. If None, try to
        determine which will be fastest for a given step. Options are:
            "numpy" - which uses np.lib.stride_tricks.sliding_window_view.
            "pandas" - which uses pandas.rolling.
        If step < 10 samples, pandas is faster for all operations other than apply.
        If step > 10 samples, or `apply` is the desired rolling operation, numpy
        is probably better.
    samples
        {sample_explination}
    **kwargs
        Used to pass dimension and window size.
        For example `time=10` represents window size of
        10*(default unit) along the time axis.

    Notes
    -----
    Here we have included some useful notes for the rolling function.

    #### Note 1: Length of the output and NaN values

    Rolling is designed to behaves like Pandas [DataFrame.rolling](
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html)
    which has some important implications:

    - First, when step is not defined or set to 1, the output patch will have the
    same shape as the input patch. The consequence of this is that NaN values
    will appear at the start of the dimension. You can use
    [`patch.dropna`](`dascore.Patch.dropna`) to remove the NaN values.

    - Second, the step parameter is equivalent applying to the output along the
    specified dimension. For example, if step=2 the output of the chosen
    dimension will be 1/2 of the input length.

    Here are a few examples to help illustrate how rolling works.

    Consider a patch with a simple 1D array in the dimension "time":
        [0, 1, 2, 3, 4, 5]

    - If time = 2 * dt the output is
        [NaN, 0.5, 1.5, 2.5, 3.5, 4.5]

    - If time = 3 * dt the output is
        [NaN, NaN, 1.0, 2.0, 3.0, 4.0]

    - if time = 3 * dt and step = 2 * dt
        [NaN, 1.0, 3.0]

    - if time = 3 * dt and step = 3 * dt
        [NaN, 2.0]


    #### Note 2: Applying custom functions with rolling operation

    When `apply` is the desired rolling operation and we are interested to use our
    own function over a desired window, we need to define our finction the way that
    it performs the operation on the last axis of the sliced matrix

    Below is an example of applying a custom zero crossing rate function (zcr_std) with
    rolling operation for a window size of 100 samples and skipping every other samples.
    It applys the desired operation (which is multiplying every sample to its next
    sample to determine the zero crossings) on the last axis of the sliced
    frame (from rolling).


    ```python
    import numpy as np
    import dascore as dc


    def zcr_std(frame, axis=-1):
        '''Compute standard deviation of zero crossing rate using rolling function.'''
        zero_crossings = (frame[..., :-1] * frame[..., 1:]) < 0
        return np.std(zero_crossings, axis=axis)


    patch = dc.get_example_patch()
    zcr_patch = patch.rolling(time=100, step=2, samples=True).apply(zcr_std)
    ```

    Examples
    --------
    >>> import dascore as dc
    >>>
    >>> # Simple example for rolling mean function
    >>> patch = dc.get_example_patch()
    >>>
    >>> # apply rolling over 1 second with 0.5 step
    >>> mean_patch = patch.rolling(time=1, step=0.5).mean()
    >>>
    >>> # drop nan at the start of the time axis.
    >>> out = mean_patch.dropna("time")
    """

    def _get_engine(step, engine, patch):
        """Get the engine."""
        engines = {"numpy": _NumpyPatchRoller, "pandas": _PandasPatchRoller}
        if cls := engines.get(engine):
            return cls
        if step < 10 and len(patch.squeeze().dims) < 2:
            return _PandasPatchRoller
        return _NumpyPatchRoller

    # get window sizes in samples
    dim, axis, value = get_dim_axis_value(patch, kwargs=kwargs)[0]
    roll_hist = f"rolling({dim}={value}, step={step}, center={center}, engine={engine})"
    coord = patch.get_coord(dim)
    window = coord.get_sample_count(value, samples=samples, enforce_lt_coord=True)
    step = (
        1
        if step is None
        else coord.get_sample_count(step, samples=samples, enforce_lt_coord=True)
    )
    if window == 0 or step == 0:
        msg = "Window or step size can't be zero. Use any positive values."
        raise ParameterError(msg)
    cls = _get_engine(step, engine, patch)
    out = cls(
        patch=patch,
        window=window,
        step=step,
        dim=dim,
        axis=axis,
        center=center,
        roll_hist=roll_hist,
    )
    return out
