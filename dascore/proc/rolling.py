"""Processing for applying roller operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

import dascore as dc
from dascore.constants import samples_arg_description
from dascore.exceptions import ParameterError
from dascore.utils.docs import compose_docstring
from dascore.utils.patch import (
    _maybe_add_history_str,
)
from dascore.utils.pd import rolling_df
from dascore.utils.window import resolve_window

rolling_apply_description = """
Apply a function over the specified moving window.

Parameters
----------
function
    The function which is applied.
*args
    Positional arguments passed to function.
**kwargs
    Keyword arguments passed to function.

Examples
--------
>>> import numpy as np
>>> import dascore as dc
>>>
>>> patch = dc.get_example_patch()
>>> out = patch.rolling(time=100, samples=True).apply(np.percentile, 80)
"""


@dataclass(frozen=True, slots=True)
class _PatchRollerInfo:
    """
    A dataclass for storing info on rolling operation.

    Should be subclassed to implement rolling methods. This is an ephemeral
    internal object created on every rolling call, so it is a plain dataclass
    rather than a validated model.
    """

    patch: Any  # cant set to patch due to circular import
    window: int
    step: int
    dim: str
    axis: int
    center: bool
    roll_hist: str = ""

    def get_coords(self):
        """
        Get the new coordinates for "rolled" patch.

        Accounts for centered or non-centered coordinates. If the window
        length is even, the first half value is used.
        """
        # Without a step the dimension is unchanged; reuse the coord manager.
        if self.step == 1:
            return self.patch.coords
        coord = self.patch.get_coord(self.dim)[:: self.step]
        return self.patch.coords.update(**{self.dim: coord})

    def _get_attrs_with_apply_history(self, func_or_str):
        """Get new attrs that has history from apply attached."""
        if callable(func_or_str):
            func_name = getattr(func_or_str, "__name__", "")
            hist_str = f"{self.roll_hist}.apply({func_name})"
        else:
            hist_str = f"{self.roll_hist}.{func_or_str}()"
        return _maybe_add_history_str(self.patch.attrs, hist_str)

    def _new_patch(self, data, func_or_str):
        """Create the output patch from rolled data."""
        coords = self.get_coords()
        attrs = self._get_attrs_with_apply_history(func_or_str)
        return self.patch.update(data=data, coords=coords, attrs=attrs)


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
        """
        Pad the reduced array with NaNs and align it to the output coordinate.

        The NaNs go at the start of the axis, except when centering, which
        moves `num_nans // 2` of them to the end. This is done with a single
        allocation rather than a pad followed by a roll.
        """
        num_nans = 1 + (self.window - 2) // self.step
        if not num_nans:  # window of one sample; nothing to pad.
            return data
        shape = list(data.shape)
        shape[self.axis] += num_nans
        out = np.full(shape, np.nan, dtype=data.dtype)
        start = num_nans - num_nans // 2 if self.center else num_nans
        slicer = [slice(None, None)] * len(shape)
        slicer[self.axis] = slice(start, start + data.shape[self.axis])
        out[tuple(slicer)] = data
        if self.step == 1:
            assert out.shape == self.patch.data.shape
        return out

    @compose_docstring(apply_description=rolling_apply_description)
    def apply(self, function, *args, **kwargs):
        """
        {apply_description}

        Notes
        -----
        The provided function must accept an ``axis`` argument.
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
        trimmed_slide_view = slide_view[tuple(step_slice)]
        raw = function(trimmed_slide_view, *args, axis=-1, **kwargs)
        out = self._pad_roll_array(np.asarray(raw, dtype=np.float64))
        return self._new_patch(out, function)

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

    def _repack_patch(self, df, func_or_str):
        """Repack patch into dataframe."""
        data = df.values if not self.axis else df.T.values
        # get rid of extra dims if original data doesn't have them.
        if len(data.shape) != len(self.patch.data.shape):
            data = np.squeeze(data)
        return self._new_patch(data, func_or_str)

    def _call_rolling_func(self, name, *args, **kwargs):
        """Helper function for calling a rolling function."""
        rolling = self._get_rolling()
        df = getattr(rolling, name)(*args, **kwargs)
        return self._repack_patch(df, name)

    @compose_docstring(apply_description=rolling_apply_description)
    def apply(self, function, *args, **kwargs):
        """
        {apply_description}
        """
        df = self._get_rolling().apply(function, args=args, kwargs=kwargs)
        return self._repack_patch(df, function)

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
    overlap=None,
    **kwargs,
) -> _NumpyPatchRoller | _PandasPatchRoller:
    """
    Apply a rolling function along a specified dimension.

    See also the
    [rolling section of the processing tutorial](/tutorial/processing.qmd#rolling)
    and the [smoothing recipe](/recipes/smoothing.qmd).

    Parameters
    ----------
    patch
        The patch to apply the rolling function to.
    step
        Evaluate every nth result, like slicing the output. This changes the
        output length and is mutually exclusive with ``overlap``.
    center
        Label each window by its center rather than its right edge.
    engine
        ``"numpy"`` uses ``sliding_window_view`` and ``"pandas"`` uses
        ``pandas.rolling``. None selects pandas only when the step is below 10
        and the squeezed patch has fewer than two dimensions; otherwise it
        selects NumPy. Explicit pandas supports at most two dimensions and
        raises `ParameterError` above that.
    samples
        {sample_explanation}
    overlap
        Window overlap in coordinate units, samples, or percent. When given,
        ``step = window - overlap``; percentages are relative to the window.
    **kwargs
        Dimension and window size, such as ``time=10``.

    Notes
    -----
    Rolling follows Pandas [DataFrame.rolling](
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html)
    semantics. With no step, the output retains the input shape. When
    ``center=False``, incomplete leading windows are NaN; when
    ``center=True``, incomplete windows at both edges are NaN. Use
    [`Patch.dropna`](`dascore.Patch.dropna`) to remove them. A step downsamples
    that output. For example, the mean of ``[0, 1, 2, 3, 4, 5]`` is:

    - window 2: ``[NaN, 0.5, 1.5, 2.5, 3.5, 4.5]``
    - window 3: ``[NaN, NaN, 1, 2, 3, 4]``
    - window 3, step 2: ``[NaN, 1, 3]``
    - window 3, step 3: ``[NaN, 2]``

    ``apply`` receives the rolling dimension as the last axis of each window;
    custom functions should reduce that axis. Extra arguments passed to
    ``apply`` are forwarded to the function.

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

    resolved = resolve_window(
        patch,
        kwargs,
        samples=samples,
        overlap=overlap,
        step=step,
        allow_multiple=False,
        enforce_lt_coord=True,
    )
    dim, axis, window = resolved.dims[0], resolved.axes[0], resolved.size[0]
    value = kwargs[dim]
    step = None if resolved.stride is None else resolved.stride[0]
    # No overlap or step given means every sample gets a window.
    step = 1 if step is None else step
    cls = _get_engine(step, engine, patch)
    roll_hist = (
        f"rolling({dim}={value}, step={step}, overlap={overlap}, "
        f"center={center}, engine={engine})"
    )
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
