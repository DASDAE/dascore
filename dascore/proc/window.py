"""
Windows of a patch, and what is done in them.

[`Patch.window`](`dascore.Patch.window`) declares a tiling of a patch and
computes nothing: which dimensions are windowed, how long each window is,
and how far each one advances. It returns a
[`PatchWindow`](`dascore.proc.window.PatchWindow`), and a call on that is
one operation, which gives back a patch:

- [`apply`](`dascore.proc.window.PatchWindow.apply`) hands the windows to a
  function and blends what it returns back into a patch shaped like the
  input, or returns the windows stacked;
- a reduction -- `mean`, `median`, `min`, `max`, `std`, `sum`, or any
  function through [`reduce`](`dascore.proc.window.PatchWindow.reduce`) --
  gives one value for each window.

Examples
--------
>>> import numpy as np
>>> import dascore as dc
>>> patch = dc.get_example_patch()
>>>
>>> # A moving average over 11 samples, shaped like the patch.
>>> smooth = patch.window(time=11, samples=True).mean()
>>> assert smooth.shape == patch.shape
>>>
>>> # The same windows, half a window apart, each scaled to unit RMS.
>>> def agc(tiles):
...     rms = np.sqrt(np.mean(tiles**2, axis=-1, keepdims=True))
...     return tiles / np.where(rms > 0, rms, 1)
>>> gained = patch.window(time=0.5).apply(agc)
>>> assert gained.shape == patch.shape
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import ConfigDict, Field

from dascore.core.processor import PatchProcessor
from dascore.exceptions import ParameterError
from dascore.proc.rolling import _pandas_agrees, _rolling_pandas
from dascore.proc.tile_apply import TileApply
from dascore.utils.patch import _call_str
from dascore.utils.window import resolve_window

__all__ = ("PatchWindow", "WindowReduce")

_ENDS = ("pad", "drop")
# How many samples of a window come before the sample which labels it.
_BEFORE: dict[str, Callable[[int], int]] = {
    "center": lambda size: size // 2,
    "right": lambda size: size - 1,
    "left": lambda size: 0,
}
# The reductions spelled by name, which a faster engine may answer for.
_REDUCERS = {
    "mean": np.mean,
    "median": np.median,
    "min": np.min,
    "max": np.max,
    "std": np.std,
    "sum": np.sum,
}


def _options(**given) -> dict[str, Any]:
    """Return the window options a call set, for its history."""
    defaults = {"samples": False, "ends": "pad", "align": "center"}
    return {k: v for k, v in given.items() if v is not None and v != defaults.get(k)}


class WindowReduce(PatchProcessor):
    """
    Reduce each window along one dimension to a single value.

    The operation behind the reductions of
    [`Patch.window`](`dascore.Patch.window`), where its parameters are
    described; the dimension to window is given as a keyword, such as
    ``time=0.5``.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.proc.window import WindowReduce
    >>> patch = dc.get_example_patch()
    >>> out = WindowReduce("mean", time=11, samples=True)(patch)
    >>> assert out.shape == patch.shape
    """

    model_config = ConfigDict(extra="allow", frozen=True, arbitrary_types_allowed=True)

    # Reached through `Patch.window`, so it has no patch method of its own.
    name = None

    function: Any
    function_args: tuple = ()
    function_kwargs: dict = Field(default_factory=dict)
    overlap: Any = None
    step: Any = None
    samples: bool = False
    ends: str = "pad"
    align: str = "center"
    fill: Any = None

    _positional_fields = ("function",)

    def _spelled(self, name: str) -> str:
        """Return the call as it is written: a window, then the reduction."""
        options = _options(
            overlap=self.overlap,
            step=self.step,
            samples=self.samples,
            ends=self.ends,
            align=self.align,
            fill=self.fill,
        )
        window = _call_str("window", {**(self.model_extra or {}), **options})
        if isinstance(self.function, str) and self.function in _REDUCERS:
            return f"{window}.{self.function}()"
        applied: dict[str, Any] = {"function": self.function}
        if self.function_args:
            applied["args"] = self.function_args
        return f"{window}.{_call_str('reduce', {**applied, **self.function_kwargs})}"

    def get_metadata(self, meta):
        """Return the reduced metadata, and where the complete windows sit."""
        window = resolve_window(
            meta,
            self.model_extra or {},
            samples=self.samples,
            overlap=self.overlap,
            step=self.step,
            # Every sample gets a window unless the call says otherwise.
            default_overlap=lambda size: size - 1,
            allow_multiple=False,
            enforce_lt_coord=True,
        )
        assert window.stride is not None
        dim, axis = window.dims[0], window.axes[0]
        size, step = window.size[0], window.stride[0]
        length = meta.shape[axis]
        before = _BEFORE[self.align](size)
        # Output `k` is labelled by sample `k * step`, and its window starts
        # `before` samples earlier. The complete ones are `first` to `last`.
        first = math.ceil(before / step)
        last = (length - size + before) // step
        valid = max(last - first + 1, 0)
        if self.ends == "drop":
            start = first * step
            kept = slice(start, start + (valid - 1) * step + 1, step)
            kept = kept if valid else slice(0, 0)
            lead, total = 0, valid
        else:
            kept = slice(None, None, step)
            lead, total = first, len(range(0, length, step))
        coords, _ = meta.coords.isel(**{dim: kept})
        # Each output sample summarizes a window, not the input's cell.
        out = meta.new(coords=coords._update_grid(dim), dtype=np.float64)
        return out, {
            "axis": axis,
            "size": size,
            "step": step,
            "before": before,
            "first": first,
            "valid": valid,
            "lead": lead,
            "total": total,
        }

    def _complete(self, data, *, axis, size, step, before, first, valid):
        """Return the reduction of each complete window, in float64."""
        named = isinstance(self.function, str)
        function = _REDUCERS[self.function] if named else self.function
        # pandas labels a window by its last sample, so its answers are these
        # when the two labels are a whole number of steps apart.
        shift, apart = divmod(size - 1 - before, step)
        if named and not apart and _pandas_agrees(data, size, step, center=False):
            # pandas' std defaults to ddof=1, NumPy's to 0
            kwargs = {"ddof": 0} if self.function == "std" else {}
            rolled = _rolling_pandas(
                data,
                self.function,
                window=size,
                step=step,
                axis=axis,
                center=False,
                args=(),
                kwargs=kwargs,
            )
            pick = [slice(None)] * data.ndim
            pick[axis] = slice(first + shift, first + shift + valid)
            return rolled[tuple(pick)]
        view = np.lib.stride_tricks.sliding_window_view(data, size, axis)
        start = first * step - before
        pick = [slice(None)] * (data.ndim + 1)
        pick[axis] = slice(start, start + (valid - 1) * step + 1, step)
        windows = view[tuple(pick)]
        raw = function(windows, *self.function_args, axis=-1, **self.function_kwargs)
        return np.asarray(raw, dtype=np.float64)

    def numpy_kernel(
        self, data, *, axis, size, step, before, first, valid, lead, total
    ):
        """Return one value for each window; NaN where a window is incomplete."""
        if self.fill is not None and total != valid:
            # Filled out at both ends, every window is complete.
            after = max((total - 1) * step + size - before - data.shape[axis], 0)
            width = [(0, 0)] * data.ndim
            width[axis] = (before, after)
            data = np.pad(data, width, constant_values=self.fill)
            return self._complete(
                data, axis=axis, size=size, step=step, before=0, first=0, valid=total
            )
        if not valid:
            shape = list(data.shape)
            shape[axis] = total
            return np.full(shape, np.nan)
        reduced = self._complete(
            data,
            axis=axis,
            size=size,
            step=step,
            before=before,
            first=first,
            valid=valid,
        )
        if total == valid:
            return reduced
        shape = list(reduced.shape)
        shape[axis] = total
        out = np.full(shape, np.nan)
        place = [slice(None)] * out.ndim
        place[axis] = slice(lead, lead + valid)
        out[tuple(place)] = reduced
        return out


@dataclass(frozen=True)
class PatchWindow:
    """
    A patch and a window over it; see [`Patch.window`](`dascore.Patch.window`).

    A call on it is one operation and returns a patch, so a chain goes on
    from the patch: ``patch.window(time=1).mean().abs()``.
    """

    patch: Any
    dims: Mapping[str, Any] = field(default_factory=dict)
    overlap: Any = None
    step: Any = None
    samples: bool = False
    taper: Any = None
    analysis: Any = None
    ends: str = "pad"
    align: str = "center"
    fill: Any = None

    def __post_init__(self):
        """Refuse a window which could not be one, before anything is done."""
        if not self.dims:
            msg = "A window needs a dimension and its size, e.g. time=0.5."
            raise ParameterError(msg)
        if self.overlap is not None and self.step is not None:
            msg = "step and overlap are mutually exclusive."
            raise ParameterError(msg)
        if self.ends not in _ENDS:
            msg = f"ends must be one of {_ENDS}; got {self.ends!r}."
            raise ParameterError(msg)
        if self.align not in _BEFORE:
            msg = f"align must be one of {tuple(_BEFORE)}; got {self.align!r}."
            raise ParameterError(msg)

    def apply(
        self,
        function: Callable,
        *,
        mode: str = "overlap_add",
        engine: str = "auto",
    ):
        """
        Apply a function to the windows, and blend or stack what it returns.

        Parameters
        ----------
        function
            What to do to the windows. On the numpy engine it is given all
            of them at once, an array of ``[n_tiles, *window]``, and returns
            one of the same shape: one vectorized call, not one per window.
            A function compiled with numba is given one window at a time by
            the numba engine, in parallel, and returns a window of the same
            shape; it compiles the first time it is used in each process,
            which takes a few seconds.
        mode
            ``"overlap_add"`` blends the windows back under the window's
            `taper` into a patch shaped like the input. ``"stack"`` returns
            them unblended: each windowed dimension becomes an axis of
            window centres, and the samples within a window become a new
            ``{dim}_offset`` dimension at the end.
            [`Patch.reassemble`](`dascore.Patch.reassemble`) blends a stack
            back.
        engine
            ``"numpy"``, ``"numba"``, or ``"auto"``, which is numba for a
            numba-compiled `function` and numpy otherwise.

        Returns
        -------
        Patch
            In ``"overlap_add"`` mode, a patch with the input's coordinates.
            In ``"stack"`` mode, the windows: the windowed dimensions carry
            the window centres, ``{dim}_start`` and ``{dim}_stop`` give each
            one's closed low and open high edges in the dimension's units,
            and ``{dim}_offset`` is the position within a window.

        Examples
        --------
        >>> import numpy as np
        >>> import dascore as dc
        >>> patch = dc.get_example_patch("example_event_2")
        >>>
        >>> # Automatic gain control: every window scaled to unit RMS, blended.
        >>> def agc(tiles):
        ...     rms = np.sqrt(np.mean(tiles**2, axis=(1, 2), keepdims=True))
        ...     return tiles / np.where(rms > 0, rms, 1)
        >>> normalized = patch.window(time=0.05, distance=50).apply(agc)
        >>>
        >>> # The windows themselves, to work on and put back.
        >>> tiles = patch.window(time=0.05, distance=50).apply(
        ...     lambda x: x, mode="stack"
        ... )
        >>> back = tiles.reassemble()

        Notes
        -----
        - A window with no overlap or step given overlaps the next by half.
          Under a taper the overlap is never more than half, since the ramps
          would cross; under an analysis window it may be as deep as scipy
          can invert.
        - Windows start before the data, and hold the window's `fill` there
          and past its end -- zeros unless one is given -- so every sample
          is under a whole taper. The function sees those values. The window's
          `ends` and `align` say nothing about blended windows, and anything
          but their defaults is refused.
        """
        if self.ends != "pad" or self.align != "center":
            msg = (
                "Windows which are blended are always padded at the ends and "
                "have no label to align; ends and align apply to reductions."
            )
            raise ParameterError(msg)
        return TileApply(
            function=function,
            mode=mode,
            overlap=self.overlap,
            step=self.step,
            taper=self.taper,
            analysis=self.analysis,
            fill=self.fill,
            samples=self.samples,
            engine=engine,
            **self.dims,
        ).run(self.patch)

    def reduce(self, function: Callable | str, *args, **kwargs):
        """
        Reduce each window to one value with a function.

        Parameters
        ----------
        function
            Given the windows with the windowed dimension last, and an
            ``axis=-1`` keyword naming it, which it reduces. One of
            ``"mean"``, ``"median"``, ``"min"``, ``"max"``, ``"std"`` or
            ``"sum"`` names the NumPy function, and lets a faster engine
            answer where it gives the same values.
        *args, **kwargs
            Passed on to the function.

        Returns
        -------
        Patch
            One value for each window, in float64, along the windowed
            dimension. A window is labelled by one of the input's samples,
            the one its `align` names, so with a step of one sample the
            patch keeps its shape and coordinates.

        Examples
        --------
        >>> import numpy as np
        >>> import dascore as dc
        >>> patch = dc.get_example_patch()
        >>>
        >>> # The 80th percentile of every window of 100 samples.
        >>> out = patch.window(time=100, samples=True).reduce(np.percentile, 80)
        >>>
        >>> # Only the complete windows, a second long and half a second apart.
        >>> means = patch.window(time=1, step=0.5, ends="drop").mean()
        >>> assert not np.isnan(means.data).any()

        Notes
        -----
        - A window with no overlap or step given advances one sample.
        - A window which reaches past either end of the data is incomplete.
          With ``ends="pad"`` its value is NaN, unless the window has a
          `fill`, which the data are then filled out with so that it is
          computed; with ``ends="drop"`` it is left out, and the result is
          shorter.
        - Only one dimension can be windowed.
        """
        if self.taper is not None or self.analysis is not None:
            msg = "A reduction blends nothing, so it takes no taper or analysis."
            raise ParameterError(msg)
        return WindowReduce(
            function=function,
            function_args=args,
            function_kwargs=kwargs,
            overlap=self.overlap,
            step=self.step,
            samples=self.samples,
            ends=self.ends,
            align=self.align,
            fill=self.fill,
            **self.dims,
        ).run(self.patch)

    def mean(self):
        """Return the mean of each window; see `reduce`."""
        return self.reduce("mean")

    def median(self):
        """Return the median of each window; see `reduce`."""
        return self.reduce("median")

    def min(self):
        """Return the minimum of each window; see `reduce`."""
        return self.reduce("min")

    def max(self):
        """Return the maximum of each window; see `reduce`."""
        return self.reduce("max")

    def std(self):
        """Return the standard deviation of each window; see `reduce`."""
        return self.reduce("std")

    def sum(self):
        """Return the sum of each window; see `reduce`."""
        return self.reduce("sum")
