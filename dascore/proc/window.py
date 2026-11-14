"""
Windows of a patch, and what is done in them.

[`Patch.window`](`dascore.Patch.window`) declares the windows, and returns
a [`PatchWindow`](`dascore.proc.window.PatchWindow`); a call on that is one
operation, which gives back a patch.
"""

from __future__ import annotations

import math
import threading
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from itertools import pairwise, product
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ConfigDict

from dascore.core.coordmanager import _cell_edge_names
from dascore.core.processor import PatchProcessor, _via_numpy, captured_operation
from dascore.exceptions import ParameterError, PatchError
from dascore.lazy.contract import (
    Halo,
    Pointwise,
    Tiles,
    Unchanged,
    WholeAxis,
    _margin,
)
from dascore.proc.basic import _pad_array
from dascore.proc.rolling import _pandas_agrees, _rolling_pandas
from dascore.proc.tile_apply import TileApply, _blended_window, _windowed, _windows
from dascore.utils.array_api import (
    LAZY_BACKEND,
    _as_numpy_dtype,
    array_namespace,
    asarray_like,
    backend_name,
    device,
    is_numpy,
)
from dascore.utils.misc import optional_import, suppress_warnings
from dascore.utils.patch import _call_str, check_patch_attrs, check_patch_coords
from dascore.utils.signal import get_taper
from dascore.utils.tiles import _taper_like, fill_dtype, get_tile_plan
from dascore.utils.window import resolve_window

__all__ = ("PatchWindow", "WindowReduce", "Windowed")

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


def _check_options(ends: str, align: str) -> None:
    """Refuse an `ends` or `align` which is not one of the choices."""
    if ends not in _ENDS:
        msg = f"ends must be one of {_ENDS}; got {ends!r}."
        raise ParameterError(msg)
    if align not in _BEFORE:
        msg = f"align must be one of {tuple(_BEFORE)}; got {align!r}."
        raise ParameterError(msg)


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
    # None for none: a default pydantic need not build on every call.
    function_kwargs: dict | None = None
    overlap: Any = None
    step: Any = None
    samples: bool = False
    ends: str = "pad"
    align: str = "center"
    fill: Any = None

    _positional_fields = ("function",)

    def _spelled(self, name: str) -> str:
        """Return the call as it is written: a window, then the reduction."""
        given = self._inputs()[0]
        function = given.pop("function")
        args = given.pop("function_args", None)
        kwargs = given.pop("function_kwargs", None) or {}
        window = _call_str("window", given)
        if isinstance(function, str) and not (args or kwargs):
            return f"{window}.{function}()"
        # Written last, so a keyword of the function's cannot stand for them.
        applied = {**kwargs, "function": function, "args": args}
        return f"{window}.{_call_str('reduce', applied)}"

    def get_metadata(self, meta):
        """Return the reduced metadata, and where the complete windows sit."""
        _check_options(self.ends, self.align)
        if isinstance(self.function, str) and self.function not in _REDUCERS:
            msg = (
                f"A reduction is a function or one of {tuple(_REDUCERS)}; "
                f"got {self.function!r}."
            )
            raise ParameterError(msg)
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
        kept, total = slice(None, None, step), len(range(0, length, step))
        if self.ends == "drop":
            kept, total = slice(first * step, (first + valid) * step, step), valid
        coords = meta.coords
        # A window at every sample, the usual one, keeps every label.
        if kept != slice(None, None, 1):
            coords, _ = coords.isel(**{dim: kept})
        dtype = np.result_type(meta.dtype, np.float64)
        # Each output sample summarizes a window, not the input's cell.
        out = meta.new(coords=coords._update_grid(dim), dtype=dtype)
        return out, {
            "axis": axis,
            "size": size,
            "step": step,
            "before": before,
            "first": first,
            "valid": valid,
            "total": total,
        }

    def _bare(self) -> bool:
        """Return True for a named reduction given nothing more."""
        # Only then pandas: it is not handed what NumPy would be.
        named = isinstance(self.function, str)
        return named and not (self.function_args or self.function_kwargs)

    def _pandas(self, data, *, axis, size, step, center):
        """Return pandas' reduction, a value wherever a window ends or centres."""
        # pandas' std defaults to ddof=1, NumPy's to 0
        kwargs = {"ddof": 0} if self.function == "std" else {}
        where = {"window": size, "step": step, "axis": axis, "center": center}
        return _rolling_pandas(data, self.function, args=(), kwargs=kwargs, **where)

    def _whole(self, data, *, axis, size, step, before, total):
        """
        Return pandas' answer where it is every window's already, else None.

        It marks the incomplete windows itself, so nothing is placed in a
        second array the size of the patch: so for a window labelled at its
        last sample, or at the middle one pandas calls the centre.
        """
        center = step == 1 and before == (size - 1) // 2
        if not (self._bare() and (center or before == size - 1)):
            return None
        if not _pandas_agrees(data, size, step, center=center):
            return None
        out = self._pandas(data, axis=axis, size=size, step=step, center=center)
        return out if out.shape[axis] == total else None

    def _complete(self, data, *, axis, size, step, before, first, valid):
        """Return the reduction of each complete window, in a float dtype."""
        named = isinstance(self.function, str)
        function = _REDUCERS[self.function] if named else self.function
        bare = self._bare()
        # pandas labels a window by its last sample, so its answers are these
        # when the two labels are a whole number of steps apart.
        shift, apart = divmod(size - 1 - before, step)
        if bare and not apart and _pandas_agrees(data, size, step, center=False):
            rolled = self._pandas(data, axis=axis, size=size, step=step, center=False)
            pick = [slice(None)] * data.ndim
            pick[axis] = slice(first + shift, first + shift + valid)
            return rolled[tuple(pick)]
        view = np.lib.stride_tricks.sliding_window_view(data, size, axis)
        start = first * step - before
        pick = [slice(None)] * (data.ndim + 1)
        pick[axis] = slice(start, start + (valid - 1) * step + 1, step)
        windows = view[tuple(pick)]
        kwargs = self.function_kwargs or {}
        raw = function(windows, *self.function_args, axis=-1, **kwargs)
        # Float64, or complex for complex data, so a window can hold NaN.
        return np.asarray(raw, dtype=np.result_type(data.dtype, np.float64))

    def numpy_kernel(self, data, *, axis, size, step, before, first, valid, total):
        """Return one value for each window; NaN where a window is incomplete."""
        where = {"axis": axis, "size": size, "step": step}
        if self.fill is not None and total != valid:
            # Filled out at both ends, every window is complete. The data
            # are widened first, so a fill they cannot hold is not cut down.
            after = max((total - 1) * step + size - before - data.shape[axis], 0)
            width = [(0, 0)] * data.ndim
            width[axis] = (before, after)
            data = data.astype(fill_dtype(data.dtype, self.fill), copy=False)
            data = np.pad(data, width, constant_values=self.fill)
            return self._complete(data, before=0, first=0, valid=total, **where)
        whole = self._whole(data, before=before, total=total, **where)
        if whole is not None:
            return whole
        reduced = self._complete(data, before=before, first=first, valid=valid, **where)
        if total == valid:
            return reduced
        shape = list(reduced.shape)
        shape[axis] = total
        out = np.full(shape, np.nan, dtype=reduced.dtype)
        place = [slice(None)] * out.ndim
        place[axis] = slice(first, first + valid)
        out[tuple(place)] = reduced
        return out


# The most samples of a dimension which is not windowed that the dtype of a
# lazy or dask patch is found from: tried in turn, before the whole dimension, for an
# operation which cannot plan or run on so few.
_PROBES = (64, 4096)
# How many windows long a piece of a dask array is computed in, at least:
# each piece is given the windows which reach into it from either side, so
# a piece of one window would compute three to keep one.
_DASK_WINDOWS = 8
# DASCore's NumPy kernels are run one at a time when dask computes pieces on
# several threads: not every one may be run from two threads at once.
_DASK_TURN = threading.Lock()


@dataclass(frozen=True)
class _Made:
    """
    Where what an operation makes of each window sits, along the windowed axes.

    An operation may keep every n-th sample of a window, and may leave
    samples at either end of one empty. This is the tiling in the samples
    it makes: of a window's result, of how far the next starts, and of the
    whole result, with how much of a window's result is blended.
    """

    size: tuple[int, ...]
    stride: tuple[int, ...]
    length: tuple[int, ...]
    # How many samples at the start of a window's result are discarded,
    # and how many after them are blended.
    lead: tuple[int, ...]
    kept: tuple[int, ...]

    @classmethod
    def of(cls, size, stride, length, factor, trim) -> _Made:
        """Return the geometry for windows given in samples of the input."""
        made = tuple(-(-z // q) for z, q in zip(size, factor, strict=True))
        return cls(
            size=made,
            stride=tuple(x // q for x, q in zip(stride, factor, strict=True)),
            length=tuple(-(-n // q) for n, q in zip(length, factor, strict=True)),
            lead=tuple(before for before, _ in trim),
            kept=tuple(z - sum(ends) for z, ends in zip(made, trim, strict=True)),
        )

    @property
    def overlap(self) -> tuple[int, ...]:
        """How far what is blended of one window reaches into the next."""
        return tuple(z - step for z, step in zip(self.kept, self.stride, strict=True))

    @property
    def pick(self) -> tuple:
        """What is blended of a window's result, whose last axes these are."""
        spans = zip(self.lead, self.kept, strict=True)
        return (..., *(slice(at, at + z) for at, z in spans))

    def span(self, tiles) -> tuple[list[int], list[slice]]:
        """Return how far the given windows' results reach, and the array in it."""
        extent, inside = [], []
        spans = zip(tiles, self.size, self.stride, self.length, strict=True)
        for (first, last), z, step, n in spans:
            low = first * step - _margin(z, step)
            high = low + (last - 1 - first) * step + z
            extent.append(high - low)
            inside.append(slice(max(low, 0) - low, min(high, n) - low))
        return extent, inside


class Windowed(PatchProcessor):
    """
    Run an operation in each window of a patch, and blend the windows.

    The operation behind a patch method called on
    [`Patch.window`](`dascore.Patch.window`), where the window's parameters
    are described; the dimensions to window are given as keywords, such as
    ``time=0.5``.

    Parameters
    ----------
    operation
        The operation each window is given, as a
        [`PatchProcessor`](`dascore.core.processor.PatchProcessor`).

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.proc.window import Windowed
    >>> patch = dc.get_example_patch()
    >>> # A trend taken out of every second of the patch, not out of all of it.
    >>> out = patch.window(time=1).detrend("time")
    >>> same = Windowed(dc.proc.Detrend(dim="time"), time=1)(patch)
    >>> assert out.equals(same)

    Notes
    -----
    - One plan is worked out, from the metadata of the first window, and
      run in every window: the kernel never changes from window to window.
      Only an operation which acts relative to its window can be run that
      way, and its class says that it is one by setting `position_free`;
      DASCore's detrend, taper, filters, normalizations and the like do.
      Any other is refused.
    - Everything is decided before any data are read. The operation must
      also give back a window with the same dimensions and coordinates,
      take no other patch or array shaped like the patch, and not settle
      its result from the data it computed; the window must fit inside its
      dimension.
    - An operation may keep every n-th sample of a windowed dimension, as
      `decimate` and `fbe` with a `step` do, if the windows advance by a
      multiple of n samples, so that every window lands on one grid. The
      result is then that much shorter. No other change of shape is
      blended.
    - An operation which leaves samples at the ends of its result empty
      says how many by its `halo`: `fbe` and `stalta` leave NaN until a
      whole rolling window is behind a sample. Those are discarded from
      every window and what is left is blended, so the windows must overlap
      by at least as much. At the start of the patch a window reaches
      before the data and holds the fill there, so the first samples are
      computed from the fill and the data together, not left empty.
    - On an array backend other than NumPy the data stay where they are.
      A dask array is computed a piece at a time, each from the windows
      which overlap it, so nothing is computed when the operation is
      called; a piece is blended on the library its chunks are held in,
      and the pieces one at a time in a process, since not every kernel
      may be run from two threads at once. On any other library the
      windows are cut, run and blended there, which takes several array
      operations for each window. Only an operation with no implementation
      for the library is converted through NumPy, as it is outside a
      window, and always with a `NumpyFallbackWarning` naming it.
    - An operation which only changes metadata leaves the data as they are.
    - On a lazy patch an output chunk needs only the windows which overlap
      it, so a windowed dimension can be cut even where the operation
      itself needs its dimension whole. For a lazy or dask patch the dtype
      is found by running the operation on a window of zeros.
    """

    model_config = ConfigDict(extra="allow", frozen=True, arbitrary_types_allowed=True)

    # Reached through `Patch.window`, so it has no patch method of its own.
    name = None

    operation: Any
    overlap: Any = None
    step: Any = None
    taper: Any = None
    analysis: Any = None
    fill: Any = None
    samples: bool = False

    _positional_fields = ("operation",)

    def _spelled(self, name: str) -> str:
        """Return the call as it is written: a window, then the operation."""
        given = self._inputs()[0]
        operation = given.pop("operation")
        called = operation.name or type(operation).__name__
        spelled = operation._spelled(called)
        # An operation which records only its name is still a call here.
        spelled = spelled if spelled.endswith(")") else f"{spelled}()"
        return f"{_call_str('window', given)}.{spelled}"

    def _tile(self, meta, window, starts):
        """Return the metadata of the window which starts at `starts`."""
        spans = zip(window.dims, starts, window.size, strict=True)
        index = {dim: slice(at, at + size) for dim, at, size in spans}
        coords, _ = meta.coords.isel(**index)
        # In the dtype the operation is handed, which is what it plans for.
        return meta.new(coords=coords, dtype=self._given_dtype(meta.dtype))

    def _planned(self, tile):
        """Return what the operation makes of one window's metadata, and its plan."""
        try:
            return self.operation.get_metadata(tile)
        except Exception as error:
            name = self.operation.name or type(self.operation).__name__
            error.add_note(f"Raised by {name} for one window of Patch.window.")
            raise

    def _refuse(self, why: str):
        """Raise for an operation which cannot run in a window."""
        operation = self.operation
        name = operation.name or type(operation).__name__
        msg = (
            f"{name} cannot run in a window: {why} A function of a window "
            "given to Patch.window(...).apply can do anything."
        )
        raise ParameterError(msg)

    def _lazy_dtype(self, window, tile, plan) -> np.dtype:
        """Return the dtype of the blended windows, by running one of zeros."""
        # A lazy or dask patch states its dtype before anything is read,
        # and an operation's metadata need not say what its kernel returns:
        # a detrend of integers is float. A dtype does not depend on how
        # long the other dimensions are, so the window is cut short along
        # those; an operation which cannot plan or run that short is given
        # it longer, and at last whole.
        for most in _PROBES:
            long = {
                dim: slice(0, most)
                for dim, n in zip(tile.dims, tile.shape)
                if dim not in window.dims and n > most
            }
            if not long:
                break
            try:
                short = tile.new(coords=tile.coords.isel(**long)[0])
                return self._probed(short, {**plan, "inner": self._planned(short)[1]})
            except Exception:
                # Whatever it could not do that short, it is asked longer.
                continue
        return self._probed(tile, plan)

    def _probed(self, tile, plan) -> np.dtype:
        """Return the dtype one window of zeros is blended to."""
        # The one window which starts where the array does, as if it were
        # all there is.
        size, stride = plan["size"], plan["stride"]
        only = tuple(-(-z // step) for z, step in zip(size, stride, strict=True))
        keys = ("axes", "size", "stride", "factor", "trim", "inner")
        where = {x: plan[x] for x in keys}
        # In NumPy's dtype: a dask array's chunks may be another library's.
        probe = np.zeros(tile.shape, _as_numpy_dtype(tile.dtype))
        with np.errstate(all="ignore"), suppress_warnings():
            out, before = self._blend(
                probe, length=size, tiles=tuple((x - 1, x) for x in only), **where
            )
        _warn_if_integers(before, out.dtype)
        return out.dtype

    def _given_dtype(self, dtype):
        """
        Return the dtype the operation's kernel is handed for data of `dtype`.

        A fill the data cannot hold widens them, and an analysis window
        makes floats of them, before the operation sees a window.
        """
        if self.fill in (None, 0) and self.analysis is None:
            return dtype
        dtype = _as_numpy_dtype(dtype)
        if self.fill is not None and self.fill != 0:
            dtype = fill_dtype(dtype, self.fill)
        if self.analysis is not None:
            dtype = np.result_type(dtype, _taper_like(np.empty(0), np.empty(0, dtype)))
        return dtype

    def _landing(self, meta, window, tile, made):
        """
        Return how the windows' results land: the factors and the coordinates.

        For each windowed axis every how many samples of a window the
        operation keeps, and the coordinates of the blended result. An
        operation which does anything else to a window's shape, dimensions
        or coordinates is refused.
        """
        changed = "it changes the window's dimensions or coordinates."
        if made is tile:
            return (1,) * len(window.axes), meta.coords
        if made.dims != tile.dims:
            self._refuse(changed)
        shapes = f"a window of shape {tile.shape} into one of {made.shape}"
        windowed = dict(zip(window.axes, window.dims, strict=True))
        for axis, (before, after) in enumerate(zip(tile.shape, made.shape)):
            if axis not in windowed and before != after:
                self._refuse(
                    f"it turns {shapes}, changing the length of "
                    f"{tile.dims[axis]}, which is not windowed."
                )
        factor, strides = [], {}
        assert window.stride is not None
        spans = zip(window.axes, window.dims, window.size, window.stride, strict=True)
        for axis, dim, size, stride in spans:
            every = _every(tile, made, dim, size, made.shape[axis])
            if every is None:
                self._refuse(
                    f"it turns {shapes}, and along a windowed dimension only "
                    "a result which keeps the window's shape, or every n-th "
                    "sample of it, is blended."
                )
            if stride % every:
                self._refuse(
                    f"it keeps every {every}th sample along {dim} and the "
                    f"windows advance by {stride} samples, which is not a "
                    f"multiple of {every}, so the windows would not land on "
                    "one grid. Give the window a step or an overlap which "
                    "makes it one."
                )
            factor.append(every)
            if every != 1:
                strides[dim] = slice(None, None, every)
        coords = tile.coords.isel(**strides)[0] if strides else tile.coords
        # The bounds of a windowed dimension's cells may go: a sample which
        # summarizes a rolling window is no longer its input's cell.
        edges = {
            x for dim in window.dims for x in _cell_edge_names(dim, coords.coord_map)
        }
        gone = (set(coords.coord_map) - set(made.coords.coord_map)) & edges
        if gone:
            coords = coords.drop_coords(*gone)[0]
        if not _same_coords(made.coords, coords):
            self._refuse(changed)
        out = meta.coords.isel(**strides)[0] if strides else meta.coords
        return tuple(factor), out.drop_coords(*gone)[0] if gone else out

    def _trim(self, window, factor, inner):
        """Return how many samples at each end of a window's result are empty."""
        halo = self.operation.halo(inner)
        trim = tuple(tuple(halo.get(axis, (0, 0))) for axis in window.axes)
        whole = not any(map(any, trim)) and all(x == 1 for x in factor)
        if whole:
            return trim
        if self.analysis is not None:
            self._refuse(
                "it does not hand back the whole of a window, and an "
                "analysis window is blended under a dual worked out for "
                "whole windows."
            )
        assert window.stride is not None
        # Of one window alone, which is all that is asked of it here.
        landed = _Made.of(window.size, window.stride, window.size, factor, trim)
        for dim, ends, overlap in zip(window.dims, trim, landed.overlap, strict=True):
            if overlap < 0:
                self._refuse(
                    f"it leaves {sum(ends)} samples at the ends of each "
                    f"window empty along {dim}, which are discarded, so the "
                    "windows must overlap by at least that many of the "
                    f"samples it makes; they overlap by {overlap + sum(ends)}."
                )
        # A taper whose ramps would cross is refused here, not at the data.
        get_taper(self._taper_name(), landed.kept, landed.overlap)
        return trim

    def _taper_name(self):
        """Return the taper windows are blended under."""
        return "hann" if self.taper is None else self.taper

    def _tapers(self, plan, made: _Made):
        """
        Return what windows are multiplied by on the way in and out, and where.

        The last is the geometry of what is blended, less the ends the
        taper gives no weight: a Hann ramp starts from zero, and nothing
        times zero is added, or an infinity or a NaN there would reach the
        sample of the next window which holds all the weight.
        """
        if made.kept == plan.size:
            overlap = tuple(z - x for z, x in zip(plan.size, plan.stride, strict=True))
            analysis, synthesis = _windows(self.analysis, self.taper, plan, overlap)
        else:
            # Complementary ramps over what is blended of each window.
            taper = get_taper(self._taper_name(), made.kept, made.overlap)
            analysis, synthesis = None, taper
        lead, kept, cut = list(made.lead), list(made.kept), []
        for axis in range(synthesis.ndim):
            others = tuple(x for x in range(synthesis.ndim) if x != axis)
            held = np.flatnonzero(synthesis.any(axis=others))
            first, stop = int(held[0]), int(held[-1]) + 1
            lead[axis], kept[axis] = lead[axis] + first, stop - first
            cut.append(slice(first, stop))
        if tuple(kept) != made.kept:
            made = replace(made, lead=tuple(lead), kept=tuple(kept))
            synthesis = synthesis[tuple(cut)]
        return analysis, synthesis, made

    def get_metadata(self, meta):
        """Return the blended metadata, the tiling, and one window's plan."""
        operation = self.operation
        if not isinstance(operation, PatchProcessor):
            msg = f"A window runs a PatchProcessor; got {operation!r}."
            raise ParameterError(msg)
        if type(operation).reconcile is not PatchProcessor.reconcile:
            self._refuse("it settles its result from the data it computed.")
        if operation._inputs()[1]:
            self._refuse("it takes another patch, which a window cannot cut yet.")
        given = operation.kwargs.values()
        if any(getattr(x, "shape", None) == meta.shape for x in given):
            self._refuse(
                "it was given an array shaped like the patch, which a window "
                "cannot cut yet."
            )
        window = _blended_window(
            meta,
            self.model_extra or {},
            samples=self.samples,
            overlap=self.overlap,
            step=self.step,
            taper=self.taper,
            analysis=self.analysis,
            whole=True,
        )
        assert window.stride is not None
        axes, size, stride = window.axes, window.size, window.stride
        # Refused here, not when the data are read: a taper whose ramps
        # would cross, or an analysis window with no dual at this stride.
        _windows(self.analysis, self.taper, window.tiles(meta.shape), window.overlap)
        # The first whole window stands for every one.
        tile = self._tile(meta, window, (0,) * len(axes))
        check_patch_coords(
            tile, dims=operation.required_dims, coords=operation.required_coords
        )
        check_patch_attrs(tile, operation.required_attrs)
        if not operation._is_position_free(window.dims):
            try:
                made = operation.get_metadata(tile)[0]
            except Exception:
                made = None
            if made is not None:
                # What it does to a window is the more useful thing to say.
                self._landing(meta, window, tile, made)
            self._refuse(
                "as called it is not known to do the same thing wherever a "
                "window sits, which its class says by position_free. One plan "
                "is worked out and run in every window, so an operation which "
                "reads where its coordinates start has no place in one. An "
                "operation which takes coordinate values and has a `relative` "
                "argument can be given them from the start of its window."
            )
        made, inner = self._planned(tile)
        factor, coords = self._landing(meta, window, tile, made)
        trim = self._trim(window, factor, inner)
        # A lazy patch and a dask one both compute later, so both are told
        # now what dtype the blended windows have.
        deferred = meta.backend in (LAZY_BACKEND, "dask")
        plan = {
            "axes": axes,
            "size": size,
            "stride": stride,
            # The whole length of each windowed axis, which places the tiles.
            "length": tuple(meta.shape[x] for x in axes),
            "others": tuple(x for x in range(len(meta.shape)) if x not in axes),
            # Every how many samples of a window the operation keeps, and
            # how many at each end of what it makes are discarded.
            "factor": factor,
            "trim": trim,
            "inner": inner,
        }
        out = meta.new(attrs=operation._result_attrs(made))
        if coords is not meta.coords:
            out = out.new(coords=coords)
        if type(operation).kernel_for("numpy") is None:
            # Nothing is computed, so nothing is blended.
            return out, {**plan, "quiet": True, "like": None}
        if not deferred:
            # In memory the array which comes back says what it is.
            return out, {**plan, "quiet": False, "like": None}
        dtype = self._lazy_dtype(window, tile, plan)
        # An empty array says the dtype: a plan holds numbers and arrays.
        return out.new(dtype=dtype), {**plan, "quiet": True, "like": np.empty(0, dtype)}

    def chunk_contract(self, dtype, plan):
        """Return the tiles a chunk needs, and what the operation reads besides."""
        # Imported here: dascore.lazy.kernels imports the lazy array, which
        # reads files through modules which import the processors.
        from dascore.lazy.kernels import _contract  # noqa: PLC0415

        if type(self.operation).kernel_for("numpy") is None:
            return Unchanged()
        axes, others = plan["axes"], plan["others"]
        spans = dict(zip(axes, zip(plan["size"], plan["stride"], plan["length"])))
        factors = {x: q for x, q in zip(axes, plan["factor"], strict=True) if q != 1}
        tiles = partial(Tiles, spans, factors=factors)
        inner = _contract(self.operation, self._given_dtype(dtype), plan["inner"])
        if isinstance(inner, Pointwise | Unchanged):
            return tiles()
        if isinstance(inner, Halo):
            widths = {x: w for x, w in inner.widths.items() if x not in axes}
            return tiles(widths=widths)
        if isinstance(inner, WholeAxis):
            return tiles(whole=tuple(x for x in inner.axes if x not in axes))
        # Nothing says how far it reaches along the dimensions not windowed.
        return tiles(whole=others)

    def numpy_kernel(
        self,
        data,
        *,
        axes,
        size,
        stride,
        length,
        others,
        factor,
        trim,
        inner,
        quiet,
        like=None,
        tiles=None,
    ):
        """
        Return the windows the operation made, blended.

        `tiles`, given by a chunk, names the windows the data hold: for each
        windowed axis the first and one past the last. Without it the data
        are the whole array and hold every window. `quiet` says the caller
        has already said what blending does to integers.
        """
        if type(self.operation).kernel_for("numpy") is None:
            return data
        where = {"axes": axes, "size": size, "stride": stride, "length": length}
        where |= {"factor": factor, "trim": trim}
        out, made = self._blend(data, inner=inner, tiles=tiles, **where)
        if not quiet:
            _warn_if_integers(made, out.dtype)
        return out

    def kernel(
        self,
        data,
        *,
        axes,
        size,
        stride,
        length,
        others,
        factor,
        trim,
        inner,
        quiet,
        like=None,
        tiles=None,
    ):
        """
        Return the windows the operation made, blended, on the data's backend.

        As `numpy_kernel`, for any other array library, so the data stay
        where they are. A dask array is computed a chunk at a time, each
        from the windows which overlap it; see `_by_dask_chunk`. For any
        other library the windows are cut by slicing, the operation's
        kernel for the library is run on each, and the results are put
        together by concatenation and addition, with nothing written in
        place, which not every array library allows.
        """
        operation = self.operation
        backend = backend_name(data)
        kernel = type(operation).kernel_for(backend)
        if kernel is None:
            return data
        where = {"axes": axes, "size": size, "stride": stride, "length": length}
        where |= {"factor": factor, "trim": trim}
        if backend == "dask" and like is not None and tiles is None:
            return self._by_dask_chunk(
                data, like=like, others=others, inner=inner, **where
            )
        if (name := getattr(kernel, "_converts", None)) is not None:
            # The operation has nothing for this backend and goes through
            # NumPy. Once, for the whole array: a window at a time, a lazy
            # backend would compute what its windows share again for each.
            convert = _via_numpy(type(self).numpy_kernel, name)
            rest = {"others": others, "inner": inner, "quiet": quiet, "tiles": tiles}
            return convert(self, data, **where, **rest)
        xp = array_namespace(data)
        plan = get_tile_plan(length, size, stride)
        tiles = tiles or tuple((0, count) for count in plan.grid)
        ndim = len(axes)
        tail = tuple(range(-ndim, 0))
        moved = xp.moveaxis(data, axes, tail)
        fill = 0 if self.fill is None else self.fill
        if fill != 0:
            wide = fill_dtype(_as_numpy_dtype(moved.dtype), fill)
            moved = xp.astype(moved, _dtype_on(moved, wide))
        # A window reaching beyond the array holds the fill there: the data
        # are lengthened at each end by as much of it as the windows reach.
        spans = zip(tiles, size, stride, plan.margin, length, strict=True)
        widths = [(0, 0)] * (moved.ndim - ndim)
        for (first, last), z, step, margin, n in spans:
            low, high = first * step - margin, (last - 1) * step - margin + z
            widths.append((max(low, 0) - low, high - min(high, n)))
        moved = _pad_array(moved, widths, fill)
        landed = _Made.of(size, stride, length, factor, trim)
        analysis, synthesis, landed = self._tapers(plan, landed)
        # Windows a whole number of colours apart never share a sample, so
        # each such class is run, laid out with nothing between its windows,
        # and added to the rest before the next is begun, so the windows of
        # one class are held at a time and not of all. The classes go in one
        # order always, by
        # where each window sits in the array, so a sample's windows are
        # summed the same way however the array was cut.
        classes: dict[tuple, list] = {}
        for index in product(*(range(first, last) for first, last in tiles)):
            colour = tuple(k % c for k, c in zip(index, plan.colours, strict=True))
            classes.setdefault(colour, []).append(index)
        total, dtype, expected = None, None, None
        reach, within = landed.span(tiles)
        keep = landed.pick
        for colour in sorted(classes):
            made = {}
            for index in classes[colour]:
                starts = ((k - t[0]) * s for k, t, s in zip(index, tiles, stride))
                pick = (..., *(slice(at, at + z) for at, z in zip(starts, size)))
                tile = moved[pick]
                if analysis is not None:
                    tile = _weighted(xp, tile, analysis)
                tile = xp.moveaxis(tile, tail, axes)
                out = kernel(operation, tile, **inner)
                if backend_name(out) != backend:
                    # A kernel which answered in NumPy whatever it was given.
                    out = asarray_like(out, tile)
                if expected is None:
                    expected = _window_shape(tile.shape, axes, landed.size)
                if tuple(out.shape) != expected:
                    _refuse_window(operation, tile.shape, out.shape, expected)
                dtype = _as_numpy_dtype(out.dtype)
                kept = xp.moveaxis(out, axes, tail)[keep]
                made[index] = _weighted(xp, kept, synthesis)
            laid = _laid_out(xp, made, tiles, landed, reach)
            total = laid if total is None else total + laid
            # Let go before the next class is run.
            made = laid = None
        assert total is not None and dtype is not None  # an axis has a window
        if not quiet:
            _warn_if_integers(dtype, _as_numpy_dtype(total.dtype))
        return xp.moveaxis(total[(..., *within)], tail, axes)

    def _by_dask_chunk(
        self, data, *, like, axes, size, stride, length, factor, trim, others, inner
    ):
        """
        Return the blended windows of a dask array, a chunk at a time.

        The result is made in pieces, each a task which is given the windows
        that overlap its piece, whole, and blends them: what a chunk of a
        lazy patch is given, by the same contract. So nothing is computed
        here, and no task holds more than its piece and the windows around
        it. A piece is blended on the library its chunks are held in, by
        the kernel the operation has for that library: the NumPy one for
        NumPy chunks, whose values are then the NumPy ones exactly.
        A piece is a chunk, or along a windowed dimension several chunks
        where they are short beside the window; a dimension the operation
        needs whole is one piece. The result is cut back into the chunks
        the data came in, each as much shorter as the operation makes it.
        """
        dask, array = optional_import("dask"), optional_import("dask.array")
        where = {"axes": axes, "size": size, "stride": stride, "length": length}
        where |= {"factor": factor, "trim": trim}
        plan = {**where, "others": others, "inner": inner}
        contract = self.chunk_contract(_as_numpy_dtype(data.dtype), plan)
        sizes = dict(zip(axes, size, strict=True))
        spans, cut = [], []
        for axis, chunks in enumerate(data.chunks):
            edges = np.cumsum((0, *chunks)).tolist()
            every = contract.factors.get(axis, 1)
            cut.append(chunks if every == 1 else _made_chunks(edges, every))
            if axis in contract.whole:
                edges = [0, edges[-1]]
            elif axis in sizes:
                # Chunks joined until a piece is several windows long.
                least, kept = _DASK_WINDOWS * sizes[axis], [0]
                for edge in edges[1:-1]:
                    if edge - kept[-1] >= least and edges[-1] - edge >= least:
                        kept.append(edge)
                edges = [*kept, edges[-1]]
            # A piece is of the result: where every n-th sample is kept, a
            # chunk's edge falls on the first sample kept at or after it.
            edges = edges if every == 1 else sorted({-(-x // every) for x in edges})
            spans.append(list(pairwise(edges)))
        # Handed to every task once, not copied into each.
        given = (contract, self, {**plan, "quiet": True}, like.dtype)
        shared = dask.delayed(given, traverse=False)

        def build(prefix: tuple):
            """Return the pieces under `prefix`, nested by axis for `block`."""
            if len(prefix) < data.ndim:
                return [build((*prefix, span)) for span in spans[len(prefix)]]
            box = tuple(slice(low, high) for low, high in prefix)
            needed = contract.needs(box, data.shape)
            task = dask.delayed(_blended_piece)(shared, data[needed], box, needed)
            shape = tuple(high - low for low, high in prefix)
            return array.from_delayed(task, shape=shape, dtype=like.dtype)

        return array.block(build(())).rechunk(tuple(cut))

    def _blend(self, data, *, axes, size, stride, length, factor, trim, inner, tiles):
        """Return the blended windows, and the dtype the operation made."""
        operation = self.operation
        kernel = type(operation).kernel_for("numpy")
        plan = get_tile_plan(length, size, stride)
        tiles = tiles or tuple((0, count) for count in plan.grid)
        ndim = len(axes)
        tail = tuple(range(-ndim, 0))
        moved = np.moveaxis(data, axes, tail)
        # The span of the windows given, and where the data sit inside it:
        # a window reaching beyond the array holds the fill there.
        spans = zip(tiles, size, stride, plan.margin, length, strict=True)
        extent, inside = [], []
        for (first, last), z, step, margin, n in spans:
            low, high = first * step - margin, (last - 1) * step - margin + z
            extent.append(high - low)
            inside.append(slice(max(low, 0) - low, min(high, n) - low))
        inside = (..., *inside)
        fill = 0 if self.fill is None else self.fill
        dtype = moved.dtype if fill == 0 else fill_dtype(moved.dtype, fill)
        whole = np.zeros if fill == 0 else partial(np.full, fill_value=fill)
        padded = whole((*moved.shape[:-ndim], *extent), dtype=dtype)
        padded[inside] = moved
        landed = _Made.of(size, stride, length, factor, trim)
        analysis, synthesis, landed = self._tapers(plan, landed)
        reach, within = landed.span(tiles)
        # All of a window's result is blended, unless the operation keeps
        # every n-th sample of it or leaves its ends empty, or the taper
        # gives its ends no weight.
        keep = None if landed.kept == tuple(size) else landed.pick
        # For each axis and each window along it: where the window is cut
        # from, where what is blended of it is put, and whether it lies
        # wholly inside the array.
        cut, put, inside = [], [], []
        spans = zip(tiles, size, stride, plan.margin, length, strict=True)
        made_spans = zip(landed.stride, landed.lead, landed.kept, strict=True)
        for ((first, last), z, x, m, n), (made_x, lead, kept) in zip(spans, made_spans):
            count = range(last - first)
            cut.append([slice(k * x, k * x + z) for k in count])
            starts = [k * made_x + lead for k in count]
            put.append([slice(at, at + kept) for at in starts])
            low, high = -(-m // x) - first, (n + m - z) // x - first
            inside.append([low <= k <= high for k in count])
        put = cut if keep is None else put
        expected, out, weights, seen = None, None, None, None
        # In one order always, so a sample's windows are summed the same
        # way however the array was cut.
        for index in product(*(range(last - first) for first, last in tiles)):
            pick = (..., *[x[k] for x, k in zip(cut, index)])
            tile = np.moveaxis(_windowed(padded[pick], analysis), tail, axes)
            if all([x[k] for x, k in zip(inside, index)]):
                made = kernel(operation, tile, **inner)
            else:
                # A window which reaches beyond the array holds fill there,
                # and what NumPy would say of a logarithm or a quotient of
                # that is not about the data.
                with np.errstate(all="ignore"):
                    made = kernel(operation, tile, **inner)
            if expected is None:
                expected = _window_shape(tile.shape, axes, landed.size)
            if np.shape(made) != expected:
                _refuse_window(operation, tile.shape, np.shape(made), expected)
            made = np.moveaxis(made, axes, tail)
            if keep is not None:
                made = made[keep]
            if seen is None or made.dtype != seen:
                # The taper in the precision of what the operation made.
                weights, seen = _taper_like(synthesis, made), made.dtype
            if out is None:
                shape = (*padded.shape[:-ndim], *reach)
                out = np.zeros(shape, dtype=np.result_type(made, weights))
            out[(..., *[x[k] for x, k in zip(put, index)])] += made * weights
        assert out is not None  # an axis with a sample has a window
        return np.moveaxis(out[(..., *within)], tail, axes), made.dtype


def _blended_piece(shared, piece, box, needed):
    """
    Return one piece of a dask array: the windows around it blended, and cut.

    Blended on the library the piece's chunks are held in, as a chunk of a
    lazy patch is on NumPy, so a piece on a GPU stays there and is given
    the kernel the operation has for its library. An operation with none
    for it is converted through NumPy with a `NumpyFallbackWarning`, as it
    is outside a window, and never silently.
    """
    contract, windowed, plan, dtype = shared
    kernel = type(windowed).kernel_for(backend_name(piece))
    with _DASK_TURN:
        out = contract.apply(partial(kernel, windowed, **plan), piece, box, needed)
    # In the dtype the result states, whatever this piece's values made.
    if is_numpy(out):
        return out.astype(dtype, copy=False)
    return array_namespace(out).astype(out, _dtype_on(out, dtype), copy=False)


def _weighted(xp, array, weights: np.ndarray):
    """Return an array times NumPy weights, on the array's own backend."""
    # In the dtype NumPy would give, stated outright: an array library may
    # refuse to mix integers with floats.
    like = np.empty(0, _as_numpy_dtype(array.dtype))
    weights = _taper_like(weights, like)
    wide = np.result_type(like, weights)
    if wide != like.dtype:
        array = xp.astype(array, _dtype_on(array, wide))
    return array * asarray_like(weights.astype(wide, copy=False), array)


def _dtype_on(array, dtype: np.dtype):
    """Return a NumPy dtype as the dtype of an array's own library."""
    try:
        return asarray_like(np.empty(0, dtype), array).dtype
    except (TypeError, ValueError):
        # A library without half precision is given single.
        return asarray_like(np.empty(0, np.result_type(dtype, np.float32)), array).dtype


def _block(xp, like, axis: int, length: int, fill):
    """Return a block of `fill` as long as `length` along one axis of `like`."""
    if not length:
        return None
    shape = list(like.shape)
    shape[axis] = length
    # On the device the data are on, where a library has more than one.
    return xp.full(tuple(shape), fill, dtype=like.dtype, device=device(like))


def _laid_out(xp, windows: dict, tiles, landed: _Made, extent, prefix: tuple = ()):
    """
    Return windows which share no sample, laid out over the whole extent.

    `windows` holds what is blended of each by its index along the windowed
    axes, which are its last ones. Along each axis in turn they are joined
    end to end, with zeros wherever none of them lies; `prefix` is the
    index, along the axes already joined, of the windows being joined along
    the next.
    """
    level, ndim = len(prefix), len(landed.kept)
    if level == ndim:
        return windows[prefix]
    axis, first = level - ndim, tiles[level][0]
    pieces, cursor = [], 0
    for k in sorted({x[level] for x in windows if x[:level] == prefix}):
        block = _laid_out(xp, windows, tiles, landed, extent, (*prefix, k))
        start = (k - first) * landed.stride[level] + landed.lead[level]
        pieces += [_block(xp, block, axis, start - cursor, 0), block]
        cursor = start + landed.kept[level]
    pieces.append(_block(xp, pieces[-1], axis, extent[level] - cursor, 0))
    return xp.concat([x for x in pieces if x is not None], axis=axis)


def _window_shape(given, axes, sizes) -> tuple[int, ...]:
    """Return the shape an operation states for a window of shape `given`."""
    expected = list(given)
    for axis, size in zip(axes, sizes, strict=True):
        expected[axis] = size
    return tuple(expected)


def _refuse_window(operation, given, made, expected):
    """Raise for a kernel which made a window of another shape than planned."""
    name = type(operation).__name__
    msg = (
        f"{name} computed a window of shape {tuple(made)} from one of "
        f"{tuple(given)}; its get_metadata stated one of {tuple(expected)}."
    )
    raise PatchError(msg)


def _made_chunks(edges: list[int], every: int) -> tuple[int, ...]:
    """Return a dask axis's chunks where every n-th sample of it is kept."""
    made = sorted({-(-x // every) for x in edges})
    return tuple(after - before for before, after in pairwise(made))


def _every(tile, made, dim: str, size: int, kept: int) -> int | None:
    """
    Return every how many samples of a window's dimension a result keeps.

    One if it keeps them all, and None if what it holds is not every n-th
    sample of the window, starting from the first.
    """
    if kept == size:
        return 1
    if not 1 < kept < size:
        return None
    before, after = tile.coords.coord_map[dim], made.coords.coord_map[dim]
    # Each n for which a window of this size keeps this many.
    for every in range(-(-size // kept), (size - 1) // (kept - 1) + 1):
        if before[::every] == after:
            return every
    return None


def _same_coords(made, expected) -> bool:
    """Whether two coordinate managers hold the same, units and all."""
    # Units and private coordinates as well, which equality passes over:
    # the result takes the patch's coordinates, so a change would be lost.
    before, after = expected.coord_map, made.coord_map
    return (
        made.dims == expected.dims
        and made == expected
        and set(before) == set(after)
        and all(before[x].units == after[x].units for x in before)
        and all(before[x] == after[x] for x in before if x.startswith("_"))
    )


def _warn_if_integers(made, blended) -> None:
    """Warn that integers or booleans an operation made are blended to floats."""
    if np.dtype(made).kind not in "biu":
        return
    msg = (
        f"Blending windows of {made} data gives {blended}: each sample is a "
        "weighted sum of the windows which hold it."
    )
    prefix = str(Path(__file__).parent.parent)
    warnings.warn(msg, UserWarning, skip_file_prefixes=(prefix,))


@dataclass(frozen=True)
class PatchWindow:
    """
    A patch and a window over it; see [`Patch.window`](`dascore.Patch.window`).

    What it does returns a patch, so a chain goes on
    from the patch: ``patch.window(time=1).mean().abs()``.

    Besides the methods written here, a patch operation called on it runs
    in each window and the windows are blended:
    ``patch.window(time=1).detrend("time")``. See
    [`Windowed`](`dascore.proc.window.Windowed`) for which operations can.
    """

    patch: Any
    dims: Mapping[str, Any] = field(default_factory=dict)
    overlap: Any = None
    step: Any = None
    samples: bool = False
    # Not `taper`, which is the patch operation of that name.
    blend_taper: Any = None
    analysis: Any = None
    ends: str = "pad"
    align: str = "center"
    fill: Any = None

    def __getattr__(self, name: str):
        """Return a patch operation which runs in each window, then blends."""
        if name.startswith("_"):
            raise AttributeError(name)
        method = getattr(type(self.patch), name, None)
        if method is None or getattr(method, "__processor__", None) is None:
            msg = (
                f"{name} is not a patch operation a window can run. A window "
                "runs the patch operations written as processors, such as "
                "detrend, taper or pass_filter; anything else is a function "
                "given to apply."
            )
            raise AttributeError(msg)

        def windowed(*args, **kwargs):
            """Run the operation in each window, and blend the windows."""
            return self._windowed(method, *args, **kwargs)

        # Its name and nothing else: the method's signature takes a patch,
        # and its bypass and processor are the operation's without a window.
        windowed.__name__ = windowed.__qualname__ = name
        return windowed

    def _windowed(self, method, /, *args, **kwargs):
        """Run a patch method's operation in each window, and blend them."""
        self._check_blended()
        operation = captured_operation(method, self.patch, *args, **kwargs)
        return Windowed(
            operation=operation,
            overlap=self.overlap,
            step=self.step,
            taper=self.blend_taper,
            analysis=self.analysis,
            fill=self.fill,
            samples=self.samples,
            **self.dims,
        ).run(self.patch)

    def _check_blended(self):
        """Refuse the options which only a reduction reads."""
        if self.ends != "pad" or self.align != "center":
            msg = (
                "Windows which are blended are always padded at the ends and "
                "have no label to align; ends and align apply to reductions."
            )
            raise ParameterError(msg)

    def __post_init__(self):
        """Refuse a window which could not be one, before anything is done."""
        if not self.dims:
            msg = "A window needs a dimension and its size, e.g. time=0.5."
            raise ParameterError(msg)
        if self.overlap is not None and self.step is not None:
            msg = "step and overlap are mutually exclusive."
            raise ParameterError(msg)
        _check_options(self.ends, self.align)
        if self.fill is not None and (isinstance(self.fill, str) or np.ndim(self.fill)):
            msg = f"fill must be a single number; got {self.fill!r}."
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
        - Windows start before the data and run past its end, holding the
          window's `fill` there, so every sample is under a whole taper. The
          function sees those values.
        """
        self._check_blended()
        return TileApply(
            function=function,
            mode=mode,
            overlap=self.overlap,
            step=self.step,
            taper=self.blend_taper,
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
            Given an array with one more axis than the data, the last,
            which holds each window's samples, and an ``axis=-1`` keyword
            naming it, which it reduces. One of ``"mean"``, ``"median"``,
            ``"min"``, ``"max"``, ``"std"`` or ``"sum"`` names the NumPy
            function.
        *args, **kwargs
            Passed on to the function.

        Returns
        -------
        Patch
            One value for each window, in float64 (complex for complex
            data), along the windowed dimension. A window is labelled by
            one of the input's samples, the one its `align` names, so with
            a step of one sample the patch keeps its shape and coordinates.

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
        - Only one dimension can be windowed.
        - A reduction named by a string, with nothing else given to it, may
          be answered by pandas when the window is long, which is much
          faster and agrees with NumPy to rounding: a few parts in ten
          million for float32, less for float64.
        """
        if self.blend_taper is not None or self.analysis is not None:
            msg = "A reduction blends nothing, so it takes no taper or analysis."
            raise ParameterError(msg)
        if self.fill is not None and self.ends == "drop":
            msg = (
                "fill completes the windows which reach past the data and "
                "ends='drop' leaves them out; give one or the other."
            )
            raise ParameterError(msg)
        return WindowReduce(
            function=function,
            function_args=args,
            function_kwargs=kwargs or None,
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
