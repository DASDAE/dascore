"""Block aggregation of a patch along one dimension."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, ClassVar

import numpy as np
from pydantic import ConfigDict

from dascore.core.coords import Grid, get_coord
from dascore.core.processor import PatchProcessor
from dascore.exceptions import ParameterError
from dascore.proc.window import _REDUCERS, WindowReduce, _Reductions
from dascore.utils.patch import _refuse_segments
from dascore.utils.window import resolve_window

_BOUNDARIES = ("trim", "pad")


def _block_centres(coord, strided, size: int, count: int):
    """Return a coordinate labelling each block at its centre."""
    if coord.step is None:
        # Labels: the mean of each block's labels, offset from the first.
        values = coord.values[: count * size]
        if values.dtype.kind in "iu":
            values = values.astype(np.float64)
        starts = np.arange(0, len(values), size)
        lengths = np.diff(np.append(starts, len(values)))
        means = np.add.reduceat(values - values[0], starts) / lengths
        return get_coord(data=values[0] + means, units=coord.units)
    run = strided.runs[0]
    if not getattr(run, "step_den", 0):
        # A float grid, or labels keeping a step: every label moves alike.
        return strided.update(min=strided.min() + coord.step * (size - 1) / 2)
    # An exact grid, in ticks: the centre of the first block, then each
    # block's step on from it, so a fractional step stays exact.
    step = Fraction(run.step_num, run.step_den)
    centre = run.ideal_origin + step * (size - 1) / (2 * size)
    if np.issubdtype(coord.dtype, np.integer) and centre.denominator != 1:
        # Integer labels cannot hold a part of one; the centres are floats.
        start, step = float(centre), float(step)
        shape, units = strided.shape, coord.units
        return get_coord(start=start, step=step, shape=shape, units=units)
    den = math.lcm(run.step_den, centre.denominator)
    origin = math.floor(centre)
    phase = int((centre - origin) * den)
    grid = Grid(origin, int(step * den), den, len(strided), phase=phase)
    return get_coord(runs=(grid,), dtype=coord.dtype, units=coord.units)


class Coarsen(PatchProcessor):
    """
    Reduce non-overlapping blocks along one dimension to one value each.

    The operation behind [`Patch.coarsen`](`dascore.Patch.coarsen`), where
    its parameters are described.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.proc.coarsen import Coarsen
    >>> patch = dc.get_example_patch()
    >>> out = Coarsen("mean", distance=2, samples=True)(patch)
    >>> assert out.shape[0] == patch.shape[0] // 2
    """

    model_config = ConfigDict(extra="allow", frozen=True, arbitrary_types_allowed=True)

    # Reached through `Patch.coarsen`, so it has no patch method of its own.
    name = None
    _call: ClassVar[str] = "coarsen"
    _spelled = WindowReduce._spelled

    function: Any
    function_args: tuple = ()
    function_kwargs: dict | None = None
    samples: bool = False
    boundary: str = "trim"

    _positional_fields = ("function",)

    def get_metadata(self, meta):
        """Return the coarsened metadata, the axis, block size and count."""
        if self.boundary not in _BOUNDARIES:
            msg = f"boundary must be one of {_BOUNDARIES}; got {self.boundary!r}."
            raise ParameterError(msg)
        if isinstance(self.function, str) and self.function not in _REDUCERS:
            msg = f"A reduction is a function or one of {tuple(_REDUCERS)}."
            raise ParameterError(msg)
        window = resolve_window(
            meta,
            self.model_extra or {},
            samples=self.samples,
            allow_multiple=False,
            require_evenly_sampled=False,
            enforce_lt_coord=True,
        )
        dim, axis, size = window.dims[0], window.axes[0], window.size[0]
        _refuse_segments(meta, dim, "coarsen")
        length = meta.shape[axis]
        if size > length:
            msg = f"A block of {size} samples is longer than {dim} ({length})."
            raise ParameterError(msg)
        count = length // size if self.boundary == "trim" else -(-length // size)
        coords, _ = meta.coords.isel(**{dim: slice(0, count * size, size)})
        centres = _block_centres(
            meta.get_coord(dim), coords.get_coord(dim), size, count
        )
        coords = coords._update_grid(dim, **{dim: centres})
        dtype = np.result_type(meta.dtype, np.float64)
        return meta.new(coords=coords, dtype=dtype), {"axis": axis, "size": size}

    def numpy_kernel(self, data, *, axis, size):
        """Return the reduction of each block; a partial last one if kept."""
        named = isinstance(self.function, str)
        function = _REDUCERS[self.function] if named else self.function
        moved = np.moveaxis(data, axis, -1)
        full = moved.shape[-1] // size
        blocks = moved[..., : full * size].reshape(*moved.shape[:-1], full, size)
        args, kwargs = self.function_args, self.function_kwargs or {}
        out = function(blocks, *args, axis=-1, **kwargs)
        if self.boundary == "pad" and full * size < moved.shape[-1]:
            tail = function(moved[..., full * size :], *args, axis=-1, **kwargs)
            out = np.concatenate([out, np.expand_dims(tail, -1)], axis=-1)
        out = np.asarray(out, dtype=np.result_type(data.dtype, np.float64))
        return np.moveaxis(out, -1, axis)


@dataclass(frozen=True)
class PatchCoarsen(_Reductions):
    """A patch and its blocks; see [`Patch.coarsen`](`dascore.Patch.coarsen`)."""

    patch: Any
    dims: Mapping[str, Any] = field(default_factory=dict)
    samples: bool = False
    boundary: str = "trim"

    def reduce(self, function: Callable | str, *args, **kwargs):
        """
        Reduce each block to one value with a function.

        The function is given an array whose last axis holds each block's
        samples and an ``axis=-1`` keyword; a name from ``mean``,
        ``median``, ``min``, ``max``, ``std`` or ``sum`` is the NumPy one.
        """
        return Coarsen(
            function=function,
            function_args=args,
            function_kwargs=kwargs or None,
            samples=self.samples,
            boundary=self.boundary,
            **self.dims,
        ).run(self.patch)
