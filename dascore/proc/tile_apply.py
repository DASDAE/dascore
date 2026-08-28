"""
Apply a function to overlapping windows of a patch, and blend them back.

Many operations on DAS data are local: a window of the data is transformed,
edited, and put back, and the windows overlap so that the seams do not
show. The adaptive spectral filter is one; local normalization, a local
f-k filter, and tile-wise rank reduction are others. What they share is
everything but the function, and this module holds the everything.

`tile_apply` cuts the patch into tiles along one or more dimensions, hands
them to a function, and either blends them back under a taper into a patch
shaped like the input (``mode="overlap_add"``) or returns them stacked, one
tile axis per windowed dimension, for the caller to work on and put back
with `reassemble` (``mode="stack"``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import ConfigDict

import dascore as dc
from dascore.constants import PatchType
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.exceptions import (
    MissingOptionalDependencyError,
    ParameterError,
    PatchError,
)
from dascore.utils.patch import patch_function
from dascore.utils.signal import get_taper
from dascore.utils.tiles import get_tile_plan
from dascore.utils.time import is_datetime64, is_timedelta64, to_float
from dascore.utils.window import Window, resolve_window
from dascore.workflow.meta import PatchMeta
from dascore.workflow.processor import PatchProcessor, register_implementation

__all__ = ("TileApply", "reassemble", "tile_apply")

_MODES = ("overlap_add", "stack")
_ENGINES = ("auto", "numpy", "numba")


def _is_jitted(func) -> bool:
    """Whether a function is a numba dispatcher, to be given a tile at a time."""
    return hasattr(func, "py_func") and hasattr(func, "nopython_signatures")


def _offset_values(coord, offsets: np.ndarray):
    """
    Return coordinate values `offsets` samples from the coordinate's first sample.

    From the first sample, not the minimum: a descending coordinate steps
    down from where it starts.
    """
    first, step = coord.values[0], coord.step
    if is_datetime64(coord.dtype) or is_timedelta64(coord.dtype):
        return first + dc.to_timedelta64(offsets * to_float(step))
    return first + offsets * step


def _engine_for(engine: str, func, ndim: int) -> str:
    """Return which engine runs `func`, or say why neither can."""
    if engine not in _ENGINES:
        msg = f"engine must be one of {_ENGINES}; got {engine!r}."
        raise ParameterError(msg)
    jitted = _is_jitted(func)
    if engine == "auto":
        engine = "numba" if jitted else "numpy"
    if engine == "numba":
        if not jitted:
            msg = (
                "engine='numba' takes a numba-compiled function of one tile; "
                "a plain function takes the whole stack and runs on numpy."
            )
            raise ParameterError(msg)
        if ndim != 2:
            msg = (
                "A numba-compiled function is given one tile at a time over "
                f"exactly two windowed dimensions; this call windows {ndim}."
            )
            raise ParameterError(msg)
        # Deferred: the driver is optional, and importing it eagerly would
        # compile it whenever dascore is imported.
        from dascore.utils._tiles_numba import _JIT_AVAILABLE  # noqa: PLC0415

        if not _JIT_AVAILABLE:
            msg = "engine='numba' requires the optional dependency numba."
            raise MissingOptionalDependencyError(msg)
    elif jitted:
        msg = (
            "A numba-compiled function is given one tile at a time by "
            "engine='numba'; the numpy engine hands the whole stack to a plain "
            "function."
        )
        raise ParameterError(msg)
    return engine


@patch_function()
def tile_apply(
    patch: PatchType,
    function: Callable,
    *,
    mode: str = "overlap_add",
    overlap: Any = None,
    taper: Any = "hann",
    samples: bool = False,
    engine: str = "auto",
    **kwargs: Any,
) -> PatchType:
    """
    Apply a function to overlapping windows of the patch.

    Parameters
    ----------
    patch
        The patch to window.
    function
        What to do to the tiles. On the numpy engine it is given the whole
        stack at once, an array of ``[n_tiles, *window]``, and returns one of
        the same shape: one vectorized call, not one per tile. A function
        compiled with numba is given one tile at a time by the numba engine
        and returns a tile of the same shape; it compiles the first time it
        is used in each process, which takes a few seconds.
    mode
        ``"overlap_add"`` blends the tiles back under `taper` into a patch
        shaped like the input. ``"stack"`` returns the tiles unblended: each
        windowed dimension becomes an axis of tile centres, and the samples
        within a tile become a new ``{dim}_offset`` dimension at the end.
        `reassemble` blends a stack back.
    overlap
        How far each window reaches into the next, in coordinate units or,
        with `samples`, in samples; a percent is a fraction of the window;
        a mapping gives each dimension its own. Half the window when not
        given. Never more than half: the taper's ramps would cross.
    taper
        The window whose edge the taper ramps take -- any name
        `dascore.utils.signal.get_window` knows. The ramps are made
        complementary, so blended tiles of an unchanged stack return the
        input exactly. Not applied in ``"stack"`` mode.
    samples
        If True, windows and overlaps are sample counts.
    engine
        ``"numpy"``, ``"numba"``, or ``"auto"``, which is numba for a
        numba-compiled `function` over two dimensions and numpy otherwise.
    **kwargs
        The dimensions to window and the window along each, such as
        ``time=0.5`` (seconds) or ``time=64, distance=16, samples=True``.

    Returns
    -------
    Patch
        In ``"overlap_add"`` mode, a patch with the input's coordinates. In
        ``"stack"`` mode, the tiles: the windowed dimensions carry the tile
        centres, ``{dim}_start`` and ``{dim}_stop`` say where each tile came
        from in samples, and ``{dim}_offset`` is the position within a tile.

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
    >>> normalized = patch.tile_apply(agc, time=0.05, distance=50)
    >>>
    >>> # The tiles themselves, to work on and put back.
    >>> tiles = patch.tile_apply(lambda x: x, mode="stack", time=0.05, distance=50)
    >>> back = tiles.reassemble()

    Notes
    -----
    - Tiles start one stride before the data and are padded with zeros, so
      every sample of the input is covered by tiles which see a full taper
      ramp on both sides.
    - The adaptive spectral filter is this with a spectral weighting as
      `function`; see
      [`Patch.adaptive_spectral_filter`](`dascore.Patch.adaptive_spectral_filter`).
    """
    return TileApply(
        function=function,
        mode=mode,
        overlap=overlap,
        taper=taper,
        samples=samples,
        engine=engine,
        **kwargs,
    )._apply(patch)


class TileApply(PatchProcessor):
    """
    Apply a function to overlapping windows of a patch.

    The dimensions to window arrive as extras carrying their window sizes,
    as the patch function takes them; `window` turns them into sample
    counts once the coordinates are known, and `derive_meta` says what a
    stack's coordinates are without touching any data.
    """

    model_config = ConfigDict(extra="allow", frozen=True, arbitrary_types_allowed=True)

    function: Callable
    mode: str = "overlap_add"
    overlap: Any = None
    taper: Any = "hann"
    samples: bool = False
    engine: str = "auto"

    def window(self, meta: PatchMeta) -> Window:
        """Return the window in samples."""
        if self.mode not in _MODES:
            msg = f"mode must be one of {_MODES}; got {self.mode!r}."
            raise ParameterError(msg)
        selected = self.model_extra or {}
        if not selected:
            msg = "tile_apply needs a dimension and its window, e.g. time=0.5."
            raise ParameterError(msg)
        # In the patch's axis order, whatever order they were named in, so a
        # stack's tile and offset axes come out in that order too.
        position = {dim: axis for axis, dim in enumerate(meta.dims)}
        selected = dict(
            sorted(selected.items(), key=lambda kv: position.get(kv[0], -1))
        )
        return resolve_window(
            meta,
            selected,
            samples=self.samples,
            overlap=self.overlap,
            # Half the window, the most the taper's ramps allow.
            default_overlap=lambda size: size // 2,
            min_samples=2,
        )

    def derive_meta(self, meta: PatchMeta) -> PatchMeta:
        """Return the coordinates of a stack; a blend keeps the input's."""
        window = self.window(meta)
        if self.mode == "overlap_add":
            return meta
        assert window.stride is not None
        # The stride the tiles were cut at travels in attrs, so a thinned
        # stack still reassembles under the taper it was cut for.
        strides = {
            f"_tile_stride_{d}": int(s) for d, s in zip(window.dims, window.stride)
        }
        attrs = meta.attrs.update(**strides)
        return meta.update(coords=_stack_coords(meta, window), attrs=attrs)

    def kernel(self, data, meta, out_meta):
        """Tile every batch over the windowed axes; blend or stack."""
        window = self.window(meta)
        assert window.stride is not None and window.overlap is not None
        engine = _engine_for(self.engine, self.function, len(window.axes))
        plan = window.tiles(data.shape)
        data = np.asarray(data)
        ndim = len(window.axes)
        tail = tuple(range(-ndim, 0))
        moved = np.moveaxis(data, window.axes, tail)
        batches = moved.reshape((-1, *moved.shape[-ndim:]))
        if self.mode == "stack":
            if engine == "numba":
                msg = (
                    "mode='stack' hands the function the whole stack of tiles; "
                    "a numba-compiled function of one tile cannot take it."
                )
                raise ParameterError(msg)
            stacks = np.stack([self.function(plan.extract(batch)) for batch in batches])
            out = stacks.reshape((*moved.shape[:-ndim], *plan.grid, *plan.size))
            # The tile axes go where the windowed dimensions were; the
            # offsets within a tile stay at the end, already in axis order.
            n_batch = moved.ndim - ndim
            return np.moveaxis(out, range(n_batch, n_batch + ndim), window.axes)
        taper = get_taper(self.taper, window.size, window.overlap)
        if engine == "numba":
            from dascore.utils._tiles_numba import apply_jit  # noqa: PLC0415

            blended = [
                apply_jit(plan, batch, self.function, taper) for batch in batches
            ]
        else:
            blended = [plan.apply(batch, self.function, taper) for batch in batches]
        out = np.stack(blended).reshape(moved.shape)
        return np.moveaxis(out, tail, window.axes)


def _stack_coords(meta: PatchMeta, window: Window):
    """Return a stack's coordinate manager: tile centres, edges, and offsets."""
    coords = meta.coords
    plan = window.tiles(meta.shape)
    new_coords = {}
    for dim, size, stride, count in zip(
        window.dims, window.size, plan.stride, plan.grid
    ):
        claimed = (
            f"{dim}_start",
            f"{dim}_stop",
            f"{dim}_offset",
            f"_tile_source_{dim}",
        )
        for name in claimed:
            if name in coords.coord_map:
                msg = f"The patch already has a coordinate called {name}."
                raise ParameterError(msg)
        coord = coords.get_coord(dim)
        starts = np.arange(count) * stride - stride
        # The tile's middle sample, in the coordinate's units.
        centres = _offset_values(coord, starts + (size - 1) / 2)
        offsets = _offset_values(coord, np.arange(size)) - coord.values[0]
        new_coords[dim] = get_coord(data=centres, units=coord.units)
        new_coords[f"{dim}_start"] = (dim, starts)
        new_coords[f"{dim}_stop"] = (dim, starts + size)
        new_coords[f"{dim}_offset"] = get_coord(data=offsets, units=coord.units)
        # The coordinate the tiles were cut from, for reassembly.
        new_coords[f"_tile_source_{dim}"] = (None, coord)
        # And every coordinate which rode along it, a quality flag say.
        for name, aux_dims in coords.dim_map.items():
            if aux_dims == (dim,) and name != dim:
                new_coords[f"_tile_source_{dim}__{name}"] = (
                    None,
                    coords.get_coord(name),
                )
        coords = coords.disassociate_coord(dim)
    dims = (*meta.dims, *(f"{dim}_offset" for dim in window.dims))
    # Coords given bare, or as (dims, values): the manager takes either.
    coord_map: dict[str, Any] = dict(coords.get_coord_tuple_map())
    coord_map.update(new_coords)
    return get_coord_manager(coords=coord_map, dims=dims)


register_implementation("tile_apply", TileApply)


def _place_each(stacks, starts, shape, size, weights):
    """
    Blend tiles by adding each where its start says, one tile at a time.

    For a stack which is not every tile in the order it was cut -- thinned,
    or reordered -- where the plan's colour classes no longer apply.
    """
    ndim = len(shape)
    origins = np.stack(np.meshgrid(*starts, indexing="ij"), axis=-1).reshape(-1, ndim)
    low = tuple(int(min(0, s.min())) for s in starts)
    high = tuple(int(max(n, s.max() + z)) for n, s, z in zip(shape, starts, size))
    out = np.zeros(
        (stacks.shape[0], *(h - lo for lo, h in zip(low, high))),
        dtype=np.result_type(stacks, weights),
    )
    for tile, origin in enumerate(origins):
        place = tuple(
            slice(int(o - lo), int(o - lo + z)) for o, lo, z in zip(origin, low, size)
        )
        out[(slice(None), *place)] += stacks[:, tile] * weights
    inner = tuple(slice(-lo, -lo + n) for lo, n in zip(low, shape))
    return out[(slice(None), *inner)]


@patch_function()
def reassemble(patch: PatchType, *, taper: Any = "hann") -> PatchType:
    """
    Blend a stack of tiles back into the patch they were cut from.

    The inverse of [`tile_apply`](`dascore.Patch.tile_apply`) in
    ``"stack"`` mode: each tile is multiplied by a taper with complementary
    ramps and added where its ``{dim}_start`` says it came from, so a stack
    which was not changed returns the original patch exactly.

    Parameters
    ----------
    patch
        A patch `tile_apply` stacked.
    taper
        The window whose edge the taper ramps take; see `tile_apply`.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> tiles = patch.tile_apply(lambda x: x, mode="stack", time=0.2, samples=False)
    >>> assert tiles.reassemble().equals(patch, close=True)
    """
    stashed = {
        name[len("_tile_source_") :]: coord
        for name, coord in patch.coords.coord_map.items()
        if name.startswith("_tile_source_")
    }
    # `dim` for the coordinate the tiles were cut from; `dim__name` for one
    # which rode along it.
    sources = {name: coord for name, coord in stashed.items() if "__" not in name}
    riders = {name: coord for name, coord in stashed.items() if "__" in name}
    if not sources:
        msg = "reassemble takes a patch tile_apply stacked; this one has no tiles."
        raise PatchError(msg)
    dims = tuple(sources)
    offset_dims = tuple(f"{dim}_offset" for dim in dims)
    size = tuple(len(patch.get_coord(name)) for name in offset_dims)
    shape = tuple(len(coord) for coord in sources.values())
    tile_axes = tuple(patch.get_axis(dim) for dim in dims)
    offset_axes = tuple(patch.get_axis(name) for name in offset_dims)
    ndim = len(dims)
    # Batches first, then the tile grid, then the samples within a tile.
    moved = np.moveaxis(
        patch.data, (*tile_axes, *offset_axes), tuple(range(-2 * ndim, 0))
    )
    batch_shape = moved.shape[: -2 * ndim]
    grid = moved.shape[-2 * ndim : -ndim]
    stacks = moved.reshape((-1, int(np.prod(grid)), *size))
    # Where each tile goes is what its start says, whatever order the tiles
    # are in and whichever of them are still here; the taper is the one the
    # tiles were cut for, which the stride in attrs says.
    starts = [patch.get_coord(f"{dim}_start").values for dim in dims]
    strides = tuple(int(patch.attrs[f"_tile_stride_{dim}"]) for dim in dims)
    overlap = tuple(z - st for z, st in zip(size, strides))
    weights = get_taper(taper, size, overlap)
    plan = get_tile_plan(shape, size, strides)
    as_cut = all(
        len(s) == count and np.array_equal(s, np.arange(count) * st - st)
        for s, count, st in zip(starts, plan.grid, strides)
    )
    if as_cut:
        # Every tile, in the order it was cut: the plan blends the stack a
        # colour class at a time rather than a tile at a time.
        blended = np.stack([plan.overlap_add(stack, weights) for stack in stacks])
    else:
        blended = _place_each(stacks, starts, shape, size, weights)
    out = blended.reshape((*batch_shape, *shape))
    out = np.moveaxis(out, tuple(range(-ndim, 0)), tile_axes)
    # Put the source coordinates back, and drop everything the stack added.
    coord_map: dict[str, Any] = dict(patch.coords.get_coord_tuple_map())
    for dim, coord in sources.items():
        coord_map[dim] = coord
        for name in (
            f"{dim}_start",
            f"{dim}_stop",
            f"{dim}_offset",
            f"_tile_source_{dim}",
        ):
            coord_map.pop(name)
    for key, coord in riders.items():
        dim, name = key.split("__", 1)
        coord_map[name] = (dim, coord)
        coord_map.pop(f"_tile_source_{key}")
    new_dims = tuple(d for d in patch.dims if d not in offset_dims)
    coords = get_coord_manager(coords=coord_map, dims=new_dims)
    attrs = {k: v for k, v in dict(patch.attrs).items() if not k.startswith("_tile_")}
    return patch.new(data=out, coords=coords, attrs=dc.PatchAttrs(**attrs))
