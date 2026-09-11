"""
Convert a spool to a lazy, dask-backed xarray DataTree.

The tree is partitioned exactly as `chunk` partitions patches; blocks
load through the same resolver path a chunked spool loads through, and
the merged dimension's coordinate is served by the lazy index in
`dascore.xarray.index`.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from dascore.config import get_config
from dascore.constants import SpoolType
from dascore.exceptions import PatchConversionError
from dascore.utils.misc import optional_import
from dascore.utils.time import to_float
from dascore.xarray.patch import _lazy_index


def _np_scalar(value):
    """A relation-row scalar as numpy; pandas scalars make object coords."""
    if isinstance(value, pd.Timestamp):
        return value.to_datetime64()
    if isinstance(value, pd.Timedelta):
        return value.to_timedelta64()
    return value


def _envelope_coord(row, dim, get_coord):
    """A dimension coordinate stated by its row's envelope, either order."""
    # function-level: patch_assembly imports the io package, which imports this
    from dascore.utils.patch_assembly import coord_from_row  # noqa: PLC0415

    # the row's exact grid, where it carries one, sizes the array as
    # loading will; the whole-tick envelope can be a sample off
    if (coord := coord_from_row(row, dim)) is not None:
        return coord
    low, high, step = (
        _np_scalar(row[f"{dim}_{end}"]) for end in ("min", "max", "step")
    )
    if pd.isnull(step) or to_float(step) == 0:
        if low == high:
            return get_coord(data=[low])
        msg = (
            f"Cannot size a lazy array: a dimension spanning {low} to "
            f"{high} records no sampling step in the spool index."
        )
        raise PatchConversionError(msg)
    # A descending coordinate starts at its max; stop is exclusive. An
    # ascending one lands here only when its units were converted.
    start, stop = (high, low + step) if to_float(step) < 0 else (low, high + step)
    return get_coord(start=start, stop=stop, step=step)


def _member_coord(low, high, source_row, dim, get_coord, units=None):
    """
    The coordinate a member presents inside its trim window.

    The member's full coordinate is rebuilt from its row and trimmed by
    the coordinate's own select, so block sizes and sample labels follow
    exactly the rule loading follows, not a parallel rounding. A time
    coordinate is rebuilt on its exact grid, which the whole-tick
    envelope can miss by a sample; a numeric one from the envelope in
    the plan's units, since members joined along it must share a dtype
    and may not share a unit or an integer grid. Units ride along so a
    unit-bearing tolerance can be read against the merged coordinate, as
    chunk reads it.

    Also returns the window as half-open sample indices on the member's
    own grid — the form `FiberIO.read_array` takes.
    """
    low, high = _np_scalar(low), _np_scalar(high)
    env_low, env_high, step = (
        _np_scalar(source_row[f"{dim}_{end}"]) for end in ("min", "max", "step")
    )
    # Rows state units as strings; anything else (absent, null) is none.
    units = units if isinstance(units, str) and units else None
    # A single-sample merge dimension never reaches here: the planner
    # refuses to chunk a dimension it cannot order.
    if pd.isnull(step) or to_float(step) == 0:
        msg = (
            f"Cannot size a lazy block: a patch spanning {low} to {high} "
            "records no sampling step in the spool index."
        )
        raise PatchConversionError(msg)
    from dascore.utils.patch_assembly import coord_from_row  # noqa: PLC0415

    full = None
    if np.asarray(step).dtype.kind == "m":
        full = coord_from_row(source_row, dim, units=units)
    if full is None:
        full = get_coord(min=env_low, max=env_high + step, step=step, units=units)
    coord, indexer = full.select((low, high))
    # plan invariant: a published member always presents at least a sample
    assert len(coord), "a plan member never presents an empty window"
    # a range select of an evenly sampled coordinate is a unit slice
    assert isinstance(indexer, slice), "range select yields a slice"
    start, stop, stride = indexer.indices(len(full))
    assert stride == 1, "range select never strides"
    return coord, (start, stop)


def _samples_per_block(block_size, dtype, sizes, dim) -> int | None:
    """
    How many samples along ``dim`` fit in one block, or None for no limit.

    A budget of zero is no limit rather than no samples: it is how a
    caller says one block per source patch.

    A block's other dimensions are whole, so its size grows only along
    the merge dimension: one sample of it costs the product of the rest.
    """
    if not block_size:
        return None
    row_bytes = dtype.itemsize
    for name, size in sizes.items():
        if name != dim:
            row_bytes *= size
    return max(1, int(block_size) // max(row_bytes, 1))


def _block_pieces(count: int, limit: int | None) -> tuple[tuple[int, int], ...]:
    """
    Cut ``count`` samples into contiguous half-open pieces.

    The pieces are as even as whole samples allow, so no block is much
    smaller than its siblings; ``limit`` is a ceiling, not a target.
    """
    if limit is None or count <= limit:
        return ((0, count),)
    pieces = -(-count // limit)
    size, extra = divmod(count, pieces)
    # the remainder is spread one sample at a time over the leading
    # pieces; repeating a rounded-up size instead would put the whole
    # deficit in the last piece (101 in 30s as 26, 26, 26, 23)
    sizes = [size + 1] * extra + [size] * (pieces - extra)
    out, start = [], 0
    for length in sizes:
        out.append((start, start + length))
        start += length
    return tuple(out)


def _segment_chunks(members, block_size, dtype, sizes, dims, dim):
    """
    Dask's chunks for one segment: member boundaries, cut by the ceiling.

    A selection reads only what it asks for whatever these are, since it
    fuses into the source's own read. They bound a *whole* read instead:
    computing the segment asks for one chunk at a time, so a chunk is
    the most one task has to hold. Members bound them because a chunk
    spanning two members reads both.
    """
    limit = _samples_per_block(block_size, dtype, sizes, dim)
    along = []
    for member in members:
        splittable = member.window is not None and member.resolver.can_read_array(
            member.row
        )
        pieces = _block_pieces(member.count, limit if splittable else None)
        along.extend(stop - start for start, stop in pieces)
    return tuple(tuple(along) if d == dim else (sizes[d],) for d in dims)


class _SegmentSource:
    """
    One merged segment's data, as an array which reads what it is asked.

    Dask fuses a selection into the read of whatever it slices, but only
    while nothing sits in between; joining one array per member with
    `concatenate` puts a layer there, and every selection then reads the
    whole of whichever blocks it touches. Spanning the segment instead
    keeps the selection and the read adjacent, so a window reaches the
    members it covers and no others, and each of those reads only its
    part of it.
    """

    def __init__(self, members, dim, dims, shape, dtype):
        self.members = tuple(members)
        self.dim = dim
        self.dims = tuple(dims)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)
        self.axis = self.dims.index(dim)

    def __getitem__(self, key) -> np.ndarray:
        """Read what ``key`` selects, reading no more of a member than that."""
        keys = key if isinstance(key, tuple) else (key,)
        keys = keys + (slice(None),) * (self.ndim - len(keys))
        pairs = [_window_and_key(k, n) for k, n in zip(keys, self.shape, strict=True)]
        bounds = [window for window, _ in pairs]
        out = self._by_window(bounds)
        rest = tuple(rest for _, rest in pairs)
        # a contiguous window is already the answer; anything else picks
        # its samples out of the window read for it
        if any(not isinstance(x, slice) or x != slice(None) for x in rest):
            out = out[rest]
        return out

    def _by_window(self, bounds) -> np.ndarray:
        """Read one contiguous window, in the tree's dims."""
        low, high = bounds[self.axis]
        parts = []
        for member in self.members:
            start, stop = (
                max(low, member.offset),
                min(high, member.offset + member.count),
            )
            if start >= stop:
                continue
            parts.append(
                member.read(
                    start - member.offset, stop - member.offset, bounds, self.shape
                )
            )
        if not parts:
            shape = tuple(hi - lo for lo, hi in bounds)
            return np.empty(shape, dtype=self.dtype)
        out = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=self.axis)
        return out.astype(self.dtype, copy=False)


def _window_and_key(key, size: int):
    """
    Split one dimension's index into a window to read and what to take.

    Reading is contiguous, so anything which is not a step-one slice --
    an integer, a list of positions, a stride, a reversal -- is read as
    the window enclosing what it selects and then picked out of it. The
    second half is `slice(None)` when the window is already the answer.
    """
    if isinstance(key, slice):
        start, stop, step = key.indices(size)
        if step == 1:
            return (start, max(stop, start)), slice(None)
    positions = np.arange(size)[key]
    if np.ndim(positions) == 0:
        # an integer drops its dimension, and must still do so here
        low = int(positions)
        return (low, low + 1), 0
    if positions.size == 0:
        return (0, 0), slice(None)
    low = int(positions.min())
    return (low, int(positions.max()) + 1), positions - low


@dataclass
class _SegmentMember:
    """One member's place in a segment, and how to read part of it."""

    resolver: Any
    row: Mapping
    dim: str
    dims: tuple[str, ...]
    coord: Any
    offset: int
    count: int
    window: tuple[int, int] | None
    dtype: np.dtype

    def read(self, start: int, stop: int, bounds, full_shape) -> np.ndarray:
        """Read samples ``start`` to ``stop`` of this member, in tree dims."""
        sub = self.coord.select((start, stop), samples=True)[0]
        shape = tuple(
            (stop - start) if d == self.dim else (hi - lo)
            for d, (lo, hi) in zip(self.dims, bounds, strict=True)
        )
        window = None
        if self.window is not None:
            window = (self.window[0] + start, self.window[0] + stop)
        lims = (sub.min(), sub.max())
        return _load_xarray_block(
            self.resolver,
            self.row,
            self.dim,
            lims,
            self.dims,
            shape,
            self.dtype,
            window,
            bounds,
            full_shape,
        )


def _load_xarray_block(
    resolver,
    row,
    dim,
    lims,
    dims,
    shape,
    dtype,
    window=None,
    bounds=None,
    full_shape=None,
):
    """
    Load one piece of a segment: a source patch trimmed to a window.

    Runs at compute time. When the row's format offers a `read_array`
    fast path, only the raw array for the sample window is read;
    otherwise member loading goes through the plan resolver — the same
    path a chunked spool loads through — so residuals, units, and nested
    plans are honored. The select re-applies the window exactly since
    read hints only reduce reading.

    ``bounds`` is a half-open sample window per tree dimension. The
    merge dimension's is already spent (``window`` and ``lims`` say it);
    the rest narrow what is read where the format can take them, and are
    applied to the array otherwise.
    """
    array = None
    stated = row.get("dims")
    src_dims = tuple(stated.split(",")) if isinstance(stated, str) else ()
    others = _other_windows(dim, dims, full_shape or shape, bounds)
    # The row must state the same dimensions the tree promises, or the
    # transpose below could not be built; disagreement means the patch
    # path, whose own errors say what is wrong.
    if window is not None and sorted(src_dims) == sorted(dims):
        array = resolver._load_member_array(row, {dim: window, **others})
    if array is None:
        patch = resolver._load_member(row).select(**{dim: lims})
        if patch.dims != dims:
            patch = patch.transpose(*dims)
        array = patch.data
        if others:
            # the reader could not take them, so they are spent here
            array = array[
                tuple(slice(*others[d]) if d in others else slice(None) for d in dims)
            ]
    elif src_dims != dims:
        array = np.transpose(array, [src_dims.index(d) for d in dims])
    if array.shape != shape:
        msg = (
            f"Loaded block from '{row.get('source_path', '<memory>')}' has shape "
            f"{array.shape}, but the spool index promised {shape}. The source "
            "file changed after indexing; run spool.update() and convert again."
        )
        raise PatchConversionError(msg)
    # A member narrower than the segment's combined dtype upcasts here so
    # the array holds the dtype its metadata states.
    return array.astype(dtype, copy=False)


def _other_windows(dim, dims, sizes, bounds) -> dict[str, tuple[int, int]]:
    """
    Sample windows for every dimension but the merged one, when narrowed.

    ``sizes`` is the whole segment's shape: a window is only worth
    stating where it asks for less than all of its dimension.
    """
    if bounds is None:
        return {}
    out = {}
    for name, (low, high), size in zip(dims, bounds, sizes, strict=True):
        if name != dim and (low, high) != (0, size):
            out[name] = (low, high)
    return out


def _xarray_group_nodes(outputs, group_attrs):
    """Name a DataTree node per output group, by the spool's naming rule."""
    # The same rule which names a repr's tracks and a coverage plot's
    # lanes; a node name must also be a valid single path component.
    from dascore.utils.display import ACQUISITION_ATTR, group_names  # noqa: PLC0415

    if not group_attrs:
        frame = pd.DataFrame(index=[0])
        codes = pd.Series(0, index=outputs.index)
    else:
        codes = outputs.groupby(group_attrs, dropna=False, sort=True).ngroup()
        frame = (
            outputs[group_attrs]
            .assign(_code=codes)
            .drop_duplicates("_code")
            .sort_values("_code")
            .drop(columns="_code")
            .reset_index(drop=True)
        )
    fallback = ACQUISITION_ATTR if ACQUISITION_ATTR in frame.columns else None
    names = group_names(frame, fallback=fallback)
    if bad := [x for x in names if "/" in x or x in ("", ".", "..")]:
        msg = (
            f"Group name(s) {bad} cannot name a DataTree node (a name may "
            "not contain '/' or be '.' or '..'). Pass different `group` "
            "attributes."
        )
        raise PatchConversionError(msg)
    # A literal value can collide with a generated fallback name (a group
    # tagged "group 0" beside an untagged one); a shared node path would
    # silently overwrite arrays.
    if len(set(names)) != len(names):
        dupes = sorted({x for x in names if names.count(x) > 1})
        msg = (
            f"Group name(s) {dupes} name more than one group of patches. "
            "Pass `group` attributes which tell the groups apart."
        )
        raise PatchConversionError(msg)
    return names, codes


def spool_to_xarray(
    spool: SpoolType,
    dim: str = "time",
    group: str | typing.Sequence[str] | None = None,
    tolerance=1.5,
    conflict: Literal["drop", "raise", "keep_first"] = "raise",
    block_size: str | int | None = None,
):
    """
    Convert a spool to a lazy, dask-backed xarray DataTree.

    Patches are partitioned exactly as [`chunk`](`dascore.Spool.chunk`)
    partitions them: one tree node per group of related patches, holding
    one child node per merged output (``segment_0`` onward, ordered along
    ``dim``), each with a dask-backed ``data`` variable. Node names
    follow the spool's own naming rule — the one its repr's tracks and a
    coverage plot's lanes use. Building the tree reads no patch data —
    every shape, dtype, and coordinate comes from the spool's metadata —
    and computing a selection loads only the source patches it touches.

    Parameters
    ----------
    spool
        The spool to convert.
    dim
        The dimension segments are merged along.
    group
        Attributes which partition patches into unrelated groups, exactly
        as `chunk` uses them. Defaults to the config's ``patch_kind_attrs``.
    tolerance
        The continuity tolerance deciding when a gap splits segments, as
        in `chunk` (not the sampling-step grouping tolerance).
    conflict
        How attribute conflicts within a segment resolve, as in `chunk`.
    block_size
        The most a single dask block may hold, as a byte count or a
        string dask parses ("256MiB", "1GB"). This bounds a *bulk* read:
        computing a whole segment asks for one block at a time, so a
        source patch bigger than this is read in several windows along
        ``dim`` rather than whole. It does not affect a selection, which
        reads only what it asks for whatever the blocks are. None takes
        the configured ``xarray_block_size`` (256 MiB by default); zero
        makes each source patch one block. A patch whose format cannot
        hand back a window (see `FiberIO.read_array`) stays one block
        whatever this says: splitting it would read the file once per
        block instead of once.

    Notes
    -----
    A selection reads only the samples it names: the arrays are backed
    by a source spanning each segment, and dask fuses the selection into
    that source's own read, which asks each member for its part of the
    window and no other member at all.

    Requires ``xarray`` and ``dask``. Coordinates associated with a
    dimension (rather than defining one) are not carried into the tree,
    and coordinates are rebuilt from their indexed envelopes, whose
    numeric values are floats — an integer-valued dimension coordinate
    comes back as floats.

    The merged dimension's coordinate, when it is a range or segmented,
    is served lazily by `dascore.xarray.index.CoordIndex`: its labels are
    computed on demand from the merged coordinate rather than stored, so an
    arbitrarily long merged time coordinate costs nothing to build, even
    when sub-tolerance gaps or slightly different sampling steps leave it
    segmented rather than one range. Label selection on it answers
    as `Patch.sel` does, and reading ``.values`` or asking for the
    pandas index materializes labels on demand. Other dimensions keep
    materialized labels, which per-channel arrays (gains, offsets) align
    with. xarray aligns a lazy index only with lazy ones: to combine a
    segment with arrays indexed along the merged dimension, or reindex
    it, give it an ordinary index first, which reads its labels, e.g.
    ``data.drop_indexes("time").set_xindex("time")``.

    A spool with pending value-range selections cannot be converted: the
    catalog states such bounds as candidacy rather than sample positions,
    so the arrays cannot be sized without reading. Convert first and
    select on the tree (e.g. ``sel(time=...)``), or select with
    ``samples=True`` on dimensions, which stays exact. Pending inventory
    enrichment is likewise refused, since the tree would omit the
    enriched attributes.
    """
    xr = optional_import("xarray")
    da = optional_import("dask.array")
    # the tree's arrays carry the `.dc` accessor, like any conversion
    from dascore.xarray.patch import _register_accessor  # noqa: PLC0415

    _register_accessor()
    if block_size is None:
        block_size = get_config().xarray_block_size
    if isinstance(block_size, str):
        block_size = optional_import("dask.utils").parse_bytes(block_size)
    if block_size < 0:
        # a negative ceiling fits no samples, which would otherwise round
        # up to one block per sample and build a task for every one
        msg = (
            f"block_size must be a size in bytes or zero, not {block_size}. "
            "Zero reads each source patch as one block."
        )
        raise PatchConversionError(msg)
    # function-level to avoid circular imports through the package root
    from dascore.core.coords import concat_coords, get_coord  # noqa: PLC0415
    from dascore.io.index.planned import PlanResolver, derived_catalog  # noqa: PLC0415
    from dascore.utils.chunk_plan import (  # noqa: PLC0415
        _normalize_chunk_units,
        build_chunk_plan,
    )
    from dascore.utils.gaps import GapTolerance  # noqa: PLC0415

    source_rows, working = spool._plan_frames(dim)
    if not len(working):
        return xr.DataTree()
    if spool._enrich_kwargs:
        msg = (
            "Cannot convert a spool with pending inventory enrichment: the "
            "tree would omit the enriched attributes. Convert the spool "
            "before enriching it."
        )
        raise PatchConversionError(msg)
    all_dims = {
        d for dims_str in working["dims"].dropna() for d in str(dims_str).split(",")
    }
    for selected, samples, relative in spool._catalog.residuals:
        # A samples selection on a dimension adjusts that dimension's
        # envelopes exactly; anything else changes what loads in ways the
        # envelopes do not state. A relative bound resolves against each
        # patch as it loads, so the relation states a candidacy superset
        # rather than the sample positions the blocks would need.
        if not selected or (samples and set(selected) <= all_dims):
            continue
        if samples:
            kind = "sample selections on associated coordinates"
        elif relative:
            kind = "relative selections"
        else:
            kind = "value selections"
        msg = (
            f"Cannot convert a spool with pending {kind} (on "
            f"{sorted(selected)}): such bounds are candidacy, not sample "
            "positions, so the lazy arrays cannot be sized. Convert "
            "first and select on the tree, or select dimensions with "
            "samples=True."
        )
        raise PatchConversionError(msg)
    steps = working.get(f"{dim}_step")
    if steps is not None and (to_float(steps.values) < 0).any():
        msg = (
            f"A descending '{dim}' coordinate is not supported by "
            f"to_xarray; sort the patches along {dim} first."
        )
        raise PatchConversionError(msg)
    chunk_kwargs: dict[str, Any] = {dim: None}
    plan = build_chunk_plan(
        working, tolerance=tolerance, conflict=conflict, group=group, **chunk_kwargs
    )
    outputs = plan.outputs
    group_attrs = list(plan.params["group"])
    names, codes = _xarray_group_nodes(outputs, group_attrs)
    # The same derivation chunk performs: its resolver is what knows how
    # to load one member row, whatever kind of spool this is.
    catalog = derived_catalog(
        source_rows=source_rows,
        plan=plan,
        parent=spool._catalog,
        merge_kwargs={
            "conflict": conflict,
            "snap_coords": True,
            "tolerance": plan.params["tolerance"],
        },
        mode="chunk",
        origin_path=spool.spool_path,
    )
    resolver = catalog.resolver
    assert isinstance(resolver, PlanResolver)  # a chunk derivation always is
    # plan.members and the resolver's member_rows are the same rows in the
    # same order; the former keeps _patch_id (for the source grid), the
    # latter is what the resolver loads from. Verify the invariant, since
    # derived_catalog does not promise it to other callers.
    member_rows = resolver.member_rows.reset_index(drop=True)
    members = plan.members.reset_index(drop=True)
    check_cols = ["output_id", f"{dim}_min", f"{dim}_max"]
    assert members[check_cols].equals(member_rows[check_cols])
    # Member grids in the plan's normalized units, so trims and envelopes
    # speak the same unit; the plan normalized the same frame identically.
    norm = _normalize_chunk_units(working, dim).set_index("_patch_id")
    # A working row which is itself a trim (a collapsed plan's member)
    # states a trimmed envelope, so a sample window measured against it
    # is not anchored on the file's grid; such members load by value.
    if "_modified" in norm.columns:
        modified = members["_patch_id"].map(norm["_modified"]).fillna(True).astype(bool)
    else:
        modified = pd.Series(False, index=members.index)
    members = members.assign(
        _pos=np.arange(len(members)),
        _env_anchored=~modified,
    )
    envelope_cols = {
        f"{d}_{end}"
        for dims_str in outputs["dims"].unique()
        for d in str(dims_str).split(",")
        for end in ("min", "max", "step")
    }
    tree = {}
    for code, sub in outputs.groupby(codes.to_numpy(), sort=True):
        node = f"/{names[code]}"
        first = sub.iloc[0]
        node_attrs = {key: first[key] for key in group_attrs if pd.notnull(first[key])}
        tree[node] = xr.Dataset(attrs=node_attrs)
        for segment, (_, out) in enumerate(sub.sort_values(f"{dim}_min").iterrows()):
            dims = tuple(str(out["dims"]).split(","))
            if (dtype_str := out["_dtype"]) is None or not str(dtype_str):
                msg = (
                    "Cannot build a lazy array without a dtype in the spool "
                    "index; re-index the spool with spool.update()."
                )
                raise PatchConversionError(msg)
            dtype = np.dtype(dtype_str)
            mem = members[members["output_id"] == out["output_id"]]
            mem = mem.sort_values(f"{dim}_min")
            # Each member's block is sized by its own coordinate: the same
            # select which trims the block at load also counts it here,
            # and its indexer is the sample window a data-only read takes.
            member_coords, member_windows = [], []
            for _, m in mem.iterrows():
                coord, window = _member_coord(
                    m[f"{dim}_min"],
                    m[f"{dim}_max"],
                    norm.loc[m["_patch_id"]],
                    dim,
                    get_coord,
                    units=m.get(f"_{dim}_units"),
                )
                member_coords.append(coord)
                member_windows.append(window)
            coords, lazy_indexes, sizes = {}, {}, {}
            for d in dims:
                if d == dim:
                    # The same construction chunk merges by: concatenate
                    # the member coordinates truth-preservingly, then
                    # absorb sub-tolerance seams. A seam beyond tolerance
                    # stays segmented here exactly as it does there.
                    coord = concat_coords(*member_coords).simplify(
                        GapTolerance.from_user(tolerance, dim)
                    )
                else:
                    coord = _envelope_coord(out, d, get_coord)
                sizes[d] = len(coord)
                # The merged coordinate stays lazy: its labels cost 8 bytes
                # a sample materialized, which for a long merge dwarfs
                # everything else the tree holds. The others are short, and
                # a materialized index is what per-channel arrays align on.
                if d == dim and (index := _lazy_index(d, coord)) is not None:
                    lazy_indexes[d] = index
                else:
                    coords[d] = coord.values
            segment_members, offset = [], 0
            zipped = zip(member_coords, member_windows, mem.iterrows(), strict=True)
            for member_coord, window, (_, m) in zipped:
                row = member_rows.iloc[int(m["_pos"])].to_dict()
                anchored = bool(m["_env_anchored"])
                segment_members.append(
                    _SegmentMember(
                        resolver=resolver,
                        row=row,
                        dim=dim,
                        dims=dims,
                        coord=member_coord,
                        offset=offset,
                        count=len(member_coord),
                        window=window if anchored else None,
                        dtype=dtype,
                    )
                )
                offset += len(member_coord)
            shape = tuple(sizes[d] for d in dims)
            source = _SegmentSource(segment_members, dim, dims, shape, dtype)
            array = da.from_array(
                source,
                chunks=_segment_chunks(
                    segment_members, block_size, dtype, sizes, dims, dim
                ),
                meta=np.empty((0,) * len(dims), dtype=dtype),
                # the source is the read; dask must not copy it into the
                # graph piecewise, nor hash a thing which opens files
                asarray=False,
                # Output ids are local to a plan; task keys must also
                # distinguish independently converted spools.
                name=f"dascore-segment-{resolver.token}-{out['output_id']}",
            )
            attrs = {
                key: value
                for key, value in out.items()
                if not str(key).startswith("_")
                and key not in envelope_cols
                and key not in ("output_id", "dims")
                and pd.notnull(value)
            }
            data = xr.DataArray(array, dims=dims, coords=coords, attrs=attrs)
            for index in lazy_indexes.values():
                data = data.assign_coords(xr.Coordinates.from_xindex(index))
            tree[f"{node}/segment_{segment}"] = xr.Dataset({"data": data})
    return xr.DataTree.from_dict(tree)
