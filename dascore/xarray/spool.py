"""
Convert a spool to a lazy, dask-backed xarray DataTree.

The tree is partitioned exactly as `chunk` partitions patches, and each
segment is described by the merge `chunk` would perform: its
coordinates and attrs come from `PatchAssembler.describe_output`, and
its members read as the windows of stored arrays that merge would read,
or load as the patches it would load. The merged dimension's coordinate
is served by the lazy index in `dascore.xarray.index`.
"""

from __future__ import annotations

import typing
from collections import defaultdict
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

import dascore as dc
from dascore.config import get_config
from dascore.constants import CONFLICT, SpoolType
from dascore.exceptions import InvalidFiberIOError, ParameterError, PatchConversionError
from dascore.utils.misc import optional_import
from dascore.utils.time import to_float
from dascore.xarray.patch import _to_dataarray


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


def _segment_chunks(meta, block_size, dim):
    """
    Dask's chunks for one segment: member boundaries, cut by the ceiling.

    A selection reads only what it asks for whatever these are, since it
    fuses into the source's own read. They bound a *whole* read instead:
    computing the segment asks for one chunk at a time, so a chunk is
    the most one task has to hold. Members bound them because a chunk
    spanning two members reads both, and a member loading as a patch
    stays whole, since splitting it would load it once per piece.
    """
    sizes = dict(zip(meta.dims, meta.coords.shape, strict=True))
    limit = _samples_per_block(block_size, meta.dtype, sizes, dim)
    axis = meta.dims.index(dim)
    along = []
    for member in meta.members:
        ceiling = limit if member.source is not None else None
        pieces = _block_pieces(member.coords.shape[axis], ceiling)
        along.extend(stop - start for start, stop in pieces)
    return tuple(tuple(along) if d == dim else (sizes[d],) for d in meta.dims)


def _window_and_pick(key, size: int):
    """
    Split one dimension's index into a window to read and what to pick.

    Dask hands a source integers and forward slices, picking positions
    and reversing itself from what they read. Reading is contiguous, so
    an integer or a stride is read as the window enclosing what it
    selects and picked out of it; the bounds are computed arithmetically,
    never by building an array the length of the axis.
    """
    if not isinstance(key, slice):
        # an integer drops its dimension, and must still do so here
        index = range(size)[key]
        return (index, index + 1), 0
    span = range(*key.indices(size))
    if not span:
        return (0, 0), slice(None)
    assert span.step > 0, "dask reverses what it reads itself"
    return (span[0], span[-1] + 1), slice(None, None, span.step)


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

    def __init__(self, meta, rows, dim, assembler, frame, units, locations):
        self.members, self.rows, self.locations = meta.members, rows, locations
        # `frame` has the member rows' columns, which is all a load asks of it
        self.assembler, self.frame, self.units = assembler, frame, units
        self.dims, self.dtype = meta.dims, meta.dtype
        self.shape = meta.coords.shape
        self.ndim = len(self.shape)
        self.axis = self.dims.index(dim)
        counts = [x.coords.shape[self.axis] for x in self.members]
        self.offsets = np.cumsum([0, *counts]).tolist()

    def __getitem__(self, key) -> np.ndarray:
        """Read what ``key`` selects, reading no more of a member than that."""
        keys = key if isinstance(key, tuple) else (key,)
        keys = keys + (slice(None),) * (self.ndim - len(keys))
        pairs = [_window_and_pick(k, n) for k, n in zip(keys, self.shape, strict=True)]
        window = [slice(*bounds) for bounds, _ in pairs]
        low, high = pairs[self.axis][0]
        parts = []
        for num, member in enumerate(self.members):
            start, stop = self.offsets[num], self.offsets[num + 1]
            if max(low, start) < min(high, stop):
                window[self.axis] = slice(
                    max(low, start) - start, min(high, stop) - start
                )
                parts.append(self._read(num, member, tuple(window)))
        if parts:
            out = parts[0] if len(parts) == 1 else np.concatenate(parts, self.axis)
        else:
            out = np.empty(tuple(hi - lo for (lo, hi), _ in pairs), dtype=self.dtype)
        # slices and integers pick orthogonally, as xarray means them
        return out[tuple(pick for _, pick in pairs)]

    def _read(self, num, member, window) -> np.ndarray:
        """Read one member's window, cast as the merge casts it."""
        if member.source is not None:
            source, path = member.source[window], self.locations[num]
            try:
                # the resolved path, which keeps a remote store's options
                with dc.io.core._open_array_reader(source, path) as load:
                    data = load(source)
            except (InvalidFiberIOError, ParameterError) as error:
                msg = self._stale(num, f"raised {error}")
                raise PatchConversionError(msg) from error
        else:
            # chunk's own member load: residuals, trims and units, as it merges
            patch = self.assembler._load_trimmed_patch(
                self.rows[num], self.frame, self.units
            )
            if patch.dims != self.dims or patch.shape != member.coords.shape:
                msg = self._stale(num, f"gave shape {patch.shape}")
                raise PatchConversionError(msg)
            data = np.asarray(patch.data)[window]
        if member.cast_via is not None:
            data = data.astype(member.cast_via)
        return data.astype(self.dtype, copy=False)

    def _stale(self, num, found) -> str:
        """Why a member which did not load as indexed is refused."""
        return (
            f"Loading '{self.rows[num].get('source_path', '<memory>')}' {found}, "
            "but the spool index promised shape "
            f"{self.members[num].coords.shape}. The source changed after "
            "indexing; run spool.update() and convert again."
        )


def _with_loaded_range(member_rows, members, source_rows, dim):
    """
    State each member's loaded range beside its cut, as a source range.

    A residual withholds the range from the rows, since a window of the
    stored array cannot stand for its selection; but a cut member's
    labels are the samples its loaded patch has inside the cut, and the
    rows planned over state that patch's range.
    """
    from dascore.utils.patch_assembly import (  # noqa: PLC0415
        SOURCE_RANGE_ENDS,
        source_range_column,
    )

    # the plan's members are the resolver's rows, in order
    loaded = source_rows.set_index("_patch_row").loc[members["_patch_row"]]
    ends = zip(SOURCE_RANGE_ENDS, ("min", "max", "step"), strict=True)
    return member_rows.assign(
        **{
            source_range_column(dim, end): loaded[f"{dim}_{name}"].to_numpy()
            for end, name in ends
        }
    )


def _misstated_merge(resolver, row, dim) -> bool:
    """Whether a member loads a chunk's snapped merge its envelope misstates."""
    path = str(row.get("source_path"))
    for prefix, plan in getattr(resolver.loader, "plan_entries", dict)().items():
        snaps = plan.mode == "chunk" and plan.merge_kwargs.get("fill_value") is None
        if path.startswith(prefix) and plan.dim == dim and snaps:
            rows = plan.member_rows.sort_values(f"{dim}_min")
            rows = rows[rows["output_id"] == int(path.removeprefix(prefix))]
            meta = plan._assembler().describe_output(rows.to_dict("records"))
            assert meta is not None, "an output described its members can be"
            coord = meta.coords.coord_map[dim]
            return not coord.evenly_sampled or coord.max() != rows[f"{dim}_max"].max()
    return False


def _location(resolver, row):
    """Where a member's file opens, with any storage options its path carries."""
    loader, path, _, _ = resolver._array_read_info(row)
    return loader.resolve_path(path)


def _store_attrs(attrs) -> dict:
    """A segment's attrs as a store holds them."""
    out = dict(attrs)
    # the index holds no history, and an empty id is one the rows cannot know
    out.pop("history", None)
    for name in ("data_id", "origin_id"):
        if not out.get(name):
            out.pop(name, None)
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
    conflict: CONFLICT = "raise",
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
        makes each source patch one block. A patch which cannot be read
        as a window of its stored array (a format without
        `FiberIO.read_array`, or one whose data units or coordinates the
        index cannot stand for) stays one block whatever this says:
        splitting it would read the file once per block instead of once.

    Notes
    -----
    A selection reads only the samples it names: the arrays are backed
    by a source spanning each segment, and dask fuses the selection into
    that source's own read, which asks each member for its part of the
    window and no other member at all.

    Requires ``xarray`` and ``dask``. Coordinates associated with a
    dimension (rather than defining one) are not carried into the tree.
    Building the tree checks every local source file the index recorded
    a stat for, and refuses one which changed (or can no longer be
    stat'ed). Any other source (a remote one, say), and any file changed
    after the tree is built, is only caught when a read finds it is not
    the shape the index promised.

    Each segment's attrs are those of the patch `chunk` merges, with
    data units as a string. They state no history, which the index does
    not hold, and state ``data_id`` or ``origin_id`` only where the index
    knows the merged patch's without loading it.

    The merged dimension's coordinate, when it is a range or segmented,
    is served lazily by `dascore.xarray.index.CoordIndex`: its labels are
    computed on demand from the merged coordinate rather than stored, so an
    arbitrarily long merged time coordinate costs nothing to build, even
    when sub-tolerance gaps or slightly different sampling steps leave it
    segmented rather than one range. Label selection on it answers
    as `Patch.sel` does, and reading ``.values`` or asking for the
    pandas index materializes labels on demand. Other dimensions keep
    materialized labels, which per-channel arrays (gains, offsets) align
    with. Before xarray 2026.9 a lazy index aligned only with lazy ones,
    and a lazy and a materialized index whose labels differ still do not:
    to combine a segment with arrays indexed along the merged dimension,
    or reindex it, give it an ordinary index first, which reads all its
    labels into memory (so select first where possible), e.g.
    ``data.drop_indexes("time").set_xindex("time")``, or give the other
    array this index type, ``other.drop_indexes("time").set_xindex("time",
    CoordIndex)``, which reads no lazy labels where the two are the same
    coordinate.

    A spool with pending value-range selections cannot be converted: the
    catalog states such bounds as candidacy rather than sample positions,
    so the arrays cannot be sized without reading. Convert first and
    select on the tree (e.g. ``sel(time=...)``), or select with
    ``samples=True`` on dimensions, which stays exact. Pending inventory
    enrichment is likewise refused, since the tree would omit the
    enriched attributes, as is a spool chunked along ``dim`` after a
    sample (or relative) selection, unless that selection was on ``dim``
    alone and every chunk is a window inside a selected patch: planning
    it again along ``dim`` would otherwise lose the selection. Convert
    the selected spool instead.
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
    from dascore.io.index.planned import (  # noqa: PLC0415
        PlanResolver,
        collapse_working_df,
        derived_catalog,
    )
    from dascore.utils.chunk_plan import build_chunk_plan  # noqa: PLC0415
    from dascore.utils.patch_assembly import plan_data_units  # noqa: PLC0415

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
    own = spool._catalog.resolver
    selected = []
    if isinstance(own, PlanResolver) and own.dim == dim:
        selected = [set(c) for c, s, r in own.parent_residuals if s or r]
    collapsed = collapse_working_df(spool._catalog) if selected else None
    if collapsed is not None and (
        any(x - {dim} for x in selected) or not collapsed["_modified"].all()
    ):
        # Re-planning along the same dimension loads the source rows
        # without the selection the plan was made after (#1373): only a
        # member trimmed inside its selected source, by a selection on
        # this dimension alone, still loads the samples it stands for.
        msg = (
            f"Cannot convert a spool chunked along '{dim}' after a sample "
            f"selection or a relative one: planning it along '{dim}' again "
            "would lose the selection. Convert the selected spool instead."
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
        working,
        tolerance=tolerance,
        conflict=conflict,
        group=group,
        _bridge_holes=True,
        **chunk_kwargs,
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
    member_rows = resolver.member_rows
    # Reads trust the index for where a window lies, which a rewritten
    # file would silently move; measuring each local file now catches it.
    if changed := resolver.changed_sources(member_rows, skip_unmeasured=True):
        msg = (
            f"{len(changed)} source file(s) changed since the spool was "
            f"indexed, e.g. '{changed[0]}'; run spool.update() and convert "
            "again."
        )
        raise PatchConversionError(msg)
    assembler = resolver._assembler(fill=False)
    if resolver.parent_residuals:
        # a residual re-trims the loaded patch, which a window read skips
        assembler = replace(assembler, array_source=None)
        member_rows = _with_loaded_range(member_rows, plan.members, source_rows, dim)
    by_output = defaultdict(list)
    for row in assembler._df_to_dict_list(member_rows):
        by_output[row["output_id"]].append(row)

    def _segment(out):
        """One output's dask array and its coordinates and attrs."""
        rows = by_output[out["output_id"]]
        units = plan_data_units([x.get("data_units") for x in rows])
        rows = sorted(rows, key=lambda x: x[f"{dim}_min"])
        meta = assembler.describe_output(rows, units)
        if meta is None:
            msg = (
                f"Cannot size a lazy array: a patch spanning {out[f'{dim}_min']} "
                f"to {out[f'{dim}_max']} records no sampling step for one of "
                "its dimensions, or no dtype, in the spool index; an index "
                "made before dtypes were recorded needs spool.update()."
            )
            raise PatchConversionError(msg)
        if any(_misstated_merge(resolver, x, dim) for x in rows):
            msg = (
                f"Cannot size a lazy array: a patch spanning {out[f'{dim}_min']} to "
                f"{out[f'{dim}_max']} loads a chunk which snapped its members "
                "onto one grid; convert the spool before chunking it."
            )
            raise PatchConversionError(msg)
        locations = [
            None if x.source is None else _location(resolver, row)
            for x, row in zip(meta.members, rows, strict=True)
        ]
        array = da.from_array(
            _SegmentSource(meta, rows, dim, assembler, member_rows, units, locations),
            chunks=_segment_chunks(meta, block_size, dim),
            meta=np.empty((0,) * len(meta.dims), dtype=meta.dtype),
            # the source is the read; dask must not copy it into the
            # graph piecewise, nor hash a thing which opens files
            asarray=False,
            # Output ids are local to a plan; task keys must also
            # distinguish independently converted spools.
            name=f"dascore-segment-{resolver.token}-{out['output_id']}",
        )
        # The merged coordinate stays lazy: its labels cost 8 bytes a
        # sample materialized, which for a long merge dwarfs everything
        # else the tree holds. The others are short, and a materialized
        # index is what per-channel arrays align on.
        attrs = _store_attrs(meta.attrs)
        return _to_dataarray(array, meta.coords, attrs, (dim,), held=False)

    tree = {}
    for code, sub in outputs.groupby(codes.to_numpy(), sort=True):
        node = f"/{names[code]}"
        first = sub.iloc[0]
        node_attrs = {key: first[key] for key in group_attrs if pd.notnull(first[key])}
        tree[node] = xr.Dataset(attrs=node_attrs)
        for segment, (_, out) in enumerate(sub.sort_values(f"{dim}_min").iterrows()):
            tree[f"{node}/segment_{segment}"] = xr.Dataset({"data": _segment(out)})
    return xr.DataTree.from_dict(tree)
