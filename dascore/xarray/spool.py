"""
Convert a spool to a lazy, dask-backed xarray DataTree.

The tree is partitioned exactly as `chunk` partitions patches; blocks
load through the same resolver path a chunked spool loads through, and
evenly sampled time coordinates are served by the lazy index in
`dascore.xarray.index`.
"""

from __future__ import annotations

import typing
from typing import Any, Literal

import numpy as np
import pandas as pd

from dascore.constants import SpoolType
from dascore.exceptions import PatchConversionError
from dascore.utils.misc import optional_import
from dascore.utils.time import to_float


def _np_scalar(value):
    """A relation-row scalar as numpy; pandas scalars make object coords."""
    if isinstance(value, pd.Timestamp):
        return value.to_datetime64()
    if isinstance(value, pd.Timedelta):
        return value.to_timedelta64()
    return value


def _envelope_coord(low, high, step, get_coord):
    """A dimension coordinate stated by its index envelope, either order."""
    low, high, step = _np_scalar(low), _np_scalar(high), _np_scalar(step)
    if pd.isnull(step) or to_float(step) == 0:
        if low == high:
            return get_coord(data=[low])
        msg = (
            f"Cannot size a lazy array: a dimension spanning {low} to "
            f"{high} records no sampling step in the spool index."
        )
        raise PatchConversionError(msg)
    if to_float(step) < 0:
        # A descending coordinate starts at its max; stop is exclusive.
        return get_coord(start=high, stop=low + step, step=step)
    return get_coord(min=low, max=high + step, step=step)


def _member_coord(low, high, step, env_low, env_high, get_coord, units=None):
    """
    The coordinate a member presents inside its trim window.

    The member's full coordinate is rebuilt from its envelope and trimmed
    by the coordinate's own select, so block sizes and sample labels
    follow exactly the rule loading follows, not a parallel rounding.
    Units ride along so a unit-bearing tolerance can be read against the
    merged coordinate, as chunk reads it.

    Also returns the window as half-open sample indices on the member's
    own grid — the form `FiberIO.read_array` takes.
    """
    low, high, step = _np_scalar(low), _np_scalar(high), _np_scalar(step)
    env_low, env_high = _np_scalar(env_low), _np_scalar(env_high)
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
    full = get_coord(min=env_low, max=env_high + step, step=step, units=units)
    coord, indexer = full.select((low, high))
    # plan invariant: a published member always presents at least a sample
    assert len(coord), "a plan member never presents an empty window"
    # a range select of an evenly sampled coordinate is a unit slice
    assert isinstance(indexer, slice), "range select yields a slice"
    start, stop, stride = indexer.indices(len(full))
    assert stride == 1, "range select never strides"
    return coord, (start, stop)


def _lazy_temporal_index(name, coord):
    """
    Return a lazy xarray index for an evenly sampled temporal coordinate.

    None when the coordinate cannot be served lazily — an irregular
    (segmented or array) coordinate, a descending one, or a numeric one,
    whose materialized values are short in practice.
    """
    from dascore.core.coords import CoordRange  # noqa: PLC0415

    step = getattr(coord, "step", None)
    if not isinstance(coord, CoordRange) or step is None or pd.isnull(step):
        return None
    if not (
        np.issubdtype(coord.dtype, np.datetime64)
        or np.issubdtype(coord.dtype, np.timedelta64)
    ):
        return None
    if np.asarray(step).astype("int64") <= 0:
        return None
    # function-level: xarray is an optional dependency
    from dascore.xarray.index import TemporalRangeIndex  # noqa: PLC0415

    return TemporalRangeIndex.from_coord(name, coord)


def _load_xarray_block(resolver, row, dim, lims, dims, shape, dtype, window=None):
    """
    Load one dask block: a source patch trimmed to its member window.

    Runs at compute time, once per block. When the row's format offers a
    `read_array` fast path, only the raw array for the sample window is
    read; otherwise member loading goes through the plan resolver — the
    same path a chunked spool loads through — so residuals, units, and
    nested plans are honored. The select re-applies the window exactly
    since read hints only reduce reading.
    """
    array = None
    stated = row.get("dims")
    src_dims = tuple(stated.split(",")) if isinstance(stated, str) else ()
    # The row must state the same dimensions the tree promises, or the
    # transpose below could not be built; disagreement means the patch
    # path, whose own errors say what is wrong.
    if window is not None and sorted(src_dims) == sorted(dims):
        array = resolver._load_member_array(row, {dim: window})
    if array is None:
        patch = resolver._load_member(row).select(**{dim: lims})
        if patch.dims != dims:
            patch = patch.transpose(*dims)
        array = patch.data
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

    Notes
    -----
    Requires ``xarray`` and ``dask``. Coordinates associated with a
    dimension (rather than defining one) are not carried into the tree,
    and coordinates are rebuilt from their indexed envelopes, whose
    numeric values are floats — an integer-valued dimension coordinate
    comes back as floats.

    An evenly sampled datetime/timedelta dimension coordinate is served
    lazily: its labels are computed from start and step on demand
    rather than stored, so an arbitrarily long merged time coordinate
    costs nothing to build. Label selection on it resolves
    arithmetically — a scalar must land on a sample (or pass
    ``method="nearest"``), a slice keeps every sample within its
    inclusive endpoints — and reading ``.values`` or asking for the
    pandas index materializes labels on demand. A merged coordinate
    which is not one even range (a sub-tolerance seam between
    differently sampled members) spells its values out, as it must.

    A spool with pending value-range selections cannot be converted: the
    catalog states such bounds as candidacy rather than sample positions,
    so the arrays cannot be sized without reading. Convert first and
    select on the tree (e.g. ``sel(time=...)``), or select with
    ``samples=True`` on dimensions, which stays exact. Pending inventory
    enrichment is likewise refused, since the tree would omit the
    enriched attributes.
    """
    xr = optional_import("xarray")
    dask = optional_import("dask")
    da = optional_import("dask.array")
    # function-level to avoid circular imports through the package root
    from dascore.core.coords import concat_coords, get_coord  # noqa: PLC0415
    from dascore.io.index.planned import PlanResolver, derived_catalog  # noqa: PLC0415
    from dascore.units import carries_units  # noqa: PLC0415
    from dascore.utils.chunk_plan import (  # noqa: PLC0415
        _normalize_chunk_units,
        build_chunk_plan,
    )
    from dascore.utils.misc import get_middle_value  # noqa: PLC0415

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
    for selected, samples in spool._catalog.residuals:
        # A samples selection on a dimension adjusts that dimension's
        # envelopes exactly; anything else changes what loads in ways the
        # envelopes do not state.
        if not selected or (samples and set(selected) <= all_dims):
            continue
        kind = (
            "sample selections on associated coordinates"
            if samples
            else ("value selections")
        )
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
        _env_low=members["_patch_id"].map(norm[f"{dim}_min"]),
        _env_high=members["_patch_id"].map(norm[f"{dim}_max"]),
        _env_anchored=~modified,
    )
    # One shared graph node for the resolver, not a copy inside every
    # block task — it can hold live patches or a large member table.
    resolver_ref = dask.delayed(resolver, pure=True)
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
            axis = dims.index(dim)
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
                    m[f"{dim}_step"],
                    m["_env_low"],
                    m["_env_high"],
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
                    merged = concat_coords(*member_coords)
                    if carries_units(tolerance):
                        merged = merged.simplify(tolerance)
                    else:
                        mem_steps = [
                            abs(c.step)
                            for c in member_coords
                            if getattr(c, "step", None) is not None
                            and not pd.isnull(c.step)
                        ]
                        if mem_steps:
                            merged = merged.simplify(
                                tolerance * get_middle_value(mem_steps)
                            )
                    coord = merged
                else:
                    coord = _envelope_coord(
                        out[f"{d}_min"], out[f"{d}_max"], out[f"{d}_step"], get_coord
                    )
                sizes[d] = len(coord)
                # An evenly sampled temporal coordinate stays lazy: its
                # labels cost 8 bytes a sample materialized, which for a
                # long merged time coordinate dwarfs everything else the
                # tree holds. Irregular (segmented) coordinates are the
                # exception that must spell out its values.
                if (index := _lazy_temporal_index(d, coord)) is not None:
                    lazy_indexes[d] = index
                else:
                    coords[d] = coord.values
            blocks = []
            zipped = zip(member_coords, member_windows, mem.iterrows(), strict=True)
            for member_coord, window, (_, m) in zipped:
                count = len(member_coord)
                shape = tuple(count if d == dim else sizes[d] for d in dims)
                lims = (m[f"{dim}_min"], m[f"{dim}_max"])
                row = member_rows.iloc[int(m["_pos"])].to_dict()
                delayed = dask.delayed(_load_xarray_block)(
                    resolver_ref,
                    row,
                    dim,
                    lims,
                    dims,
                    shape,
                    dtype,
                    window if m["_env_anchored"] else None,
                )
                blocks.append(da.from_delayed(delayed, shape=shape, dtype=dtype))
            array = da.concatenate(blocks, axis=axis)
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
