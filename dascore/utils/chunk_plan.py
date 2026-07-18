"""
Chunk planning over the flat patch relation.

Implements the "Chunking formalities" spec: the planner consumes the
catalog's flat relation (one row per patch: `{dim}_min/max/step` envelopes,
`_{dim}_def_key` structural identity, attr columns) and produces a
[`ChunkPlan`](`dascore.utils.chunk_plan.ChunkPlan`) — an outputs table (one row
per output patch) plus a members table binding each output to trimmed
slices of source patches. No patch data is touched; assembly happens later.

Portions of the interval/instruction math were ported from the old
`ChunkManager` (now removed) with the spec's adjudicated corrections
applied; `Spool.chunk` runs on these plans.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

import dascore as dc
from dascore.exceptions import (
    ChunkError,
    CoordMergeError,
    InvalidSpoolQueryError,
    ParameterError,
)
from dascore.utils.chunk import get_intervals
from dascore.utils.misc import get_middle_value, is_range
from dascore.utils.pd import _remove_overlaps, get_interval_columns
from dascore.utils.time import is_datetime64, is_timedelta64, to_float, to_timedelta64

# Columns which never participate in conflict policing and never carry to
# outputs: source bookkeeping (outputs are not file rows).
_SOURCE_COLUMNS = ("path", "file_format", "file_version", "source_patch_id")
# The default continuity tolerance; looser values warn when they force
# merges (#662).
_DEFAULT_TOLERANCE = 1.5


@dataclass(frozen=True)
class ChunkPlan:
    """
    A materialization-free description of a chunk operation.

    Attributes
    ----------
    outputs
        One row per output patch: `{dim}_min/max/step` for the chunked
        dimension, an `output_id`, and all carried columns (group attrs,
        dims, structural def keys, conflict-policed attrs).
    members
        Instruction rows binding outputs to sources: `output_id`,
        `_patch_id`, the exact `{dim}_min/max` trim for that member, and
        `_modified` (False when the member loads whole).
    dim
        The chunked dimension.
    value
        The requested chunk length (None for merge mode).
    params
        Resolved parameters (group attrs, tolerances, overlap,
        keep_partial, conflict, snap_coords, missing_dim) — recorded, not
        referencing config.
    """

    outputs: pd.DataFrame
    members: pd.DataFrame
    dim: str
    value: Any
    params: dict = field(default_factory=dict)

    @property
    def merge_mode(self) -> bool:
        """Return True when this plan merges (no segmenting length)."""
        return self.value is None


def _resolve_group_attrs(group, columns) -> tuple[str, ...]:
    """Resolve the group attrs: per-call > config; explicit names must exist."""
    if group is not None:
        group = (group,) if isinstance(group, str) else tuple(group)
        if missing := [x for x in group if x not in columns]:
            msg = (
                f"group attribute(s) {missing} do not exist on any patch "
                "in the spool."
            )
            raise InvalidSpoolQueryError(msg)
        return group
    # Config (and default) names are best-effort.
    return tuple(x for x in dc.get_config().groupby_attrs if x in columns)


def samples_adjusted_envelopes(df: pd.DataFrame, residuals) -> pd.DataFrame:
    """
    Adjust envelope columns for patch-local samples residuals.

    A ``samples=True`` index window trims each patch at load, so the
    planner must consume the trimmed envelopes or it publishes outputs
    that lie entirely outside the selected samples (phantom empties).
    Only non-negative index windows adjust (Python-slice clamping per
    patch); anything else leaves the envelope as a candidacy superset —
    exactness is always re-applied at load, and the adjustment only
    exists so plans reflect the truth.
    """

    def _usable_index(value) -> bool:
        return value is None or (isinstance(value, int | np.integer) and value >= 0)

    df = df.copy(deep=False)
    for coords, samples in residuals:
        if not samples:
            continue
        for name, value in coords.items():
            cols = [f"{name}_min", f"{name}_max", f"{name}_step"]
            if not set(cols).issubset(df.columns) or not is_range(value):
                continue
            lo_idx, hi_idx = value
            if not (_usable_index(lo_idx) and _usable_index(hi_idx)):
                continue
            mins, maxs, steps = (df[c] for c in cols)
            new_min = mins if lo_idx is None else mins + lo_idx * steps
            new_max = maxs if hi_idx is None else mins + hi_idx * steps
            df[cols[0]] = new_min.where(new_min <= maxs, other=maxs)
            df[cols[1]] = new_max.where(new_max <= maxs, other=maxs)
            # rows whose window starts past their end contribute nothing
            df = df[df[cols[0]] <= df[cols[1]]]
    return df


def _ensure_patch_id(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the positional identity fallback for plain dataframes."""
    if "_patch_id" in df.columns:
        return df
    return df.assign(_patch_id=np.arange(len(df)))


def _dim_def_key_columns(df: pd.DataFrame, name: str) -> list[str]:
    """Return def-key column names for every non-chunked dimension."""
    dim_names: set[str] = set()
    if "dims" in df.columns:
        for dims_str in df["dims"].dropna().unique():
            dim_names.update(str(dims_str).split(","))
    dim_names.discard(name)
    return [f"_{x}_def_key" for x in sorted(dim_names)]


def _sampling_group(step: pd.Series, tolerance: float) -> pd.Series:
    """Label rows whose steps are within relative tolerance (spec 2.3)."""
    col = to_float(step.values)
    order = np.argsort(col)
    sorted_col = col[order]
    prev = np.roll(sorted_col, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = (sorted_col - prev) / sorted_col
    out_of_threshold = diff > tolerance
    group = np.cumsum(out_of_threshold)
    return pd.Series(group[np.argsort(order)], index=step.index)


def _continuity_group(start, stop, step, tolerance) -> pd.Series:
    """Label maximal near-contiguous runs (spec 2.4)."""
    args = np.argsort(start.to_numpy())
    start_sorted = start.iloc[args]
    stop_sorted = stop.iloc[args]
    step_sorted = step.iloc[args]
    stop_cum_max = stop_sorted.cummax()
    end_markers = stop_cum_max.shift() + step_sorted * tolerance
    has_gap = start_sorted > end_markers
    group = has_gap.astype(np.int64).cumsum()
    return group[start.index]


def _partition(
    df, name, group_attrs, tolerance, sampling_tolerance
) -> tuple[pd.Series, bool]:
    """
    Return (partition labels, forced_merge): rows sharing a label may
    combine (spec 2).

    Components: group attrs, dims signature, structural def keys of
    non-chunked coords, sampling group, and continuity group. Continuity
    is evaluated *within* each other-component cell so unrelated patches
    can never bridge a gap. `forced_merge` is True when a loosened
    tolerance merged patches the default would have kept apart (#662);
    the caller owns warning about it.
    """
    start, stop, step = get_interval_columns(df, name)
    cols = [x for x in group_attrs if x in df.columns]
    if "dims" in df.columns:
        cols.append("dims")
    # Structural identity: def keys of non-chunked *dimensions* only
    # (spec 2.2). Non-dimensional coordinate conflicts are policed at
    # assembly per the `conflict` argument, never partitioned on.
    cols += [x for x in _dim_def_key_columns(df, name) if x in df.columns]
    base = (
        df.groupby(cols, dropna=False, sort=False).ngroup()
        if cols
        else pd.Series(0, index=df.index)
    )
    samp = _sampling_group(step, sampling_tolerance)
    cell = base.astype(str) + "_" + samp.astype(str)
    cont = pd.Series(0, index=df.index, dtype=np.int64)
    forced_merge = False
    for _, index in df.groupby(cell, sort=False).groups.items():
        sub = df.loc[index]
        s, e, st = get_interval_columns(sub, name)
        labels = _continuity_group(s, e, st, tolerance).astype(np.int64)
        cont.loc[index] = labels
        if tolerance > _DEFAULT_TOLERANCE and not forced_merge:
            default = _continuity_group(s, e, st, _DEFAULT_TOLERANCE)
            forced_merge = default.nunique() > labels.nunique()
    return cell + "_" + cont.astype(str), forced_merge


def _user_stacklevel() -> int:
    """Return the warn stacklevel pointing at the first non-dascore frame.

    Plans are built at several call depths (spool.chunk, spool.chunk_plan,
    build_chunk_plan directly), so a fixed stacklevel would blame library
    frames for some entries.
    """
    import inspect

    # The dascore package directory, resolved from the package itself so
    # this does not depend on this module's location within it.
    package_dir = str(Path(dc.__file__).resolve().parent)
    # Frames after this helper's own align exactly with warn's numbering:
    # level 1 is the frame calling warn.
    for level, frame_info in enumerate(inspect.stack()[1:], start=1):
        filename = str(Path(frame_info.filename).resolve())
        if not filename.startswith(package_dir):
            return level
    return 1


def _coerce_length_overlap(value, overlap, start_dtype):
    """Coerce the chunk length/overlap to the dimension's span dtype."""
    time_like = is_datetime64(start_dtype) or is_timedelta64(start_dtype)
    if time_like:
        value = to_timedelta64(value) if value is not None else None
        overlap = to_timedelta64(overlap) if overlap is not None else None
    return value, overlap


def _coord_owner(col: str, coord_names: set[str]) -> str | None:
    """
    Return the coordinate owning an envelope column, if any.

    Ownership is decided by matching the full name against known
    coordinates with an interval suffix; splitting on the first
    underscore would mis-assign columns of dims like `event_time`.
    """
    for suffix in ("_min", "_max", "_step", "_units"):
        if col.endswith(suffix):
            base = col[: -len(suffix)]
            if base in coord_names:
                return base
    return None


def _police_columns(sub: pd.DataFrame, name, conflict) -> dict:
    """
    Return the carried column values for one partition (spec 2.5/6.4).

    Group attrs, dims, and def keys are single-valued by construction.
    Remaining public attrs must be single-valued, policed by `conflict`.
    """
    dims = set(str(sub.iloc[0].get("dims", "")).split(","))
    coord_names = dims | {name}
    carried: dict[str, Any] = {}
    for col in sub.columns:
        if col.startswith("_") or col in _SOURCE_COLUMNS:
            continue
        owner = _coord_owner(col, coord_names)
        if owner == name:  # chunk-dim envelope columns are rebuilt
            continue
        values = sub[col].unique()
        single = len(values) == 1 or (len(values) and pd.isnull(values).all())
        if single:
            carried[col] = values[0]
            continue
        # Group attrs and dims are partition keys, so they are always
        # single-valued above and never reach the conflict policy here.
        if owner is not None or conflict == "raise":
            msg = (
                f"Cannot merge on dim {name} because all values for "
                f"{col} are not equal. Consider using the `conflict` "
                "argument to loosen this restriction."
            )
            raise CoordMergeError(msg)
        if conflict == "keep_first":
            carried[col] = sub[col].iloc[0]
        # conflict == "drop": omit the column entirely.
    # Structural (dimension) def keys carry — single-valued by partitioning.
    for col in _dim_def_key_columns(sub, name):
        if col in sub.columns:
            carried[col] = sub[col].iloc[0]
    return carried


def _build_members(sub: pd.DataFrame, outputs: pd.DataFrame, name) -> pd.DataFrame:
    """
    Bind one partition's outputs to trimmed source slices.

    Sources are ordered by (start, _patch_id); overlapping coverage is
    deduplicated so the earlier source owns the overlap (D3: complete
    overlaps keep the first member, deterministically).
    """
    min_name, max_name = f"{name}_min", f"{name}_max"
    step_name = f"{name}_step"
    sub = sub.sort_values([min_name, "_patch_id"], kind="stable")
    original = sub[[min_name, max_name]].reset_index(drop=True)
    sub = _remove_overlaps(sub, name)
    # Fully-covered sources become degenerate after start correction; they
    # contribute nothing (deterministic keep-first dedup).
    keep = sub[min_name].values <= sub[max_name].values
    sub = sub[keep]
    original = original[keep].reset_index(drop=True)
    # sub and outputs are always non-empty here: a partition too short to
    # yield an interval raises ChunkError in the caller (and the earliest
    # source is never fully covered), so both keep at least one row.
    steps = sub[step_name].values
    src1 = sub[min_name].values
    src2 = sub[max_name].values
    chu1 = outputs[min_name].values
    chu2 = outputs[max_name].values
    out_ids = outputs["output_id"].to_numpy()
    patch_ids = sub["_patch_id"].to_numpy()
    orig_min = original[min_name].to_numpy()
    orig_max = original[max_name].to_numpy()
    modified_src = (
        sub["_modified"].to_numpy()
        if "_modified" in sub
        else np.zeros(len(sub), dtype=bool)
    )
    # Map each output onto the source rows it draws from.
    starts_ind = np.searchsorted(src1, chu1, side="right") - 1
    ends_ind = np.searchsorted(src2, chu2, side="left")
    rows = []
    for out_num, (a, b) in enumerate(zip(starts_ind, ends_ind)):
        a = max(int(a), 0)
        for src_num in range(a, int(b) + 1):
            if src_num >= len(sub):
                continue
            lo = max(src1[src_num], chu1[out_num])
            hi = min(src2[src_num], chu2[out_num])
            if lo > hi:  # pragma: no cover -- searchsorted boundary guard
                # Sources within a partition are continuous (partitioning
                # splits on gaps) and start-corrected, so searchsorted does
                # not offer a non-overlapping source in practice; this guards
                # against a boundary off-by-one rather than a reachable state.
                continue
            unchanged = (
                lo == orig_min[src_num]
                and hi == orig_max[src_num]
                and not modified_src[src_num]
            )
            rows.append(
                {
                    "output_id": out_ids[out_num],
                    "_patch_id": patch_ids[src_num],
                    min_name: lo,
                    max_name: hi,
                    step_name: steps[src_num],
                    "_modified": not unchanged,
                }
            )
    return pd.DataFrame(rows)


def build_chunk_plan(
    df: pd.DataFrame,
    *,
    overlap=None,
    keep_partial: bool = False,
    snap_coords: bool = True,
    tolerance: float = 1.5,
    conflict: Literal["drop", "raise", "keep_first"] = "raise",
    group=None,
    missing_dim: Literal["raise", "drop"] = "raise",
    **kwargs,
) -> ChunkPlan:
    """
    Build a chunk plan from a flat patch relation.

    Parameters mirror `Spool.chunk` (see the chunking formalities spec);
    exactly one keyword names the dimension to chunk and its length
    (`None`/`...` merges).
    """
    if len(kwargs) != 1:
        msg = (
            "Chunking only supported along one dimension. You passed "
            f"kwargs: {kwargs}"
        )
        raise ParameterError(msg)
    ((name, value),) = kwargs.items()
    value = None if value is Ellipsis else value
    merge_mode = pd.isnull(value)
    if merge_mode and (keep_partial or overlap):
        msg = (
            "When chunk value is None (ie chunking is used for merging) "
            "keep_partial and overlap are not supported."
        )
        raise ParameterError(msg)
    if not merge_mode:
        zero = to_timedelta64(0) if is_timedelta64(value) else 0
        if value <= zero:
            msg = "Chunk value must be greater than 0."
            raise ParameterError(msg)
    if missing_dim not in ("raise", "drop"):
        msg = f"missing_dim must be 'raise' or 'drop', got {missing_dim!r}"
        raise ParameterError(msg)
    if conflict not in ("drop", "raise", "keep_first"):
        msg = "conflict must be 'drop', 'raise', or 'keep_first', " f"got {conflict!r}"
        raise ParameterError(msg)

    min_name, max_name = f"{name}_min", f"{name}_max"
    if min_name not in df.columns and not df.empty:
        msg = f"No patch in the spool has a {name!r} dimension to chunk."
        raise ChunkError(msg)
    empty_members = pd.DataFrame(
        columns=["output_id", "_patch_id", min_name, max_name, "_modified"]
    )
    params = dict(
        overlap=overlap,
        keep_partial=keep_partial,
        snap_coords=snap_coords,
        tolerance=tolerance,
        conflict=conflict,
        missing_dim=missing_dim,
        group=_resolve_group_attrs(group, set(df.columns)),
        sampling_group_tolerance=dc.get_config().sampling_group_tolerance,
    )
    if df.empty:
        outputs = pd.DataFrame(columns=[min_name, max_name, "output_id"])
        return ChunkPlan(outputs, empty_members, name, value, params)
    df = _ensure_patch_id(df)
    # Missing chunk-dim envelopes (spec 7 / D2).
    null_rows = pd.isnull(df[min_name]) | pd.isnull(df[max_name])
    if null_rows.any():
        if missing_dim == "raise":
            bad = df.loc[null_rows, "_patch_id"].tolist()
            msg = (
                f"{int(null_rows.sum())} patch(es) lack the chunk dimension "
                f"{name!r} (patch ids {bad[:5]}...). Pass missing_dim='drop' "
                "to exclude them."
            )
            raise ChunkError(msg)
        df = df[~null_rows]
    if df.empty:
        outputs = pd.DataFrame(columns=[min_name, max_name, "output_id"])
        return ChunkPlan(outputs, empty_members, name, value, params)

    labels, forced_merge = _partition(
        df, name, params["group"], tolerance, params["sampling_group_tolerance"]
    )
    if forced_merge:
        msg = (
            f"There is a gap in the patch along dimension {name} but a "
            f"merge tolerance of {tolerance} was used to force merging "
            "the patches. As a result, some patches in the chunked spool "
            "may be unevenly sampled, or have their sampling rate increased."
        )
        warnings.warn(msg, UserWarning, stacklevel=_user_stacklevel())
    value_c, overlap_c = _coerce_length_overlap(value, overlap, df[min_name].dtype)
    out_frames, member_frames = [], []
    next_id = 0
    # Deterministic partition order (spec 8): by (partition min, smallest
    # member patch id) — never by anything derived from input row order.
    grouped = df.groupby(labels, sort=False)
    stats = grouped.agg(_min=(min_name, "min"), _pid=("_patch_id", "min"))
    part_order = stats.sort_values(["_min", "_pid"], kind="stable").index
    groups = grouped.groups
    for label in part_order:
        sub = df.loc[groups[label]]
        start, stop, step = get_interval_columns(sub, name)
        part_step = get_middle_value(step.values)  # D7: one step everywhere
        g_start, g_stop = start.min(), stop.max()
        if merge_mode:
            start_stop = np.atleast_2d(np.asarray([g_start, g_stop]))
        else:
            try:
                start_stop = get_intervals(
                    g_start,
                    g_stop,
                    value_c,
                    overlap=overlap_c,
                    # interval arithmetic is over (direction-free) envelope
                    # values; a descending coordinate's negative step would
                    # invert the final partial interval
                    step=abs(part_step),
                    keep_partials=keep_partial,
                )
            except ChunkError:  # partition too short; skip (D8)
                continue
        sub_sorted = sub.sort_values([min_name, "_patch_id"], kind="stable")
        carried = _police_columns(sub_sorted, name, conflict)
        outputs = pd.DataFrame(start_stop, columns=[min_name, max_name])
        outputs[f"{name}_step"] = part_step
        for col, val in carried.items():
            outputs[col] = val
        outputs["output_id"] = np.arange(next_id, next_id + len(outputs))
        next_id += len(outputs)
        members = _build_members(sub, outputs, name)
        # Plan invariant: every published output has at least one member.
        # An advertised row that cannot assemble is never surfaced as a
        # runtime error; it is not surfaced at all.
        fed = set(members["output_id"]) if not members.empty else set()
        outputs = outputs[outputs["output_id"].isin(fed)]
        out_frames.append(outputs)
        member_frames.append(members)
    if not out_frames or all(x.empty for x in out_frames):
        msg = "Could not chunk. No segments with sufficient length found."
        raise ChunkError(msg)
    outputs = pd.concat(out_frames, ignore_index=True)
    members = pd.concat(
        [x for x in member_frames if not x.empty] or [empty_members],
        ignore_index=True,
    )
    return ChunkPlan(outputs, members, name, value, params)
