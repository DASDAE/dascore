"""
Derived catalogs: chunk/concat plans as first-class catalog rows.

A restructuring operation materializes the current view's membership
into a fresh in-memory catalog whose *patch rows are the plan outputs*;
a `PlanResolver` turns an output row back into a Patch by loading the
member source patches through the parent's resolver and trimming or
merging them (the existing assembly engine). Every catalog operation —
select, order, window, union, equality, serialization — then runs the
identical code path for planned and identity spools.

Single-writer rule: derived catalogs are always fresh in-memory
databases; the on-disk index is only ever written by the directory
syncer, and views never write.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from dataclasses import replace
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from upath import UPath

import dascore as dc
from dascore.core.coords import _EXACT_GRID_FIELDS, CoordSummary
from dascore.core.source import ArraySource, _text_offsets
from dascore.exceptions import UnknownFiberFormatError
from dascore.io.core import FiberIO
from dascore.io.index.backend import get_backend
from dascore.io.index.catalog import (
    CompositeResolver,
    FileResolver,
    PatchCatalog,
    PatchResolver,
    _row_source_patch_key,
    apply_exact_residuals,
    directory_decodes,
)
from dascore.io.index.indexer import scan_unit_stats
from dascore.io.index.ingest import (
    CoordRecord,
    PatchRecord,
    SourceRecord,
    _coord_record,
    index_runs,
    stored_runs,
    typed_value,
)
from dascore.io.index.schema import SOURCE_STAT_COLUMNS
from dascore.units import get_quantity
from dascore.utils.attrs import _is_missing
from dascore.utils.chunk_plan import (
    _SOURCE_COLUMNS,
    PIECE_COLUMNS,
    _concatenated_steps,
    _ensure_patch_row,
    _SampleTracker,
    grid_fits,
    patch_local_adjusted_envelopes,
    stretch_coordinate,
)
from dascore.utils.gaps import GapTolerance, RunRecord
from dascore.utils.misc import is_range
from dascore.utils.misc import stated_units as _stated_units
from dascore.utils.patch import concatenate_planned
from dascore.utils.patch_assembly import (
    SOURCE_RANGE_ENDS,
    PatchAssembler,
    fill_on_grid,
    source_range_column,
)
from dascore.utils.paths import is_local_path, is_memory_uri
from dascore.utils.time import to_float

# Row columns which name dc.read's own keyword arguments; passing one along
# as a trim hint would collide with the value the loader already supplies.
_READ_KWARGS = ("path", "file_format", "file_version")

PLAN_SCHEME = "plan://"
# columns that are structural/positional rather than patch attributes
_NON_ATTR = {"output_id", "dims", "coord_names", "patch"}


def _source_units_column(name: str) -> str:
    """Return the private column recording a member's own unit spelling.

    Deliberately not ``_{name}_source_units``: a coordinate may be named
    ``{name}_source``, whose own unit column has exactly that spelling.
    Every coordinate unit column ends in ``_units``, so a name ending in
    ``_source`` instead can never be one.
    """
    return f"_{name}_units_source"


def _def_key_data_id(key) -> str | None:
    """Recover the coordinate's data_id from a stored def key.

    Value-identity keys are ``fp:{data_id}`` (see CoordRecord.def_key);
    any other prefix names a summary, not the values.
    """
    if not (isinstance(key, str) and key.startswith("fp:")):
        return None
    return key[3:]


def _ns(value) -> int | None:
    """Convert a datetime/timedelta-like envelope value to ns int."""
    if value is None or pd.isnull(value):
        return None
    if isinstance(value, pd.Timedelta | np.timedelta64):
        return int(pd.Timedelta(value).value)
    return int(pd.Timestamp(value).value)


def _num(value) -> float | None:
    """Convert a numeric envelope value to float."""
    if value is None or pd.isnull(value):
        return None
    return float(value)


def _is_local(path: str) -> bool:
    """Whether a stored path names a local file this process can stat."""
    try:
        return is_local_path(path)
    except ValueError:
        # a scheme no filesystem here claims is not a local file
        return False


# What a stat column holds where the index recorded none. No file's mtime
# or size can be it, and it is a value rather than a null because the
# column must be read as exact int64; a nullable integer column (the
# planned Arrow move) would hold the null itself and drop the sentinel.
NO_STAT = np.iinfo(np.int64).min


def _stat_column(series: pd.Series) -> np.ndarray:
    """One stat column as exact int64, with what it never measured as NO_STAT."""
    # Never through float64: a nanosecond mtime does not survive it.
    if series.dtype == np.dtype(np.int64):
        return series.to_numpy()
    return series.fillna(NO_STAT).to_numpy(np.int64)


def _row_str(value) -> str:
    """A row cell as a string, with the frame's nulls (None/NaN) as ""."""
    return "" if value is None or pd.isnull(value) else str(value)


def _numeric_dtype_str(stored) -> str:
    """The numeric dtype a row states for a coordinate, else float64."""
    if not isinstance(stored, str) or not stored:
        return "float64"
    # the column is written from a real dtype, as _row_values also assumes
    dtype = np.dtype(stored)
    return str(dtype) if np.issubdtype(dtype, np.number) else "float64"


def _coord_record_from_row(
    row: Mapping,
    name: str,
    dims: tuple[str, ...] | None = None,
    name_is_held: bool = False,
    planned: bool = False,
) -> CoordRecord | None:
    """
    Build the envelope coord record for one output coordinate.

    Delegates to the ingest converter through a range CoordSummary so
    virtual outputs carry the same identities real patches would: a
    carried ``fp:`` def key survives for non-planned dims, and the
    planned dim's range data_id is reconstructed exactly. ``dims``
    names the dimensions the coordinate rides (itself by default).
    `name_is_held` says the members hold this coordinate, so a record is
    written even when nothing about its values can be stated: the patch
    will carry it, and a catalog which omitted it would deny it.
    `planned` says the row is the plan's own output along ``name``, whose
    grid describes its samples without a def key.
    """
    dims = (name,) if dims is None else dims
    lo, hi = row.get(f"{name}_min"), row.get(f"{name}_max")
    if lo is None or (pd.isnull(lo) and pd.isnull(hi)):
        # a coordinate without values (a dimension a plan created, or one
        # its members carry blank) has its identity and nothing else, as
        # ingesting it would record. A "cat:" digest — what such a
        # dimension is worth after a concatenation — identifies it here
        # the same way, matching another output only when the members it
        # joined were the same.
        key = row.get(f"_{name}_def_key")
        data_id = _def_key_data_id(key)
        if data_id is None and isinstance(key, str) and key.startswith("cat:"):
            data_id = key[4:]
        if data_id is None and not name_is_held:
            return None
        units = row.get(f"_{name}_units")
        if units == "" or (units is not None and pd.isnull(units)):
            units = None
        return CoordRecord(
            coord_name=name,
            value_kind="num",
            dtype="float64",
            coord_dims=",".join(dims),
            length=None,
            units=units,
            data_id=data_id,
        )
    step = row.get(f"{name}_step")
    step = None if step is None or pd.isnull(step) else step
    if isinstance(lo, str):
        # string coords have no range representation; store the
        # lexicographic envelope directly
        key = row.get(f"_{name}_def_key")
        data_id = _def_key_data_id(key)
        return CoordRecord(
            coord_name=name,
            value_kind="str",
            dtype="str",
            coord_dims=",".join(dims),
            length=None,
            units=None,
            min_str=str(lo),
            max_str=None if hi is None or pd.isnull(hi) else str(hi),
            data_id=data_id,
        )
    # Only the str envelope above represents a missing max. Every producer
    # writes {name}_min and {name}_max together -- _output_records feeds
    # whole dataframe rows (a missing value is NaN, not None) and the aux
    # info dict always sets both -- so a max cannot be absent past here.
    assert hi is not None
    if isinstance(lo, pd.Timestamp | np.datetime64):
        lo, hi = pd.Timestamp(lo).to_datetime64(), pd.Timestamp(hi).to_datetime64()
        dtype = "datetime64[ns]"
    elif isinstance(lo, pd.Timedelta | np.timedelta64):
        # dascore's converter (unlike Timedelta.to_timedelta64) handles NaT.
        lo, hi = dc.to_timedelta64(lo), dc.to_timedelta64(hi)
        dtype = "timedelta64[ns]"
    else:
        lo, hi = float(lo), float(hi)
        # the sign says which way the coordinate runs, as ingest records it
        step = None if step is None else float(step)
        # The frame holds every numeric envelope as float, but the row
        # says what the values really are. Publishing float64 regardless
        # would make an output rebuilt from its row disagree with the
        # same coordinate on the patches beside it.
        dtype = _numeric_dtype_str(row.get(f"_{name}_coord_dtype"))
    if isinstance(step, pd.Timedelta):
        step = step.to_timedelta64()
    if step is not None and not step:
        # a degenerate (zero) step is not a range; drop it rather than
        # letting range reconstruction divide by it
        step = None
    # envelope values and the units column are both the coordinate's
    # ORIGINAL spelling, never converted, so reattaching the pivot's
    # unit reconstructs exactly what was ingested
    units = row.get(f"_{name}_units")
    if units == "" or (units is not None and pd.isnull(units)):
        units = None
    length = None
    if step is not None:
        # lo, hi, and step always share a time kind (or are all floats), but
        # ty unions the branch types and rejects the mixed combinations.
        span = (hi - lo) / step  # ty: ignore[unsupported-operator]
        length = round(abs(span)) + 1
    key = row.get(f"_{name}_def_key")
    data_id = _def_key_data_id(key)
    # the grid is the source's; once the def key (value identity) is gone,
    # so are the values it described
    grid = row.get(f"_{name}_grid")
    if not (data_id or (planned and grid_fits(grid, lo, hi))):
        grid = None
    exact = {}
    if isinstance(grid, tuple):
        *terms, length = grid
        exact = dict(zip(_EXACT_GRID_FIELDS, terms))
    summary = CoordSummary(
        dtype=dtype,
        min=lo,
        max=hi,
        step=step,
        units=units,
        dims=dims,
        len=length,
        data_id=data_id,
        **exact,
    )
    return _coord_record(name, summary)


def _extrema(grouped, how: str) -> np.ndarray:
    """
    One end of each group's envelope, None where the values do not compare.

    Groups are taken one at a time because the column may hold more than
    one kind of value (a name numeric in one output and text in another),
    which pandas cannot aggregate whole.
    """
    values: list = []
    for _, group in grouped:
        stated = group.dropna()
        try:
            values.append(getattr(stated, how)() if len(stated) else None)
        except TypeError:
            # even within the output the values do not compare
            values.append(None)
    return np.array(values, dtype=object)


def _trim_to_member_windows(
    joined: pd.DataFrame,
    members: pd.DataFrame,
    coord_dims_map: Mapping[str, str],
    source_trimmed: frozenset[str],
) -> pd.DataFrame:
    """
    Trim each member's coordinates to the samples its windows keep.

    A member cut to a window on one coordinate loads only those samples of
    every coordinate on that coordinate's dimension. Only a source row
    describing the whole of that dimension, in the unit the window is in,
    says which samples those are; any other keeps its source envelopes as
    candidacy supersets.
    """
    windows = [
        x
        for x, dims in coord_dims_map.items()
        if {f"{x}_min", f"{x}_max"}.issubset(members.columns)
        and f"{x}_min" in joined.columns
        and str(dims) not in source_trimmed
        and "," not in str(dims)
    ]
    if not windows:
        return joined
    cut = joined.copy(deep=False)
    for name in windows:
        native = joined.get(_source_units_column(name), joined.get(f"_{name}_units"))
        planned = members.get(f"_{name}_units")
        same = np.ones(len(joined), dtype=bool)
        if native is not None and planned is not None:
            pairs = zip(native.to_numpy(), planned.to_numpy(), strict=True)
            same = np.array([_is_missing(a) or a == b for a, b in pairs], dtype=bool)
        for end in ("min", "max"):
            column = f"{name}_{end}"
            window = pd.Series(members[column].to_numpy(), index=joined.index)
            cut[column] = window.where(same, joined[column])
    tracker = _SampleTracker(joined)
    tracker.follow_values(joined, cut, dict.fromkeys(windows), False, {})
    return tracker.trim(cut, windows)


def _fill_held(outputs: pd.DataFrame, members: pd.DataFrame, dim: str):
    """
    Each output of a fill: the samples its members hold in it, and its size.

    A member a window inside a gap reads lies
    outside the window, so holds none of it.
    """
    spec = members[f"_{dim}_stretch"]
    ids = members["output_id"]
    bounds = outputs.set_index("output_id").reindex(ids.to_numpy())
    lows, highs = (members[f"{dim}_{x}"].to_numpy() for x in ("min", "max"))
    low = np.maximum(lows, bounds[f"{dim}_min"].to_numpy())
    high = np.minimum(highs, bounds[f"{dim}_max"].to_numpy())
    steps = members[f"{dim}_step"]
    if low.dtype.kind in "Mm":
        steps = pd.to_timedelta(steps)
    spans = pd.Series(high - low, index=members.index) / steps.abs()
    held = np.where(high >= low, np.rint(spans.to_numpy(np.float64)) + 1, 0)
    size = spec.map(lambda x: x[6] if isinstance(x, tuple) else np.nan)
    held = pd.Series(held, index=members.index).groupby(ids, sort=True).sum()
    return held.to_numpy(), size.groupby(ids, sort=True).first().to_numpy()


def _aux_coord_info(
    source_rows: pd.DataFrame,
    members: pd.DataFrame,
    plan_dim: str,
    coord_dims_map: Mapping[str, str],
    trimmed_dims: frozenset[str] = frozenset(),
    concat: bool = False,
    conflict: str | None = None,
    source_trimmed: frozenset[str] = frozenset(),
    filled=None,
) -> dict[int, dict[str, dict]]:
    """
    Aggregate per-output envelope info for auxiliary coordinates.

    Aggregated from the *member source rows* (authoritative, unlike the
    planner's carried columns), trimmed to the samples each member's
    windows keep (see `_trim_to_member_windows`; ``source_trimmed`` names
    the dimensions the source rows were already trimmed on). A coordinate
    a window trims claims no identity. Structural identity (def key and step,
    which permit value-identity claims) is kept only when every member
    shares one def key and the values provably survive assembly: a
    coordinate riding the planned dimension is trimmed/merged with it,
    so only a lone unmodified member keeps identity there. Envelopes
    always aggregate — the catalog contract is candidacy, with exact
    values re-established at load.

    A concatenation joins a rider's segments rather than merging them, so
    a rider whose members share one step and meet end to end keeps that
    step (its values still differ member by member, so not its identity).

    A merge's `conflict` settles a coordinate the members do not hold
    alike (see `merge_coord_managers`): "drop", or a rider some member
    lacks, leaves it out; "keep_first" describes the first member's.

    A fill (``filled``, from `_fill_held`) gives a rider NaN or NaT where
    it fills, so one holding neither is left out of an output it fills,
    and an output only of fill states no values for the rest.
    """
    out: dict[int, dict[str, dict]] = {}
    if not len(members) or not coord_dims_map:
        return out
    cols = [c for c in ("output_id", "_patch_row", "_modified") if c in members.columns]
    whole = members[cols].merge(source_rows, on="_patch_row", how="left")
    joined = _trim_to_member_windows(whole, members, coord_dims_map, source_trimmed)
    grouped = joined.groupby("output_id", sort=True)
    output_ids = grouped.size().index.to_numpy()
    single = (grouped.size() == 1).to_numpy()
    modified = (
        grouped["_modified"].any().to_numpy()
        if "_modified" in joined.columns
        else np.zeros(len(output_ids), dtype=bool)
    )
    for name, dims_str in coord_dims_map.items():
        cmin, cmax = f"{name}_min", f"{name}_max"
        if cmin not in joined.columns or name == plan_dim:
            # the planned dimension is described by its output row (a
            # coordinate of that name a concatenation replaces is gone)
            continue
        dims = tuple(d for d in str(dims_str).split(",") if d)
        rides = plan_dim in dims
        # a residual selection trims the dims it rides at load, changing
        # the values of every coordinate on those dims
        trimmed = bool(set(dims) & trimmed_dims)
        key_col, step_col = f"_{name}_def_key", f"{name}_step"
        unit_col, grid_col = f"_{name}_units", f"_{name}_grid"
        lows = _extrema(grouped[cmin], "min")
        highs = _extrema(grouped[cmax], "max")
        # the *_first arrays are only read where their gate is True, and
        # a gate can only be True when its column exists
        no_gate = np.zeros(len(output_ids), dtype=bool)
        keep, key_first = no_gate, None
        grid_first = None
        if key_col in joined.columns:
            keep = grouped[key_col].nunique().to_numpy() == 1
            key_first = grouped[key_col].first().to_numpy()
            if grid_col in joined.columns:
                grid_first = grouped[grid_col].first().to_numpy()
        keep = keep & (not trimmed)
        # a window cut the values the source's identity describes
        windowed = (joined[cmin] != whole[cmin]) | (joined[cmax] != whole[cmax])
        windowed &= joined[cmin].notna()
        windowed = windowed.groupby(joined["output_id"], sort=True).any().to_numpy()
        keep = keep & ~windowed
        all_null = pd.isnull(lows) & pd.isnull(highs)
        if rides:
            # a rider's values differ member by member, so only a lone
            # member keeps identity — unless nobody states any values,
            # which every member says the same way
            keep = keep & ((single & ~modified) | all_null)
        step_ok, step_first = no_gate, None
        if step_col in joined.columns:
            step_ok = keep & (grouped[step_col].nunique().to_numpy() == 1)
            step_first = grouped[step_col].first().to_numpy()
            if rides and concat:
                # the joined coordinate is a range when the segments are
                # (the members are already in the order they join in)
                order = {v: i for i, v in enumerate(output_ids)}
                codes = joined["output_id"].map(order).to_numpy()
                step_first = _concatenated_steps(joined, codes, name)
                step_ok = ~pd.isnull(step_first)
        unit_ok, unit_first = no_gate, None
        if unit_col in joined.columns:
            unit_ok = grouped[unit_col].nunique().to_numpy() == 1
            unit_first = grouped[unit_col].first().to_numpy()
            # members spelling one coordinate two ways (seconds beside
            # milliseconds) have no envelope in common: the magnitudes mean
            # different things, and only the loaded patch says which
            # spelling wins. No units at all is one spelling, not two.
            undecided = grouped[unit_col].nunique().to_numpy() > 1
            if undecided.any():
                lows = np.array(
                    [None if u else v for v, u in zip(lows, undecided)], dtype=object
                )
                highs = np.array(
                    [None if u else v for v, u in zip(highs, undecided)], dtype=object
                )
                step_ok = step_ok & ~undecided
        # a coordinate no member holds contributes nothing; one the members
        # do hold is always named, even when nothing about its values can
        # be stated — the patch will have it, so the catalog says so
        holds = joined[cmin].notna()
        if key_col in joined.columns:
            holds = holds | joined[key_col].notna()
        by_output = holds.groupby(joined["output_id"], sort=True)
        absent = ~by_output.any().to_numpy()
        if conflict in ("keep_first", "drop") and not concat:
            ends = [c for c in (cmin, cmax, key_col, unit_col) if c in joined]
            lacking = ~by_output.all().to_numpy()
            differs = (grouped[ends].nunique(dropna=False) > 1).any(axis=1)
            settled = lacking if rides else lacking | np.asarray(differs)
            heads = joined.drop_duplicates("output_id").sort_values("output_id")
            first = heads[ends].to_numpy().T
            gone = ~holds[heads.index].to_numpy() | (conflict == "drop") | rides
            absent |= settled & gone
            lows = np.where(settled, first[0], lows)
            highs = np.where(settled, first[1], highs)
            keep, step_ok = keep & ~settled, step_ok & ~settled
            if unit_col in joined:  # the kept coordinate carries its units
                unit_first = np.where(settled, first[-1], unit_first)
                unit_ok = unit_ok | settled
        if rides and filled is not None:
            held, size = filled
            kind = joined.get(f"_{name}_coord_dtype", pd.Series(dtype=object))
            unfillable = ~kind.dropna().map(lambda x: np.dtype(x).kind in "fcMm")
            absent |= (held < size) & unfillable.any()
            none = np.full(len(held), None, dtype=object)
            lows, highs = (np.where(held > 0, x, none) for x in (lows, highs))
        for index in np.flatnonzero(~absent):
            step = step_first[index] if step_first is not None else None
            key = key_first[index] if key_first is not None else None
            unit = unit_first[index] if unit_first is not None else None
            grid = grid_first[index] if grid_first is not None else None
            info = {
                cmin: lows[index],
                cmax: highs[index],
                step_col: step if step_ok[index] else None,
                key_col: key if keep[index] else None,
                grid_col: grid if keep[index] else None,
                unit_col: unit if unit_ok[index] else None,
                "dims": dims,
                "windowed": bool(windowed[index]),
            }
            out.setdefault(int(output_ids[index]), {})[name] = info
    return out


def _with_windowed_dims(outputs: pd.DataFrame, aux_info, plan_dim: str):
    """
    Give each output the envelopes its windows left on its other dimensions.

    A window on a coordinate riding a dimension cuts that dimension too,
    which the output row, carried from its sources, does not say; nor
    can its identity still describe the values.
    """
    spans: dict[str, dict[int, tuple]] = {}
    for output_id, info in aux_info.items():
        for name, values in info.items():
            if name != plan_dim and values["windowed"] and values["dims"] == (name,):
                ends = (values[f"{name}_min"], values[f"{name}_max"])
                spans.setdefault(name, {})[output_id] = ends
    if not spans:
        return outputs
    outputs = outputs.copy(deep=False)
    ids = outputs["output_id"].astype(int)
    for name, ends in spans.items():
        hit = ids.isin(list(ends)).to_numpy()
        for position, end in enumerate(("min", "max")):
            new = ids.map({k: v[position] for k, v in ends.items()})
            outputs[f"{name}_{end}"] = outputs[f"{name}_{end}"].where(~hit, new)
        if (key := f"_{name}_def_key") in outputs:
            outputs[key] = outputs[key].astype(object).where(~hit, None)
    return outputs


def _cell_key(value):
    """A cell as a cache key: a float by its bits, so -0.0 is not 0.0."""
    return value.hex() if isinstance(value, float) else (type(value), value)


def _record_once(cache: dict, row: Mapping, name: str, **kwargs):
    """
    `_coord_record_from_row`, built once per distinct coordinate.

    Outputs cut to the same window state the same coordinate, and
    rebuilding a range's identity is the costly part. Records are frozen,
    so outputs can share one. The key is every cell the record reads.
    """
    cells = [row.get(f"{name}_{x}") for x in ("min", "max", "step")]
    cells += [row.get(f"_{name}_{x}") for x in ("units", "coord_dtype", "def_key")]
    cells += [row.get(f"_{name}_grid")]
    key = (name, *sorted(kwargs.items()), *map(_cell_key, cells))
    if key not in cache:
        cache[key] = _coord_record_from_row(row, name, **kwargs)
    return cache[key]


def _output_records(
    outputs: pd.DataFrame,
    token: str,
    aux_info: Mapping[int, Mapping[str, Mapping]] | None = None,
    sizes: Mapping[int, int] | None = None,
    plan_dim: str | None = None,
) -> list[SourceRecord]:
    """
    Convert plan output rows into ingestible source records.

    ``sizes`` names the outputs whose sample count is known (see
    `_whole_member_sizes`); every other output states none.
    """
    sizes = sizes or {}
    records = []
    aux_info = aux_info or {}
    # Envelope columns belong to coordinates actually present in a row;
    # an attr that merely looks envelope-shaped (channel_step with no
    # channel coord) is ordinary metadata and must be preserved. The
    # def-key columns are frame-wide, so the envelope-key sets repeat
    # across rows and are cached by (dims, aux coord names).
    base_names = {
        key[1 : -len("_def_key")]
        for key in outputs.columns
        if key.startswith("_") and key.endswith("_def_key")
    }
    base_names |= {"time", "distance"}  # fixed patches-table envelopes
    envelope_cache: dict[tuple, set[str]] = {}
    coord_cache: dict[tuple, CoordRecord | None] = {}
    for row in outputs.to_dict("records"):
        output_id = int(row["output_id"])
        dims = str(row.get("dims") or "")
        dim_names = [d for d in dims.split(",") if d]
        aux = aux_info.get(output_id, {})
        coords = []
        for name in dim_names:
            record = _record_once(coord_cache, row, name, planned=name == plan_dim)
            if record is not None:
                coords.append(record)
        # auxiliary (non-dimension) coordinates remain on the assembled
        # patches, so the catalog must keep describing them
        for name, info in aux.items():
            if name in dim_names:
                continue
            record = _record_once(
                coord_cache, info, name, dims=info["dims"], name_is_held=True
            )
            if record is not None:
                coords.append(record)
        cache_key = (dims, tuple(aux))
        envelope_keys = envelope_cache.get(cache_key)
        if envelope_keys is None:
            coord_names = set(dim_names) | set(aux) | base_names
            envelope_keys = {
                f"{name}_{sfx}"
                for name in coord_names
                for sfx in ("min", "max", "step", "units")
            }
            envelope_cache[cache_key] = envelope_keys
        attrs = {}
        for key, value in row.items():
            if (
                key in _NON_ATTR
                or key.startswith("_")
                or key in envelope_keys
                or value is None
                or (np.isscalar(value) and pd.isnull(value))
            ):
                continue
            typed = typed_value(value)
            if typed is not None:
                attrs[key] = typed
        patch = PatchRecord(
            source_patch_key=str(output_id),
            dims=dims,
            # the plan carries the element dtype privately so a chained
            # chunk can still size patches by their memory footprint.
            # NaN is truthy, so `or ""` alone would store the string
            # "nan" and poison every later np.dtype() of this column.
            dtype=_row_str(row.get("_dtype")),
            data_size=sizes.get(output_id),
            time_min=_ns(row.get("time_min")),
            time_max=_ns(row.get("time_max")),
            time_step=_ns(row.get("time_step")),
            distance_min=_num(row.get("distance_min")),
            distance_max=_num(row.get("distance_max")),
            distance_step=_num(row.get("distance_step")),
            attrs=attrs,
            coords=tuple(coords),
            # Plan attributes are resolved through members, not this row.
            attrs_complete=False,
        )
        records.append(
            SourceRecord(
                source_path=f"{PLAN_SCHEME}{token}/{output_id}",
                source_format="plan",
                format_version="",
                patches=(patch,),
            )
        )
    return records


def _trim_loaded_member(patch: dc.Patch, trim: Mapping) -> dc.Patch:
    """Apply the plan's window to a member the reader loaded whole."""
    ranges = {
        key: tuple(value)
        for key, value in trim.items()
        if key in set(patch.dims)
        and isinstance(value, list | tuple | np.ndarray)
        and len(value) == 2
    }
    return patch.select(**ranges) if ranges else patch


class PlanResolver(PatchResolver):
    """
    Assemble plan-output rows from their member source patches.

    ``member_rows`` carries, per output, the full source row (path,
    format, identity, attrs) with the planned dimension's envelope
    replaced by the member's trim range; loading goes through ``loader``
    (live registry, files, and nested plan rows), applies the parent
    view's residual selections, then trims/merges via the assembly
    engine ("chunk" mode) or concatenates in order ("concat" mode).
    """

    def __init__(
        self,
        *,
        token: str,
        dim: str,
        member_rows: pd.DataFrame,
        loader: PatchResolver,
        merge_kwargs: Mapping,
        parent_residuals: tuple = (),
        mode: str = "chunk",
        aux_coords: frozenset[str] = frozenset(),
        origin_path=None,
        stamped: tuple[str, ...] = (),
        lossy: bool = False,
        trim_dims: tuple[str, ...] = (),
    ):
        if "output_id" not in member_rows.columns:
            msg = "member_rows must carry an output_id column."
            raise ValueError(msg)
        # plan invariant: outputs without members must never be published
        self.token = token
        self.dim = dim
        self.trim_dims = tuple(trim_dims)  # dims besides `dim` windows trim
        rows = member_rows.reset_index(drop=True)
        # What the index measured of each member's source, held beside
        # the rows rather than in them: every output slices this frame,
        # and a column is charged for at every slice. Row number is the
        # key, which the slices keep.
        self._source_stats = {
            name: _stat_column(rows[name])
            for name in SOURCE_STAT_COLUMNS
            if name in rows.columns
        }
        self.member_rows = rows.drop(columns=list(self._source_stats))
        self.loader = loader
        self.aux_coords = frozenset(aux_coords)
        self.merge_kwargs = dict(merge_kwargs)
        self.parent_residuals = tuple(parent_residuals)
        self.mode = mode
        # informational only: the directory/file the plan derived from
        self.origin_path = origin_path
        # attrs the outputs state about themselves rather than inherit
        self.stamped = tuple(stamped)
        # Whether the outputs leave samples of their sources out. A lossy
        # plan must never be collapsed: its members do not cover their
        # sources, so re-planning over them would load back what it
        # dropped. See `collapse_working_df`.
        self.lossy = bool(lossy)

    def live_entries(self) -> dict[str, dc.Patch]:
        """Expose the loader's live registry (for absorption/transfer)."""
        return self.loader.live_entries()

    def plan_entries(self) -> Mapping[str, PlanResolver]:
        """Route plan:// paths with this resolver's token to it."""
        nested = dict(getattr(self.loader, "plan_entries", dict)())
        nested[f"{PLAN_SCHEME}{self.token}/"] = self
        return nested

    def _assembler(self):
        # the fill is written once the members merge (`resolve`), so the
        # merge, and the ids it stamps, never see it
        merge = {k: v for k, v in self.merge_kwargs.items() if k != "fill_value"}
        if self.merge_kwargs.get("fill_value") is not None:
            # members keep their labels, which the fill places as a patch's
            merge |= {"tolerance": GapTolerance.samples(0), "max_shift": 0}
        return PatchAssembler(
            load_patch=self._load_member,
            merge_kwargs=merge,
            plan_dim=self.dim,
            array_source=self._member_array_source,
            can_use_index=self._can_load_member_from_index,
            sources_unchanged=self._sources_unchanged,
            trim_dims=self.trim_dims,
            aux_coords=bool(self.aux_coords),
            stamped=self.stamped,
        )

    def _can_load_member_from_index(self, row: Mapping) -> bool:
        """Whether a member reads as a window of its stored array, reading nothing."""
        if row.get("_modified"):
            # A trim is a window of the source, which is only placeable
            # when the row keeps the source's own range beside it --
            # which `_with_source_range` withholds from every row a
            # residual this plan cannot stand for would trim again.
            if _is_missing(row.get(source_range_column(self.dim, "low"))):
                return False
        else:
            # An unmodified row lies wholly inside any value selection on
            # the plan's dimension; other residuals need the loaded patch.
            for coords, samples, relative in self.parent_residuals:
                if samples or relative or set(coords) - {self.dim}:
                    return False
        return self._array_read_info(row) is not None

    def _sources_unchanged(self, rows: pd.DataFrame, strict=True) -> bool:
        """
        Whether every member source still is what the index recorded.

        A recipe reads the window the index promised rather than the whole
        array, so a file rewritten longer under the same key would come
        back the shape its row predicted and go unnoticed. Only a source
        measured now and found to be what was recorded keeps the recipe:
        a source `changed_sources` cannot measure refuses it too, and sends
        the merge down the patch path. Unless ``strict``, only a source
        measured and found changed counts.
        """
        return self.changed_sources(rows, skip_unmeasured=not strict) == []

    def changed_sources(self, rows: pd.DataFrame, skip_unmeasured=False):
        """
        The member sources which are no longer what the index recorded.

        What the index measured of each source came from the same join as
        the member rows and is held under their row numbers, so the two
        describe one revision of the index however long ago it was planned
        and nothing is asked of the index here. Each distinct source is
        measured once, by the same function the indexer records -- a file
        by its own stat, a directory-format unit by its manifest.

        A remote store is never touched. A path this process cannot stat,
        or one the index recorded nothing for, makes this None, or is left
        out with ``skip_unmeasured``, which also follows a member loading
        another plan's output to that plan's members. A recorded file which
        no longer answers (deleted, unreadable) is listed as changed.
        """
        stats, named = self._source_stats, "source_path" in rows.columns
        changed = self._nested_changes(rows) if skip_unmeasured and named else []
        if len(stats) != len(SOURCE_STAT_COLUMNS) or not named:
            return changed if skip_unmeasured else None
        # the row numbers this slice kept, which is what the stats are under
        taken = rows.index.to_numpy()
        paths = rows["source_path"].to_numpy()
        mtimes = stats[SOURCE_STAT_COLUMNS[0]][taken]
        sizes = stats[SOURCE_STAT_COLUMNS[1]][taken]
        seen = set()
        for path, mtime, size in zip(paths, mtimes, sizes, strict=True):
            path = _row_str(path)
            if path in seen:
                continue
            seen.add(path)
            if not path or not _is_local(path) or NO_STAT in (mtime, size):
                if skip_unmeasured:
                    continue
                return None
            if scan_unit_stats(path) != (int(mtime), int(size)):
                changed.append(path)
        return changed

    def _nested_changes(self, rows: pd.DataFrame) -> list[str]:
        """The changed sources of the plans whose outputs these rows load."""
        paths = rows["source_path"].map(_row_str)
        changed = []
        for prefix, plan in getattr(self.loader, "plan_entries", dict)().items():
            ids = paths[paths.str.startswith(prefix)].str.removeprefix(prefix)
            if len(ids):
                members = plan.member_rows
                members = members[members["output_id"].isin(ids.astype(int))]
                changed += plan.changed_sources(members, skip_unmeasured=True)
        return changed

    def source_stats_of(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Put what the index measured back beside these member rows."""
        taken = rows.index.to_numpy()
        return rows.assign(**{k: v[taken] for k, v in self._source_stats.items()})

    @cached_property
    def _stats_by_path(self) -> dict[str, tuple[int, int]]:
        """What the index measured of each member's source, by its stored path."""
        none = np.empty(0, np.int64)  # an index which measured nothing
        stats = [self._source_stats.get(x, none).tolist() for x in SOURCE_STAT_COLUMNS]
        paths = self.member_rows["source_path"].map(_row_str)
        return dict(zip(paths, zip(*stats)))

    def _member_array_source(self, row: Mapping, shape) -> ArraySource | None:
        """
        Name the whole stored array a member is part of; nothing is read.

        The row states the shape its caller passes and the dtype the
        array comes back as, so a source which names the whole of it is
        enough to window the member out of it later. A row which states
        no dtype cannot be named and sends the merge down the patch path.

        A source's origin is the id of the array itself, which is the
        patch's `data_id` -- not the lineage `origin_id` beside it, which
        a processed patch keeps from what it was made of.
        """
        info = self._array_read_info(row)
        assert info is not None, "the preflight resolved every member's reader"
        loader, path, _, key = info
        dtype = row.get("_dtype")
        if not isinstance(dtype, str) or not dtype:
            return None
        origin = row.get("data_id")
        offsets = _row_str(row.get("_data_offsets"))
        unknown = (None, None)
        mtime, size = self._stats_by_path.get(path, unknown) if offsets else unknown
        source = ArraySource(
            path=str(loader.resolve_path(path)),
            format=_row_str(row.get("source_format")),
            version=_row_str(row.get("source_version")),
            key=key,
            origin_id="" if origin is None or pd.isnull(origin) else str(origin),
            offsets=_text_offsets(offsets),
            chunk_rows=int(row["_data_chunk_rows"]) if offsets else 0,
            mtime_ns=mtime,
            size_bytes=size,
        )
        return source.describe(shape, dtype)

    def _load_member(self, kwargs: Mapping) -> dc.Patch:
        """Load one member source patch, applying parent residuals."""
        trim = {}
        if kwargs.get("_modified"):
            # The index already selected the source row. Only coordinate
            # ranges are read hints: attrs may have been overridden by the
            # path and would reject the file before those overrides apply.
            dims = set(str(kwargs.get("dims", "")).split(","))
            trim = {
                k: v
                for k, v in kwargs.items()
                if k in dims
                and not str(k).startswith("_")
                and k not in _SOURCE_COLUMNS
                and k not in _READ_KWARGS
            }
            # Trim magnitudes are in the plan's unit; when the source file
            # spells the coordinate differently, a bare read hint would
            # trim the wrong physical interval, so it is dropped (hints
            # are optional — slower, never wrong) and the exact trim is
            # applied above in plan units.
            plan_units = _stated_units(kwargs.get(f"_{self.dim}_units"))
            source_units = _stated_units(
                kwargs.get(_source_units_column(self.dim), plan_units)
            )
            if plan_units is not None and source_units != plan_units:
                trim.pop(self.dim, None)
        patch_local = any(s or r for _, s, r in self.parent_residuals)
        if trim and patch_local:
            # A patch-local residual resolves against the source patch, so
            # the plan's window cannot narrow the member before it runs;
            # otherwise select(time=(1, -1), relative=True).chunk(time=2)
            # would trim a second off each chunk instead of each file.
            patch = self.loader.resolve(kwargs)
            patch = apply_exact_residuals(patch, self.parent_residuals)
            patch = _trim_loaded_member(patch, trim)
        else:
            patch = self.loader.resolve(kwargs, **trim)
            patch = apply_exact_residuals(patch, self.parent_residuals)
        return self._in_plan_units(patch, kwargs)

    def _array_read_info(self, row: Mapping):
        """Resolve array-reader metadata without opening the source file."""
        path = _row_str(row.get("source_path"))
        if not path:
            return None
        if is_memory_uri(path) or path.startswith(PLAN_SCHEME):
            return None
        fmt = _row_str(row.get("source_format"))
        version = _row_str(row.get("source_version"))
        if not fmt or not version:
            return None
        loader = self.loader
        if not isinstance(loader, FileResolver):
            loader = getattr(loader, "file", None)
        if not isinstance(loader, FileResolver):
            return None
        try:
            # An exact version is required: without one the manager hands
            # back the newest reader, which may not match this file.
            fiber_io = FiberIO.manager.get_fiberio(format=fmt, version=version)
        except UnknownFiberFormatError:
            return None
        key = _row_source_patch_key(row)
        if key.isdigit():
            # A synthesized positional key binds against the full source
            # read (see FileResolver.resolve); an override which resolves
            # keys natively could match the wrong patch, or nothing.
            return None
        return loader, path, fiber_io, key

    def _in_plan_units(self, patch: dc.Patch, kwargs: Mapping) -> dc.Patch:
        """
        Re-express the chunked dimension in the unit the plan advertises.

        A partition mixing compatible spellings is planned in one of
        them, and the output rows say so, so a member kept in its own
        spelling would make the catalog describe an envelope no patch it
        yields actually has. Merging already converted its members; this
        covers the single-member outputs merging never visits.
        Identity mode is exempt: it promises the untouched patch.
        """
        if self.mode == "identity":
            return patch
        plan_units = _stated_units(kwargs.get(f"_{self.dim}_units"))
        if plan_units is None:
            return patch
        coord = patch.coords.coord_map.get(self.dim)
        current = getattr(coord, "units", None) if coord is not None else None
        # `==`, not `units_match`: the plan asks whether the values are
        # already at the right scale, and `m` and `100 cm` label the same
        # ones. Converting between them would relabel and change nothing.
        if current is None or get_quantity(current) == get_quantity(plan_units):
            return patch
        # raw_function: the conversion serves the plan's own bookkeeping,
        # and a history entry on some members but not others would make
        # the merge warn about histories differing.
        return dc.proc.units.convert_units.raw_function(patch, **{self.dim: plan_units})

    def resolve(self, row: Mapping, **trim) -> dc.Patch:
        """Assemble the output patch a plan row describes."""
        output_id = int(_row_source_patch_key(row))
        members = self.member_rows[self.member_rows["output_id"] == output_id]
        dim, merge = self.dim, self.merge_kwargs
        if self.mode == "identity":
            # one untouched member per output; residuals apply at load
            assert len(members) == 1
            patch = self._load_member(members.iloc[0].to_dict())
            if (snap := merge.get("snap")) is not None:
                patch = patch.snap_coords(*snap[0], **snap[1])
        elif self.mode == "concat":
            # the plan decided what fits together; assembly executes it
            # each member trimmed as the plan cut it, as assembly trims one
            load = self._assembler()._load_trimmed_patch
            loaded = [load(x, members) for x in members.to_dict("records")]
            patch = concatenate_planned(
                loaded,
                dim,
                count=merge.get("count"),
                conflict=merge.get("conflict", "raise"),
            )
        else:
            # a fill's members fill onto their output's slice of the
            # stretch's grid, or the part a trim asks for, loading only
            # the members reaching it
            if fill := (merge.get("fill_value") is not None):
                spec = members[f"_{dim}_stretch"].iloc[0]
                units = _stated_units(members[f"_{dim}_units"].iloc[0])
                # a whole stretch keeps the grid its members fill onto themselves
                whole = not isinstance(spec, tuple) or spec[5:7] == (0, spec[3])
                grid = None if whole and not trim else stretch_coordinate(spec, units)
                if (window := trim.get(dim)) is not None:
                    members = reached(members, dim, *window)
                    grid = None if grid is None else grid.select(tuple(window))[0]
            patch = self._assembled(output_id, members)
            if fill:
                patch = fill_on_grid(patch, dim, grid, merge, row.get("_dtype"))
        return self._stamp(patch, row)

    def _assembled(self, output_id: int, members: pd.DataFrame) -> dc.Patch:
        """The one patch a chunk plan's members assemble into."""
        joined = members.assign(current_index=output_id)
        (patch,) = self._assembler()._patch_from_instruction_df(joined)
        return patch

    def _stamp(self, patch: dc.Patch, row: Mapping) -> dc.Patch:
        """
        Apply the attrs the outputs state about themselves, if any.

        An output is assembled from its members, so it carries their
        attrs and knows nothing of why it was cut out. `stamped` is how
        an operation which does know says so -- `Spool.expand_by`
        recording which value each patch was split on -- and it keeps
        the patch which comes out agreeing with the row `get_contents`
        shows for it. Nothing else needs filling: under every `conflict`
        policy the row and the assembled patch reach the same values --
        the members agree, or both take the first member's, or both
        carry nothing.
        """
        if not self.stamped:
            return patch
        return patch.update_attrs(**{x: row[x] for x in self.stamped})


def _trimmed_dims(residuals, coord_dims_map: Mapping) -> frozenset[str]:
    """The dimensions a residual (load-time) selection trims."""
    names = {n for coords, _, _ in residuals for n in coords}
    return frozenset(
        d for n in names for d in str(coord_dims_map.get(n, n)).split(",") if d
    )


def stale_def_keys(residuals, coord_dims_map: Mapping, columns) -> list[str]:
    """
    The def-key columns which describe coordinates a residual will trim.

    A residual selection is applied when a patch is loaded, so until then
    the identity claims (def keys) of coordinates on the trimmed
    dimensions describe the untrimmed values and must not be compared or
    published.
    """
    trimmed = _trimmed_dims(residuals, coord_dims_map)
    return [
        f"_{c}_def_key"
        for c, dims_str in coord_dims_map.items()
        if set(str(dims_str).split(",")) & trimmed and f"_{c}_def_key" in columns
    ]


def _whole_member_sizes(trims: pd.DataFrame, sources: pd.DataFrame) -> dict[int, int]:
    """
    The sample count of each output which is one whole member.

    Such an output *is* that patch, so it holds the samples the patch
    states. An output assembled from more than one member, or from a
    member a trim cuts down, has a size only the loaded patch knows, and
    states none rather than a member's.
    """
    if trims.empty or "_data_size" not in sources.columns:
        return {}
    counts = trims.groupby("output_id")["_patch_row"].transform("size").to_numpy()
    modified = np.asarray(trims.get("_modified", False), dtype=bool)
    whole = trims[(counts == 1) & ~modified]
    lookup = sources.drop_duplicates("_patch_row").set_index("_patch_row")["_data_size"]
    out = {}
    whole_rows = zip(whole["output_id"], whole["_patch_row"], strict=True)
    for output_id, patch_row in whole_rows:
        size = lookup.get(patch_row)
        if not pd.isnull(size):
            out[int(output_id)] = int(size)
    return out


def _value_residuals_on(residuals, name: str) -> bool:
    """
    Whether every residual is one plain value range on ``name``.

    Such a selection is projected onto the relation's envelopes before
    anything is planned, so a row it narrows says so (`_modified`) and a
    row it leaves whole still states its source's own range. Nothing
    else can: a sample or relative selection resolves against the loaded
    patch, one naming another coordinate trims an axis the row still
    describes whole, one which is not a range never reached the
    envelopes at all, and a unit-bearing one is clipped by a different
    conversion than the patch itself makes.
    """
    for coords, samples, relative in residuals:
        if samples or relative or set(coords) - {name}:
            return False
        value = coords.get(name)
        if getattr(value, "magnitudes", None) is not None or not is_range(value):
            return False
        if any(hasattr(bound, "units") for bound in value):
            return False
    return True


def _with_source_range(sources: pd.DataFrame, name: str, residuals) -> pd.DataFrame:
    """
    Keep each source's own range on the plan dimension beside its trim.

    A member the plan cuts states the window it was cut to, and placing
    that window back on the file's samples takes the range the file
    itself spans. Only a row which describes the whole of its source
    states one: a row a residual selection already trimmed no longer
    does, and neither does one which arrived trimmed for a reason this
    plan cannot name. A residual selection a window cannot stand for
    takes the range from every row, cut or not.
    """
    columns = [source_range_column(name, x) for x in SOURCE_RANGE_ENDS]
    if residuals and not _value_residuals_on(residuals, name):
        # Such a residual re-trims whatever the member load returns, so
        # its trimmed rows must not take the recipe; dropping the range
        # is what refuses them (`_can_load_member_from_index`).
        return sources.drop(columns=columns, errors="ignore")
    if set(columns).issubset(sources.columns):
        # already carried: these rows are the members of a plan on this
        # same dimension, and the range beside them is still the file's
        return sources
    envelope = [f"{name}_{x}" for x in ("min", "max", "step")]
    if not set(envelope).issubset(sources.columns):
        return sources
    modified = sources.get("_modified", pd.Series(False, index=sources.index))
    whole = pd.Series(~np.asarray(modified, dtype=bool), index=sources.index)
    values = {
        column: sources[source].where(whole)
        for column, source in zip(columns, envelope, strict=True)
    }
    return sources.assign(**values)


def derived_catalog(
    *,
    source_rows: pd.DataFrame,
    plan,
    parent: PatchCatalog | None,
    merge_kwargs: Mapping,
    mode: str = "chunk",
    origin_path=None,
    stamped: tuple[str, ...] | Mapping[str, str] = (),
    lossy: bool = False,
    snapped: Mapping[int, Mapping[str, Any]] | None = None,
) -> PatchCatalog:
    """
    Materialize a plan into a fresh in-memory catalog.

    ``source_rows`` are the full member source rows (path/format/
    identity plus envelopes and attrs) keyed by ``_patch_row`` matching
    ``plan.members``; ``parent`` supplies the resolver (live registry,
    file root, nested plans) and the residual selections its view
    carried, which member loading re-applies; ``stamped`` may map to dtypes.
    ``snapped`` maps an output to the coordinates a snap gives it as it
    loads, which its record states in place of its source's.
    """
    token = secrets.token_hex(8)
    name = plan.dim
    # a plan over stamped rows keeps stating the stamps
    stamps = dict(parent._stamps) if parent is not None else {}
    stamps |= stamped if isinstance(stamped, Mapping) else {}
    # a plan's diagnostics describe it; loading never reads them
    trims = plan.members.drop(columns=list(PIECE_COLUMNS), errors="ignore")
    trim_cols = [c for c in trims.columns if c not in ("_patch_row",)]
    # rows re-planned from an earlier chunk's members carry its grid slices
    sources = source_rows.drop(columns=[f"_{name}_stretch"], errors="ignore")
    if "_patch_row" not in sources.columns:
        sources = _ensure_patch_row(sources)
    # Trim magnitudes are in the plan's (partition-normalized) unit; the
    # source's own spelling survives under a renamed column so member
    # loading can tell when a bare read hint would mean the wrong unit.
    unit_col = f"_{name}_units"
    source_unit_col = _source_units_column(name)
    if unit_col in trim_cols and unit_col in sources.columns:
        if source_unit_col in sources.columns:
            # Re-chunking a derived view: these rows already record the
            # file's own spelling, and that — not the previous plan's
            # unit — is what the loader will meet. Renaming again would
            # also duplicate the column, which silently drops one of the
            # two when the rows become load kwargs.
            sources = sources.drop(columns=[unit_col])
        else:
            sources = sources.rename(columns={unit_col: source_unit_col})
    parent_residuals = () if parent is None else parent.residuals
    residuals = tuple(parent_residuals)
    if plan.trim_dims:  # explicit windows' trims act as residuals here
        residuals += ((dict.fromkeys(plan.trim_dims), False, False),)
    sources = _with_source_range(sources, name, residuals)
    # Resolve source paths once before deriving both members and fill anchors.
    root = getattr(parent.resolver, "_root", None) if parent is not None else None
    upaths = {}
    if root is not None and "source_path" in sources.columns:
        file_resolver = FileResolver(root)
        resolved = [file_resolver.resolve_path(x) for x in sources["source_path"]]
        sources = sources.assign(source_path=[str(x) for x in resolved])
        # a remote root's storage options (e.g. auth headers) live on its
        # UPaths, which the loader reopens the rows by
        upaths = {
            str(x): x for x in resolved if isinstance(x, UPath) and not is_local_path(x)
        }
    member_rows = trims[["_patch_row", *[c for c in trim_cols]]].merge(
        sources.drop(columns=[c for c in trim_cols if c in sources], errors="ignore"),
        on="_patch_row",
        how="left",
    )
    # the member's trimmed range replaces the source envelope for loading
    member_rows = member_rows.drop(columns=["_patch_row"])
    loader = CompositeResolver()
    if parent is not None:
        member_paths = set(
            member_rows.get("source_path", pd.Series(dtype=str)).astype(str)
        )
        loader.absorb(parent.resolver, paths=member_paths)
        loader.file._upaths.update(upaths)
        # a directory's members read under the decode it scanned them with
        columns = ["source_path", "source_format", "source_version"]
        if set(columns) <= set(member_rows.columns):
            stated = member_rows[columns].drop_duplicates().itertuples(index=False)
            loader.file.hold_decodes(directory_decodes(parent.resolver, stated))
    coord_dims_map = {} if parent is None else parent.backend.coord_dims_map()
    # a coordinate riding a dimension it is not named for, on any member,
    # is associated; the array path rebuilds dimension coordinates only
    aux_coords = frozenset(
        () if parent is None else parent.backend.associated_coord_names()
    )
    resolver = PlanResolver(
        token=token,
        dim=name,
        member_rows=member_rows,
        loader=loader,
        merge_kwargs=merge_kwargs,
        parent_residuals=parent_residuals,
        mode=mode,
        aux_coords=aux_coords,
        origin_path=origin_path,
        stamped=tuple(dict.fromkeys([*stamped, *stamps])),
        lossy=lossy,
        trim_dims=plan.trim_dims,
    )
    backend = get_backend(":memory:")
    # residual selections trim at load; identity claims (def keys) for
    # coordinates on the trimmed dims would describe the untrimmed values
    trimmed_dims = _trimmed_dims(residuals, coord_dims_map)
    outputs = plan.outputs
    stale_keys = stale_def_keys(residuals, coord_dims_map, outputs.columns)
    if stale_keys:
        outputs = outputs.drop(columns=stale_keys)
    fills = merge_kwargs.get("fill_value") is not None
    aux_info = _aux_coord_info(
        sources,
        trims,
        name,
        coord_dims_map,
        trimmed_dims,
        concat=mode == "concat",
        conflict=merge_kwargs.get("conflict"),
        source_trimmed=_trimmed_dims(parent_residuals, coord_dims_map),
        filled=_fill_held(outputs, trims, name) if fills else None,
    )
    outputs = _with_windowed_dims(outputs, aux_info, name)
    records = _output_records(
        outputs,
        token,
        aux_info=aux_info,
        sizes=_whole_member_sizes(trims, sources),
        plan_dim=name,
    )
    if parent is not None and "_index_row" in sources.columns:
        concat = mode == "concat"
        records = _with_parent_runs(
            records, parent.backend, trims, sources, name, concat
        )
    if snapped:
        records = _with_snapped(records, snapped)
    backend.write_sources(records)
    return PatchCatalog(backend=backend, resolver=resolver, stamps=stamps)


def _with_snapped(records, snapped) -> list:
    """The records with each output's snapped coordinates stated as they load."""
    out = []
    for source in records:
        patches = []
        for patch in source.patches:
            new = snapped.get(int(patch.source_patch_key), {})
            coords = []
            for coord in patch.coords:
                if (snap := new.get(coord.coord_name)) is not None:
                    restated = _coord_record(coord.coord_name, snap.to_summary())
                    assert restated is not None  # a snapped coordinate has a dtype
                    coord = replace(restated, coord_dims=coord.coord_dims)
                coords.append(coord)
            patches.append(replace(patch, coords=tuple(coords)))
        out.append(replace(source, patches=tuple(patches)))
    return out


def _with_parent_runs(
    records, parent_backend, trims, sources, name: str, concat: bool = False
) -> list:
    """
    The records with the runs their members' coordinates hold in the parent.

    An output of one member holds that member's values, perhaps trimmed,
    so it takes the member's runs; the reports clip runs to each row's
    envelope. An output of several takes runs only for a coordinate which
    kept its identity, and so equals each member's. None takes runs along
    the planned dimension, which the plan cut at its holes (or filled),
    except a concatenation's, which keeps each member's runs as segments.
    Members find their runs through the parent's patch rows (``_index_row``).
    """
    index_rows = dict(zip(sources["_patch_row"], sources["_index_row"], strict=True))
    members: dict[str, list] = {}
    for output_id, patch_row in zip(trims["output_id"], trims["_patch_row"]):
        members.setdefault(str(int(output_id)), []).append(index_rows.get(patch_row))
    known = {x for ids in members.values() for x in ids if pd.notna(x)}
    runs = parent_backend.patch_runs(known)
    if not runs and not concat:
        return records
    rows = trims.groupby("output_id", sort=False)
    out = []
    for source in records:
        patches = []
        for patch in source.patches:
            ids = list(dict.fromkeys(members.get(patch.source_patch_key, [])))
            held = runs.get(ids[0], {}) if ids and pd.notna(ids[0]) else {}
            coords = []
            for coord in patch.coords:
                planned = name in str(coord.coord_dims).split(",")
                parent = held.get(coord.coord_name)
                several = len(ids) > 1 and not coord.data_id
                if concat and coord.coord_name == name and len(ids) > 1:
                    group = rows.get_group(int(patch.source_patch_key))
                    coord = _segment_runs(coord, group, runs, index_rows, name)
                if parent is None or planned or several:
                    coords.append(coord)
                    continue
                same = (parent.value_kind, parent.is_relative, parent.units)
                assert same == (coord.value_kind, coord.is_relative, coord.units)
                carried = {x: getattr(parent, x) for x in _CARRIED_RUN_FIELDS}
                coords.append(replace(coord, **carried))
            patches.append(replace(patch, coords=tuple(coords)))
        out.append(replace(source, patches=tuple(patches)))
    return out


def _segment_runs(coord, members, runs, index_rows, name: str):
    """A concatenated coordinate's runs: each member's, or the member as one."""
    kind, absolute = coord.value_kind, not coord.is_relative
    if f"{name}_step" not in members:  # a dimension the concatenation made
        return coord
    records = []
    for row in members.to_dict("records"):
        held = runs.get(index_rows.get(row["_patch_row"]), {}).get(coord.coord_name)
        if held is not None and held.runs:
            records.extend(index_runs(held.runs, kind, absolute))
            continue
        low, high, step = (row[f"{name}_{x}"] for x in ("min", "max", "step"))
        if pd.isnull(step):  # its count is unknown, so its runs are too
            return coord
        count = round(to_float(high - low) / abs(to_float(step))) + 1
        if records and records[-1].step == step and low - records[-1].max == step:
            prev = records.pop()  # it continues the grid before it exactly
            count, low = prev.count + count, prev.min
        records.append(RunRecord(count, low, high, step))
    if len(records) < 2:
        return coord
    grid = dict.fromkeys(("step_float", "step_int", *_EXACT_GRID_FIELDS))
    return replace(
        coord,
        **grid,
        is_exact=False,
        run_count=len(records),
        runs=stored_runs(records, kind, absolute),
    )


# What a coordinate's runs say of it, beyond its envelope.
_CARRIED_RUN_FIELDS = (
    "step_float",
    "step_int",
    "jittered",
    "gapped",
    "offset_min",
    "offset_max",
    "run_count",
    "runs",
)


def reached(members: pd.DataFrame, dim: str, start, stop) -> pd.DataFrame:
    """The fill members within half a step of a window, or else the nearest."""
    low, high = members[f"{dim}_min"], members[f"{dim}_max"]
    start, stop = (
        x if y is None else y for x, y in ((low.min(), start), (high.max(), stop))
    )
    half = members[f"{dim}_step"].abs().max() / 2
    if not (reach := (high >= start - half) & (low <= stop + half)).any():
        reach = (low - start).abs() == (low - start).abs().min()
    return members[reach]


def collapse_working_df(catalog: PatchCatalog) -> pd.DataFrame | None:
    """
    Return the re-planning frame for a derived catalog, or None.

    Re-planning the *same* dimension collapses: it plans over the
    current view's *members* — the trimmed source rows — restricted to
    outputs the view still presents, with the view's value residuals
    applied to the envelopes. (Planning a different dimension must keep
    the assembled boundaries, so its caller plans over the output rows
    instead and never collapses.)

    A *lossy* plan is the exception and never collapses. Collapsing is
    sound because the members of a chunk or a subdivision together cover
    their sources, so a re-plan which merges them back is entitled to
    load a source whole. A plan which drops samples — channel selection
    keeping some channels of a patch and not others — breaks exactly
    that, and collapsing it would quietly load back what it removed. A
    plan which *adds* samples, by filling a bridged hole, breaks it the
    same way and is the second exception.
    """
    resolver = catalog.resolver
    if not isinstance(resolver, PlanResolver) or resolver.lossy:
        return None
    # A fill plan is the same exception from the other side: its outputs
    # hold samples no member has, so a re-plan over the members would
    # quietly drop the ones it added.
    if resolver.merge_kwargs.get("fill_value") is not None:
        return None
    if any(samples or relative for _, samples, relative in catalog.residuals):
        return None
    members = resolver.member_rows
    if catalog.is_view:
        present = {
            int(_row_source_patch_key(row))
            for row in catalog.to_df().to_dict("records")
        }
        members = members[members["output_id"].isin(present)]
    # `_modified` carries: it says the member is a *trim* of its source
    # rather than the whole of it, which is exactly what the re-plan needs
    # to know. Dropping it left `_build_members` to assume no source was
    # modified, so a member which was a slice of a file came back marked
    # "load whole" and the loader read all of it.
    # the re-plan builds new member rows from these, so what the index
    # measured of each source has to travel with them
    members = resolver.source_stats_of(members)
    # each member stands in for the output holding it, so its index row is
    # that output's patch in this catalog, which carries the member's runs
    ids = catalog.backend.patch_ids_by_key()
    index_ids = [ids.get(str(int(x))) for x in members["output_id"]]
    working = members.assign(_index_row=pd.array(index_ids, dtype="Int64"))
    working = working.drop(columns=["output_id"])
    working = patch_local_adjusted_envelopes(
        working, catalog.residuals, drop_empty=True
    )
    return working.reset_index(drop=True)
