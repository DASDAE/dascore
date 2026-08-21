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
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress

import numpy as np
import pandas as pd

import dascore as dc
from dascore.core.coord_join import join_summaries
from dascore.core.coords import CoordSummary
from dascore.io.index.backend import get_backend
from dascore.io.index.catalog import (
    CompositeResolver,
    PatchCatalog,
    PatchResolver,
    _adjust_unit_segments,
    _row_source_patch_key,
    apply_exact_residuals,
)
from dascore.io.index.ingest import (
    CoordRecord,
    PatchRecord,
    SourceRecord,
    _coord_record,
    coord_summary,
    typed_value,
)
from dascore.units import get_quantity, get_quantity_str
from dascore.utils.attrs import _is_missing
from dascore.utils.chunk_plan import (
    _SOURCE_COLUMNS,
    _ensure_patch_id,
)
from dascore.utils.misc import _CanonicalRange, is_range
from dascore.utils.patch import concatenate_planned
from dascore.utils.patch_assembly import PatchAssembler
from dascore.utils.pd import adjust_segments
from dascore.utils.time import to_float

# Row columns which name dc.read's own keyword arguments; passing one along
# as a trim hint would collide with the value the loader already supplies.
_READ_KWARGS = ("path", "file_format", "file_version")

PLAN_SCHEME = "plan://"
# columns that are structural/positional rather than patch attributes
_NON_ATTR = {"output_id", "dims", "coord_names", "patch"}


def _stated_units(value) -> str | None:
    """Return a unit string, or None when the row states none.

    Row values arrive from dataframes, so an absent unit is NaN rather
    than None — and NaN never equals itself, which would make an unstated
    unit look like a mismatch.
    """
    if value is None or value == "" or pd.isnull(value):
        return None
    return str(value)


def _source_units_column(name: str) -> str:
    """Return the private column recording a member's own unit spelling.

    Deliberately not ``_{name}_source_units``: a coordinate may be named
    ``{name}_source``, whose own unit column has exactly that spelling.
    Every coordinate unit column ends in ``_units``, so a name ending in
    ``_source`` instead can never be one.
    """
    return f"_{name}_units_source"


def _def_key_fingerprint(key) -> str | None:
    """Recover the semantic fingerprint from a stored def key.

    Fingerprinted keys are ``fp:{hash}`` with the unit spelling riding
    after ``|`` (see CoordRecord.def_key); the spelling belongs to
    storage deduplication only, never to value identity.
    """
    if not (isinstance(key, str) and key.startswith("fp:")):
        return None
    return key[3:].split("|", maxsplit=1)[0]


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


def _dtype_str(value) -> str:
    """Convert a stored element dtype to its string, "" when unknown."""
    return "" if value is None or pd.isnull(value) else str(value)


def _coord_record_from_row(
    row: Mapping,
    name: str,
    dims: tuple[str, ...] | None = None,
    name_is_held: bool = False,
) -> CoordRecord | None:
    """
    Build the envelope coord record for one output coordinate.

    Delegates to the ingest converter through a range CoordSummary so
    virtual outputs carry the same identities real patches would: a
    carried ``fp:`` def key survives for non-planned dims, and the
    planned dim's range fingerprint is reconstructed exactly. ``dims``
    names the dimensions the coordinate rides (itself by default).
    `name_is_held` says the members hold this coordinate, so a record is
    written even when nothing about its values can be stated: the patch
    will carry it, and a catalog which omitted it would deny it.
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
        fingerprint = _def_key_fingerprint(key)
        if fingerprint is None and isinstance(key, str) and key.startswith("cat:"):
            fingerprint = key[4:]
        if fingerprint is None and not name_is_held:
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
            coord_hash=fingerprint,
        )
    step = row.get(f"{name}_step")
    step = None if step is None or pd.isnull(step) else step
    if isinstance(lo, str):
        # string coords have no range representation; store the
        # lexicographic envelope directly
        key = row.get(f"_{name}_def_key")
        fingerprint = _def_key_fingerprint(key)
        return CoordRecord(
            coord_name=name,
            value_kind="str",
            dtype="str",
            coord_dims=",".join(dims),
            length=None,
            units=None,
            min_str=str(lo),
            max_str=None if hi is None or pd.isnull(hi) else str(hi),
            coord_hash=fingerprint,
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
        dtype = "float64"
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
    fingerprint = _def_key_fingerprint(key)
    summary = CoordSummary(
        dtype=dtype,
        min=lo,
        max=hi,
        step=step,
        units=units,
        dims=dims,
        len=length,
        fingerprint=fingerprint,
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


def _member_summaries(backend, members: pd.DataFrame) -> dict:
    """Every member's coordinates, as the index recorded them."""
    if not len(members) or backend is None:
        return {}
    ids = [int(x) for x in members["_patch_id"].dropna().unique()]
    assert ids, "a plan's members name the patches they load"
    out: dict[int, dict[str, CoordSummary]] = {}
    for row in backend.coord_frame(ids).to_dict("records"):
        summary = coord_summary(row)
        if summary is not None:
            out.setdefault(int(row["patch_id"]), {})[str(row["coord_name"])] = summary
    if set(out) != set(ids):
        # Re-planning a derived view collapses to the *grandparent's*
        # members, whose ids this index does not use; matching them here
        # would describe the wrong patches. The plan's own rows then say
        # what the outputs hold, as they did before.
        return {}
    return out


def _is_cut(stored: Mapping, row: Mapping, plan_dim: str) -> bool:
    """Whether this member loads less than the whole of its dimension."""
    if row.get("_modified"):
        return True
    summary = stored.get(int(row["_patch_id"]), {}).get(plan_dim)
    low, high = row.get(f"{plan_dim}_min"), row.get(f"{plan_dim}_max")
    if summary is None or (pd.isnull(low) and pd.isnull(high)):
        return False
    return bool(low != summary.min or high != summary.max)


def _trimmed_summary(summary: CoordSummary, row: Mapping, name: str) -> CoordSummary:
    """
    The member's summary as its trim leaves it.

    A trimmed member holds fewer samples than the index recorded, so the
    stored envelope, length and identity all describe something the
    output will not contain; the plan's own range replaces them.
    """
    low, high = row.get(f"{name}_min"), row.get(f"{name}_max")
    if pd.isnull(low) and pd.isnull(high):
        return summary
    if low == summary.min and high == summary.max:
        return summary  # the whole of it, so its identity still holds
    step = row.get(f"{name}_step", summary.step)
    step = summary.step if pd.isnull(step) else step
    length = None
    if step is not None and not pd.isnull(step):
        with suppress(TypeError, ValueError, ZeroDivisionError):
            span = to_float(high) - to_float(low)
            length = round(abs(span / to_float(step))) + 1
    # built rather than copied: these values come from the plan's frame,
    # so they need the conforming a validated summary does (a pandas
    # Timestamp where the rest of the join speaks numpy would not compare)
    units = row.get(f"_{name}_units", summary.units)
    units = summary.units if units is None or pd.isnull(units) else units
    return CoordSummary(
        dtype=summary.dtype,
        min=low,
        max=high,
        step=step,
        units=units,
        dims=summary.dims,
        len=length,
    )


def _union_summary(summaries: Sequence[CoordSummary]) -> CoordSummary:
    """
    What can be said of members which cannot be joined from summaries.

    The envelope spans them all — that much any join preserves — and
    nothing else is claimed: no step, and no identity.
    """
    stated = [x for x in summaries if not pd.isnull(x.min)]
    # a member which states nothing describes nothing, not even its dtype
    template = stated[0] if stated else summaries[0]
    blank = dict(step=None, len=None, fingerprint=None)
    kinds = {_summary_kind(x) for x in stated}
    spellings = {get_quantity(x.units) for x in stated}
    silent = len(stated) != len(summaries)
    if len(kinds) > 1 or len(spellings) > 1 or (silent and "str" in kinds):
        # No envelope covers these members. Two spellings of one
        # coordinate (2000 milliseconds beside 3 seconds) cannot be
        # compared until one is chosen, which only the loaded patch does;
        # two kinds of value cannot be compared at all; and a member
        # which states no labels cannot be given one, since text has no
        # missing value to stand in.
        null = _null_like(template.min)
        return template.model_copy(update=dict(min=null, max=null, **blank))
    lows = [x.min for x in stated]
    highs = [x.max for x in summaries if not pd.isnull(x.max)]
    return template.model_copy(
        update=dict(
            min=min(lows) if lows else template.min,
            max=max(highs) if highs else template.max,
            **blank,
        )
    )


def _unvouched(trimmed: bool) -> dict:
    """
    What a summary may still say once its values are not vouched for.

    A residual selection trims these values when the patch loads, so the
    step and the sample count describe something the output will not
    contain — and a summary which still looked evenly sampled would have
    its identity recomputed from those very values by `_coord_record`.
    The envelope stays: it still bounds where the output lies.
    """
    void: dict = {"fingerprint": None}
    if trimmed:
        void.update(step=None, len=None)
    return void


def _summary_kind(summary: CoordSummary) -> str:
    """Whether a summary holds times, numbers or labels."""
    kind = np.dtype(summary.dtype).kind if summary.dtype else ""
    if kind in "mM":
        return "time"
    return "str" if kind in "USO" else "num"


def _null_like(value):
    """The missing value of whatever kind this one is."""
    if isinstance(value, np.datetime64 | pd.Timestamp):
        return np.datetime64("NaT", "ns")
    if isinstance(value, np.timedelta64 | pd.Timedelta):
        return np.timedelta64("NaT", "ns")
    return np.nan


def predicted_coords(
    backend,
    members: pd.DataFrame,
    plan_dim: str,
    *,
    trimmed_dims: frozenset[str] = frozenset(),
    snap_tolerance: float | None = None,
) -> dict[int, dict[str, CoordSummary]]:
    """
    Per output, the summary of every coordinate its members hold.

    This is what the plan claims about an output, and it is decided by
    running the *real* join over the members' summaries
    ([`join_summaries`](`dascore.core.coord_join.join_summaries`)), so a
    row cannot describe a coordinate differently from the patch assembly
    will build. Where the join cannot be decided from summaries alone the
    envelope still spans the members and nothing else is claimed.

    Parameters
    ----------
    backend
        The parent index, which holds the members' coordinate rows.
    members
        The plan's member table: which patches feed which output, with
        each member's trim.
    plan_dim
        The dimension being chunked or concatenated. Coordinates riding
        it are joined along it; the others must already agree.
    trimmed_dims
        Dimensions a residual selection trims at load. A coordinate on
        one of them describes untrimmed values, so it keeps no identity.
    snap_tolerance
        Passed to the join, bounding how far a seam may be absorbed.
    """
    stored = _member_summaries(backend, members)
    if not stored:
        return {}
    out: dict[int, dict[str, CoordSummary]] = {}
    for output_id, rows in members.groupby("output_id", sort=True):
        records = rows.to_dict("records")
        names: dict[str, None] = {}  # an ordered set
        for row in records:
            names.update(dict.fromkeys(stored.get(int(row["_patch_id"]), {})))
        described: dict[str, CoordSummary] = {}
        # the plan's member rows are what will be *loaded*: they carry each
        # member's trim, in the unit the plan settled on, so along the
        # planned dimension they outrank what the index recorded
        cut = any(_is_cut(stored, row, plan_dim) for row in records)
        for name in names:
            summaries = []
            for row in records:
                summary = stored.get(int(row["_patch_id"]), {}).get(name)
                if summary is None:
                    continue
                if name == plan_dim:
                    summary = _trimmed_summary(summary, row, name)
                elif cut and plan_dim in summary.dims:
                    # a coordinate riding a dimension being cut loses the
                    # values the cut removes, which its summary still counts
                    summary = summary.model_copy(
                        update=dict(step=None, len=None, fingerprint=None)
                    )
                summaries.append(summary)
            assert summaries, "a name comes from the members which state it"
            stated = _describe(name, summaries, plan_dim, trimmed_dims, snap_tolerance)
            if stated is not None:
                described[name] = stated
        out[int(str(output_id))] = described
    return out


def _describe(
    name: str,
    summaries: Sequence[CoordSummary],
    plan_dim: str,
    trimmed_dims: frozenset[str],
    snap_tolerance: float | None,
) -> CoordSummary | None:
    """
    State one coordinate of one output, claiming only what holds.

    None means the output does not carry it at all.
    """
    first = summaries[0]
    if plan_dim == name and name not in first.dims:
        # the members hold this as an ordinary coordinate and the
        # concatenation replaces it with a dimension of its own, so
        # nothing the members say about it survives
        return None
    rides = plan_dim == name or plan_dim in first.dims
    trimmed = bool(set(first.dims) & trimmed_dims)
    if not rides:
        # every member states the same coordinate, or assembly refuses to
        # build the output at all; the identity survives when they agree
        agreed = len({x.fingerprint for x in summaries}) == 1
        if agreed and not trimmed:
            return first
        return first.model_copy(update=_unvouched(trimmed))
    blank = all(pd.isnull(x.min) and pd.isnull(x.max) for x in summaries)
    if blank and name == plan_dim:
        # Nobody states any values along the dimension being joined, so
        # there is nothing to join and nothing to say the plan's own row
        # does not already say: it carries the identity the planner works
        # out for such a dimension (see _member_key_digests). An
        # auxiliary coordinate has no such row, so it is still described.
        return None
    joined = join_summaries(summaries, snap_tolerance=snap_tolerance)
    if joined is None:
        return _union_summary(summaries)
    if trimmed:
        joined = joined.model_copy(update=_unvouched(True))
    return joined.model_copy(update=dict(dims=first.dims))


def _aux_coord_info(
    source_rows: pd.DataFrame,
    members: pd.DataFrame,
    plan_dim: str,
    coord_dims_map: Mapping[str, str],
    trimmed_dims: frozenset[str] = frozenset(),
) -> dict[int, dict[str, dict]]:
    """
    Aggregate per-output envelope info for auxiliary coordinates.

    Aggregated from the *member source rows* (authoritative, unlike the
    planner's carried columns). Structural identity (def key and step,
    which permit fingerprint claims) is kept only when every member
    shares one def key and the values provably survive assembly: a
    coordinate riding the planned dimension is trimmed/merged with it,
    so only a lone unmodified member keeps identity there. Envelopes
    always aggregate — the catalog contract is candidacy, with exact
    values re-established at load.

    Used where an output's coordinates cannot be predicted from the
    members' own summaries — a re-plan whose members this index does not
    know — so the member rows are all there is to describe them with.

    """
    out: dict[int, dict[str, dict]] = {}
    if not len(members) or not coord_dims_map:
        return out
    cols = [c for c in ("output_id", "_patch_id", "_modified") if c in members.columns]
    joined = members[cols].merge(source_rows, on="_patch_id", how="left")
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
        unit_col = f"_{name}_units"
        lows = _extrema(grouped[cmin], "min")
        highs = _extrema(grouped[cmax], "max")
        # the *_first arrays are only read where their gate is True, and
        # a gate can only be True when its column exists
        no_gate = np.zeros(len(output_ids), dtype=bool)
        keep, key_first = no_gate, None
        if key_col in joined.columns:
            keep = grouped[key_col].nunique().to_numpy() == 1
            key_first = grouped[key_col].first().to_numpy()
        keep = keep & (not trimmed)
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
        unit_ok, unit_first = no_gate, None
        if unit_col in joined.columns:
            unit_ok = grouped[unit_col].nunique().to_numpy() == 1
            unit_first = grouped[unit_col].first().to_numpy()
        # a coordinate no member holds contributes nothing; one the members
        # do hold is always named, even when nothing about its values can
        # be stated — the patch will have it, so the catalog says so
        held = grouped[cmin].count().to_numpy() > 0
        if key_col in joined.columns:
            held = held | (grouped[key_col].count().to_numpy() > 0)
        absent = ~held
        for index in np.flatnonzero(~absent):
            step = step_first[index] if step_first is not None else None
            key = key_first[index] if key_first is not None else None
            unit = unit_first[index] if unit_first is not None else None
            info = {
                cmin: lows[index],
                cmax: highs[index],
                step_col: step if step_ok[index] else None,
                key_col: key if keep[index] else None,
                unit_col: unit if unit_ok[index] else None,
                "dims": dims,
            }
            out.setdefault(int(output_ids[index]), {})[name] = info
    return out


def _apply_predictions(
    outputs: pd.DataFrame,
    predicted: Mapping[int, Mapping[str, CoordSummary]],
    name: str,
) -> pd.DataFrame:
    """
    Restate the planned dimension's envelope from what the join predicts.

    The frame's envelope columns feed selection and the patches table, so
    they must say what the records say; otherwise the row and its own
    coordinate would disagree, which is the drift this predicts away.
    """
    if not predicted:
        return outputs
    min_name, max_name, step_name = f"{name}_min", f"{name}_max", f"{name}_step"
    unit_col = f"_{name}_units"
    out = outputs.copy(deep=False)
    described = [predicted.get(int(x), {}).get(name) for x in out["output_id"]]
    if not any(x is not None for x in described):
        return out
    fields = {
        min_name: lambda x: x.min,
        max_name: lambda x: x.max,
        step_name: lambda x: x.step,
        unit_col: lambda x: None if x.units is None else get_quantity_str(x.units),
    }
    for column, read in fields.items():
        if column not in out.columns:
            continue
        # an output the join could not describe keeps what the row said;
        # only what was predicted is restated
        kept = out[column].to_numpy(dtype=object, copy=True)
        for index, summary in enumerate(described):
            if summary is not None:
                kept[index] = read(summary)
        out[column] = pd.Series(kept, index=out.index, dtype=object)
    return out


def _output_records(
    outputs: pd.DataFrame,
    token: str,
    aux_info: Mapping[int, Mapping[str, Mapping]] | None = None,
    predicted: Mapping[int, Mapping[str, CoordSummary]] | None = None,
) -> list[SourceRecord]:
    """
    Convert plan output rows into ingestible source records.

    `predicted` states what an output's coordinates will be, decided by
    joining its members' summaries. A coordinate it describes is written
    from that summary; the row is consulted only for what it cannot know,
    such as a dimension the plan creates out of the member count.
    """
    records = []
    aux_info = aux_info or {}
    predicted = predicted or {}
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
    for row in outputs.to_dict("records"):
        output_id = int(row["output_id"])
        dims = str(row.get("dims") or "")
        dim_names = [d for d in dims.split(",") if d]
        aux = aux_info.get(output_id, {})
        known = predicted.get(output_id, {})
        coords = []
        for name in dim_names:
            summary = known.get(name)
            if summary is not None:
                record = _coord_record(
                    name, summary.model_copy(update={"dims": (name,)})
                )
            else:
                record = _coord_record_from_row(row, name)
            if record is not None:
                coords.append(record)
        for name, summary in known.items():
            if name in dim_names:
                continue
            record = _coord_record(name, summary)
            if record is not None:
                coords.append(record)
        # auxiliary (non-dimension) coordinates remain on the assembled
        # patches, so the catalog must keep describing them
        for name, info in aux.items():
            if name in dim_names or name in known:
                continue
            record = _coord_record_from_row(
                info, name, dims=info["dims"], name_is_held=True
            )
            if record is not None:
                coords.append(record)
        cache_key = (dims, tuple(aux), tuple(known))
        envelope_keys = envelope_cache.get(cache_key)
        if envelope_keys is None:
            coord_names = set(dim_names) | set(aux) | set(known) | base_names
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
            dtype=_dtype_str(row.get("_dtype")),
            time_min=_ns(row.get("time_min")),
            time_max=_ns(row.get("time_max")),
            time_step=_ns(row.get("time_step")),
            distance_min=_num(row.get("distance_min")),
            distance_max=_num(row.get("distance_max")),
            distance_step=_num(row.get("distance_step")),
            attrs=attrs,
            coords=tuple(coords),
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
        origin_path=None,
        stamped: tuple[str, ...] = (),
        lossy: bool = False,
    ):
        if "output_id" not in member_rows.columns:
            msg = "member_rows must carry an output_id column."
            raise ValueError(msg)
        # plan invariant: outputs without members must never be published
        self.token = token
        self.dim = dim
        self.member_rows = member_rows.reset_index(drop=True)
        self.loader = loader
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

        return PatchAssembler(
            load_patch=self._load_member,
            merge_kwargs=self.merge_kwargs,
        )

    def _load_member(self, kwargs: Mapping) -> dc.Patch:
        """Load one member source patch, applying parent residuals."""
        trim = {}
        if kwargs.get("_modified"):
            trim = {
                k: v
                for k, v in kwargs.items()
                if not str(k).startswith("_")
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
                for suffix in ("_min", "_max", "_step"):
                    trim.pop(f"{self.dim}{suffix}", None)
        patch = self.loader.resolve(kwargs, **trim)
        patch = apply_exact_residuals(patch, self.parent_residuals)
        return self._in_plan_units(patch, kwargs)

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
        assert len(members), "no plan members found for output row"
        if self.mode == "identity":
            # one untouched member per output; residuals apply at load
            assert len(members) == 1
            patch = self._load_member(members.iloc[0].to_dict())
        elif self.mode == "concat":
            # the plan decided what fits together; assembly executes it
            loaded = [
                self._load_member(kwargs) for kwargs in members.to_dict("records")
            ]
            patch = concatenate_planned(
                loaded,
                self.dim,
                count=self.merge_kwargs.get("count"),
                conflict=self.merge_kwargs.get("conflict", "raise"),
            )
        else:
            joined = members.assign(current_index=output_id)
            assembled = self._assembler()._patch_from_instruction_df(joined)
            assert len(assembled) == 1
            patch = assembled[0]
        return self._stamp(patch, row)

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
    names = {n for coords, _ in residuals for n in coords}
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


def _residual_ranges(residuals) -> dict:
    """Envelope-applicable value ranges from a residual tuple.

    Bare ranges apply to the native envelope columns directly; a
    `_CanonicalRange` passes through whole so the caller can convert it
    per row unit.
    """
    out = {}
    for coords, samples in residuals:
        if samples:
            continue
        for name, value in coords.items():
            if getattr(value, "magnitudes", None) is not None:
                out[name] = value
            elif is_range(value) and not any(
                hasattr(b, "units") for b in value if b is not None
            ):
                out[name] = value
    return out


def derived_catalog(
    *,
    source_rows: pd.DataFrame,
    plan,
    parent: PatchCatalog | None,
    merge_kwargs: Mapping,
    mode: str = "chunk",
    origin_path=None,
    stamped: tuple[str, ...] = (),
    lossy: bool = False,
) -> PatchCatalog:
    """
    Materialize a plan into a fresh in-memory catalog.

    ``source_rows`` are the full member source rows (path/format/
    identity plus envelopes and attrs) keyed by ``_patch_id`` matching
    ``plan.members``; ``parent`` supplies the resolver (live registry,
    file root, nested plans) and the residual selections its view
    carried, which member loading re-applies.
    """
    token = secrets.token_hex(8)
    name = plan.dim
    trims = plan.members
    trim_cols = [c for c in trims.columns if c not in ("_patch_id",)]
    sources = source_rows.copy(deep=False)
    if "_patch_id" not in sources.columns:
        sources = _ensure_patch_id(sources)
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
    member_rows = trims[["_patch_id", *[c for c in trim_cols]]].merge(
        sources.drop(columns=[c for c in trim_cols if c in sources], errors="ignore"),
        on="_patch_id",
        how="left",
    )
    # the member's trimmed range replaces the source envelope for loading
    member_rows = member_rows.drop(columns=["_patch_id"])
    parent_residuals = () if parent is None else parent.residuals
    # resolve stored-relative paths once; the derived catalog is
    # root-independent afterwards
    root = getattr(parent.resolver, "_root", None) if parent is not None else None
    if root is not None and "source_path" in member_rows.columns:
        member_rows = member_rows.assign(
            source_path=[
                str(p)
                if "://" in str(p) or str(p).startswith("/")
                else str(root / str(p))
                for p in member_rows["source_path"]
            ]
        )
    loader = CompositeResolver()
    if parent is not None:
        member_paths = set(
            member_rows.get("source_path", pd.Series(dtype=str)).astype(str)
        )
        loader.absorb(parent.resolver, paths=member_paths)
    coord_dims_map = {} if parent is None else parent.backend.coord_dims_map()
    resolver = PlanResolver(
        token=token,
        dim=name,
        member_rows=member_rows,
        loader=loader,
        merge_kwargs=merge_kwargs,
        parent_residuals=parent_residuals,
        mode=mode,
        origin_path=origin_path,
        stamped=stamped,
        lossy=lossy,
    )
    backend = get_backend(":memory:")
    # residual selections trim at load; identity claims (def keys) for
    # coordinates on the trimmed dims would describe the untrimmed values
    trimmed_dims = _trimmed_dims(parent_residuals, coord_dims_map)
    outputs = plan.outputs
    stale_keys = stale_def_keys(parent_residuals, coord_dims_map, outputs.columns)
    if stale_keys:
        outputs = outputs.drop(columns=stale_keys)
    # what an output will hold is decided by joining its members'
    # summaries, the same join assembly runs on their values
    snap = (
        merge_kwargs.get("tolerance") if merge_kwargs.get("snap_coords", True) else None
    )
    predicted = predicted_coords(
        None if parent is None else parent.backend,
        trims,
        name,
        trimmed_dims=trimmed_dims,
        snap_tolerance=snap,
    )
    outputs = _apply_predictions(outputs, predicted, name)
    aux_info = {}
    if not predicted:
        # a re-plan whose members this index does not know: the auxiliary
        # coordinates are described from the member rows, as before
        aux_info = _aux_coord_info(sources, trims, name, coord_dims_map, trimmed_dims)
    records = _output_records(outputs, token, aux_info=aux_info, predicted=predicted)
    backend.write_sources(records)
    return PatchCatalog(backend=backend, resolver=resolver)


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
    that, and collapsing it would quietly load back what it removed.
    """
    resolver = catalog.resolver
    if not isinstance(resolver, PlanResolver) or resolver.lossy:
        return None
    members = resolver.member_rows
    if catalog.is_view:
        present = {
            int(_row_source_patch_key(row))
            for row in catalog.to_df().to_dict("records")
        }
        members = members[members["output_id"].isin(present)]
    ranges = _residual_ranges(catalog.residuals)
    # `_modified` carries: it says the member is a *trim* of its source
    # rather than the whole of it, which is exactly what the re-plan needs
    # to know. Dropping it left `_build_members` to assume no source was
    # modified, so a member which was a slice of a file came back marked
    # "load whole" and the loader read all of it.
    working = members.drop(columns=["output_id"], errors="ignore")
    bare = {
        name: value
        for name, value in ranges.items()
        if not isinstance(value, _CanonicalRange)
    }
    if bare:
        working = adjust_segments(working, ignore_bad_kwargs=True, **bare)
    for name, canonical in ranges.items():
        if isinstance(canonical, _CanonicalRange):
            working = _adjust_unit_segments(working, name, canonical)
    return working.reset_index(drop=True)
