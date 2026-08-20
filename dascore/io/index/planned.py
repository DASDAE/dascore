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
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

import dascore as dc
from dascore.constants import WARN_LEVELS
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
    typed_value,
)
from dascore.units import get_quantity
from dascore.utils.attrs import _is_missing
from dascore.utils.chunk_plan import _SOURCE_COLUMNS, _ensure_patch_id
from dascore.utils.misc import _CanonicalRange, is_range
from dascore.utils.patch import concatenate_patches
from dascore.utils.patch_assembly import PatchAssembler
from dascore.utils.pd import adjust_segments

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
    row: Mapping, name: str, dims: tuple[str, ...] | None = None
) -> CoordRecord | None:
    """
    Build the envelope coord record for one output coordinate.

    Delegates to the ingest converter through a range CoordSummary so
    virtual outputs carry the same identities real patches would: a
    carried ``fp:`` def key survives for non-planned dims, and the
    planned dim's range fingerprint is reconstructed exactly. ``dims``
    names the dimensions the coordinate rides (itself by default).
    """
    dims = (name,) if dims is None else dims
    lo, hi = row.get(f"{name}_min"), row.get(f"{name}_max")
    if lo is None or (pd.isnull(lo) and pd.isnull(hi)):
        return None
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
        step = None if step is None else abs(float(step))
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
        length = round((hi - lo) / step) + 1  # ty: ignore[unsupported-operator]
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
        if cmin not in joined.columns:
            continue
        dims = tuple(d for d in str(dims_str).split(",") if d)
        rides = plan_dim in dims
        # a residual selection trims the dims it rides at load, changing
        # the values of every coordinate on those dims
        trimmed = bool(set(dims) & trimmed_dims)
        key_col, step_col = f"_{name}_def_key", f"{name}_step"
        unit_col = f"_{name}_units"
        lows = grouped[cmin].min().to_numpy()
        highs = grouped[cmax].max().to_numpy()
        # the *_first arrays are only read where their gate is True, and
        # a gate can only be True when its column exists
        no_gate = np.zeros(len(output_ids), dtype=bool)
        keep, key_first = no_gate, None
        if key_col in joined.columns:
            keep = grouped[key_col].nunique().to_numpy() == 1
            key_first = grouped[key_col].first().to_numpy()
        keep = keep & (not trimmed)
        if rides:
            keep = keep & single & ~modified
        step_ok, step_first = no_gate, None
        if step_col in joined.columns:
            step_ok = keep & (grouped[step_col].nunique().to_numpy() == 1)
            step_first = grouped[step_col].first().to_numpy()
        unit_ok, unit_first = no_gate, None
        if unit_col in joined.columns:
            unit_ok = grouped[unit_col].nunique().to_numpy() == 1
            unit_first = grouped[unit_col].first().to_numpy()
        # a coordinate absent from every member contributes nothing
        absent = pd.isnull(lows) & pd.isnull(highs)
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


def _output_records(
    outputs: pd.DataFrame,
    token: str,
    aux_info: Mapping[int, Mapping[str, Mapping]] | None = None,
) -> list[SourceRecord]:
    """Convert plan output rows into ingestible source records."""
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
    for row in outputs.to_dict("records"):
        output_id = int(row["output_id"])
        dims = str(row.get("dims") or "")
        dim_names = [d for d in dims.split(",") if d]
        aux = aux_info.get(output_id, {})
        coords = []
        for name in dim_names:
            record = _coord_record_from_row(row, name)
            if record is not None:
                coords.append(record)
        # auxiliary (non-dimension) coordinates remain on the assembled
        # patches, so the catalog must keep describing them
        for name, info in aux.items():
            if name in dim_names:
                continue
            record = _coord_record_from_row(info, name, dims=info["dims"])
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
        check_behavior: WARN_LEVELS = "warn",
        origin_path=None,
        stamped: tuple[str, ...] = (),
        lossy: bool = False,
        attr_units: Mapping[str, str | None] | None = None,
        coord_names: Iterable[str] = (),
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
        self.check_behavior = check_behavior
        # The kind rule which produced the plan decides assembly too,
        # whatever the config says by the time a patch is asked for.
        self.kind_attrs = dc.get_config().patch_kind_attrs
        # informational only: the directory/file the plan derived from
        self.origin_path = origin_path
        # attrs the outputs state about themselves rather than inherit
        self.stamped = tuple(stamped)
        # the units the index stores each numeric attr column in (None for
        # a plain number), resolved at planning so a row's magnitude can
        # be turned back into the attr it came from
        self.attr_units = dict(attr_units or {})
        # every coordinate the parent index knows, so a row's envelope
        # columns are told from attrs which merely look like one
        self.coord_names = frozenset(coord_names)
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
        with dc.config_context(patch_kind_attrs=self.kind_attrs):
            return self._resolve(row, **trim)

    def _resolve(self, row: Mapping, **trim) -> dc.Patch:
        """Assemble under the plan's own kind rule."""
        output_id = int(_row_source_patch_key(row))
        members = self.member_rows[self.member_rows["output_id"] == output_id]
        assert len(members), "no plan members found for output row"
        if self.mode == "identity":
            # one untouched member per output; residuals apply at load
            assert len(members) == 1
            patch = self._load_member(members.iloc[0].to_dict())
        elif self.mode == "concat":
            loaded = [
                self._load_member(kwargs) for kwargs in members.to_dict("records")
            ]
            out = concatenate_patches(
                loaded, check_behavior=self.check_behavior, **{self.dim: None}
            )
            assert len(out) == 1
            patch = out[0]
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
        shows for it.
        """
        if self.stamped:
            patch = patch.update_attrs(**{x: row[x] for x in self.stamped})
        # A row carries its partition's known attr values; the members
        # which survived into this output may lack some (a missing value
        # matches, and overlap removal may keep the member which lacked
        # it). Fill them so the patch agrees with the row.
        attrs = patch.attrs
        # every public attr column the row carries, extras included; the
        # coordinate envelope columns and the row's own bookkeeping are not attrs
        # the row may carry an envelope for a coordinate the assembled patch
        # no longer has, so every coordinate the index knows counts
        coords = set(patch.coords.coord_map) | set(patch.dims) | {self.dim}
        coords |= self.coord_names
        envelope = {f"{x}_{y}" for x in coords for y in ("min", "max", "step", "units")}
        skip = {*_SOURCE_COLUMNS, *coords, *envelope, "dims", "output_id"}
        names = {x for x in row if not x.startswith("_") and x not in skip}
        # A number in a row may be a quantity the index stored as a base-SI
        # magnitude with its units kept in the attr metadata resolved at
        # planning; a number whose units the plan does not know is left
        # alone rather than stamped as a bare magnitude.
        fill = {}
        for x in names:
            value = row[x]
            if _is_missing(value) or not _is_missing(attrs.get(x)):
                continue
            if _is_number(value):
                if x not in self.attr_units:
                    continue
                if units := self.attr_units[x]:
                    value = value * get_quantity(units)
            fill[x] = value
        if fill:
            patch = patch.new(attrs=attrs.update(**fill))
        return patch


def _plan_attr_units(parent, outputs: pd.DataFrame) -> dict[str, str | None]:
    """
    The units the parent index stores each public output column in.

    One read of the index's attr metadata; None marks a plain value, and
    columns the index holds no attr metadata for (envelopes, bookkeeping)
    are None too — `_stamp` never fills those anyway.
    """
    if parent is None:
        return {}
    known = parent.backend.attr_units_map()
    return {x: known.get(x) for x in outputs.columns if not x.startswith("_")}


def _is_number(value) -> bool:
    """True for a numeric scalar other than a bool."""
    return isinstance(value, int | float | np.number) and not isinstance(
        value, bool | np.bool_
    )


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
    check_behavior: WARN_LEVELS = "warn",
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
    resolver = PlanResolver(
        token=token,
        dim=name,
        member_rows=member_rows,
        loader=loader,
        merge_kwargs=merge_kwargs,
        parent_residuals=parent_residuals,
        mode=mode,
        check_behavior=check_behavior,
        origin_path=origin_path,
        stamped=stamped,
        lossy=lossy,
        attr_units=_plan_attr_units(parent, plan.outputs),
        coord_names=() if parent is None else parent.backend.coord_names(),
    )
    backend = get_backend(":memory:")
    coord_dims_map = {} if parent is None else parent.backend.coord_dims_map()
    # residual selections trim at load; identity claims (def keys) for
    # coordinates on the trimmed dims would describe the untrimmed values
    trimmed_dims = _trimmed_dims(parent_residuals, coord_dims_map)
    outputs = plan.outputs
    stale_keys = stale_def_keys(parent_residuals, coord_dims_map, outputs.columns)
    if stale_keys:
        outputs = outputs.drop(columns=stale_keys)
    aux_info = _aux_coord_info(sources, trims, name, coord_dims_map, trimmed_dims)
    backend.write_sources(_output_records(outputs, token, aux_info=aux_info))
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
