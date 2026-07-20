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

import numpy as np
import pandas as pd

import dascore as dc
from dascore.io.index.backend import get_backend
from dascore.io.index.catalog import (
    CompositeResolver,
    PatchCatalog,
    PatchResolver,
    _row_source_patch_id,
    apply_exact_residuals,
)
from dascore.io.index.ingest import (
    CoordRecord,
    PatchRecord,
    SourceRecord,
    typed_value,
)
from dascore.utils.misc import is_range
from dascore.utils.pd import adjust_segments

PLAN_SCHEME = "plan://"
# columns that are structural/positional rather than patch attributes
_NON_ATTR = {"output_id", "dims", "coord_names", "patch"}


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
    from dascore.core.coords import CoordSummary
    from dascore.io.index.ingest import _coord_record

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
        fingerprint = (
            key[3:] if isinstance(key, str) and key.startswith("fp:") else None
        )
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
    if isinstance(lo, pd.Timestamp | np.datetime64):
        lo, hi = pd.Timestamp(lo).to_datetime64(), pd.Timestamp(hi).to_datetime64()
        dtype = "datetime64[ns]"
    elif isinstance(lo, pd.Timedelta | np.timedelta64):
        lo, hi = pd.Timedelta(lo).to_timedelta64(), pd.Timedelta(hi).to_timedelta64()
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
    # numeric envelope values are stored canonical-SI, so only the
    # canonical (base) unit carried by the pivot may be attached —
    # ingest's re-conversion is then the identity (time kinds attach
    # without conversion)
    units = row.get(f"_{name}_units")
    if units == "" or (units is not None and pd.isnull(units)):
        units = None
    length = None
    if step is not None:
        length = int(round((hi - lo) / step)) + 1
    key = row.get(f"_{name}_def_key")
    fingerprint = None
    if isinstance(key, str) and key.startswith("fp:"):
        fingerprint = key[3:]
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
        for output_id, sub in joined.groupby("output_id"):
            lo, hi = sub[cmin].min(), sub[cmax].max()
            if pd.isnull(lo) and pd.isnull(hi):
                continue
            keys = set(sub[key_col].dropna()) if key_col in sub.columns else set()
            modified = bool(sub["_modified"].any()) if "_modified" in sub else False
            keep = (
                len(keys) == 1
                and not trimmed
                and (not rides or (len(sub) == 1 and not modified))
            )
            steps = (
                set(sub[step_col].dropna())
                if keep and step_col in sub.columns
                else set()
            )
            unit_col = f"_{name}_units"
            units = set(sub[unit_col].dropna()) if unit_col in sub.columns else set()
            info = {
                cmin: lo,
                cmax: hi,
                step_col: steps.pop() if len(steps) == 1 else None,
                key_col: keys.pop() if keep else None,
                unit_col: units.pop() if len(units) == 1 else None,
                "dims": dims,
            }
            out.setdefault(int(output_id), {})[name] = info
    return out


def _output_records(
    outputs: pd.DataFrame,
    token: str,
    aux_info: Mapping[int, Mapping[str, Mapping]] | None = None,
) -> list[SourceRecord]:
    """Convert plan output rows into ingestible source records."""
    records = []
    aux_info = aux_info or {}
    for row in outputs.to_dict("records"):
        output_id = int(row["output_id"])
        dims = str(row.get("dims") or "")
        dim_names = [d for d in dims.split(",") if d]
        coords = []
        for name in dim_names:
            record = _coord_record_from_row(row, name)
            if record is not None:
                coords.append(record)
        # auxiliary (non-dimension) coordinates remain on the assembled
        # patches, so the catalog must keep describing them
        for name, info in aux_info.get(output_id, {}).items():
            if name in dim_names:
                continue
            record = _coord_record_from_row(info, name, dims=info["dims"])
            if record is not None:
                coords.append(record)
        # Envelope columns belong to coordinates actually present in the
        # row; an attr that merely looks envelope-shaped (channel_step with
        # no channel coord) is ordinary metadata and must be preserved.
        coord_names = set(dim_names) | set(aux_info.get(output_id, {}))
        coord_names |= {
            key[1 : -len("_def_key")]
            for key in row
            if key.startswith("_") and key.endswith("_def_key")
        }
        coord_names |= {"time", "distance"}  # fixed patches-table envelopes
        envelope_keys = {
            f"{name}_{sfx}"
            for name in coord_names
            for sfx in ("min", "max", "step", "units")
        }
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
            source_patch_id=str(output_id),
            dims=dims,
            shape="",
            n_dims=len(dim_names),
            sample_count_total=None,
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
        check_behavior: str = "warn",
        origin_path=None,
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
        # informational only: the directory/file the plan derived from
        self.origin_path = origin_path

    def live_entries(self) -> Mapping[str, dc.Patch]:
        """Expose the loader's live registry (for absorption/transfer)."""
        return self.loader.live_entries()

    def plan_entries(self) -> Mapping[str, PlanResolver]:
        """Route plan:// paths with this resolver's token to it."""
        nested = dict(getattr(self.loader, "plan_entries", dict)())
        nested[f"{PLAN_SCHEME}{self.token}/"] = self
        return nested

    def _assembler(self):
        from dascore.utils.patch_assembly import PatchAssembler

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
                and k not in ("path", "file_format", "file_version", "source_patch_id")
            }
        patch = self.loader.resolve(kwargs, **trim)
        return apply_exact_residuals(patch, self.parent_residuals)

    def resolve(self, row: Mapping, **trim) -> dc.Patch:
        """Assemble the output patch a plan row describes."""
        output_id = int(_row_source_patch_id(row))
        members = self.member_rows[self.member_rows["output_id"] == output_id]
        assert len(members), "no plan members found for output row"
        if self.mode == "identity":
            # one untouched member per output; residuals apply at load
            assert len(members) == 1
            return self._load_member(members.iloc[0].to_dict())
        if self.mode == "concat":
            from dascore.utils.patch import concatenate_patches

            patches = [
                self._load_member(kwargs) for kwargs in members.to_dict("records")
            ]
            out = concatenate_patches(
                patches, check_behavior=self.check_behavior, **{self.dim: None}
            )
            assert len(out) == 1
            return out[0]
        joined = members.assign(current_index=output_id)
        patches = self._assembler()._patch_from_instruction_df(joined)
        assert len(patches) == 1
        return patches[0]


def _residual_ranges(residuals) -> dict:
    """Envelope-applicable value ranges from a residual tuple."""
    out = {}
    for coords, samples in residuals:
        if samples:
            continue
        for name, value in coords.items():
            magnitudes = getattr(value, "magnitudes", None)
            if magnitudes is not None:
                out[name] = magnitudes
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
    check_behavior: str = "warn",
    origin_path=None,
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
        from dascore.utils.chunk_plan import _ensure_patch_id

        sources = _ensure_patch_id(sources)
    member_rows = trims[["_patch_id", *[c for c in trim_cols]]].merge(
        sources.drop(columns=[c for c in trim_cols if c in sources], errors="ignore"),
        on="_patch_id",
        how="left",
    )
    # the member's trimmed range replaces the source envelope for loading
    member_rows = member_rows.drop(columns=["_patch_id"])
    parent_residuals = () if parent is None else parent._residuals
    # resolve stored-relative paths once; the derived catalog is
    # root-independent afterwards
    root = getattr(parent.resolver, "_root", None) if parent is not None else None
    if root is not None and "path" in member_rows.columns:
        member_rows = member_rows.assign(
            path=[
                str(p)
                if "://" in str(p) or str(p).startswith("/")
                else str(root / str(p))
                for p in member_rows["path"]
            ]
        )
    loader = CompositeResolver()
    if parent is not None:
        member_paths = set(member_rows.get("path", pd.Series(dtype=str)).astype(str))
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
    )
    backend = get_backend(":memory:")
    coord_dims_map = {} if parent is None else parent.backend.coord_dims_map()
    # residual selections trim at load; identity claims (def keys) for
    # coordinates on the trimmed dims would describe the untrimmed values
    residual_names = {n for coords, _ in parent_residuals for n in coords}
    trimmed_dims = frozenset(
        d for n in residual_names for d in str(coord_dims_map.get(n, n)).split(",") if d
    )
    outputs = plan.outputs
    stale_keys = [
        f"_{c}_def_key"
        for c, dims_str in coord_dims_map.items()
        if set(str(dims_str).split(",")) & trimmed_dims
        and f"_{c}_def_key" in outputs.columns
    ]
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
    """
    resolver = catalog.resolver
    if not isinstance(resolver, PlanResolver):
        return None
    members = resolver.member_rows
    if catalog.is_view:
        present = {
            int(_row_source_patch_id(row)) for row in catalog.to_df().to_dict("records")
        }
        members = members[members["output_id"].isin(present)]
    ranges = _residual_ranges(catalog._residuals)
    working = members.drop(columns=["output_id", "_modified"], errors="ignore")
    if ranges:
        working = adjust_segments(working, ignore_bad_kwargs=True, **ranges)
    return working.reset_index(drop=True)
