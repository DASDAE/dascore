"""
Convert patch summaries into normalized index records.

This module is backend-independent: it turns `PatchSummary` objects into
plain records (dicts/dataclasses) using only the four primitive storage
types. All unit-bearing numeric values are normalized to pint base SI
units here, so cross-patch comparisons in the index are always valid.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from dascore.core.summary import PatchSummary, normalize_source_patch_id
from dascore.io.index.schema import KINDS, RESERVED_ATTR_COLUMNS
from dascore.units import get_quantity
from dascore.utils.time import to_datetime64, to_int, to_timedelta64

_SANITIZE_RE = re.compile(r"[^a-z0-9_]+")

# Attrs handled structurally or intentionally excluded from the index.
_SKIPPED_ATTRS = frozenset({"history", "dims", "coords"})

# Suffixes of the per-coordinate envelope columns the flat relation
# emits ({name}_min/{name}_max/{name}_step). An attr shaped like one of
# these is reserved catalog-wide — not just against the ingesting
# patch's own coords — so a flat-relation envelope column (e.g.
# "event_time_min") can never collide with a same-named attr contributed
# by a different patch, which would make one get_contents() column's
# meaning depend on which other patches share the catalog. (units/dtype
# do not become per-coord columns, so a real attr like "data_units"
# stays queryable, matching Patch.update_attrs.)
_ENVELOPE_SUFFIXES = ("min", "max", "step")


def _is_envelope_shaped(name: str) -> bool:
    """True if name looks like a ``{coord}_{min,max,step}`` envelope column."""
    prefix, _, suffix = name.rpartition("_")
    return bool(prefix) and suffix in _ENVELOPE_SUFFIXES


@dataclass(frozen=True)
class TypedValue:
    """A value with its index kind and (canonical) units."""

    kind: str
    value: float | int | str | bool
    units: str | None = None

    def __post_init__(self):
        assert self.kind in KINDS


@dataclass(frozen=True)
class CoordRecord:
    """One patch-coord entry (typed columns split by kind)."""

    coord_name: str
    value_kind: str
    dtype: str
    coord_dims: str
    length: int | None
    units: str | None
    min_num: float | None = None
    max_num: float | None = None
    step_num: float | None = None
    min_ns: int | None = None
    max_ns: int | None = None
    step_ns: int | None = None
    min_str: str | None = None
    max_str: str | None = None
    is_monotonic: bool | None = None
    is_relative: bool | None = None
    coord_hash: str | None = None

    @property
    def def_key(self) -> str:
        """
        Deduplication key for the coord definition.

        The CoordSummary fingerprint when available ("fp:" prefix; exact
        value identity), otherwise a hash of the stored summary fields
        ("sum:" prefix; lossless for the index but too weak for
        value-identity claims). Name and dims are patch-level and
        excluded.
        """
        if self.coord_hash:
            # truncated: 128 bits is ample and key size shows up in the
            # def_key index for archives with mostly-unique time coords
            return f"fp:{self.coord_hash[:32]}"
        fields = (
            self.value_kind,
            self.dtype,
            self.length,
            self.units,
            self.min_num,
            self.max_num,
            self.step_num,
            self.min_ns,
            self.max_ns,
            self.step_ns,
            self.min_str,
            self.max_str,
            self.is_monotonic,
            self.is_relative,
        )
        digest = hashlib.sha256(repr(fields).encode()).hexdigest()[:32]
        return f"sum:{digest}"


@dataclass(frozen=True)
class PatchRecord:
    """One patch: structural fields, typed attrs, coord rows."""

    source_patch_id: str
    dims: str
    shape: str
    n_dims: int
    sample_count_total: int | None
    time_min: int | None
    time_max: int | None
    time_step: int | None
    distance_min: float | None
    distance_max: float | None
    distance_step: float | None
    attrs: dict[str, TypedValue] = field(default_factory=dict)
    coords: tuple[CoordRecord, ...] = ()


@dataclass(frozen=True)
class SourceRecord:
    """One source (scan unit) and the patches it emitted."""

    source_path: str
    source_format: str
    format_version: str
    base_uri: str | None = None
    mtime_ns: int | None = None
    size_bytes: int | None = None
    patches: tuple[PatchRecord, ...] = ()


def sanitize_attr_name(name: str) -> str:
    """Return a lowercase [a-z0-9_] identifier for an attr name."""
    out = _SANITIZE_RE.sub("_", name.lower()).strip("_")
    if not out or out[0].isdigit():
        out = f"a_{out}"
    return out


def attr_column_name(name: str, kind: str) -> str:
    """Return the attrs-table column for an attr name and kind."""
    return f"{sanitize_attr_name(name)}__{kind}"


def _base_unit_info(value, unit_str: str | None = None) -> tuple[float, str]:
    """Return a value's base-unit magnitude and canonical unit string."""
    quant = value if unit_str is None else value * get_quantity(unit_str)
    quant = get_quantity(quant).to_base_units()
    return float(quant.magnitude), str(quant.units)


def _is_missing(value) -> bool:
    """Return True for values that mean 'not present'."""
    if value is None or (isinstance(value, str) and value == ""):
        return True
    try:
        return bool(pd.isnull(value))
    except (TypeError, ValueError):
        return False


def typed_value(value) -> TypedValue | None:
    """
    Classify a python/numpy scalar into a TypedValue, or None to skip.

    Unit-bearing quantities are converted to base SI units; the canonical
    unit string is recorded so queries can convert consistently.
    """
    if _is_missing(value):
        return None
    # containers/arrays are complex attrs: never indexable scalars (and
    # they must not reach the datetime fallback, which accepts arrays).
    if isinstance(value, np.ndarray | list | tuple | set | frozenset | dict | bytes):
        return None
    # bool must precede int (bool is a subclass of int).
    if isinstance(value, bool | np.bool_):
        return TypedValue("bool", bool(value))
    if isinstance(value, np.datetime64):
        return TypedValue("time", to_int(value))
    if isinstance(value, np.timedelta64):
        return TypedValue("dur", to_int(value))
    # pint scalar quantity or unit.
    if hasattr(value, "units"):
        magnitude = getattr(value, "magnitude", 1)
        if isinstance(magnitude, np.ndarray):
            return None  # array quantities are not scalar attrs
        magnitude, base = _base_unit_info(value)
        return TypedValue("num", magnitude, units=base)
    if isinstance(value, int | np.integer | float | np.floating):
        return TypedValue("num", float(value))
    if isinstance(value, str):
        return TypedValue("str", value)
    # datetime/timedelta and anything datetime-like numpy missed.
    try:
        return TypedValue("time", to_int(to_datetime64(value)))
    except Exception:
        pass
    return None  # complex attrs (sequences, dicts, ...) are skipped


def _extract_attrs(summary: PatchSummary) -> dict[str, TypedValue]:
    """Get indexable typed attrs from a patch summary."""
    raw = summary.attrs.model_dump()
    out = {}
    for name, value in raw.items():
        if name in _SKIPPED_ATTRS or name.startswith("_"):
            continue
        # Reserve structural columns and any coordinate-envelope-shaped
        # name (catalog-wide, not just this patch's own coords) so the
        # meaning of a flat-relation column never depends on which other
        # patches share the catalog.
        if sanitize_attr_name(name) in RESERVED_ATTR_COLUMNS or (
            _is_envelope_shaped(name)
        ):
            msg = (
                f"Skipping reserved attr name {name!r}; it collides with a "
                "structural index column. The attr stays on the patch but "
                "is not queryable through the spool."
            )
            warnings.warn(msg, UserWarning)
            continue
        typed = typed_value(value)
        if typed is not None:
            out[name] = typed
    return out


def _coord_record(name: str, summary) -> CoordRecord | None:
    """Convert one CoordSummary into a CoordRecord."""
    fingerprint = getattr(summary, "fingerprint", None)
    if fingerprint is None and getattr(summary, "is_range_like", False):
        # A range summary contains its complete representation, so recover the
        # same exact identity a loaded CoordRange would have produced.
        fingerprint = summary.to_coord().fingerprint()
    # str() on a pint Quantity is comparatively expensive; do it once and
    # reuse it below (a Quantity-keyed cache is unsafe — 1 m == 100 cm with
    # equal hashes but different strings).
    units_str = str(summary.units) if summary.units is not None else None
    common = dict(
        coord_name=name,
        dtype=summary.dtype,
        coord_dims=",".join(summary.dims),
        length=summary.len,
        units=units_str,
        coord_hash=fingerprint,
    )
    dtype = np.dtype(summary.dtype) if summary.dtype else None
    if dtype is None:
        return None  # unsupported coord representation: skip, per design
    if dtype.kind in "mM":  # datetime64 ("M") / timedelta64 ("m")
        is_datetime = dtype.kind == "M"
        convert = to_datetime64 if is_datetime else to_timedelta64
        step = summary.step
        return CoordRecord(
            value_kind="time",
            is_relative=not is_datetime,
            min_ns=to_int(convert(summary.min)),
            max_ns=to_int(convert(summary.max)),
            step_ns=None if pd.isnull(step) else to_int(to_timedelta64(step)),
            **common,
        )
    if np.issubdtype(dtype, np.number):
        min_num = float(summary.min)
        max_num = float(summary.max)
        step = summary.step
        step_num = None if pd.isnull(step) else float(step)
        if units_str is not None:
            min_num, base = _base_unit_info(summary.min, units_str)
            max_num, _ = _base_unit_info(summary.max, units_str)
            if step_num is not None:
                step_end, _ = _base_unit_info(summary.min + summary.step, units_str)
                step_num = step_end - min_num
            common["units"] = base
        return CoordRecord(
            value_kind="num",
            min_num=min_num,
            max_num=max_num,
            step_num=step_num,
            **common,
        )
    if dtype.kind in "US" or dtype == object:
        return CoordRecord(
            value_kind="str",
            min_str=str(summary.min),
            max_str=str(summary.max),
            **common,
        )
    return None  # unsupported coord representation: skip, per design


def _envelope(coords: tuple[CoordRecord, ...], name: str, kind: str):
    """Pull the (min, max, step) envelope for one coord if present."""
    for rec in coords:
        if rec.coord_name != name or rec.value_kind != kind:
            continue
        if kind == "time" and not rec.is_relative:
            return rec.min_ns, rec.max_ns, rec.step_ns
        if kind == "num":
            return rec.min_num, rec.max_num, rec.step_num
    return None, None, None


def patch_record(summary: PatchSummary) -> PatchRecord:
    """Convert one PatchSummary into a PatchRecord."""
    coords = tuple(
        rec
        for name, csum in summary.coords.items()
        if (rec := _coord_record(name, csum)) is not None
    )
    time_min, time_max, time_step = _envelope(coords, "time", "time")
    dist_min, dist_max, dist_step = _envelope(coords, "distance", "num")
    shape = tuple(int(x) for x in summary.shape)
    return PatchRecord(
        source_patch_id=normalize_source_patch_id(summary.source_patch_id),
        dims=",".join(summary.dims),
        shape=",".join(str(x) for x in shape),
        n_dims=len(summary.dims),
        sample_count_total=int(np.prod(shape)) if shape else None,
        time_min=time_min,
        time_max=time_max,
        time_step=time_step,
        distance_min=dist_min,
        distance_max=dist_max,
        distance_step=dist_step,
        attrs=_extract_attrs(summary),
        coords=coords,
    )


def summaries_to_records(
    summaries: list[PatchSummary],
    base_uri: str | None = None,
    relative_to: str | None = None,
    mtimes_ns: dict[str, int] | None = None,
    sizes_bytes: dict[str, int] | None = None,
) -> list[SourceRecord]:
    """
    Group patch summaries by source and convert to SourceRecords.

    Parameters
    ----------
    summaries
        Patch summaries, e.g. from `dc.scan`.
    base_uri
        Optional common root persisted with each source (remote spools);
        source paths are stored relative to it.
    relative_to
        Optional local spool root: source paths are stored relative to it
        but the root itself is *not* persisted (local directory spools
        resolve against their current root, per the design doc).
    mtimes_ns, sizes_bytes
        Optional maps of source_path -> stat values. When omitted the
        caller is responsible for change detection.
    """
    # Group by the original (OS-native) source path so the mtimes_ns /
    # sizes_bytes maps, which the caller keys by that same path, still
    # resolve. Index paths themselves are stored as POSIX so comparison
    # and deletion are separator-agnostic across platforms.
    by_source: dict[str, list[PatchSummary]] = {}
    for summary in summaries:
        by_source.setdefault(str(summary.source_path), []).append(summary)
    root = base_uri or relative_to
    root_posix = str(root).replace("\\", "/") if root else None
    root_prefix = root_posix.rstrip("/") if root_posix else None
    out = []
    for path, group in by_source.items():
        first = group[0]
        patches = []
        for num, summary in enumerate(group):
            record = patch_record(summary)
            if record.source_patch_id == "" and len(group) > 1:
                # positional identity within the source, per design doc
                record = replace(record, source_patch_id=str(num))
            patches.append(record)
        posix_path = path.replace("\\", "/")
        store_path = posix_path
        if root_prefix is not None and (
            posix_path == root_prefix or posix_path.startswith(f"{root_prefix}/")
        ):
            # "." (not "") marks a source that IS the root (directory units)
            store_path = posix_path[len(root_prefix) :].lstrip("/") or "."
        out.append(
            SourceRecord(
                source_path=store_path,
                base_uri=base_uri,
                source_format=first.source_format,
                format_version=first.source_version,
                mtime_ns=(mtimes_ns or {}).get(path),
                size_bytes=(sizes_bytes or {}).get(path),
                patches=tuple(patches),
            )
        )
    return out


def _py_scalar(value):
    """Convert a fetched cell to the plain python scalar records use."""
    if value is None or pd.isnull(value):
        return None
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.integer | int):
        return int(value)
    if isinstance(value, np.floating | float):
        return float(value)
    return value


def assemble_source_records(
    sources: pd.DataFrame,
    patches: pd.DataFrame,
    attrs: pd.DataFrame,
    links: pd.DataFrame,
    defs: pd.DataFrame,
    meta: pd.DataFrame,
) -> list[SourceRecord]:
    """
    Assemble source records from already-fetched index frames.

    This is the transfer format for merging catalogs: feeding the result
    to another backend's `write_sources` re-ingests the metadata with
    fresh ids, coord-def deduplication (def keys are preserved), and
    replace-semantics on (base_uri, source_path) identity. The caller
    (an index backend's export_records) is responsible for narrowing the
    frames — filtering by patch id belongs in SQL, not here.
    """
    if sources.empty:
        return []
    col_info = {
        row.column_name: (row.attr_name, row.value_kind, _py_scalar(row.units))
        for row in meta.itertuples()
    }
    def_map = {int(row.coord_def_id): row for row in defs.itertuples()}
    attr_rows = (
        {int(k): v for k, v in attrs.set_index("patch_id").to_dict("index").items()}
        if not attrs.empty
        else {}
    )
    link_groups = (
        {int(k): v for k, v in links.groupby("patch_id")} if not links.empty else {}
    )
    patches_by_source = (
        {int(k): v for k, v in patches.groupby("source_id")}
        if not patches.empty
        else {}
    )
    out = []
    for src in sources.itertuples():
        sub = patches_by_source.get(int(src.source_id))
        if sub is None:
            continue
        patch_records = []
        for patch in sub.itertuples():
            pid = int(patch.patch_id)
            typed = {}
            for col, value in attr_rows.get(pid, {}).items():
                if col in col_info and not pd.isnull(value):
                    name, kind, units = col_info[col]
                    typed[name] = TypedValue(
                        kind=kind, value=_py_scalar(value), units=units
                    )
            coords = []
            for link in link_groups.get(pid, pd.DataFrame()).itertuples():
                cdef = def_map[int(link.coord_def_id)]
                coords.append(
                    CoordRecord(
                        coord_name=link.coord_name,
                        coord_dims=link.coord_dims,
                        value_kind=cdef.value_kind,
                        dtype=_py_scalar(cdef.dtype),
                        length=_py_scalar(cdef.length),
                        units=_py_scalar(cdef.units),
                        min_num=_py_scalar(cdef.min_num),
                        max_num=_py_scalar(cdef.max_num),
                        step_num=_py_scalar(cdef.step_num),
                        min_ns=_py_scalar(cdef.min_ns),
                        max_ns=_py_scalar(cdef.max_ns),
                        step_ns=_py_scalar(cdef.step_ns),
                        min_str=_py_scalar(cdef.min_str),
                        max_str=_py_scalar(cdef.max_str),
                        is_monotonic=_py_scalar(cdef.is_monotonic),
                        is_relative=_py_scalar(cdef.is_relative),
                        coord_hash=_py_scalar(cdef.fingerprint),
                    )
                )
            patch_records.append(
                PatchRecord(
                    source_patch_id=normalize_source_patch_id(patch.source_patch_id),
                    dims=_py_scalar(patch.dims) or "",
                    shape=_py_scalar(patch.shape) or "",
                    n_dims=_py_scalar(patch.n_dims),
                    sample_count_total=_py_scalar(patch.sample_count_total),
                    time_min=_py_scalar(patch.time_min),
                    time_max=_py_scalar(patch.time_max),
                    time_step=_py_scalar(patch.time_step),
                    distance_min=_py_scalar(patch.distance_min),
                    distance_max=_py_scalar(patch.distance_max),
                    distance_step=_py_scalar(patch.distance_step),
                    attrs=typed,
                    coords=tuple(coords),
                )
            )
        out.append(
            SourceRecord(
                source_path=_py_scalar(src.source_path) or "",
                base_uri=_py_scalar(src.base_uri) or None,
                source_format=_py_scalar(src.source_format) or "",
                format_version=_py_scalar(src.format_version) or "",
                mtime_ns=_py_scalar(src.mtime_ns),
                size_bytes=_py_scalar(src.size_bytes),
                patches=tuple(patch_records),
            )
        )
    return out
