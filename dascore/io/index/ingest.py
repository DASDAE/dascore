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
from functools import cache

import numpy as np
import pandas as pd

from dascore.core.summary import PatchSummary
from dascore.io.index.schema import KINDS, RESERVED_ATTR_COLUMNS
from dascore.units import get_quantity
from dascore.utils.time import to_datetime64, to_int, to_timedelta64

_SANITIZE_RE = re.compile(r"[^a-z0-9_]+")

# Attrs handled structurally or intentionally excluded from the index.
_SKIPPED_ATTRS = frozenset({"history", "dims", "coords"})


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


@cache
def _base_unit_info(unit_str: str) -> tuple[float, str]:
    """Return (scale factor to SI base, canonical base unit string)."""
    quant = get_quantity(unit_str).to_base_units()
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
        factor, base = _base_unit_info(str(value.units))
        return TypedValue("num", float(magnitude) * factor, units=base)
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
        if sanitize_attr_name(name) in RESERVED_ATTR_COLUMNS:
            warnings.warn(f"Skipping reserved attr name {name!r}.", UserWarning)
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
    common = dict(
        coord_name=name,
        dtype=summary.dtype,
        coord_dims=",".join(summary.dims),
        length=summary.len,
        units=str(summary.units) if summary.units is not None else None,
        coord_hash=fingerprint,
    )
    dtype = np.dtype(summary.dtype) if summary.dtype else None
    if dtype is not None and np.issubdtype(dtype, np.datetime64):
        step = summary.step
        return CoordRecord(
            value_kind="time",
            is_relative=False,
            min_ns=to_int(to_datetime64(summary.min)),
            max_ns=to_int(to_datetime64(summary.max)),
            step_ns=None if pd.isnull(step) else to_int(to_timedelta64(step)),
            **common,
        )
    if dtype is not None and np.issubdtype(dtype, np.timedelta64):
        step = summary.step
        return CoordRecord(
            value_kind="time",
            is_relative=True,
            min_ns=to_int(to_timedelta64(summary.min)),
            max_ns=to_int(to_timedelta64(summary.max)),
            step_ns=None if pd.isnull(step) else to_int(to_timedelta64(step)),
            **common,
        )
    if dtype is not None and np.issubdtype(dtype, np.number):
        factor = 1.0
        if summary.units is not None:
            factor, base = _base_unit_info(str(summary.units))
            common["units"] = base
        step = summary.step
        return CoordRecord(
            value_kind="num",
            min_num=float(summary.min) * factor,
            max_num=float(summary.max) * factor,
            step_num=None if pd.isnull(step) else float(step) * factor,
            **common,
        )
    if dtype is not None and (dtype.kind in "US" or dtype == object):
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
        source_patch_id=summary.source_patch_id or "",
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
    by_source: dict[str, list[PatchSummary]] = {}
    for summary in summaries:
        by_source.setdefault(str(summary.source_path), []).append(summary)
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
        store_path = path
        root = base_uri or relative_to
        if root and path.startswith(root):
            # "." (not "") when the source IS the root (directory units)
            store_path = path[len(root) :].lstrip("/") or "."
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
