"""
Convert patch summaries into normalized index records.

This module is backend-independent: it turns `PatchSummary` objects into
plain records (dicts/dataclasses) using only the four primitive storage
types. Coordinate envelopes are stored in each coordinate's ORIGINAL
units beside the original unit string — never converted — so the index
shows what the patch shows and a bare numeric query means native units.
Unit-bearing attr values are still normalized to pint base SI (one
canonical unit per attr column); unit-bearing queries convert themselves
per stored coordinate unit at query time.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any, SupportsInt, TypedDict, cast

import numpy as np
import pandas as pd

from dascore.core.coords import CoordSummary
from dascore.core.summary import PatchSummary, normalize_source_patch_key
from dascore.exceptions import InvalidInventoryError
from dascore.io.index.schema import (
    KINDS,
    RESERVED_ATTR_COLUMNS,
    CoordDefRow,
    PatchCoordRow,
    PatchRow,
    SourceRow,
)
from dascore.units import get_quantity, get_quantity_str
from dascore.utils.misc import validate_acquisition_key
from dascore.utils.paths import parse_hive_path_attrs
from dascore.utils.pd import iter_rows
from dascore.utils.time import to_datetime64, to_int, to_timedelta64

_SANITIZE_RE = re.compile(r"[^a-z0-9_]+")

# Attrs handled structurally or intentionally excluded from the index.
# The ids are not indexed until the schema version which adds columns for
# them; an index is for finding data, and an id is not a search term.
# Attrs which are structure rather than metadata, and are never indexed.
# `patch_id` is not here: it is an ordinary string, one per patch, and
# indexing it is what lets a spool find a patch by the id it carries
# rather than by loading every patch to look.
#
# `processing_id` is. It advances on every operation, and a spool applies
# operations as it loads -- a residual trim is a real `select` on the
# patch -- so what the index recorded is not what the patch which comes
# back carries. `patch_id` survives those same operations by definition,
# which is what makes it, and not this, the one worth indexing.
_SKIPPED_ATTRS = frozenset({"history", "dims", "coords", "processing_id"})


@dataclass(frozen=True)
class TypedValue:
    """A value with its index kind and (canonical) units."""

    kind: str
    value: float | int | str | bool
    units: str | None = None

    def __post_init__(self):
        assert self.kind in KINDS


class _CommonCoordFields(TypedDict):
    """The CoordRecord fields every value kind carries."""

    coord_name: str
    dtype: str
    coord_dims: str
    length: int | None
    units: str | None
    coord_hash: str | None


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
        excluded. The unit spelling rides after "|" on the fp form:
        fingerprints simplify units before hashing, so one physical
        coordinate spelled in metres and in feet hashes identically —
        but the stored def rows carry spelling-dependent units and
        envelopes, so deduplicating them together would let the first
        spelling's row lie for the second.
        """
        if self.coord_hash:
            # truncated: 128 bits is ample and key size shows up in the
            # def_key index for archives with mostly-unique time coords
            key = f"fp:{self.coord_hash[:32]}"
            return f"{key}|{self.units}" if self.units else key
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
            self.is_relative,
        )
        digest = hashlib.sha256(repr(fields).encode()).hexdigest()[:32]
        return f"sum:{digest}"


@dataclass(frozen=True)
class PatchRecord:
    """One patch: structural fields, typed attrs, coord rows."""

    source_patch_key: str
    dims: str
    dtype: str
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
    path_attrs: dict[str, str] | None = None
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


def _base_unit_info(value) -> tuple[float, str]:
    """Return an attr quantity's base-unit magnitude and unit string.

    Attrs only: coordinate envelopes store original units untouched.
    """
    quant = get_quantity(value)
    assert quant is not None  # unit-bearing values only
    quant = quant.to_base_units()
    return float(quant.magnitude), str(quant.units)


def _to_ns(value) -> int:
    """
    Return a time or duration as a plain int of nanoseconds.

    `to_int` hands back a numpy scalar. Records hold plain python
    scalars, and CoordRecord.def_key hashes their reprs, so a numpy
    scalar here would give a coord one summary key on a fresh scan and
    another when the same coord is read back out of an index.
    """
    return int(to_int(value))


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
        return TypedValue("time", _to_ns(value))
    if isinstance(value, np.timedelta64):
        return TypedValue("dur", _to_ns(value))
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
        return TypedValue("time", _to_ns(to_datetime64(value)))
    except Exception:
        pass
    return None  # complex attrs (sequences, dicts, ...) are skipped


def _attr_name_status(name: str) -> str:
    """
    Classify an attr name for indexing: "ok", "skip", or "reserved".

    Structural/underscore names skip silently; names that sanitize to a
    structural storage/contract column are reserved (callers warn).
    Coordinate-envelope-shaped names (e.g. "channel_step") are ordinary
    attrs; a genuine collision with a coord envelope column is handled
    (warn, coord wins) when the flat view is built.
    """
    if name in _SKIPPED_ATTRS or name.startswith("_"):
        return "skip"
    if sanitize_attr_name(name) in RESERVED_ATTR_COLUMNS:
        return "reserved"
    return "ok"


def _extract_attrs(summary: PatchSummary) -> dict[str, TypedValue]:
    """Get indexable typed attrs from a patch summary."""
    raw = summary.attrs.model_dump()
    out = {}
    for name, value in raw.items():
        status = _attr_name_status(name)
        if status == "reserved":
            msg = (
                f"Skipping reserved attr name {name!r}; it collides with a "
                "structural index column. The attr stays on the patch but "
                "is not queryable through the spool."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
        if status != "ok":
            continue
        typed = typed_value(value)
        if typed is not None:
            out[name] = typed
    return out


# What a directory name may never claim to be. A path says where data is
# kept, which is how renaming a directory corrects metadata; it does not
# get to say which data it is, or a directory called `patch_id=x` would
# rewrite the lineage of everything under it.
_UNCLAIMABLE_BY_PATH = frozenset({"patch_id"})


def hive_path_attrs(rel_posix: str, warn: bool = True) -> dict[str, str]:
    """
    Return the indexable hive-style path attrs for a stored relative path.

    Applies the same name rules as file attrs: structural/underscore
    names are dropped silently, reserved column collisions warn and drop.
    A name only the data itself may state is dropped with a warning too.
    """
    out = {}
    for name, value in parse_hive_path_attrs(rel_posix).items():
        if name in _UNCLAIMABLE_BY_PATH:
            if warn:
                msg = (
                    f"Ignoring hive-style path key {name!r} in {rel_posix!r}; "
                    "a path says where data is kept, not which data it is."
                )
                warnings.warn(msg, UserWarning, stacklevel=2)
            continue
        status = _attr_name_status(name)
        if status == "reserved" and warn:
            msg = (
                f"Skipping hive-style path key {name!r} in {rel_posix!r}; "
                "it collides with a structural index column."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
        if status != "ok":
            continue
        if name == "acquisition_key":
            # The patch validates this on the way in, so a path which
            # cannot produce a legal id would index and then fail at every
            # load; refuse it here, where the fix is to rename a directory.
            try:
                validate_acquisition_key(value)
            except InvalidInventoryError as error:
                if warn:
                    msg = (
                        f"Skipping hive-style path key {name!r} in "
                        f"{rel_posix!r}: {error}"
                    )
                    warnings.warn(msg, UserWarning, stacklevel=2)
                continue
        out[name] = value
    return out


def _states_something_else(existing: TypedValue, path_value: str) -> bool:
    """
    Return True when a path segment changes what an attr says.

    The comparison is on meaning rather than spelling: a directory named
    `gauge_length=10` over a file stating `10.0` restates it, and warning
    there would send someone looking for a disagreement the layout does
    not contain. A path value which cannot be read as the stored kind at
    all does change what the attr says, whatever it then says.
    """
    if existing.kind == "num":
        try:
            return float(path_value) != float(existing.value)
        except ValueError:
            return True
    if existing.kind == "bool":
        return path_value.strip().lower() != str(existing.value).lower()
    if existing.kind in {"time", "dur"}:
        # Stored as integer nanoseconds, which a path segment never spells
        # that way; read the segment as an instant before comparing.
        convert = to_datetime64 if existing.kind == "time" else to_timedelta64
        try:
            stated = typed_value(convert(path_value))
        except (ValueError, TypeError):
            return True
        return stated is None or stated.value != existing.value
    return str(existing.value) != path_value


def hive_typed_attrs(path_attrs: dict[str, str]) -> dict[str, TypedValue]:
    """Wrap hive path attrs as string TypedValues (no type inference)."""
    return {name: TypedValue("str", value) for name, value in path_attrs.items()}


def dump_path_attrs(path_attrs: dict[str, str] | None) -> str | None:
    """Serialize a path-attrs dict for the sources table (None when empty)."""
    return json.dumps(path_attrs, sort_keys=True) if path_attrs else None


def coord_summary(row: Mapping) -> CoordSummary | None:
    """
    Rebuild a coordinate summary from one stored coordinate row.

    The inverse of `_coord_record`: a row of `patch_coords` joined to its
    `coord_defs` definition (see `SQLiteIndexBackend.coord_frame`) states
    everything a summary carries, so a member's coordinate can be
    described without loading the patch it belongs to.

    Returns None for a row whose value kind the index does not represent.

    Built without re-validating: every value is converted here to the type
    a validator would coerce it to, and this runs per coordinate per member
    while a plan is made, where validation measured ~28us a call.
    """
    kind = row.get("value_kind")
    units = row.get("units")
    units = None if units is None or pd.isnull(units) else units
    fingerprint = row.get("fingerprint")
    fingerprint = None if pd.isnull(fingerprint) else str(fingerprint)
    length = row.get("length")
    length = None if length is None or pd.isnull(length) else int(length)
    dims = str(row.get("coord_dims") or "")
    common: dict[str, Any] = dict(
        dtype=str(row.get("dtype") or "").split("[")[0],
        units=units,
        dims=tuple(x for x in dims.split(",") if x),
        len=length,
        fingerprint=fingerprint,
    )
    if kind == "time":
        # stored as integer nanoseconds; a relative coord is a duration
        stamp = _ns_timedelta if row.get("is_relative") else _ns_datetime
        step = row.get("step_ns")
        return CoordSummary.model_construct(
            min=stamp(row.get("min_ns")),
            max=stamp(row.get("max_ns")),
            step=None if pd.isnull(step) else _ns_timedelta(step),
            **common,
        )
    if kind == "num":
        # an envelope nobody stated is NaN, as validation would make it;
        # only a step is genuinely absent
        low, high = _opt_float(row.get("min_num")), _opt_float(row.get("max_num"))
        return CoordSummary.model_construct(
            min=np.nan if low is None else low,
            max=np.nan if high is None else high,
            step=_opt_float(row.get("step_num")),
            **common,
        )
    if kind == "str":
        low, high = row.get("min_str"), row.get("max_str")
        return CoordSummary.model_construct(
            min=None if low is None else str(low),
            max=None if high is None else str(high),
            **common,
        )
    return None


def _opt_float(value) -> float | None:
    """A stored number, or None where the row states none."""
    return None if value is None or pd.isnull(value) else float(value)


def _ns_datetime(value) -> np.datetime64:
    """A stored nanosecond count as a datetime, NaT when the row states none."""
    if value is None or pd.isnull(value):
        return np.datetime64("NaT", "ns")
    return np.datetime64(int(value), "ns")


def _ns_timedelta(value) -> np.timedelta64:
    """A stored nanosecond count as a duration, NaT when the row states none."""
    if value is None or pd.isnull(value):
        return np.timedelta64("NaT", "ns")
    return np.timedelta64(int(value), "ns")


def _coord_record(name: str, summary) -> CoordRecord | None:
    """Convert one CoordSummary into a CoordRecord."""
    fingerprint = getattr(summary, "fingerprint", None)
    if fingerprint is None and getattr(summary, "is_range_like", False):
        # A range summary contains its complete representation, so recover the
        # same exact identity a loaded CoordRange would have produced.
        fingerprint = summary.to_coord().fingerprint()
    # Stringify the pint Quantity once (a Quantity-keyed cache is unsafe —
    # 1 m == 100 cm with equal hashes but different strings). This is the
    # ORIGINAL unit, cleanly spelled: get_quantity_str drops the "1 " a
    # magnitude-one Quantity would print while a scaled unit keeps its
    # factor, so the string still means exactly what the patch stated.
    units_str = get_quantity_str(summary.units) if summary.units is not None else None
    common = _CommonCoordFields(
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
            min_ns=_to_ns(convert(summary.min)),
            max_ns=_to_ns(convert(summary.max)),
            step_ns=None if pd.isnull(step) else _to_ns(to_timedelta64(step)),
            **common,
        )
    if np.issubdtype(dtype, np.number):
        # Envelopes are stored in the coordinate's ORIGINAL units, never
        # converted, beside the original unit string — what the patch
        # shows is what the index shows, and a bare query means these
        # native magnitudes. Only unit-bearing *queries* convert (per
        # stored unit, in the query layer).
        step = summary.step
        return CoordRecord(
            value_kind="num",
            min_num=float(summary.min),
            max_num=float(summary.max),
            step_num=None if pd.isnull(step) else float(step),
            **common,
        )
    if dtype.kind in "USO":
        # a summary which states no labels has none to store; stringifying
        # the missing value would write the label "nan"
        return CoordRecord(
            value_kind="str",
            min_str=None if pd.isnull(summary.min) else str(summary.min),
            max_str=None if pd.isnull(summary.max) else str(summary.max),
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
    return PatchRecord(
        source_patch_key=normalize_source_patch_key(summary.source_patch_key),
        dims=",".join(summary.dims),
        dtype=str(summary.dtype or ""),
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
    # Names a directory renamed out from under the file which states them,
    # with one path each: the layout is the correction mechanism, so this
    # is legal, but a whole archive silently disagreeing with its own
    # headers is worth hearing about once. Only sources being scanned are
    # compared -- a detected move rewrites the stored attrs without
    # reopening the file, so what the file itself says is not on hand.
    overridden: dict[str, str] = {}
    for path, group in by_source.items():
        first = group[0]
        patches = []
        for num, summary in enumerate(group):
            record = patch_record(summary)
            if record.source_patch_key == "" and len(group) > 1:
                # positional identity within the source, per design doc
                record = replace(record, source_patch_key=str(num))
            patches.append(record)
        posix_path = path.replace("\\", "/")
        store_path = posix_path
        path_attrs = None
        if root_prefix is not None and (
            posix_path == root_prefix or posix_path.startswith(f"{root_prefix}/")
        ):
            # "." (not "") marks a source that IS the root (directory units)
            store_path = posix_path[len(root_prefix) :].lstrip("/") or "."
            # Hive-style key=value directory segments become string attrs
            # that override same-named file attrs (renaming a directory is
            # the user's way of correcting metadata). The source's own
            # name never contributes; see parse_hive_path_attrs.
            path_attrs = hive_path_attrs(store_path) or None
            if path_attrs:
                typed = hive_typed_attrs(path_attrs)
                for patch in patches:
                    for name in typed.keys() & patch.attrs.keys():
                        if _states_something_else(patch.attrs[name], path_attrs[name]):
                            overridden.setdefault(name, store_path)
                patches = [replace(p, attrs={**p.attrs, **typed}) for p in patches]
        out.append(
            SourceRecord(
                source_path=store_path,
                base_uri=base_uri,
                source_format=first.source_format,
                format_version=first.source_version,
                mtime_ns=(mtimes_ns or {}).get(path),
                size_bytes=(sizes_bytes or {}).get(path),
                path_attrs=path_attrs,
                patches=tuple(patches),
            )
        )
    if overridden:
        shown = ", ".join(f"{k!r} (e.g. {v!r})" for k, v in sorted(overridden.items()))
        msg = (
            f"Hive-style path segments override attrs the files themselves "
            f"state: {shown}. The path wins, which is how a directory rename "
            "corrects metadata; check the layout if that was not intended."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
    return out


# Record fields read straight off an index row of the same name. The
# remaining fields need per-field handling: a coord's name/dims are
# patch-level (they come from the link row, not the shared definition),
# its hash is stored as "fingerprint", and a patch's id/dims get
# normalized below.
_COORD_DEF_FIELDS = tuple(
    f.name
    for f in fields(CoordRecord)
    if f.name not in ("coord_name", "coord_dims", "coord_hash")
)
_PATCH_ROW_FIELDS = tuple(
    f.name
    for f in fields(PatchRecord)
    if f.name not in ("source_patch_key", "dims", "dtype", "attrs", "coords")
)
# Backends with no boolean type hand these columns back as 0/1, which
# def_key would hash differently than the bool a scan produced.
_COORD_DEF_BOOLS = frozenset(
    f.name for f in fields(CoordRecord) if str(f.type).startswith("bool")
)


def _int_key(key: Hashable) -> int:
    """Return a groupby key as an int; the id columns grouped on are integral."""
    return int(cast("SupportsInt", key))


def _py_scalar(value, as_bool: bool = False):
    """Convert a fetched cell to the plain python scalar records use."""
    if value is None or pd.isnull(value):
        return None
    if as_bool or isinstance(value, np.bool_ | bool):
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
    # Records transfer in catalog order: re-ingesting assigns fresh
    # sequential ordinals, so record order IS the ordering contract.
    if "ordinal" in sources.columns:
        sources = sources.sort_values(["ordinal", "source_id"])
    if "patch_id" in patches.columns:
        patches = patches.sort_values("patch_id")
    col_info = {
        column: (name, kind, _py_scalar(units))
        for column, name, kind, units in zip(
            meta["column_name"],
            meta["attr_name"],
            meta["value_kind"],
            meta["units"],
            strict=True,
        )
    }
    def_map = {int(row.coord_def_id): row for row in iter_rows(defs, CoordDefRow)}
    attr_rows = (
        {int(k): v for k, v in attrs.set_index("patch_id").to_dict("index").items()}
        if not attrs.empty
        else {}
    )
    link_groups = (
        {_int_key(k): v for k, v in links.groupby("patch_id")}
        if not links.empty
        else {}
    )
    patches_by_source = (
        {_int_key(k): v for k, v in patches.groupby("source_id")}
        if not patches.empty
        else {}
    )
    out = []
    for src in iter_rows(sources, SourceRow):
        sub = patches_by_source.get(int(src.source_id))
        if sub is None:
            continue
        patch_records = []
        for patch in iter_rows(sub, PatchRow):
            pid = int(patch.patch_id)
            typed = {}
            for col, value in attr_rows.get(pid, {}).items():
                if col in col_info and not pd.isnull(value):
                    name, kind, units = col_info[col]
                    typed[name] = TypedValue(
                        kind=kind, value=_py_scalar(value), units=units
                    )
            coords = []
            for link in iter_rows(link_groups.get(pid, pd.DataFrame()), PatchCoordRow):
                cdef = def_map[int(link.coord_def_id)]
                coords.append(
                    CoordRecord(
                        coord_name=link.coord_name,
                        coord_dims=link.coord_dims,
                        coord_hash=_py_scalar(cdef.fingerprint),
                        **{
                            f: _py_scalar(getattr(cdef, f), f in _COORD_DEF_BOOLS)
                            for f in _COORD_DEF_FIELDS
                        },
                    )
                )
            patch_records.append(
                PatchRecord(
                    source_patch_key=normalize_source_patch_key(patch.source_patch_key),
                    dims=_py_scalar(patch.dims) or "",
                    dtype=_py_scalar(patch.dtype) or "",
                    attrs=typed,
                    coords=tuple(coords),
                    **{f: _py_scalar(getattr(patch, f)) for f in _PATCH_ROW_FIELDS},
                )
            )
        raw_path_attrs = _py_scalar(getattr(src, "path_attrs", None))
        out.append(
            SourceRecord(
                source_path=_py_scalar(src.source_path) or "",
                base_uri=_py_scalar(src.base_uri) or None,
                source_format=_py_scalar(src.source_format) or "",
                format_version=_py_scalar(src.format_version) or "",
                mtime_ns=_py_scalar(src.mtime_ns),
                size_bytes=_py_scalar(src.size_bytes),
                path_attrs=json.loads(raw_path_attrs) if raw_path_attrs else None,
                patches=tuple(patch_records),
            )
        )
    return out
