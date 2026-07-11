"""
Query model and SQL generation for the spool index.

Implements the selector semantics spec (see
`.scratch/selector_semantics_spec.md`): the index only produces
candidates — predicates the summary cannot evaluate exactly are the
caller's responsibility at patch-load time. Predicates SQLite cannot evaluate
exactly are applied as pandas residual filters.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dascore.exceptions import InvalidSpoolQueryError, ParameterError, UnitError
from dascore.io.index.dialect import BaseDialect
from dascore.io.index.ingest import typed_value
from dascore.units import convert_units
from dascore.utils.misc import sanitize_range_param

_GLOB_CHARS = frozenset("*?[")
_UNSET = object()


@dataclass(frozen=True)
class Query:
    """
    A resolved spool query.

    Name resolution (bare kwargs -> attrs first, then coords) happens
    above this layer; a Query already knows which namespace each
    predicate belongs to.
    """

    attrs: dict = field(default_factory=dict)
    coords: dict = field(default_factory=dict)


def _is_collection(value) -> bool:
    """True for non-string collections (membership predicates)."""
    if isinstance(value, str | bytes):
        return False
    if isinstance(value, np.ndarray):
        return True
    return isinstance(value, list | tuple | set | frozenset)


def _is_range(value) -> bool:
    """True for a 2-tuple range (possibly with open bounds)."""
    return isinstance(value, tuple) and len(value) == 2


def normalize_range_forms(value):
    """
    Normalize the patch-level slice range form to a 2-tuple.

    Only slices are converted: bare None/Ellipsis keep their own errors,
    and a fully-open range is rejected downstream as having no usable
    bounds (per the selector spec).
    """
    if isinstance(value, slice):
        return sanitize_range_param(value)
    return value


def _coerce_scalar(value, target_kinds: set[str]):
    """
    Coerce a query scalar to (kind, storable value).

    Follows the coercion table in the selector spec; datetime-like
    strings become time queries only when the target has a time kind.
    """
    typed = typed_value(value)
    if typed is None:
        msg = f"Cannot use {value!r} as a spool query value."
        raise InvalidSpoolQueryError(msg)
    if typed.kind == "str" and "time" in target_kinds:
        try:
            retyped = typed_value(np.datetime64(pd.Timestamp(value), "ns"))
            return retyped
        except (ValueError, TypeError):
            pass
    return typed


def _normalize_unit(value) -> str | None:
    """Return a nullable unit string from a dataframe value."""
    return None if value is None or pd.isnull(value) else str(value)


def _to_target_unit(typed, target_units: str | None, name: str):
    """Validate/convert a numeric query value for one stored unit."""
    if typed.kind != "num" or typed.units is None:
        return typed.value
    if target_units is None:
        msg = f"Cannot query unitless {name!r} with units {typed.units!r}."
        raise UnitError(msg)
    return convert_units(typed.value, to_units=target_units, from_units=typed.units)


def _range_bounds(
    value,
    target_kinds: set[str],
    target_units: str | None | object = _UNSET,
    name: str = "value",
):
    """
    Return (kind, lo, hi, typed_values) from a range tuple.

    Open bounds (None/Ellipsis) are skipped; typed_values carries the
    coerced usable bounds so callers don't coerce twice.
    """
    lo_raw, hi_raw = value
    lo = hi = None
    kind = None
    typed_values = []
    for raw, side in ((lo_raw, "lo"), (hi_raw, "hi")):
        if raw is None or raw is Ellipsis:
            continue
        typed = _coerce_scalar(raw, target_kinds)
        typed_values.append(typed)
        knd = typed.kind
        val = (
            typed.value
            if target_units is _UNSET
            else _to_target_unit(typed, target_units, name)
        )
        if kind is not None and knd != kind:
            msg = f"Range bounds {value!r} have mixed kinds ({kind}, {knd})."
            raise InvalidSpoolQueryError(msg)
        kind = knd
        if side == "lo":
            lo = val
        else:
            hi = val
    if kind is None:
        msg = f"Range {value!r} has no usable bounds."
        raise InvalidSpoolQueryError(msg)
    if lo is not None and hi is not None and lo > hi:
        msg = f"Range {value!r} has lo > hi after coercion."
        raise InvalidSpoolQueryError(msg)
    return kind, lo, hi, typed_values


def _compatible_coord_units(
    rows: pd.DataFrame, typed_values: list, name: str
) -> set[str] | None:
    """
    Return stored units compatible with quantity-valued coord selectors.

    None means the query carries no units (no unit constraint at all); a
    set constrains matching to those units plus NULL-unit definitions
    (which can never be proven incompatible, so they stay candidates).
    Raises UnitError only when every stored definition has units and none
    are compatible.
    """
    query_units = {
        x.units for x in typed_values if x is not None and x.units is not None
    }
    if not query_units:
        return None
    first = next(iter(query_units))
    for other in query_units - {first}:
        convert_units(1.0, to_units=first, from_units=other)
    stored = {_normalize_unit(x) for x in rows.get("units", ())}
    compatible = set()
    for unit in stored - {None}:
        try:
            convert_units(1.0, to_units=unit, from_units=first)
        except UnitError:
            continue
        compatible.add(unit)
    if not compatible and None not in stored:
        raise UnitError(f"Coordinate {name!r} has no units compatible with {first!r}.")
    return compatible


@dataclass
class _Where:
    """Accumulates WHERE clauses and parameters."""

    clauses: list[str] = field(default_factory=list)
    params: list = field(default_factory=list)

    def add(self, clause: str, *params):
        self.clauses.append(clause)
        self.params.extend(params)

    @property
    def sql(self) -> str:
        return " AND ".join(self.clauses) if self.clauses else "TRUE"


def build_attr_clause(
    where: _Where,
    dialect: BaseDialect,
    attr_meta: pd.DataFrame,
    name: str,
    value,
) -> re.Pattern | None:
    """
    Add SQL for one attr predicate; return a residual pattern if the
    predicate must be re-applied in pandas (regex).
    """
    rows = attr_meta[attr_meta["attr_name"] == name]
    if rows.empty:
        msg = f"{name!r} is not an attribute of any patch in this spool."
        raise InvalidSpoolQueryError(msg)
    kinds = set(rows["value_kind"])
    columns = dict(zip(rows["value_kind"], rows["column_name"]))
    units = {row.value_kind: _normalize_unit(row.units) for row in rows.itertuples()}

    def col(kind):
        return f"a.{dialect.quote(columns[kind])}"

    if isinstance(value, re.Pattern):
        # Regex is a residual filter; SQL only requires the attr be
        # present (str kind) so candidates are a superset.
        if "str" not in kinds:
            where.add("FALSE")
            return None
        where.add(f"{col('str')} IS NOT NULL")
        return value
    if _is_range(value):
        # Attr metadata has one canonical unit per typed column.
        probe = next(
            (
                _coerce_scalar(x, kinds)
                for x in value
                if x is not None and x is not Ellipsis
            ),
            None,
        )
        target_units = units.get(probe.kind) if probe is not None else None
        kind, lo, hi, _ = _range_bounds(value, kinds, target_units, name)
        if kind not in kinds:
            where.add("FALSE")
            return None
        if lo is not None:
            where.add(f"{col(kind)} >= ?", lo)
        if hi is not None:
            where.add(f"{col(kind)} <= ?", hi)
        return None
    if _is_collection(value):
        coerced = [_coerce_scalar(v, kinds) for v in value]
        by_kind: dict[str, list] = {}
        for typed in coerced:
            val = _to_target_unit(typed, units.get(typed.kind), name)
            by_kind.setdefault(typed.kind, []).append(val)
        subclauses = []
        params = []
        for kind, vals in by_kind.items():
            if kind not in kinds:
                continue
            marks = ", ".join("?" for _ in vals)
            subclauses.append(f"{col(kind)} IN ({marks})")
            params.extend(vals)
        if not subclauses:
            where.add("FALSE")
        else:
            where.add("(" + " OR ".join(subclauses) + ")", *params)
        return None
    if isinstance(value, str) and _GLOB_CHARS & set(value):
        if "str" not in kinds:
            where.add("FALSE")
            return None
        where.add(dialect.glob(col("str")), value)
        return None
    typed = _coerce_scalar(value, kinds)
    kind = typed.kind
    val = _to_target_unit(typed, units.get(kind), name)
    if kind not in kinds:
        where.add("FALSE")
        return None
    where.add(f"{col(kind)} = ?", val)
    return None


def build_coord_clause(
    where: _Where,
    dialect: BaseDialect,
    coord_meta: pd.DataFrame,
    name: str,
    value,
) -> None:
    """
    Add an EXISTS clause over patch_coords/coord_defs for one coord
    predicate.

    Candidacy only: envelope overlap, never false negatives. Exact
    membership/boolean masks are applied at patch load, above this layer.
    """
    rows = coord_meta[coord_meta["coord_name"] == name]
    if isinstance(value, tuple) and len(value) != 2:
        msg = f"Coordinate range for {name!r} must be a length 2 sequence."
        raise ParameterError(msg)
    kinds = set(rows["value_kind"]) or {"time", "num", "str"}
    typed_values = []
    if _is_range(value):
        kind, lo, hi, typed_values = _range_bounds(value, kinds)
    elif _is_collection(value):
        raw_values = list(value)
        if not raw_values:
            raise InvalidSpoolQueryError("Coordinate membership cannot be empty.")
        arr = np.asarray(raw_values)
        if arr.dtype == bool:
            # boolean masks are patch-local; no index predicate at all,
            # but the coord must exist on the patch.
            kind = lo = hi = None
        else:
            typed_values = [_coerce_scalar(x, kinds) for x in raw_values]
            value_kinds = {x.kind for x in typed_values}
            if len(value_kinds) != 1:
                raise InvalidSpoolQueryError(
                    f"Coordinate values for {name!r} have mixed kinds."
                )
            kind = typed_values[0].kind
            values = [x.value for x in typed_values]
            lo, hi = min(values), max(values)
    else:
        typed = _coerce_scalar(value, kinds)
        typed_values = [typed]
        kind = typed.kind
        lo = hi = typed.value

    compatible_units = _compatible_coord_units(rows, typed_values, name)

    min_col, max_col = {
        "time": ("min_ns", "max_ns"),
        "dur": ("min_ns", "max_ns"),
        "num": ("min_num", "max_num"),
        "str": ("min_str", "max_str"),
        None: (None, None),
    }[kind]
    conditions = ["pc.patch_id = p.patch_id", "pc.coord_name = ?"]
    params: list = [name]
    if kind is not None:
        if kind in ("time", "dur"):
            # absolute queries match absolute coords, durations relative.
            conditions.append("cd.is_relative = ?")
            params.append(kind == "dur")
            kind_match = "time"
        else:
            kind_match = kind
        conditions.append("cd.value_kind = ?")
        params.append(kind_match)
        if compatible_units is not None:
            # NULL-unit defs stay candidates: IN () never matches NULL and
            # unitless values cannot be proven dimensionally incompatible.
            if compatible_units:
                marks = ", ".join("?" for _ in compatible_units)
                conditions.append(f"(cd.units IN ({marks}) OR cd.units IS NULL)")
                params.extend(sorted(compatible_units))
            else:
                conditions.append("cd.units IS NULL")
        if lo is not None:
            conditions.append(f"cd.{max_col} >= ?")
            params.append(lo)
        if hi is not None:
            conditions.append(f"cd.{min_col} <= ?")
            params.append(hi)
    where.add(
        "EXISTS (SELECT 1 FROM patch_coords pc "
        "JOIN coord_defs cd ON cd.coord_def_id = pc.coord_def_id "
        "WHERE " + " AND ".join(conditions) + ")",
        *params,
    )


def build_query_sql(
    query: Query | Sequence[Query],
    dialect: BaseDialect,
    attr_meta: pd.DataFrame,
    coord_meta: pd.DataFrame,
) -> tuple[str, list, list[tuple[str, re.Pattern]]]:
    """
    Build the flat-relation SELECT for one or more AND-composed queries.

    coord_meta must cover every coordinate the queries reference (it may
    be empty for attr-only queries). Returns (sql, params, residuals)
    where residuals maps attr names to regex patterns that must be
    re-applied to the resulting dataframe.
    """
    queries = [query] if isinstance(query, Query) else list(query)
    where = _Where()
    residuals: list[tuple[str, re.Pattern]] = []
    for one in queries:
        for name, value in one.attrs.items():
            residual = build_attr_clause(where, dialect, attr_meta, name, value)
            if residual is not None:
                residuals.append((name, residual))
        for name, value in one.coords.items():
            build_coord_clause(where, dialect, coord_meta, name, value)
    # attr columns selected explicitly: `a.*` would duplicate patch_id and
    # engines disagree on how to dedupe result column names.
    attr_cols = "".join(
        f", a.{dialect.quote(col)}" for col in attr_meta["column_name"].unique()
    )
    sql = (
        "SELECT s.source_path, s.base_uri, s.source_format, s.format_version, "
        f"p.*{attr_cols} "
        "FROM patches p "
        "JOIN sources s ON s.source_id = p.source_id "
        "LEFT JOIN attrs a ON a.patch_id = p.patch_id "
        f"WHERE {where.sql} "
        "ORDER BY p.time_min NULLS LAST, p.patch_id"
    )
    return sql, where.params, residuals


def apply_residuals(
    df: pd.DataFrame, residuals: list[tuple[str, re.Pattern]]
) -> pd.DataFrame:
    """Apply regex residual filters to the flat relation."""
    for name, pattern in residuals:
        col = df[name]
        keep = col.map(
            lambda x: bool(pattern.search(x)) if isinstance(x, str) else False
        )
        df = df[keep]
    return df


def relative_offset(gmin, gmax, value):
    """
    Resolve one relative bound against a global [gmin, gmax] envelope.

    Positive offsets measure from the start, negative from the end;
    None/Ellipsis bounds stay open. Datetime envelopes take numeric
    seconds offsets.
    """
    import dascore as dc

    if value is None or value is Ellipsis:
        return None
    if isinstance(gmin, pd.Timestamp) or isinstance(gmin, np.datetime64):
        delta = dc.to_timedelta64(abs(float(value)))
        return (gmin + delta) if value >= 0 else (gmax - delta)
    return (gmin + value) if value >= 0 else (gmax + value)


def glob_match(value, pattern: str) -> bool:
    """Reference glob semantics (used by pandas fallbacks and tests)."""
    return isinstance(value, str) and fnmatch.fnmatch(value, pattern)
