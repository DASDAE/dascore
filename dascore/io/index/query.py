"""
Query model and SQL generation for the spool index.

Implements the selector semantics spec (see
`.scratch/selector_semantics_spec.md`): the index only produces
candidates — predicates the summary cannot evaluate exactly are the
caller's responsibility at patch-load time. SQL generation is shared by
all backends; anything a dialect cannot push down is applied as a pandas
residual filter with identical semantics.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dascore.exceptions import ParameterError
from dascore.io.index.dialect import BaseDialect
from dascore.io.index.ingest import typed_value

_GLOB_CHARS = frozenset("*?[")


class InvalidSpoolQueryError(ParameterError):
    """Raised when a spool query references unknown names or bad values."""


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
            return retyped.kind, retyped.value
        except (ValueError, TypeError):
            pass
    return typed.kind, typed.value


def _range_bounds(value, target_kinds: set[str]):
    """Return (kind, lo, hi) from a range tuple, handling open bounds."""
    lo_raw, hi_raw = value
    lo = hi = None
    kind = None
    for raw, name in ((lo_raw, "lo"), (hi_raw, "hi")):
        if raw is None or raw is Ellipsis:
            continue
        knd, val = _coerce_scalar(raw, target_kinds)
        if kind is not None and knd != kind:
            msg = f"Range bounds {value!r} have mixed kinds ({kind}, {knd})."
            raise InvalidSpoolQueryError(msg)
        kind = knd
        if name == "lo":
            lo = val
        else:
            hi = val
    if kind is None:
        msg = f"Range {value!r} has no usable bounds."
        raise InvalidSpoolQueryError(msg)
    if lo is not None and hi is not None and lo > hi:
        msg = f"Range {value!r} has lo > hi after coercion."
        raise InvalidSpoolQueryError(msg)
    return kind, lo, hi


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
        kind, lo, hi = _range_bounds(value, kinds)
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
        for kind, val in coerced:
            by_kind.setdefault(kind, []).append(val)
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
    kind, val = _coerce_scalar(value, kinds)
    if kind not in kinds:
        where.add("FALSE")
        return None
    where.add(f"{col(kind)} = ?", val)
    return None


def build_coord_clause(
    where: _Where,
    dialect: BaseDialect,
    name: str,
    value,
) -> None:
    """
    Add an EXISTS clause over patch_coords/coord_defs for one coord
    predicate.

    Candidacy only: envelope overlap, never false negatives. Exact
    membership/boolean masks are applied at patch load, above this layer.
    """
    if _is_range(value):
        kind, lo, hi = _range_bounds(value, {"time", "num", "str"})
    elif _is_collection(value):
        arr = np.asarray(list(value) if isinstance(value, set) else value)
        if arr.dtype == bool:
            # boolean masks are patch-local; no index predicate at all,
            # but the coord must exist on the patch.
            kind = lo = hi = None
        else:
            kind, lo = _coerce_scalar(arr.min(), {"time", "num", "str"})
            _, hi = _coerce_scalar(arr.max(), {"time", "num", "str"})
    else:
        kind, val = _coerce_scalar(value, {"time", "num", "str"})
        lo = hi = val

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
) -> tuple[str, list, dict[str, re.Pattern]]:
    """
    Build the flat-relation SELECT for one or more AND-composed queries.

    Returns (sql, params, residuals) where residuals maps attr names to
    regex patterns that must be re-applied to the resulting dataframe.
    """
    queries = [query] if isinstance(query, Query) else list(query)
    where = _Where()
    residuals: dict[str, re.Pattern] = {}
    for one in queries:
        for name, value in one.attrs.items():
            residual = build_attr_clause(where, dialect, attr_meta, name, value)
            if residual is not None:
                residuals[name] = residual
        for name, value in one.coords.items():
            build_coord_clause(where, dialect, name, value)
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


def apply_residuals(df: pd.DataFrame, residuals: dict[str, re.Pattern]) -> pd.DataFrame:
    """Apply regex residual filters to the flat relation."""
    for name, pattern in residuals.items():
        col = df[name]
        keep = col.map(
            lambda x: bool(pattern.search(x)) if isinstance(x, str) else False
        )
        df = df[keep]
    return df


def glob_match(value, pattern: str) -> bool:
    """Reference glob semantics (used by pandas fallbacks and tests)."""
    return isinstance(value, str) and fnmatch.fnmatch(value, pattern)
