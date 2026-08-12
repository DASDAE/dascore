"""
Query model and SQL generation for the spool index.

Implements the selector semantics spec (see
`.scratch/selector_semantics_spec.md`): the index only produces
candidates — predicates the summary cannot evaluate exactly are the
caller's responsibility at patch-load time. Predicates SQLite cannot evaluate
exactly are applied as pandas residual filters.
"""

from __future__ import annotations

import json
import operator
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import pandas as pd

from dascore.exceptions import InvalidSpoolQueryError, ParameterError, UnitError
from dascore.io.index.dialect import SQLiteDialect
from dascore.io.index.ingest import typed_value
from dascore.units import convert_units
from dascore.utils.misc import is_range

_GLOB_CHARS = frozenset("*?[")


class _Unset(Enum):
    """Sentinel for "no target units given", which None cannot express.

    None means "this coordinate is unitless", which _to_target_unit
    rejects for a query carrying units, so the two must stay distinct. An
    enum member rather than object() so that testing against it narrows
    the parameter to the str | None it otherwise holds.
    """

    UNSET = auto()


_UNSET = _Unset.UNSET


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


# Shared join skeleton for the patch relation. The attrs join is 1:1 (one
# attrs row per patch), so it is safe for both the projection and the count.
_FROM = (
    "FROM patches p "
    "JOIN sources s ON s.source_id = p.source_id "
    "LEFT JOIN attrs a ON a.patch_id = p.patch_id "
)


def _as_query_list(query: Query | Sequence[Query]) -> list[Query]:
    """Normalize a single Query or a sequence of them to a list."""
    return [query] if isinstance(query, Query) else list(query)


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
            retyped = typed_value(pd.Timestamp(value).to_datetime64())
            return retyped
        except (ValueError, TypeError):
            pass
    return typed


def _normalize_unit(value) -> str | None:
    """Return a nullable unit string from a dataframe value."""
    if value is None or pd.isnull(value) or value == "":
        return None
    return str(value)


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
    target_units: str | _Unset | None = _UNSET,
    name: str = "value",
):
    """
    Return (kind, lo, hi, typed_values) from a range tuple.

    Open bounds (None/Ellipsis) are skipped; typed_values carries the
    coerced usable bounds so callers don't coerce twice.
    """
    lo_raw, hi_raw = value
    kind = None
    typed_values = []
    typed_bounds = []
    for raw, side in ((lo_raw, "lo"), (hi_raw, "hi")):
        if raw is None or raw is Ellipsis:
            continue
        typed = _coerce_scalar(raw, target_kinds)
        typed_values.append(typed)
        knd = typed.kind
        if kind is not None and knd != kind:
            msg = f"Range bounds {value!r} have mixed kinds ({kind}, {knd})."
            raise InvalidSpoolQueryError(msg)
        kind = knd
        typed_bounds.append((side, typed))
    if kind is None:
        msg = f"Range {value!r} has no usable bounds."
        raise InvalidSpoolQueryError(msg)

    # Validate all bound kinds before attempting unit conversion. An
    # unsupported but internally consistent kind is a valid no-match query;
    # mixed kinds remain an invalid range.
    lo = hi = None
    for side, typed in typed_bounds:
        val = (
            typed.value
            if target_units is _UNSET or kind not in target_kinds
            else _to_target_unit(typed, target_units, name)
        )
        if side == "lo":
            lo = val
        else:
            hi = val
    # A coordinate range mixing a bare bound with a unit-bearing one holds
    # magnitudes in two frames of reference — the bare bound means the
    # stored coordinate's units, the other its own base unit — so the two
    # are not comparable until each is expressed in the unit of the
    # definition being tested, which happens per stored unit in
    # build_coord_clause. Attr ranges convert both bounds to the column's
    # one canonical unit above, so they stay comparable here.
    comparable = (
        target_units is not _UNSET or len({x.units is None for x in typed_values}) < 2
    )
    if (
        kind in target_kinds
        and comparable
        and lo is not None
        and hi is not None
        and lo > hi
    ):
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
    compatible: set[str] = set()
    for unit in stored:
        # Skipped rather than differenced out so the unitless rows, which
        # are handled by the check below, stay out of the result set.
        if unit is None:
            continue
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
    dialect: SQLiteDialect,
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
    units = {
        kind: _normalize_unit(unit)
        for kind, unit in zip(rows["value_kind"], rows["units"], strict=True)
    }

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
    if is_range(value):
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
        for bound, op in ((lo, ">="), (hi, "<=")):
            if bound is not None:
                where.add(f"{col(kind)} {op} ?", bound)
        return None
    if _is_collection(value):
        coerced = [_coerce_scalar(v, kinds) for v in value]
        by_kind: dict[str, list] = {}
        for typed in coerced:
            if typed.kind not in kinds:
                continue
            val = _to_target_unit(typed, units.get(typed.kind), name)
            by_kind.setdefault(typed.kind, []).append(val)
        subclauses = []
        params = []
        for kind, vals in by_kind.items():
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
    if kind not in kinds:
        where.add("FALSE")
        return None
    val = _to_target_unit(typed, units.get(kind), name)
    where.add(f"{col(kind)} = ?", val)
    return None


def glob_to_regex(pattern: str) -> re.Pattern:
    """
    Translate a glob to the regex which matches what SQLite's GLOB does.

    SQLite is the authority on what a glob selector means, since that is
    what the index applies, and it is not `fnmatch`: a character class is
    negated with `[^...]`, where fnmatch spells that `[!...]` and reads a
    leading `!` as a literal. Translating here rather than reaching for
    fnmatch is what keeps one pattern from selecting opposite halves of a
    spool depending on which side answered it. An unterminated class
    matches nothing, as it does in SQLite; a class a regex cannot express
    at all (a reversed range, which SQLite reads leniently) matches
    nothing here rather than being guessed at.
    """
    out, index, size = [], 0, len(pattern)
    while index < size:
        char = pattern[index]
        if char in "*?":
            out.append(".*" if char == "*" else ".")
        elif char == "[":
            # A class may open with '^' to negate and may hold ']' as its
            # first member; the class ends at the next ']' after those.
            end = index + 1 + pattern[index + 1 : index + 2].count("^")
            end += pattern[end : end + 1].count("]")
            end = pattern.find("]", end)
            if end < 0:
                return _MATCHES_NOTHING
            body = pattern[index + 1 : end]
            negate, body = body.startswith("^"), body.removeprefix("^")
            # A ']' opening a class is one of its members, and SQLite takes
            # it as a plain one: it is not the low end of a range, so the
            # dash which may follow it is a member too.
            leading = ""
            if body.startswith("]"):
                leading, body = re.escape("]"), body[1:]
            out.append(f"[{'^' if negate else ''}{leading}{_class_body(body)}]")
            index = end
        else:
            out.append(re.escape(char))
        index += 1
    # DOTALL, since SQLite's wildcards cross a newline like any other byte.
    return re.compile("".join(out) + r"\Z", re.DOTALL)


# A pattern which matches nothing, for a glob SQLite would not read.
_MATCHES_NOTHING = re.compile(r"(?!)")


def _class_body(body: str) -> str:
    """
    Escape the members of a glob character class for a regex one.

    Ranges stay ranges, since a glob class means them, with each endpoint
    escaped on its own — escaping the body wholesale would turn the dash
    of a range into a member. A range's low endpoint is also emitted as a
    member: SQLite tests it before testing the range, so `[z-a]` matches
    `z` where the reversed range alone matches nothing.
    """
    out, index, size = [], 0, len(body)
    while index < size:
        low = body[index]
        if index + 2 < size and body[index + 1] == "-":
            high = body[index + 2]
            out.append(re.escape(low))
            if low <= high:
                out.append(f"{re.escape(low)}-{re.escape(high)}")
            index += 3
            continue
        out.append(re.escape(low))
        index += 1
    return "".join(out)


def evaluate_attr_predicate(values, name: str, value, units=None) -> np.ndarray:
    """
    Evaluate one attr selector against values held in memory.

    The in-memory twin of `build_attr_clause`, for values which never
    reach the index — those an inventory states rather than a file. It
    takes the same selector shapes and means the same thing by each, so a
    selector cannot mean one thing for a patch which states the attr and
    another for a patch the inventory answers for.

    Parameters
    ----------
    values
        The value each row holds; every one of them states something.
    name
        The attr the selector names, for error messages.
    value
        The selector, in any shape `build_attr_clause` accepts.
    units
        The canonical units the index stores each kind of this attr in,
        for converting a unit-bearing selector exactly as the index side
        would. A kind the index knows nothing about is unitless, and a
        selector carrying units against one is the error it is there.

    Returns
    -------
    A boolean array of the same length as ``values``.
    """
    # Typed the way the index types a stored value, so a predicate
    # compares within one kind here too: SQL reaches for the typed column
    # a selector names and matches nothing when the values are of another
    # kind, and `1` must not match a stored `True` on either side.
    typed = [typed_value(x) for x in values]
    kinds = {x.kind for x in typed if x is not None}
    units = units or {}

    def selector(item):
        """Type one selector value the way the index would."""
        out = _coerce_scalar(item, kinds)
        _to_target_unit(out, units.get(out.kind), name)
        return out

    def each(func) -> np.ndarray:
        return np.array([x is not None and bool(func(x)) for x in typed], dtype=bool)

    def compare(item, op):
        """Match rows of the selector's own kind under one operator."""
        wanted = selector(item)
        return each(lambda x, w=wanted, o=op: x.kind == w.kind and o(x.value, w.value))

    if isinstance(value, re.Pattern):
        # `search`, which is what the index's regex residual applies.
        return each(lambda x: x.kind == "str" and value.search(x.value))
    if is_range(value):
        # The same reading of a range the index performs, so a range with
        # no usable bounds, mixed-kind bounds, or lo above hi is the error
        # it is there rather than a selector which quietly keeps every row.
        probe = next(
            (
                _coerce_scalar(x, kinds)
                for x in value
                if x is not None and x is not Ellipsis
            ),
            None,
        )
        target = units.get(probe.kind) if probe is not None else None
        kind, low, high, _ = _range_bounds(value, kinds, target, name)
        out = each(lambda x, k=kind: x.kind == k)
        for bound, op in ((low, operator.ge), (high, operator.le)):
            if bound is not None:
                out &= each(
                    lambda x, o=op, b=bound, k=kind: x.kind == k and o(x.value, b)
                )
        return out
    if _is_collection(value):
        out = np.zeros(len(values), dtype=bool)
        for item in value:
            out |= compare(item, operator.eq)
        return out
    if isinstance(value, str) and _GLOB_CHARS & set(value):
        pattern = glob_to_regex(value)
        return each(lambda x: x.kind == "str" and pattern.match(x.value))
    return compare(value, operator.eq)


def build_coord_clause(
    where: _Where,
    dialect: SQLiteDialect,
    coord_meta: pd.DataFrame,
    name: str,
    value,
) -> None:
    """
    Add a patch_coords/coord_defs semi-join clause for one coord
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
    if is_range(value):
        kind, lo, hi, typed_values = _range_bounds(value, kinds)
    else:
        # Scalars, value membership, and boolean masks have no exact
        # patch-level meaning spool-wide; resolve_query rejects them
        # before SQL composition, so only a hand-built Query can reach
        # this.
        msg = f"Coordinate {name!r} accepts range selectors; got {value!r}."
        raise InvalidSpoolQueryError(msg)

    compatible_units = _compatible_coord_units(rows, typed_values, name)

    min_col, max_col = {
        "time": ("min_ns", "max_ns"),
        "dur": ("min_ns", "max_ns"),
        "num": ("min_num", "max_num"),
        "str": ("min_str", "max_str"),
        None: (None, None),
    }[kind]
    conditions = ["pc.coord_name = ?"]
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
        # lo bounds the coord max (overlap), hi bounds the coord min.
        # typed_values holds the coerced non-open bounds in (lo, hi) order,
        # so each side pairs with its TypedValue (which knows the bound's
        # own units) here.
        sides = []
        for bound, bound_col, op in ((lo, max_col, ">="), (hi, min_col, "<=")):
            if bound is None:
                continue
            sides.append((bound, typed_values[len(sides)], bound_col, op))
        if compatible_units is None:
            # A bare range means each definition's native units: one plain
            # predicate against the envelopes, which are stored in the
            # coordinate's original units.
            for bound, _typed, bound_col, op in sides:
                conditions.append(f"cd.{bound_col} {op} ?")
                params.append(bound)
        elif not compatible_units:
            conditions.append("cd.units IS NULL")
        else:
            # A unit-bearing range converts itself once per distinct
            # compatible stored unit into an OR branch; a bare bound in a
            # mixed range stays native per branch, exactly how the
            # residual's per-bound _CanonicalRange reads it at load.
            # NULL-unit defs stay unconstrained candidates: unitless
            # values cannot be proven dimensionally incompatible.
            branches = ["cd.units IS NULL"]
            for unit in sorted(compatible_units):
                sub_clauses = ["cd.units = ?"]
                sub_params: list = [unit]
                for bound, typed, bound_col, op in sides:
                    if typed.units is None:
                        val = bound
                    else:
                        val = convert_units(
                            bound, to_units=unit, from_units=typed.units
                        )
                    sub_clauses.append(f"cd.{bound_col} {op} ?")
                    sub_params.append(val)
                branches.append("(" + " AND ".join(sub_clauses) + ")")
                params.extend(sub_params)
            conditions.append("(" + " OR ".join(branches) + ")")
    # A semi-join the engine can evaluate once (idx_pcoords_name) beats a
    # correlated EXISTS probed per patch row (~2.5x on a 200k-source index).
    where.add(
        "p.patch_id IN (SELECT pc.patch_id FROM patch_coords pc "
        "JOIN coord_defs cd ON cd.coord_def_id = pc.coord_def_id "
        "WHERE " + " AND ".join(conditions) + ")",
        *params,
    )


def _build_where(
    queries: list[Query],
    dialect: SQLiteDialect,
    attr_meta: pd.DataFrame,
    coord_meta: pd.DataFrame,
) -> tuple[_Where, list[tuple[str, re.Pattern]]]:
    """Compose the shared WHERE clause and any regex residuals."""
    where = _Where()
    residuals: list[tuple[str, re.Pattern]] = []
    for one in queries:
        for name, value in one.attrs.items():
            residual = build_attr_clause(where, dialect, attr_meta, name, value)
            if residual is not None:
                residuals.append((name, residual))
        for name, value in one.coords.items():
            build_coord_clause(where, dialect, coord_meta, name, value)
    return where, residuals


# typed coord_defs envelope-minimum column per value kind
_COORD_MIN_COLUMNS = {"num": "min_num", "time": "min_ns", "str": "min_str"}
# the two conventional dims cached as columns on the patches table
_HOT_COORDS = ("time", "distance")


def _order_clause(
    order_by, dialect: SQLiteDialect, attr_meta: pd.DataFrame, coord_meta: pd.DataFrame
) -> tuple[str, list]:
    """
    Resolve an order spec into an ORDER BY clause and its parameters.

    ``order_by`` is ``(kind, name, ascending)`` where kind is "attr"
    (an attrs-table column ordered by its typed column) or "coord"
    (ordered by the coordinate's envelope minimum — the hot patches
    column when cached, otherwise the linked coord_defs typed minimum).
    The ordinal contract supplies the deterministic tiebreak.
    """
    kind, name, ascending = order_by
    direction = "ASC" if ascending else "DESC"
    params: list = []
    if kind == "coord" and name in _HOT_COORDS:
        column = f"p.{dialect.quote(f'{name}_min')}"
    elif kind == "coord":
        rows = coord_meta[coord_meta["coord_name"] == name]
        # a coord observed under several kinds orders by its first kind
        value_kind = str(rows["value_kind"].iloc[0])
        min_col = _COORD_MIN_COLUMNS[value_kind]
        column = (
            f"(SELECT cd.{min_col} FROM patch_coords pc "
            "JOIN coord_defs cd ON cd.coord_def_id = pc.coord_def_id "
            "WHERE pc.patch_id = p.patch_id AND pc.coord_name = ?)"
        )
        params.append(name)
    else:
        rows = attr_meta[attr_meta["attr_name"] == name]
        columns = [dialect.quote(c) for c in rows["column_name"]]
        # an attr observed under several kinds orders by its first column
        column = f"a.{columns[0]}"
    # rows without a value sort last regardless of direction (matching
    # the ordinal renumberer's missing-time-last rule); the null key
    # repeats the column expression, so its parameters repeat too
    params = [*params, *params]
    sql = f"ORDER BY {column} IS NULL, {column} {direction}, s.ordinal, p.patch_id"
    return sql, params


def build_sql(
    query: Query | Sequence[Query],
    dialect: SQLiteDialect,
    attr_meta: pd.DataFrame,
    coord_meta: pd.DataFrame,
    count: bool = False,
    order_by=None,
    patch_ids=None,
    ids_only: bool = False,
) -> tuple[str, list, list[tuple[str, re.Pattern]]]:
    """
    Build SQL for one or more AND-composed queries.

    By default this projects the flat relation; with count=True the same
    WHERE is reused for a COUNT with no projection, coordinate pivot, or
    ordering; with ids_only=True only ordered patch ids are projected
    (the cheap realization slices/windows use). coord_meta must cover
    every coordinate the queries reference (it may be empty for
    attr-only queries). ``order_by`` overrides the default ordinal
    ordering (see `_order_clause`); ``patch_ids`` restricts rows to an
    id membership (one JSON parameter, so the SQLite bound-variable cap
    does not limit membership size).

    Returns (sql, params, residuals), where residuals pairs attr names
    with regex patterns that must be re-applied to the resulting
    dataframe. For a count a non-empty residual means the count is not
    SQL-resolvable (regex must inspect rows) and the caller must fall
    back to a projected count.
    """
    queries = _as_query_list(query)
    where, residuals = _build_where(queries, dialect, attr_meta, coord_meta)
    if patch_ids is not None:
        where.add(
            "p.patch_id IN (SELECT value FROM json_each(?))",
            json.dumps([int(x) for x in patch_ids]),
        )
    if count:
        # COUNT(p.patch_id) counts patches; a WHERE may reference a.<column>.
        sql = f"SELECT COUNT(p.patch_id) AS n {_FROM}WHERE {where.sql}"
        return sql, where.params, residuals
    if order_by is not None:
        order, order_params = _order_clause(order_by, dialect, attr_meta, coord_meta)
    else:
        # the ordering contract: source ordinal, then file-internal order
        order, order_params = "ORDER BY s.ordinal, p.patch_id", []
    params = [*where.params, *order_params]
    if ids_only:
        sql = f"SELECT p.patch_id {_FROM}WHERE {where.sql} {order}"
        return sql, params, residuals
    # attr columns selected explicitly: `a.*` would duplicate patch_id and
    # engines disagree on how to dedupe result column names.
    attr_cols = "".join(
        f", a.{dialect.quote(col)}" for col in attr_meta["column_name"].unique()
    )
    sql = (
        "SELECT s.source_path, s.base_uri, s.source_format, s.format_version, "
        f"s.path_attrs, p.*{attr_cols} "
        f"{_FROM}"
        f"WHERE {where.sql} "
        f"{order}"
    )
    return sql, params, residuals


def apply_residuals(
    df: pd.DataFrame,
    residuals: list[tuple[str, re.Pattern]],
    attr_columns: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """
    Apply regex residual filters to the flat relation.

    Residuals are produced only for attr predicates, so when `attr_columns`
    is given the original attr series is preferred over the flat column of
    the same name — the flat column can instead hold a coordinate envelope
    when the attr name collided with one.
    """
    attr_columns = attr_columns or {}
    for name, pattern in residuals:
        col = attr_columns[name].loc[df.index] if name in attr_columns else df[name]
        keep = col.map(
            lambda x: bool(pattern.search(x)) if isinstance(x, str) else False
        )
        df = df[keep]
    return df
