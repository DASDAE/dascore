"""
Logical schema for the spool index.

Each stored table is declared once, as a NamedTuple naming its columns in
order. The column types are the four primitive storage types (int64,
float64, str, bool), written as the python type each surfaces as and
marked with None where the column is nullable; times and durations are
always epoch/plain nanoseconds stored as int64, never engine-native
timestamp types.

The row classes are the single source of truth: `TABLES` (the logical
column types the DDL is built from) is derived from them, and index code
reads rows through them with `iter_rows(df, Row)` (see dascore.utils.pd),
which names — for readers and type checkers, not at runtime — the row
shape pandas builds dynamically in `itertuples`. The classes are never
instantiated. A frame holding only some of a table's columns
still uses its table's row class; only the columns actually fetched can
be read, and nullable ones may arrive as NaN rather than None (reading
code guards with `pd.isnull`).

The bottom of the module holds the other half of that translation: the
SQLite spellings the logical types become, and the DDL built from them.
It lives here because both the query builder and the backend need it and
this module imports nothing which could import them back.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple, get_args, get_type_hints

# Version of the index schema, independent of dascore's version. Bump it
# when an index written by an older dascore would be read wrongly rather
# than merely incompletely -- including when what a *stored value* means
# changes, not only when a column does. Version 15 stores source
# coordinate and numeric attribute dtypes so index-built patches recover
# their original types. Earlier indexes lack the metadata required for reconstruction
# and are rebuilt when opened.
INDEX_VERSION = 15
# Identity string so any tool can sanity-check what it opened.
WHAT_IS_THIS = "dascore_spool_index"

# Value kinds for typed attr columns and coord rows.
KINDS = ("num", "str", "bool", "time", "dur")
# Storage type (logical) backing each kind.
KIND_STORAGE = MappingProxyType(
    {
        "num": "float64",
        "str": "str",
        "bool": "bool",
        "time": "int64",  # epoch ns
        "dur": "int64",  # ns
    }
)

# The logical storage type each declared python type maps to.
_STORAGE_TYPES = MappingProxyType(
    {int: "int64", float: "float64", str: "str", bool: "bool"}
)


class MetaDataRow(NamedTuple):
    """A row of the meta_data table (the index's identity and version)."""

    what_is_this: str
    index_version: int
    dascore_version: str
    last_indexed_ns: int


class SourceRow(NamedTuple):
    """A row of the sources table (one scan unit)."""

    source_id: int
    base_uri: str
    source_path: str
    source_format: str
    format_version: str
    mtime_ns: int | None
    size_bytes: int | None
    # JSON dict of hive-style key=value attrs parsed from the stored
    # path's directory segments; NULL when the path carries none.
    # Records which attr values are path-derived so moves can rewrite
    # them and patch loading can stamp them without re-parsing paths
    # (derived/union catalogs absolutize paths, losing the segments).
    path_attrs: str | None
    last_indexed_ns: int
    # The catalog's explicit ordering contract: patch rows present in
    # (ordinal, patch_id) order. Assigned at ingest (insertion
    # sequence); a replaced source keeps its position while new
    # sources append, so merging catalogs concatenates and
    # deduplication keeps first-occurrence position with
    # last-occurrence metadata (dict-merge semantics). The directory
    # syncer renumbers to time order after each sync, preserving the
    # conventional time-ordered presentation of file archives.
    ordinal: int


class PatchRow(NamedTuple):
    """
    A row of the patches table.

    Frozen structural table; nothing dynamic is ever added here. The
    time/distance envelopes are cached summaries of the two conventional
    dims (hot path), not attr promotion.
    """

    patch_id: int
    source_id: int
    source_patch_key: str
    dims: str
    dtype: str  # the data array's dtype, eg "float64"
    # Samples in the data array (the product of its shape); NULL when the
    # summary stated no shape.
    data_size: int | None
    time_min: int | None  # epoch ns; NULL for relative-time patches
    time_max: int | None
    time_step: int | None
    distance_min: float | None  # original units, as the coord states them
    distance_max: float | None
    distance_step: float | None
    # 1 when every attr the patch defines is a column; 0 when ingest had
    # to leave one out (an array, a name a structural column owns), so a
    # row built into a patch would be missing what the file holds.
    attrs_complete: int
    # JSON mapping for numeric attrs whose scalar dtype SQL would erase.
    attr_dtypes: str | None


class AttrsRow(NamedTuple):
    """
    The fixed part of an attrs row.

    The table starts with only the key; typed columns (`<name>__<kind>`)
    are added lazily at ingest, so a fetched row carries more than this.
    """

    patch_id: int


class AttrMetaRow(NamedTuple):
    """A row of the attr_meta table (one indexed attr name and kind)."""

    attr_name: str  # original (unsanitized) attr name
    value_kind: str
    column_name: str  # sanitized column in the attrs table
    units: str | None  # canonical unit for num kinds


class CoordDefRow(NamedTuple):
    """
    A row of the coord_defs table.

    Unique coordinate summaries, deduplicated across patches. Range
    coordinates use a semantic fingerprint supplied by the scan or
    reconstructed exactly from the range summary. Non-range coordinates
    without a fingerprint use a summary hash for storage deduplication,
    but it is not exposed as value identity.
    """

    coord_def_id: int
    def_key: str
    fingerprint: str | None  # semantic hash from CoordSummary
    value_kind: str  # num | time | str
    dtype: str
    length: int | None
    # The ORIGINAL unit string as ingested, never converted — and the
    # numeric envelope below is stored in these same original units, so
    # the index shows what the patch shows. Unit-bearing queries convert
    # themselves per stored unit at query time.
    units: str | None
    min_num: float | None
    max_num: float | None
    step_num: float | None
    min_ns: int | None
    max_ns: int | None
    step_ns: int | None
    min_str: str | None
    max_str: str | None
    is_relative: bool | None


class PatchCoordRow(NamedTuple):
    """
    A row of the patch_coords table.

    Links a patch to its coord defs; the name and dims are patch-level
    semantics (two patches can share values under different names).
    """

    patch_id: int
    coord_name: str
    coord_dims: str
    coord_def_id: int
    dtype: str  # source representation; shared definitions identify values


# The row class declaring each stored table.
TABLE_ROWS = MappingProxyType(
    {
        "meta_data": MetaDataRow,
        "sources": SourceRow,
        "patches": PatchRow,
        "attrs": AttrsRow,
        "attr_meta": AttrMetaRow,
        "coord_defs": CoordDefRow,
        "patch_coords": PatchCoordRow,
    }
)


def _columns(row_type: type[NamedTuple]) -> MappingProxyType[str, str]:
    """Return a row class's {column: logical storage type} mapping."""
    out = {}
    for name, hint in get_type_hints(row_type).items():
        # Nullability is not part of the storage type; a nullable column
        # is declared as `<type> | None` for readers of the row.
        types = set(get_args(hint)) - {type(None)} or {hint}
        assert len(types) == 1, f"{row_type.__name__}.{name} needs one storage type"
        out[name] = _STORAGE_TYPES[types.pop()]
    return MappingProxyType(out)


# The logical columns of each table, in declaration order; the DDL and
# every insert's column list are built from these.
TABLES = MappingProxyType({name: _columns(row) for name, row in TABLE_ROWS.items()})

# Keeping constraints beside the logical columns makes the stored contract
# explicit and keeps dynamic attr-column DDL separate from table identity.
TABLE_CONSTRAINTS = MappingProxyType(
    {
        "meta_data": (
            "PRIMARY KEY (what_is_this)",
            f"CHECK (what_is_this = '{WHAT_IS_THIS}')",
        ),
        "sources": (
            "PRIMARY KEY (source_id)",
            "UNIQUE (base_uri, source_path)",
            "CHECK (base_uri IS NOT NULL)",
            "CHECK (source_path IS NOT NULL)",
        ),
        "patches": (
            "PRIMARY KEY (patch_id)",
            # Every ingest path joins a string here, so this cannot fire
            # today; it states the invariant where the schema states its
            # other invariants, for a writer which does not yet exist.
            "CHECK (dims IS NOT NULL)",
            "UNIQUE (source_id, source_patch_key)",
            "FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE",
        ),
        "attrs": (
            "PRIMARY KEY (patch_id)",
            "FOREIGN KEY (patch_id) REFERENCES patches(patch_id) ON DELETE CASCADE",
        ),
        "attr_meta": (
            "PRIMARY KEY (attr_name, value_kind)",
            "UNIQUE (column_name)",
            "CHECK (value_kind IN ('num', 'str', 'bool', 'time', 'dur'))",
        ),
        "coord_defs": (
            "PRIMARY KEY (coord_def_id)",
            "UNIQUE (def_key)",
            "CHECK (value_kind IN ('num', 'time', 'str'))",
            "CHECK (is_relative IS NULL OR is_relative IN (0, 1))",
        ),
        "patch_coords": (
            "PRIMARY KEY (patch_id, coord_name)",
            "FOREIGN KEY (patch_id) REFERENCES patches(patch_id) ON DELETE CASCADE",
            "FOREIGN KEY (coord_def_id) REFERENCES coord_defs(coord_def_id)",
        ),
    }
)

# Attr names which would collide with structural storage or flat-relation
# columns. Attrs with these (sanitized) names stay on the patch but are
# not indexed; ingest warns about them.
RESERVED_ATTR_COLUMNS = frozenset(
    {
        # storage tables. `patch_id` is deliberately absent: the row id it
        # names here is renamed private (SPOOL_EARLY_RENAMES) before attr
        # columns are applied, so a patch's own `patch_id` claims the
        # public spelling rather than colliding with it. A path may still
        # not claim it; see `_UNCLAIMABLE_BY_PATH`.
        "source_id",
        "source_patch_key",
        "source_path",
        "source_format",
        "format_version",
        "base_uri",
        "mtime_ns",
        "size_bytes",
        "path_attrs",
        "dims",
        "dtype",
        "data_size",
        "attrs_complete",
        "attr_dtypes",
        # These named structural columns until version 9 dropped them.
        # They stay reserved: the spool gives them a meaning whether or
        # not a column holds one, and un-reserving a name silently turns
        # it into an ordinary attr, which chunk then refuses to merge
        # patches over.
        "n_dims",
        "shape",
        "sample_count_total",
        "coord_def_id",
        "def_key",
        # fixed time/distance envelope columns on the patches table: these
        # exist for every row regardless of the patch's coords, so an attr
        # with one of these names could not be told apart from structural
        # metadata downstream (unlike other coords' envelope columns, which
        # only exist when the coord does and are handled by the flat-view
        # collision warning).
        "time_min",
        "time_max",
        "time_step",
        "distance_min",
        "distance_max",
        "distance_step",
        # flat-relation (spool-facing) names; source_path and source_format
        # keep their storage-table spelling out in the flat relation, so the
        # entries above already reserve them.
        "source_version",
        # the keyword names the row is loaded with: an attr called one of
        # these would be splatted into dc.read alongside the row's own
        # value and collide with it.
        "path",
        "file_format",
        "file_version",
        # spool instruction machinery
        "current_index",
        "source_index",
        "output_id",
        "patch",
    }
)

# Structural columns the spool relation carries *privately*. The leading
# underscore is load-bearing, not cosmetic: chunk's merge-compatibility
# grouping and conflict policing both compare all non-private columns, so
# a public `dtype` would raise CoordMergeError on every merge of patches
# with differing element types, and a public `data_size` would do the
# same to every merge of patches of different lengths. `present_columns`
# gives both back their public spelling on the way out to a caller.
# `patch_id` is the one of these an attr also legitimately claims: it
# names a row in this schema and a datum on a patch. It is renamed in the
# backend, before attr columns land, so the attr finds the public spelling
# free; the rest are renamed where the spool relation is built.
SPOOL_EARLY_RENAMES = MappingProxyType({"patch_id": "_patch_id"})
SPOOL_LATE_RENAMES = MappingProxyType(
    {
        "dtype": "_dtype",
        "data_size": "_data_size",
        "attrs_complete": "_attrs_complete",
        "attr_dtypes": "_attr_dtypes",
    }
)

# Explicit secondary indexes. Every other access path is covered by a
# PRIMARY KEY or UNIQUE autoindex above — patch_coords(patch_id,
# coord_name), sources(base_uri, source_path), patches(source_id,
# source_patch_key), coord_defs(def_key) — and duplicating them measured
# ~25% extra file size and slower writes for no query gain.
INDEXES = (("idx_pcoords_name", "patch_coords", "coord_name"),)


# --- SQL generation -----------------------------------------------------
#
# The engine-specific half of the schema: type names, quoting, glob
# matching, and DDL. It lives here, beside the logical types it
# translates, because both the query builder and the backend need it and
# this module imports nothing which could import them back.

# Logical type -> engine type. SQLite STRICT tables accept
# INTEGER/REAL/TEXT (and INTEGER for bool).
TYPE_MAP = MappingProxyType(
    {
        "int64": "INTEGER",
        "float64": "REAL",
        "str": "TEXT",
        "bool": "INTEGER",
    }
)


def quote(identifier: str) -> str:
    """Quote an identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def glob_expr(column_sql: str) -> str:
    """Return a parametrized unix-glob match expression."""
    return f"{column_sql} GLOB ?"


def create_table_sql(
    name: str, columns: Mapping[str, str], constraints: tuple[str, ...] = ()
) -> str:
    """Return DDL for one table from logical column types."""
    definitions = [f"{quote(col)} {TYPE_MAP[typ]}" for col, typ in columns.items()]
    definitions.extend(constraints)
    return f"CREATE TABLE IF NOT EXISTS {quote(name)} ({', '.join(definitions)}) STRICT"


def add_column_sql(table: str, column: str, logical_type: str) -> str:
    """Return DDL to add one nullable column."""
    return (
        f"ALTER TABLE {quote(table)} "
        f"ADD COLUMN {quote(column)} {TYPE_MAP[logical_type]}"
    )
