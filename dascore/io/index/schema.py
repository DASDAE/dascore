"""
Logical schema for the spool index.

The schema uses four primitive storage types (int64, float64, str, bool).
Times and durations are always epoch/plain nanoseconds stored as int64,
never engine-native timestamp types.
"""

from __future__ import annotations

from types import MappingProxyType

# Version of the index schema, independent of dascore's version.
INDEX_VERSION = 2
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

META_DATA = MappingProxyType(
    {
        "what_is_this": "str",
        "index_version": "int64",
        "dascore_version": "str",
        "last_indexed_ns": "int64",
    }
)

SOURCES = MappingProxyType(
    {
        "source_id": "int64",
        "base_uri": "str",
        "source_path": "str",
        "source_format": "str",
        "format_version": "str",
        "mtime_ns": "int64",
        "size_bytes": "int64",
        "last_indexed_ns": "int64",
    }
)

# Frozen structural table; nothing dynamic is ever added here. The
# time/distance envelopes are cached summaries of the two conventional
# dims (hot path), not attr promotion.
PATCHES = MappingProxyType(
    {
        "patch_id": "int64",
        "source_id": "int64",
        "source_patch_id": "str",
        "n_dims": "int64",
        "dims": "str",
        "shape": "str",
        "sample_count_total": "int64",
        "time_min": "int64",  # epoch ns; NULL for relative-time patches
        "time_max": "int64",
        "time_step": "int64",
        "distance_min": "float64",  # canonical SI (m)
        "distance_max": "float64",
        "distance_step": "float64",
    }
)

# attrs table starts with only the key; typed columns (`<name>__<kind>`)
# are added lazily at ingest.
ATTRS_BASE = MappingProxyType({"patch_id": "int64"})

ATTR_META = MappingProxyType(
    {
        "attr_name": "str",  # original (unsanitized) attr name
        "value_kind": "str",
        "column_name": "str",  # sanitized column in the attrs table
        "units": "str",  # canonical unit for num kinds, nullable
    }
)

# Unique coordinate summaries, deduplicated across patches. Range coordinates
# use a semantic fingerprint supplied by the scan or reconstructed exactly
# from the range summary. Non-range coordinates without a fingerprint use a
# summary hash for storage deduplication, but it is not exposed as value identity.
COORD_DEFS = MappingProxyType(
    {
        "coord_def_id": "int64",
        "def_key": "str",
        "fingerprint": "str",  # nullable; semantic hash from CoordSummary
        "value_kind": "str",  # num | time | str
        "dtype": "str",
        "length": "int64",
        "units": "str",  # original unit string; numeric values stored SI
        "min_num": "float64",
        "max_num": "float64",
        "step_num": "float64",
        "min_ns": "int64",
        "max_ns": "int64",
        "step_ns": "int64",
        "min_str": "str",
        "max_str": "str",
        "is_monotonic": "bool",
        "is_relative": "bool",
    }
)

# Links a patch to its coord defs; the name and dims are patch-level
# semantics (two patches can share values under different names).
PATCH_COORDS = MappingProxyType(
    {
        "patch_id": "int64",
        "coord_name": "str",
        "coord_dims": "str",
        "coord_def_id": "int64",
    }
)

TABLES = MappingProxyType(
    {
        "meta_data": META_DATA,
        "sources": SOURCES,
        "patches": PATCHES,
        "attrs": ATTRS_BASE,
        "attr_meta": ATTR_META,
        "coord_defs": COORD_DEFS,
        "patch_coords": PATCH_COORDS,
    }
)

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
            "UNIQUE (source_id, source_patch_id)",
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
            "CHECK (is_monotonic IS NULL OR is_monotonic IN (0, 1))",
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
        # storage tables
        "patch_id",
        "source_id",
        "source_patch_id",
        "source_path",
        "source_format",
        "format_version",
        "base_uri",
        "mtime_ns",
        "size_bytes",
        "n_dims",
        "dims",
        "shape",
        "sample_count_total",
        "coord_def_id",
        "def_key",
        # flat-relation (spool-facing) names
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

# Explicit secondary indexes. Every other access path is covered by a
# PRIMARY KEY or UNIQUE autoindex above — patch_coords(patch_id,
# coord_name), sources(base_uri, source_path), patches(source_id,
# source_patch_id), coord_defs(def_key) — and duplicating them measured
# ~25% extra file size and slower writes for no query gain.
INDEXES = (("idx_pcoords_name", "patch_coords", "coord_name"),)
