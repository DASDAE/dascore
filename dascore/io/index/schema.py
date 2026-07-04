"""
Logical schema for the spool index.

The schema is defined in backend-neutral terms; only four primitive
storage types are used (int64, float64, str, bool) so any SQL-ish backend
can represent it. Times and durations are always epoch/plain nanoseconds
stored as int64 — never engine-native timestamp types.
"""

from __future__ import annotations

from types import MappingProxyType

# Version of the index schema, independent of dascore's version.
INDEX_VERSION = 1
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

COORDS = MappingProxyType(
    {
        "patch_id": "int64",
        "coord_name": "str",
        "value_kind": "str",  # num | time | str
        "dtype": "str",
        "coord_dims": "str",
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
        "coord_hash": "str",
    }
)

TABLES = MappingProxyType(
    {
        "meta_data": META_DATA,
        "sources": SOURCES,
        "patches": PATCHES,
        "attrs": ATTRS_BASE,
        "attr_meta": ATTR_META,
        "coords": COORDS,
    }
)

# Names which can never be dynamic attr columns.
RESERVED_ATTR_COLUMNS = frozenset({"patch_id"})

# Secondary indexes: without these, engines that use nested-loop plans
# (SQLite) go quadratic on the correlated coords EXISTS subquery.
INDEXES = (
    ("idx_coords_patch", "coords", "patch_id"),
    ("idx_coords_name", "coords", "coord_name"),
    ("idx_attrs_patch", "attrs", "patch_id"),
    ("idx_patches_source", "patches", "source_id"),
    ("idx_sources_path", "sources", "source_path"),
)
