"""
SQLite spool index package.

Provides a normalized, summary-only index of patch metadata (sources,
patches, attrs, and coordinates). `PatchCatalog` is the spool-facing
metadata engine and the only thing outside this package needs; the
modules below it are the index implementation and are imported directly
by the few callers (tests, mostly) which have business there.
"""

from __future__ import annotations

from dascore.io.index.catalog import PatchCatalog
from dascore.io.index.query import Query

__all__ = ["PatchCatalog", "Query"]
