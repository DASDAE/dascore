"""
SQLite spool index package.

Provides a normalized, summary-only index of patch metadata (sources,
patches, attrs, and coordinates). `PatchCatalog` is the spool-facing
metadata engine, and `Query` the shape a selection takes on the way in.
The modules below them are the index implementation; the few callers
with business there import from those modules directly.
"""

from __future__ import annotations

from dascore.io.index.catalog import PatchCatalog
from dascore.io.index.query import Query

__all__ = ["PatchCatalog", "Query"]
