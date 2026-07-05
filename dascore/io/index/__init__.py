"""
Backend-agnostic spool index package.

Provides a normalized, summary-only index of patch metadata (sources,
patches, attrs, coords) with interchangeable storage backends. See
`.scratch/spool_index_design.md` on the spool-index-backend branch and
GitHub discussion #648 for the design.
"""

from __future__ import annotations

from dascore.io.index.backend import AbstractIndexBackend, get_backend
from dascore.io.index.catalog import (
    FileResolver,
    LiveResolver,
    PatchCatalog,
    PatchResolver,
)
from dascore.io.index.ingest import summaries_to_records
from dascore.io.index.query import Query

__all__ = [
    "AbstractIndexBackend",
    "FileResolver",
    "LiveResolver",
    "PatchCatalog",
    "PatchResolver",
    "Query",
    "get_backend",
    "summaries_to_records",
]
