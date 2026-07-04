"""
A directory indexer backed by the generic spool index.

Drop-in alternative to `dascore.io.indexer.DirectoryIndexer`: it walks a
directory, detects new/changed/removed sources by per-source
(mtime, size) comparison, scans only what changed, and answers content
queries from the index backend.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from typing_extensions import Self

import dascore as dc
from dascore.constants import PROGRESS_LEVELS
from dascore.io.index.backend import get_backend, resolve_query
from dascore.io.index.ingest import summaries_to_records
from dascore.io.indexer import AbstractIndexer
from dascore.utils.misc import _iter_filesystem

# Structural columns the spool machinery must not see: unique-per-patch
# values block chunk merge-compatibility grouping, which compares all
# non-private columns.
_SPOOL_HIDDEN_COLUMNS = ("n_dims", "sample_count_total", "shape")


class DBDirectoryIndexer(AbstractIndexer):
    """
    Index a directory of fiber files with a database backend.

    Parameters
    ----------
    path
        The directory to index.
    engine
        The backend kind: "duckdb", "sqlite", or "parquet".
    index_path
        Where to keep the index; defaults to a hidden entry at the top of
        the data directory.
    """

    ext: str | None = None

    def __init__(
        self,
        path: str | Path,
        engine: str = "duckdb",
        index_path: str | Path | None = None,
    ):
        self.path = Path(path).absolute()
        self.engine = engine
        if index_path is None:
            index_path = self.path / f".dascore_index_{engine}"
        self.index_path = Path(index_path)
        self._backend = get_backend(self.index_path, kind=engine)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} ({self.engine}) managing: {self.path}"

    __repr__ = __str__

    def __deepcopy__(self, memo) -> Self:
        """
        Derived spools share the indexer (and its live DB connection).

        DataFrameSpool copies spool state on select/chunk; the index
        connection is read-shared, matching the single-writer model.
        """
        return self

    def _current_files(self) -> dict[str, tuple[int, int, Path]]:
        """Map relative posix path -> (mtime_ns, size, absolute path)."""
        out = {}
        for file_path in _iter_filesystem(self.path, ext=self.ext):
            stat = file_path.stat()
            rel = file_path.relative_to(self.path).as_posix()
            out[rel] = (stat.st_mtime_ns, stat.st_size, file_path)
        return out

    def update(self, paths=None, progress: PROGRESS_LEVELS = "standard") -> Self:
        """
        Update the index: scan new/changed sources, drop removed ones.

        Change detection compares each source's stored (mtime_ns,
        size_bytes) against the filesystem — never a global watermark —
        and stale-source removal is folded in (the walk is the dominant
        cost; removal afterwards is nearly free).
        """
        current = self._current_files()
        stored = {
            row.source_path: (row.mtime_ns, row.size_bytes)
            for row in self._backend.get_sources().itertuples()
        }
        stale = [path for path in stored if path not in current]
        changed = [
            rel
            for rel, (mtime, size, _) in current.items()
            if stored.get(rel) != (mtime, size)
        ]
        if stale:
            self._backend.delete_sources(stale)
        if changed:
            abs_paths = [current[rel][2] for rel in changed]
            summaries = dc.scan(abs_paths, progress=progress)
            # scan reports absolute source paths; stat maps use them too
            records = summaries_to_records(
                summaries,
                relative_to=str(self.path),
                mtimes_ns={str(current[r][2]): current[r][0] for r in changed},
                sizes_bytes={str(current[r][2]): current[r][1] for r in changed},
            )
            if records:
                self._backend.write_sources(records)
        return self

    def get_contents(self, _attrs=None, _coords=None, **kwargs) -> pd.DataFrame:
        """
        Query the index, returning the spool-facing flat relation.

        Bare kwargs resolve attrs-first then coords; `_attrs`/`_coords`
        disambiguate explicitly (see the selector semantics spec).
        """
        query = resolve_query(self._backend, _attrs=_attrs, _coords=_coords, **kwargs)
        df = self._backend.query(query)
        df = df.drop(columns=list(_SPOOL_HIDDEN_COLUMNS), errors="ignore")
        return df.rename(columns={"patch_id": "_patch_id"})

    __call__ = get_contents

    def close(self) -> None:
        """Close the backend."""
        self._backend.close()
