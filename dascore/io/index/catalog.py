"""
PatchCatalog: one metadata engine for every spool type.

The catalog owns the index tables (through a backend) and the composed
selection state; a resolver turns flat-relation rows into patches
(from files via dc.read, or from a live registry for in-memory spools);
a syncer (the directory indexer) keeps directory-backed catalogs in step
with the filesystem. See the spool index design doc and discussion #648.

Laziness contract: creating a catalog from patches does no metadata work
until the first metadata operation (select/len/iteration), because
backend bootstrap costs ~10s of ms while holding a patch list is free.
Selection composes Query predicates without running SQL; realization
(len, to_df, iteration) runs exactly one query per view.
"""

from __future__ import annotations

import abc
import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

import dascore as dc
from dascore.constants import PROGRESS_LEVELS
from dascore.io.index.backend import get_backend, resolve_query
from dascore.io.index.ingest import SourceRecord, patch_record
from dascore.io.index.query import InvalidSpoolQueryError, Query

_MEMORY_ENGINES = frozenset({"sqlite", "duckdb"})
_counter = itertools.count()


class PatchResolver(abc.ABC):
    """Turn one flat-relation row into a Patch."""

    @abc.abstractmethod
    def resolve(self, row: Mapping, **trim) -> dc.Patch:
        """
        Return the patch for a row.

        Trim kwargs are hints (a slice of the plan, not the query):
        implementations may use them to read less, but exact trimming is
        re-applied above, so ignoring them is slower, never wrong.
        """


class LiveResolver(PatchResolver):
    """Serve patches from an in-memory registry."""

    def __init__(self):
        self._registry: dict[tuple[str, str], dc.Patch] = {}

    def register(self, path: str, source_patch_id: str, patch: dc.Patch) -> None:
        """Register a live patch under its synthetic source identity."""
        self._registry[(path, source_patch_id)] = patch

    def resolve(self, row: Mapping, **trim) -> dc.Patch:
        """Look the patch up; live patches ignore trim hints."""
        key = (row["path"], row.get("source_patch_id") or "")
        return self._registry[key]


class FileResolver(PatchResolver):
    """Load patches through dc.read; remoteness is the path layer's job."""

    def __init__(self, root: Path | str | None = None):
        self._root = Path(root) if root is not None else None

    def resolve(self, row: Mapping, **trim) -> dc.Patch:
        """Read the patch, passing range trims down as read hints."""
        path = row["path"]
        # relative paths resolve against the catalog root; URIs and
        # absolute paths pass through untouched.
        if self._root is not None and "://" not in str(path):
            if not Path(path).is_absolute():
                path = self._root / path
        kwargs = {"path": path}
        if row.get("file_format"):
            kwargs["file_format"] = row["file_format"]
        if row.get("file_version"):
            kwargs["file_version"] = row["file_version"]
        return dc.read(**kwargs, **trim)[0]


def _live_records(patches: Sequence[dc.Patch], resolver: LiveResolver):
    """Build source records for live patches with synthetic identities."""
    token = next(_counter)
    records = []
    for num, patch in enumerate(patches):
        summary = dc.PatchSummary.from_patch(patch)
        path = f"memory://catalog_{token}/{num}"
        record = patch_record(summary)
        records.append(
            SourceRecord(
                source_path=path,
                source_format="memory",
                format_version="",
                patches=(record,),
            )
        )
        resolver.register(path, record.source_patch_id, patch)
    return records


class PatchCatalog:
    """
    Query-composable metadata catalog over the spool index tables.

    Instances are lightweight views: `select` returns a new catalog
    sharing the backend and resolver with composed predicates. Mutation
    (`add`, `update`, `remove`) is only allowed on the root view.
    """

    def __init__(
        self,
        *,
        backend=None,
        backend_factory=None,
        resolver: PatchResolver | None = None,
        syncer=None,
        queries: tuple[Query, ...] = (),
        residuals: tuple[tuple[dict, bool, bool], ...] = (),
    ):
        self._backend = backend
        self._backend_factory = backend_factory
        self.resolver = resolver
        self._syncer = syncer
        self._queries = tuple(queries)
        self._residuals = tuple(residuals)
        self._df_cache: pd.DataFrame | None = None

    # --- construction -------------------------------------------------

    @classmethod
    def from_patches(
        cls, patches: Sequence[dc.Patch] = (), engine: str = "sqlite"
    ) -> PatchCatalog:
        """
        Catalog over live patches. No backend work happens until the
        first metadata operation.
        """
        if engine not in _MEMORY_ENGINES:
            msg = f"In-memory catalogs support {sorted(_MEMORY_ENGINES)}, not {engine}."
            raise ValueError(msg)
        resolver = LiveResolver()
        pending = tuple(patches)

        def factory():
            backend = get_backend(":memory:", kind=engine)
            if pending:
                backend.write_sources(_live_records(pending, resolver))
            return backend

        return cls(backend_factory=factory, resolver=resolver)

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        engine: str = "sqlite",
        index_path: str | Path | None = None,
    ) -> PatchCatalog:
        """Catalog over a directory of fiber files."""
        from dascore.io.index.indexer import DBDirectoryIndexer

        syncer = DBDirectoryIndexer(path, engine=engine, index_path=index_path)
        return cls(
            backend=syncer._backend,
            resolver=FileResolver(root=syncer.path),
            syncer=syncer,
        )

    # --- internals ------------------------------------------------------

    @property
    def backend(self):
        """The index backend, bootstrapping lazily on first use."""
        if self._backend is None:
            self._backend = self._backend_factory()
        return self._backend

    def _view(self, queries, residuals) -> PatchCatalog:
        out = PatchCatalog(
            backend=self.backend,
            resolver=self.resolver,
            syncer=self._syncer,
            queries=queries,
            residuals=residuals,
        )
        return out

    def _invalidate(self) -> None:
        self._df_cache = None

    @property
    def is_view(self) -> bool:
        """True when this catalog carries selection state."""
        return bool(self._queries or self._residuals)

    def _require_root(self, operation: str) -> None:
        if self.is_view:
            msg = f"{operation} is only allowed on a root catalog, not a view."
            raise InvalidSpoolQueryError(msg)

    # --- selection ------------------------------------------------------

    def select(
        self,
        *,
        _attrs: dict | None = None,
        _coords: dict | None = None,
        samples: bool = False,
        relative: bool = False,
        **kwargs,
    ) -> PatchCatalog:
        """
        Compose a selection; validation is eager, execution is lazy.

        samples=True selectors are patch-local (never index predicates);
        relative=True bounds resolve against the current view's global
        envelope, then behave as absolute ranges.
        """
        if samples:
            # names must be coords; validated against the index
            unknown = set(kwargs) - self.coord_names()
            if unknown or _attrs or _coords:
                msg = (
                    f"samples=True selections are coordinate-only; "
                    f"unknown coordinates: {sorted(unknown)}"
                )
                raise InvalidSpoolQueryError(msg)
            residual = (dict(kwargs), True, False)
            return self._view(self._queries, (*self._residuals, residual))
        if relative:
            kwargs = self._relative_to_absolute(kwargs)
        query = resolve_query(self.backend, _attrs=_attrs, _coords=_coords, **kwargs)
        # coord range predicates are re-applied exactly at patch load
        residuals = self._residuals
        if query.coords:
            residuals = (*residuals, (dict(query.coords), False, False))
        return self._view((*self._queries, query), residuals)

    def _relative_to_absolute(self, kwargs: dict) -> dict:
        """Resolve relative bounds against the view's global envelopes."""
        df = self.to_df()
        out = {}
        for name, value in kwargs.items():
            lo_col, hi_col = f"{name}_min", f"{name}_max"
            if lo_col not in df.columns or df.empty:
                msg = f"Cannot use relative select on unknown coord {name!r}."
                raise InvalidSpoolQueryError(msg)
            gmin, gmax = df[lo_col].min(), df[hi_col].max()
            if not (isinstance(value, tuple) and len(value) == 2):
                msg = f"relative=True requires (start, stop) ranges, got {value!r}."
                raise InvalidSpoolQueryError(msg)
            lo, hi = value
            out[name] = (
                _offset(gmin, gmax, lo),
                _offset(gmin, gmax, hi),
            )
        return out

    # --- realization ------------------------------------------------------

    def to_df(self) -> pd.DataFrame:
        """The flat patch-row relation under the composed selection."""
        if self._df_cache is None:
            self._df_cache = self.backend.query(list(self._queries) or None)
        return self._df_cache

    def __len__(self) -> int:
        return len(self.to_df())

    def get_patch(self, index: int) -> dc.Patch:
        """Materialize one patch: resolve, then exact two-stage trim."""
        row = self.to_df().iloc[index].to_dict()
        trim_hint = {}
        for coords, samples, _ in self._residuals:
            if not samples:
                trim_hint.update(
                    {k: v for k, v in coords.items() if isinstance(v, tuple)}
                )
        patch = self.resolver.resolve(row, **trim_hint)
        for coords, samples, relative in self._residuals:
            usable = {k: v for k, v in coords.items() if k in patch.coords.coord_map}
            if usable:
                patch = patch.select(**usable, samples=samples, relative=relative)
        return patch

    def __iter__(self):
        for index in range(len(self)):
            yield self.get_patch(index)

    # --- mutation (root only) ----------------------------------------------

    def add(self, patches: Sequence[dc.Patch] | dc.Patch) -> PatchCatalog:
        """Add live patches to the catalog."""
        self._require_root("add")
        if not isinstance(self.resolver, LiveResolver):
            msg = "add() currently supports in-memory catalogs only."
            raise NotImplementedError(msg)
        patches = [patches] if isinstance(patches, dc.Patch) else list(patches)
        self.backend.write_sources(_live_records(patches, self.resolver))
        self._invalidate()
        return self

    def update(self, progress: PROGRESS_LEVELS = "standard") -> PatchCatalog:
        """Sync a directory-backed catalog with the filesystem."""
        self._require_root("update")
        if self._syncer is not None:
            self._syncer.update(progress=progress)
        self._invalidate()
        return self

    def remove(self, source_paths: Sequence[str], base_uri: str = "") -> PatchCatalog:
        """Remove sources (and their patches) from the catalog."""
        self._require_root("remove")
        self.backend.delete_sources(list(source_paths), base_uri=base_uri)
        self._invalidate()
        return self

    # --- introspection -------------------------------------------------------

    def attr_names(self) -> set[str]:
        """Attr names known to the index."""
        return self.backend.attr_names()

    def coord_names(self) -> set[str]:
        """Coord names known to the index."""
        return self.backend.coord_names()

    def sources(self) -> pd.DataFrame:
        """The sources table."""
        return self.backend.get_sources()

    def get_metadata(self) -> dict:
        """Index-level metadata."""
        return self.backend.get_metadata()

    def close(self) -> None:
        """Close the backend (root and all views share it)."""
        if self._backend is not None:
            self._backend.close()


def _offset(gmin, gmax, value):
    """Resolve one relative bound against a global envelope."""
    if value is None or value is Ellipsis:
        return None
    if isinstance(gmin, pd.Timestamp) or isinstance(gmin, np.datetime64):
        delta = dc.to_timedelta64(abs(value))
        return (gmin + delta) if value >= 0 else (gmax - delta)
    return (gmin + value) if value >= 0 else (gmax + value)
