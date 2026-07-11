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
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import dascore as dc
from dascore.constants import PROGRESS_LEVELS
from dascore.io.index.backend import get_backend, resolve_query
from dascore.io.index.ingest import SourceRecord, patch_record
from dascore.io.index.query import (
    InvalidSpoolQueryError,
    Query,
    relative_offset,
)
from dascore.utils.pd import adjust_segments

_counter = itertools.count()


def _row_source_patch_id(row: Mapping) -> str:
    """Return the row's source_patch_id as a string ("" when missing).

    Rows fetched through pandas represent missing text values as NaN,
    which is truthy, so a plain `or ""` does not normalize them.
    """
    value = row.get("source_patch_id")
    return "" if value is None or pd.isnull(value) else str(value)


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
        key = (row["path"], _row_source_patch_id(row))
        return self._registry[key]


class FileResolver(PatchResolver):
    """Load patches through dc.read; remoteness is the path layer's job."""

    def __init__(self, root: Path | str | None = None):
        self._root = Path(root) if root is not None else None

    def _read(self, path, row: Mapping, trim: dict, source_patch_id: str):
        """Use a known FiberIO directly, falling back to format detection."""
        from dascore.core.spool import MemorySpool

        file_format = row.get("file_format")
        file_version = row.get("file_version")
        id_kwargs = {"source_patch_id": source_patch_id} if source_patch_id else {}
        if file_format and file_version:
            fiber_io = dc.io.FiberIO.manager.get_fiberio(
                format=file_format, version=file_version
            )
            spool = fiber_io.read(path, **id_kwargs, **trim)
            if isinstance(spool, MemorySpool):
                return spool
        kwargs = {"path": path}
        if file_format:
            kwargs["file_format"] = file_format
        if file_version:
            kwargs["file_version"] = file_version
        return dc.read(**kwargs, **id_kwargs, **trim)

    def resolve(self, row: Mapping, **trim) -> dc.Patch:
        """Read the patch, passing range trims down as read hints."""
        from dascore.io.core import _select_patch_from_spool

        path = row["path"]
        # relative paths resolve against the catalog root; URIs and
        # absolute paths pass through untouched.
        if self._root is not None and "://" not in str(path):
            if not Path(path).is_absolute():
                path = self._root / path
        source_patch_id = _row_source_patch_id(row)
        if source_patch_id.isdigit():
            # Positional (synthesized) ids index the full source read; a
            # trimmed read would shift or drop patches and bind the wrong
            # one, so these rows read the whole source.
            trim = {}
        spool = self._read(path, row, trim, source_patch_id)
        # Readers that consume source_patch_id return the one requested
        # patch, sometimes without preserving reload metadata on it. Only
        # trust that when the patch doesn't claim a different identity.
        if source_patch_id and len(spool) == 1:
            found = str(spool[0].attrs.get("_source_patch_id", "") or "")
            if found in ("", source_patch_id):
                return spool[0]
        return _select_patch_from_spool(spool, source_patch_id=source_patch_id)


def _live_records(patches: Sequence[dc.Patch], resolver: LiveResolver):
    """Build source records for live patches with synthetic identities."""
    token = next(_counter)
    records = []
    for num, patch in enumerate(patches):
        # patch.summary is a cached_property: reuse fingerprints and
        # summaries the patch already computed instead of rebuilding.
        summary = patch.summary
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


@dataclass
class _CatalogRevision:
    """Shared mutation revision for live catalog views."""

    value: int = 0


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
        resolver: PatchResolver | None = None,
        syncer=None,
        pending: tuple = (),
        queries: tuple[Query, ...] = (),
        residuals: tuple[tuple[dict, bool, bool], ...] = (),
        revision: _CatalogRevision | None = None,
    ):
        self._backend = backend
        self.resolver = resolver
        self._syncer = syncer
        # live patches not yet ingested; kept (not a closure) so catalogs
        # pickle and can rebuild their backend after unpickling.
        self._pending = tuple(pending)
        self._queries = tuple(queries)
        self._residuals = tuple(residuals)
        self._revision = revision or _CatalogRevision()
        self._df_cache: pd.DataFrame | None = None
        self._df_cache_revision = -1

    # --- construction -------------------------------------------------

    @classmethod
    def from_patches(cls, patches: Sequence[dc.Patch] = ()) -> PatchCatalog:
        """
        Catalog over live patches. No backend work happens until the
        first metadata operation.
        """
        return cls(resolver=LiveResolver(), pending=tuple(patches))

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        index_path: str | Path | None = None,
    ) -> PatchCatalog:
        """Catalog over a directory of fiber files."""
        from dascore.io.index.indexer import DBDirectoryIndexer

        syncer = DBDirectoryIndexer(path, index_path=index_path)
        return cls(
            backend=syncer._backend,
            resolver=FileResolver(root=syncer.path),
            syncer=syncer,
        )

    # --- internals ------------------------------------------------------

    @property
    def backend(self):
        """
        The index backend, bootstrapping lazily on first use.

        Every metadata operation funnels through here, so this is also
        where a brand-new directory index gets its one automatic update.
        """
        if self._backend is None:
            self._backend = get_backend(":memory:")
            if self._pending:
                self._backend.write_sources(_live_records(self._pending, self.resolver))
        if self._syncer is not None and self._syncer.ensure_updated():
            self._invalidate()
        return self._backend

    def __getstate__(self) -> dict:
        """
        Pickle without the live DB connection.

        Live catalogs rebuild their backend from pending patches on next
        use; the resolver registry (which pickled rows reference) rides
        along unchanged, so already-realized views keep resolving.
        """
        state = dict(self.__dict__)
        state["_backend"] = None
        if isinstance(self.resolver, LiveResolver) and not state["_pending"]:
            # allow rebuilding the backend from the registered patches
            registry = self.resolver._registry
            state["_pending"] = tuple(registry.values())
        return state

    def _view(self, queries, residuals) -> PatchCatalog:
        out = PatchCatalog(
            backend=self.backend,
            resolver=self.resolver,
            syncer=self._syncer,
            queries=queries,
            residuals=residuals,
            revision=self._revision,
        )
        return out

    def _invalidate(self) -> None:
        self._revision.value += 1
        self._df_cache = None
        self._df_cache_revision = -1

    def __deepcopy__(self, memo) -> PatchCatalog:
        """
        Derived spools share the catalog (live registry + connection).

        DataFrameSpool copies spool state on select/chunk; catalog state
        is read-shared, matching the single-writer model.
        """
        return self

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
        query = resolve_query(self.backend, _attrs=_attrs, _coords=_coords, **kwargs)
        if samples:
            if query.attrs:
                msg = (
                    "samples=True selections are coordinate-only; got attrs "
                    f"{sorted(query.attrs)}."
                )
                raise InvalidSpoolQueryError(msg)
            residual = (dict(query.coords), True, False)
            return self._view(self._queries, (*self._residuals, residual))
        if relative and query.coords:
            query = Query(
                attrs=query.attrs,
                coords=self._relative_to_absolute(query.coords),
            )
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
                relative_offset(gmin, gmax, lo),
                relative_offset(gmin, gmax, hi),
            )
        return out

    # --- realization ------------------------------------------------------

    def to_df(self) -> pd.DataFrame:
        """
        The spool-facing flat patch-row relation under the selection.

        Unique-per-patch structural columns (patch_id and friends) are
        hidden or renamed private so chunk merge-compatibility (which
        compares all non-private columns) is not spuriously blocked.
        """
        if self._df_cache is None or self._df_cache_revision != self._revision.value:
            df = self.backend.query(list(self._queries) or None)
            df = df.drop(
                columns=["n_dims", "sample_count_total", "shape"], errors="ignore"
            ).rename(columns={"patch_id": "_patch_id"})
            # SQL identifies overlapping source patches. Expose the selected
            # envelopes, matching spool.get_contents() and the exact trim
            # applied when each patch is materialized. Each pass copies the
            # frame, so disjoint-name range sets collapse into one pass.
            range_dicts = [
                ranges
                for query in self._queries
                if (
                    ranges := {
                        name: value
                        for name, value in query.coords.items()
                        if isinstance(value, tuple) and len(value) == 2
                    }
                )
            ]
            names = [name for ranges in range_dicts for name in ranges]
            if range_dicts and len(set(names)) == len(names):
                range_dicts = [{k: v for d in range_dicts for k, v in d.items()}]
            for ranges in range_dicts:
                df = adjust_segments(df, ignore_bad_kwargs=True, **ranges)
            self._df_cache = df
            self._df_cache_revision = self._revision.value
        return self._df_cache

    def __len__(self) -> int:
        return len(self.to_df())

    def get_patch(self, index: int) -> dc.Patch:
        """Materialize one patch: resolve, then exact two-stage trim."""
        row = self.to_df().iloc[index].to_dict()
        return self.resolve_row(row)

    def resolve_row(self, row: Mapping, extra_trim: Mapping | None = None) -> dc.Patch:
        """
        Resolve one flat-relation row and apply exact residual selects.

        extra_trim carries caller-side read hints (e.g. chunk instruction
        ranges) merged over the view's own residual ranges; like all trim
        hints they only reduce reading, exactness is re-applied above.
        """
        trim_hint = {}
        for coords, samples, _ in self._residuals:
            if not samples:
                trim_hint.update(
                    {k: v for k, v in coords.items() if isinstance(v, tuple)}
                )
        trim_hint.update(extra_trim or {})
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
