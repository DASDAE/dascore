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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import dascore as dc
from dascore.constants import PROGRESS_LEVELS
from dascore.exceptions import MissingPatchError
from dascore.io.index.backend import get_backend, resolve_query
from dascore.io.index.ingest import SourceRecord, patch_record
from dascore.io.index.query import (
    InvalidSpoolQueryError,
    Query,
)
from dascore.utils.paths import is_memory_uri
from dascore.utils.pd import adjust_segments, relative_ranges_to_absolute


class _CanonicalRange:
    """
    A numeric coordinate range resolved to canonical SI magnitudes.

    The exact per-patch re-select defers its representation until the
    target patch is known: unit-bearing coordinates get quantities in
    the canonical unit (`Patch.select` converts them to native units),
    unitless coordinates get the bare magnitudes. A single eager form
    cannot serve both — raw numbers trim the wrong physical interval
    on non-SI patches, quantities break unitless coordinates.
    """

    __slots__ = ("magnitudes",)

    def __init__(self, magnitudes: tuple):
        self.magnitudes = magnitudes

    def __repr__(self) -> str:
        return f"_CanonicalRange({self.magnitudes!r})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, _CanonicalRange) and other.magnitudes == self.magnitudes
        )

    def for_patch_coord(self, coord) -> tuple:
        """Return the range in the representation this coord needs."""
        from dascore.units import get_quantity

        units = getattr(coord, "units", None)
        if units is None:
            return self.magnitudes
        base = get_quantity(str(units)).to_base_units().units
        return tuple(None if mag is None else mag * base for mag in self.magnitudes)


def _canonical_range(value) -> _CanonicalRange | None:
    """Return the canonical SI form of a numeric range, or None."""
    if not (isinstance(value, tuple) and len(value) == 2):
        return None
    magnitudes = []
    for bound in value:
        if bound is None or bound is Ellipsis:
            magnitudes.append(None)
        elif hasattr(bound, "units"):  # pint quantity -> SI magnitude
            magnitudes.append(float(bound.to_base_units().magnitude))
        elif isinstance(bound, bool | np.bool_):
            return None
        elif isinstance(bound, int | float | np.integer | np.floating):
            magnitudes.append(float(bound))
        else:  # datetimes, strings: not a numeric range
            return None
    if all(mag is None for mag in magnitudes):
        return None
    return _CanonicalRange(tuple(magnitudes))


def _canonical_coord_selectors(backend, coords: dict) -> tuple[dict, dict]:
    """
    Split coordinate selectors into query-side and residual-side forms.

    Numeric coordinate summaries are stored in canonical SI units, so
    numeric range bounds resolve to SI magnitudes for the index and
    dataframe side: bare numbers are already canonical SI (the index
    contract), quantities convert. The residual keeps the range as a
    `_CanonicalRange` so each patch decides its own representation at
    load time, which keeps mixed unitful/unitless populations correct.

    Selectors on non-numeric coordinates (time ranges, string ranges)
    and boolean masks pass through unchanged.
    """
    meta = backend._coord_meta(set(coords))
    numeric = set(meta.loc[meta["value_kind"] == "num", "coord_name"])
    si_coords, residual_coords = {}, {}
    for name, value in coords.items():
        canonical = _canonical_range(value) if name in numeric else None
        if canonical is None:
            si_coords[name] = residual_coords[name] = value
        else:
            si_coords[name] = canonical.magnitudes
            residual_coords[name] = canonical
    return si_coords, residual_coords


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

    def live_entries(self) -> Mapping[str, dc.Patch]:
        """Return the live patches this resolver serves (path -> patch)."""
        return {}


class LiveResolver(PatchResolver):
    """
    Serve patches from an in-memory registry.

    The registry *is* the store for live catalogs: a dict from each
    patch's synthetic path (`memorypatch://<instance_id>`) to the patch
    itself. Dict construction deduplicates identical patch instances
    (set semantics by lineage), and merging catalogs unions the dicts.
    """

    def __init__(self, patches: Sequence[dc.Patch] = ()):
        self._registry: dict[str, dc.Patch] = {
            _patch_path(patch): patch for patch in patches
        }

    def live_entries(self) -> Mapping[str, dc.Patch]:
        """Return the live patch registry."""
        return self._registry

    def resolve(self, row: Mapping, **trim) -> dc.Patch:
        """Look the patch up; live patches ignore trim hints."""
        path = str(row["path"])
        try:
            return self._registry[path]
        except KeyError:
            msg = (
                f"The in-memory patch for {path} is not available in this "
                "session (e.g. the row came from a reopened index). "
                "In-memory patches only persist by writing them to files."
            )
            raise MissingPatchError(msg) from None


def _patch_path(patch: dc.Patch) -> str:
    """Return the synthetic source path identifying a live patch."""
    return f"memorypatch://{patch._instance_id}"


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
        from dascore.io.core import _resolve_read_spool

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
        return _resolve_read_spool(spool, source_patch_id)


class CompositeResolver(PatchResolver):
    """
    Route rows to a live registry or the filesystem by path scheme.

    Union catalogs mix file-backed rows (absolute paths) with in-memory
    rows (memory:// paths); this resolver dispatches accordingly.
    """

    def __init__(self):
        self.live = LiveResolver()
        self.file = FileResolver(root=None)

    def live_entries(self) -> Mapping[str, dc.Patch]:
        """Return the merged live patch registry."""
        return self.live._registry

    def absorb(self, resolver: PatchResolver) -> None:
        """Take over the live registry entries of another resolver."""
        self.live._registry.update(resolver.live_entries())

    def resolve(self, row: Mapping, **trim) -> dc.Patch:
        """Dispatch to the live registry or the file reader."""
        if is_memory_uri(row.get("path", "")):
            return self.live.resolve(row, **trim)
        return self.file.resolve(row, **trim)


def _live_records(registry: Mapping[str, dc.Patch]):
    """Build source records for live patches keyed by their identity."""
    records = []
    for path, patch in registry.items():
        # patch.summary is a cached_property: reuse fingerprints and
        # summaries the patch already computed instead of rebuilding.
        record = patch_record(patch.summary)
        records.append(
            SourceRecord(
                source_path=path,
                source_format="memory",
                format_version="",
                patches=(record,),
            )
        )
    return records


def _absolutize_record(record, root):
    """Return a source record whose relative path is resolved against root."""
    from dataclasses import replace

    path = record.source_path
    if "://" in path or Path(path).is_absolute():
        return record
    resolved = str(Path(root) / (path if path != "." else ""))
    return replace(record, source_path=resolved, base_uri=None)


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
        queries: tuple[Query, ...] = (),
        residuals: tuple[tuple[dict, bool, bool], ...] = (),
        revision: _CatalogRevision | None = None,
    ):
        self._backend = backend
        self.resolver = resolver
        self._syncer = syncer
        self._queries = tuple(queries)
        self._residuals = tuple(residuals)
        self._revision = revision or _CatalogRevision()
        self._df_cache: pd.DataFrame | None = None
        self._df_cache_revision = -1
        # Source records for rebuilding an in-memory backend (set by
        # __getstate__ so pickled catalogs survive losing the connection).
        self._rebuild_records: tuple = ()

    # --- construction -------------------------------------------------

    @classmethod
    def from_patches(cls, patches: Sequence[dc.Patch] = ()) -> PatchCatalog:
        """
        Catalog over live patches. No backend work happens until the
        first metadata operation.

        The resolver's registry is the store; identical patch instances
        collapse to a single entry (set semantics by lineage).
        """
        return cls(resolver=LiveResolver(patches))

    @classmethod
    def union(cls, catalogs: Sequence[PatchCatalog]) -> PatchCatalog:
        """
        Materialize several catalogs into one in-memory catalog.

        Metadata rows are merged table-to-table (coord definitions
        deduplicate by def key); file-backed rows get absolute paths so
        members with different roots coexist, and the same source
        appearing in several members keeps a single entry (last one
        wins). For catalog views, only the selected patches transfer —
        note this respects row membership, not range trims; re-select
        on the result for exact envelopes.
        """
        resolver = CompositeResolver()
        out = cls(resolver=resolver)
        backend = out.backend
        for member in catalogs:
            catalog, patch_ids = member if isinstance(member, tuple) else (member, None)
            if patch_ids is None and catalog.is_view:
                patch_ids = catalog.to_df()["_patch_id"].tolist()
            records = catalog.backend.export_records(patch_ids=patch_ids)
            root = getattr(catalog.resolver, "_root", None)
            if root is not None:
                records = [_absolutize_record(x, root) for x in records]
            backend.write_sources(records)
            resolver.absorb(catalog.resolver)
        out._invalidate()
        return out

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
            if self._syncer is not None:
                # Directory catalogs re-adopt the (unpickled) syncer's
                # backend; both must keep sharing one connection.
                self._backend = self._syncer._backend
            else:
                self._backend = get_backend(":memory:")
                if self._rebuild_records:
                    self._backend.write_sources(list(self._rebuild_records))
                    self._rebuild_records = ()
                elif registry := getattr(self.resolver, "_registry", None):
                    self._backend.write_sources(_live_records(registry))
        if self._syncer is not None and self._syncer.ensure_updated():
            self._invalidate()
        return self._backend

    def __getstate__(self) -> dict:
        """
        Pickle without the live DB connection.

        In-memory backends (live and union catalogs) ride along as
        source records and are re-ingested on next use; the resolver
        registry (the store for live patches) pickles with its patches.
        Directory catalogs rebuild from their index file instead.
        """
        state = dict(self.__dict__)
        state["_backend"] = None
        # Live catalogs rebuild from their registry without touching the
        # connection (which may belong to another thread during pickling);
        # other in-memory catalogs (e.g. unions) capture their rows.
        needs_records = (
            self._backend is not None
            and self._syncer is None
            and not isinstance(self.resolver, LiveResolver)
        )
        if needs_records:
            state["_rebuild_records"] = tuple(self._backend.export_records())
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
        # coord range predicates are re-applied exactly at patch load;
        # the residual carries canonical quantities so per-patch native
        # units are respected while the query side stays SI.
        residuals = self._residuals
        if query.coords:
            si_coords, residual_coords = _canonical_coord_selectors(
                self.backend, query.coords
            )
            query = Query(attrs=query.attrs, coords=si_coords)
            residuals = (*residuals, (residual_coords, False, False))
        return self._view((*self._queries, query), residuals)

    def _relative_to_absolute(self, kwargs: dict) -> dict:
        """Resolve relative bounds against the view's global envelopes."""
        return relative_ranges_to_absolute(self.to_df(), kwargs)

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
        # Count in SQL when the relation is not already realized: coord
        # range residuals only drop patches the SQL candidacy already
        # excludes and samples/relative residuals never drop patches, so
        # the count matches len(to_df()) without projecting or pivoting.
        if (
            self._df_cache is not None
            and self._df_cache_revision == self._revision.value
        ):
            return len(self._df_cache)
        return self.backend.count(list(self._queries) or None)

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
                # Canonical-SI and quantity bounds stay out of reader
                # hints: readers take numbers in their native units, so
                # a converted-narrower hint could drop data exactness
                # cannot restore.
                trim_hint.update(
                    {
                        k: v
                        for k, v in coords.items()
                        if isinstance(v, tuple)
                        and not any(hasattr(b, "units") for b in v)
                    }
                )
        trim_hint.update(extra_trim or {})
        patch = self.resolver.resolve(row, **trim_hint)
        for coords, samples, relative in self._residuals:
            coord_map = patch.coords.coord_map
            usable = {
                k: (
                    v.for_patch_coord(coord_map[k])
                    if isinstance(v, _CanonicalRange)
                    else v
                )
                for k, v in coords.items()
                if k in coord_map
            }
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
        additions = {_patch_path(x): x for x in patches}
        self.resolver._registry.update(additions)
        # Re-adding a patch replaces its row (same identity), so this
        # stays idempotent.
        self.backend.write_sources(_live_records(additions))
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
        source_paths = list(source_paths)
        self.backend.delete_sources(source_paths, base_uri=base_uri)
        # The live registry is the store for in-memory patches; it must
        # stay in step with the backend rows (pickling rebuilds from it).
        registry = self.resolver.live_entries() if self.resolver else {}
        for path in source_paths:
            registry.pop(path, None)
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
