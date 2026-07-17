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
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import dascore as dc
from dascore.constants import PROGRESS_LEVELS
from dascore.core.summary import normalize_source_patch_id
from dascore.exceptions import MissingPatchError
from dascore.io.index.backend import get_backend, resolve_query
from dascore.io.index.ingest import SourceRecord, patch_record
from dascore.io.index.query import (
    InvalidSpoolQueryError,
    Query,
)
from dascore.io.index.schema import SPOOL_HIDDEN_COLUMNS
from dascore.utils.misc import is_range
from dascore.utils.paths import is_memory_uri
from dascore.utils.pd import adjust_segments, relative_ranges_to_absolute


class _CanonicalRange:
    """
    A numeric coordinate range resolved to canonical SI magnitudes.

    The exact per-patch re-select defers its representation until the
    target patch is known: unit-bearing coordinates get quantities
    (`Patch.select` converts them to native units), unitless
    coordinates get the bare magnitudes. A single eager form cannot
    serve both — raw numbers trim the wrong physical interval on non-SI
    patches, quantities break unitless coordinates.

    ``units`` records the query's own base unit when the original
    bounds carried one, so the residual preserves the query's
    dimensionality instead of adopting each patch coordinate's — a
    metre query must never trim a seconds coordinate as 1-2 s.
    """

    __slots__ = ("magnitudes", "units")

    def __init__(self, magnitudes: tuple, units: str | None = None):
        self.magnitudes = magnitudes
        self.units = units

    def __eq__(self, other) -> bool:
        """Value equality so equal selections compare equal (spool __eq__)."""
        if not isinstance(other, _CanonicalRange):
            return NotImplemented
        return (self.magnitudes, self.units) == (other.magnitudes, other.units)

    def __hash__(self) -> int:
        return hash((self.magnitudes, self.units))

    def for_patch_coord(self, coord) -> tuple:
        """Return the range in the representation this coord needs."""
        from dascore.units import get_quantity

        coord_units = getattr(coord, "units", None)
        if coord_units is None:
            # unitless coords: bare canonical magnitudes (documented policy)
            return self.magnitudes
        # a unit-bearing query keeps its own dimensionality; a bare
        # numeric query means canonical SI in the coord's dimension
        base = (
            get_quantity(self.units)
            if self.units is not None
            else get_quantity(str(coord_units)).to_base_units().units
        )
        return tuple(None if mag is None else mag * base for mag in self.magnitudes)


def _canonical_range(value) -> _CanonicalRange | None:
    """Return the canonical SI form of a numeric range, or None."""
    if not is_range(value):
        return None
    magnitudes = []
    units = None
    for bound in value:
        if bound is None or bound is Ellipsis:
            magnitudes.append(None)
        elif hasattr(bound, "units"):  # pint quantity -> SI magnitude
            base = bound.to_base_units()
            magnitudes.append(float(base.magnitude))
            units = str(base.units)
        elif isinstance(bound, bool | np.bool_):
            return None
        elif isinstance(bound, int | float | np.integer | np.floating):
            magnitudes.append(float(bound))
        else:  # datetimes, strings: not a numeric range
            return None
    if all(mag is None for mag in magnitudes):
        return None
    return _CanonicalRange(tuple(magnitudes), units)


def _envelope_range(value):
    """Return a range with quantity bounds as SI magnitudes.

    Stored envelope columns are canonical SI, so the presented-envelope
    adjustment needs bare magnitudes; non-numeric ranges pass through.
    """
    canonical = _canonical_range(value)
    return value if canonical is None else canonical.magnitudes


def _canonical_coord_selectors(backend, coords: dict) -> tuple[dict, dict]:
    """
    Split coordinate selectors into query-side and residual-side forms.

    The query side keeps the *original* values: the SQL builder coerces
    quantities itself and needs their units to constrain candidacy to
    dimensionally compatible coordinate definitions (a metre query must
    exclude — or raise on — a seconds coordinate, never trim it).
    The residual keeps the range as a `_CanonicalRange` (canonical SI
    magnitudes plus the query's base unit) so each patch decides its
    own representation at load time, which keeps mixed unitful/unitless
    populations correct.

    Selectors on non-numeric coordinates (time ranges, string ranges)
    pass through unchanged.
    """
    meta = backend._coord_meta(set(coords))
    numeric = set(meta.loc[meta["value_kind"] == "num", "coord_name"])
    query_coords, residual_coords = {}, {}
    for name, value in coords.items():
        canonical = _canonical_range(value) if name in numeric else None
        query_coords[name] = value
        residual_coords[name] = value if canonical is None else canonical
    return query_coords, residual_coords


def _row_source_patch_id(row: Mapping) -> str:
    """Return the row's source_patch_id as a normalized string."""
    return normalize_source_patch_id(row.get("source_patch_id"))


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
        """
        Read one row's patch through dc.read.

        The recorded format/version are forwarded so dc.read skips format
        probing; it reads the file exactly once (an earlier fast path that
        called the reader directly re-read the file whenever the reader
        returned patches lazily).
        """
        id_kwargs = {"source_patch_id": source_patch_id} if source_patch_id else {}
        kwargs = {"path": path}
        if row.get("file_format"):
            kwargs["file_format"] = row["file_format"]
        if row.get("file_version"):
            kwargs["file_version"] = row["file_version"]
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

    def absorb(self, resolver: PatchResolver, paths=None) -> None:
        """
        Take over another resolver's live registry entries.

        ``paths`` restricts absorption to the given synthetic paths
        (the entries a transfer actually references); None takes all.
        """
        entries = resolver.live_entries()
        if paths is not None:
            entries = {k: v for k, v in entries.items() if k in paths}
        self.live._registry.update(entries)

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
    path = record.source_path
    if "://" in path or Path(path).is_absolute():
        return record
    resolved = str(Path(root) / (path if path != "." else ""))
    return replace(record, source_path=resolved, base_uri=None)


def _membership_resolver(resolver: PatchResolver, keep: dict) -> PatchResolver:
    """Return a copy of resolver whose live registry holds only `keep`."""
    if isinstance(resolver, LiveResolver):
        out = LiveResolver()
        out._registry = dict(keep)
        return out
    out = CompositeResolver()
    out.live._registry = dict(keep)
    return out


def _merge_source_records(existing, new):
    """
    Merge two partial records for the same source.

    Union members export only their selected patches, so two members can
    hold disjoint (or overlapping) slices of one multi-patch file. The
    merged record unions the patch lists by source_patch_id: a patch
    keeps its first-occurrence position, a duplicate identity takes the
    last occurrence's metadata (dict-merge semantics, matching the
    ordering contract), and the source-level metadata (mtime, size)
    comes from the last record.
    """
    if existing is None:
        return new
    patches = {p.source_patch_id: p for p in existing.patches}
    patches.update({p.source_patch_id: p for p in new.patches})
    return replace(new, patches=tuple(patches.values()))


@dataclass
class _CatalogRevision:
    """Shared mutation revision for live catalog views."""

    value: int = 0


# sentinel: _view keeps the current order/ids spec unless told otherwise
_KEEP = object()


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
        residuals: tuple[tuple[dict, bool], ...] = (),
        revision: _CatalogRevision | None = None,
        order: tuple | None = None,
        ids: tuple | None = None,
    ):
        self._backend = backend
        self.resolver = resolver
        self._syncer = syncer
        self._queries = tuple(queries)
        self._residuals = tuple(residuals)
        # presentation specs (D2): an order override ("attr"|"coord",
        # name, ascending) and/or an ordered patch-id membership
        self._order = order
        self._ids = None if ids is None else tuple(int(x) for x in ids)
        self._revision = revision or _CatalogRevision()
        self._df_cache: pd.DataFrame | None = None
        self._df_cache_revision = -1
        self._live_cache: tuple | None = None
        self._live_cache_revision = -1
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
        # Collect and merge every member's records before writing:
        # write_sources replaces at (base_uri, source_path) grain, so
        # partial records for the same source — two members selecting
        # different patches of one multi-patch file — must merge into a
        # complete record or the later write would delete the earlier
        # member's patches. Dict insertion order keeps first-occurrence
        # position; the merge keeps last-occurrence metadata.
        merged: dict[tuple, SourceRecord] = {}
        for member in catalogs:
            catalog, patch_ids = member if isinstance(member, tuple) else (member, None)
            if patch_ids is None and catalog.is_view:
                patch_ids = catalog.to_df()["_patch_id"].tolist()
            records = catalog.backend.export_records(patch_ids=patch_ids)
            root = getattr(catalog.resolver, "_root", None)
            if root is not None:
                records = [_absolutize_record(x, root) for x in records]
            for record in records:
                identity = (record.base_uri or "", record.source_path)
                merged[identity] = _merge_source_records(merged.get(identity), record)
            # only the live entries this member actually transfers ride
            # along; the rest of the registry stays with its own catalog
            member_paths = {record.source_path for record in records}
            resolver.absorb(catalog.resolver, paths=member_paths)
        backend.write_sources(list(merged.values()))
        out._invalidate()
        return out

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        file_format: str | None = None,
        file_version: str | None = None,
    ) -> PatchCatalog:
        """
        Catalog over a single fiber file.

        The file is scanned eagerly (one row per contained patch) into an
        in-memory backend; patches load through the file resolver on
        demand. There is no syncer — a changed file needs a new catalog.
        """
        from dascore.io.index.ingest import summaries_to_records

        summaries = dc.scan(
            path, file_format=file_format, file_version=file_version, progress=None
        )
        records = summaries_to_records(summaries)
        out = cls(resolver=FileResolver())
        out.backend.write_sources(records)
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
        # In-memory backends are rebuilt on the other side with FRESH
        # patch ids, so a stored id membership would bind to the wrong
        # rows. Restrict the rebuilt content to the current membership
        # instead (records/registry in presentation order, so re-ingest
        # ordinals preserve it) and drop the id spec; the syncer case
        # reopens the same database file, where ids stay valid.
        rebuilt_membership = self._syncer is None and self._ids is not None
        if rebuilt_membership:
            state["_ids"] = None
        # Live catalogs rebuild from their registry without touching the
        # connection (which may belong to another thread during pickling);
        # other in-memory catalogs (e.g. unions) capture their rows.
        needs_records = (
            self._backend is not None
            and self._syncer is None
            and not isinstance(self.resolver, LiveResolver)
        )
        if needs_records:
            patch_ids = self._ids if rebuilt_membership else None
            state["_rebuild_records"] = tuple(
                self._backend.export_records(patch_ids=patch_ids)
            )
        # A view shares the root's resolver, but must not drag the whole
        # live registry across the wire: keep only the entries its rows
        # reference (a one-patch view of an N-patch spool serializes one
        # patch, not N — the payload Spool.map ships per task) — in
        # presentation order, so a rebuilt registry keeps the view's
        # ordering.
        if self.is_view and self.resolver.live_entries():
            df = self.to_df()
            paths = list(dict.fromkeys(df["path"].astype(str)))
            entries = self.resolver.live_entries()
            keep = {k: entries[k] for k in paths if k in entries}
            state["resolver"] = _membership_resolver(self.resolver, keep)
        return state

    def _view(self, queries, residuals, order=_KEEP, ids=_KEEP) -> PatchCatalog:
        out = PatchCatalog(
            backend=self.backend,
            resolver=self.resolver,
            syncer=self._syncer,
            queries=queries,
            residuals=residuals,
            revision=self._revision,
            order=self._order if order is _KEEP else order,
            ids=self._ids if ids is _KEEP else ids,
        )
        return out

    def order_by(self, attribute: str, ascending: bool = True) -> PatchCatalog:
        """
        Return a view presenting rows ordered by an attribute or coord.

        A lazy presentation spec (D2): realization adds ORDER BY with
        the ordinal contract as the deterministic tiebreak; no rows are
        copied and no relation is realized here.
        """
        name = str(attribute)
        coords = self.backend.coord_names()
        if name in coords:
            spec = ("coord", name, ascending)
        elif name in self.backend.attr_names():
            spec = ("attr", name, ascending)
        elif name.endswith("_min") and name.removesuffix("_min") in coords:
            spec = ("coord", name.removesuffix("_min"), ascending)
        else:
            msg = "Invalid attribute. Please use a valid attribute such as: 'time'"
            raise IndexError(msg)
        return self._view(self._queries, self._residuals, order=spec)

    def _ordered_ids(self) -> tuple[int, ...]:
        """The view's patch ids in presentation order (ids only, cheap)."""
        if self._ids is not None and self._order is None:
            return self._ids
        return tuple(
            self.backend.query_ids(
                list(self._queries) or None,
                order_by=self._order,
                patch_ids=self._ids,
            )
        )

    def window(self, item: slice) -> PatchCatalog:
        """
        Return a view restricted to a slice of the presented rows.

        Membership realizes as an ordered id list (ids only — never the
        flat relation); subsequent selections compose within the window
        per the D2 rules.
        """
        ids = self._ordered_ids()[item]
        return self._view(self._queries, self._residuals, ids=tuple(ids))

    def restrict(self, indices) -> PatchCatalog:
        """
        Return a view keeping the presented rows an array selects.

        ``indices`` is a boolean mask over rows or an array of integer
        positions (order-preserving; duplicate positions collapse to
        one row, matching the spool's set semantics).
        """
        ids = np.asarray(self._ordered_ids())
        picked = ids[np.asarray(indices)]
        deduped = tuple(dict.fromkeys(int(x) for x in picked))
        return self._view(self._queries, self._residuals, ids=deduped)

    def _invalidate(self) -> None:
        self._revision.value += 1
        self._df_cache = None
        self._df_cache_revision = -1
        self._live_cache = None
        self._live_cache_revision = -1

    def _cold_live_values(self) -> tuple | None:
        """
        The patches, in registry (construction) order, when the registry
        alone defines contents — a root live catalog whose backend was
        never realized. None whenever the registry is not authoritative.

        This keeps len/iteration/indexing on freshly-built patch-list
        spools allocation-free: no ingest, no SQL, no flat relation.
        """
        cold = (
            self._backend is None
            and self._syncer is None
            and not self.is_view
            and not self._rebuild_records
            and isinstance(self.resolver, LiveResolver)
        )
        if not cold:
            return None
        if (
            self._live_cache is None
            or self._live_cache_revision != self._revision.value
        ):
            self._live_cache = tuple(self.resolver.live_entries().values())
            self._live_cache_revision = self._revision.value
        return self._live_cache

    def __deepcopy__(self, memo) -> PatchCatalog:
        """
        Derived spools share the catalog (live registry + connection).

        Spool copies its state on select/chunk; catalog state
        is read-shared, matching the single-writer model.
        """
        return self

    @property
    def is_view(self) -> bool:
        """True when this catalog carries selection or presentation state."""
        return bool(
            self._queries
            or self._residuals
            or self._order is not None
            or self._ids is not None
        )

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
            residual = (dict(query.coords), True)
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
            residuals = (*residuals, (residual_coords, False))
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
            df = self.backend.query(
                list(self._queries) or None,
                order_by=self._order,
                patch_ids=self._ids,
            )
            if self._ids is not None and self._order is None:
                # id membership presents in its own (window/array) order
                position = {pid: i for i, pid in enumerate(self._ids)}
                df = df.sort_values(
                    "patch_id", key=lambda s: s.map(position), kind="stable"
                ).reset_index(drop=True)
            df = df.drop(columns=list(SPOOL_HIDDEN_COLUMNS), errors="ignore").rename(
                columns={"patch_id": "_patch_id"}
            )
            # SQL identifies overlapping source patches. Expose the selected
            # envelopes, matching spool.get_contents() and the exact trim
            # applied when each patch is materialized. Each pass copies the
            # frame, so disjoint-name range sets collapse into one pass.
            range_dicts = [
                ranges
                for query in self._queries
                if (
                    ranges := {
                        name: _envelope_range(value)
                        for name, value in query.coords.items()
                        if is_range(value)
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
        if (live := self._cold_live_values()) is not None:
            return len(live)
        # Count in SQL when the relation is not already realized: coord
        # range residuals only drop patches the SQL candidacy already
        # excludes and samples/relative residuals never drop patches, so
        # the count matches len(to_df()) without projecting or pivoting.
        if (
            self._df_cache is not None
            and self._df_cache_revision == self._revision.value
        ):
            return len(self._df_cache)
        return self.backend.count(list(self._queries) or None, patch_ids=self._ids)

    def get_patch(self, index: int) -> dc.Patch:
        """Materialize one patch: resolve, then exact two-stage trim."""
        if (live := self._cold_live_values()) is not None:
            return live[index]
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
        for coords, samples in self._residuals:
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
        for coords, samples in self._residuals:
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
                # residual bounds are already absolute (relative queries
                # resolve to absolute before the residual is recorded).
                patch = patch.select(**usable, samples=samples, relative=False)
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
