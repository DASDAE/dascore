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
import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import pandas as pd

import dascore as dc
from dascore.constants import PROGRESS_LEVELS, namespace_select_type
from dascore.core.summary import normalize_source_patch_key
from dascore.exceptions import MissingPatchError
from dascore.io.core import _resolve_read_spool
from dascore.io.index.backend import get_backend
from dascore.io.index.indexer import DBDirectoryIndexer
from dascore.io.index.ingest import SourceRecord, patch_record, summaries_to_records
from dascore.io.index.query import (
    CoordExists,
    InvalidSpoolQueryError,
    Query,
    resolve_query,
)
from dascore.io.index.schema import SPOOL_LATE_RENAMES
from dascore.utils.misc import (
    _canonical_range,
    express_range_for_coord,
    is_range,
    order_range_tuple,
)
from dascore.utils.patch import record_call
from dascore.utils.paths import is_memory_uri
from dascore.utils.pd import yield_range_tuple_from_kwargs

# Directory archives present in per-patch time order (source ordinals
# alone cannot interleave multi-patch files); ordinal and patch id stay
# the deterministic tiebreak inside the ORDER BY.
_DIRECTORY_ORDER = ("coord", "time", True)


def _canonical_coord_selectors(backend, coords: dict) -> tuple[dict, dict]:
    """
    Split coordinate selectors into query-side and residual-side forms.

    The query side keeps the *original* values: the SQL builder coerces
    quantities itself and needs their units to constrain candidacy to
    dimensionally compatible coordinate definitions (a metre query must
    exclude — or raise on — a seconds coordinate, never trim it).

    A bare numeric range means each coordinate's native units, so it is
    its own residual: `Patch.select` reads it natively at load, exactly
    matching the index candidacy. A range with any unit-bearing bound
    becomes a `_CanonicalRange` (canonical SI magnitude and base unit
    per bound, bare bounds staying raw) so each patch decides its own
    representation at load time, which keeps mixed unitful/unitless
    populations correct.

    Selectors on non-numeric coordinates (time ranges, string ranges)
    pass through unchanged.
    """
    meta = backend.coord_meta(set(coords))
    numeric = set(meta.loc[meta["value_kind"] == "num", "coord_name"])
    query_coords, residual_coords = {}, {}
    for name, value in coords.items():
        canonical = _canonical_range(value) if name in numeric else None
        query_coords[name] = value
        residual_coords[name] = value if canonical is None else canonical
    return query_coords, residual_coords


def _reject_string_coords(backend, coords: dict) -> None:
    """Refuse a relative selection a string coordinate cannot answer.

    `Patch.select` has no offset arithmetic for strings, so such a view
    could only raise at load. The index records each definition's kind,
    and a name spelled as a string everywhere can be answered now.
    """
    meta = backend.coord_meta(set(coords))
    for name in coords:
        kinds = set(meta.loc[meta["coord_name"] == name, "value_kind"])
        if kinds and kinds <= {"str"}:
            msg = f"String coordinate {name!r} does not support relative selection."
            raise InvalidSpoolQueryError(msg)


def _row_source_patch_key(row: Mapping) -> str:
    """Return the row's source_patch_key as a normalized string."""
    return normalize_source_patch_key(row.get("source_patch_key"))


def apply_exact_residuals(patch: dc.Patch, residuals, hinted=()) -> dc.Patch:
    """
    Apply a view's exact residual selections to a loaded patch.

    Shared by catalog row resolution and plan-member loading so the
    two-stage select contract has exactly one implementation.

    ``hinted`` runs parallel to ``residuals``, naming per residual the
    coordinates whose bounds went to the reader and cut this row. Such a
    selection leaves its `select` nothing to do, and a call which
    changes nothing records nothing, so it is recorded here instead and
    a trimmed patch does not state the untrimmed patch's id. A shorter
    sequence records nothing for the residuals past its end, which is
    what a caller reading no hints out of the row wants.
    """
    for index, (coords, samples, relative) in enumerate(residuals):
        coord_map = patch.coords.coord_map
        usable = {
            k: express_range_for_coord(v, coord_map[k])
            for k, v in coords.items()
            if k in coord_map
        }
        if usable:
            called = dict(usable)
            if samples:
                called["samples"] = True
            if relative:
                called["relative"] = True
            out = patch.select(**called)
            cut = hinted[index] if index < len(hinted) else ()
            if out is patch and any(name in cut for name in usable):
                out = record_call(out, patch, dc.Patch.select, (), called)
            patch = out
    return patch


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

    def live_entries(self) -> dict[str, dc.Patch]:
        """
        Return the live patches this resolver serves (path -> patch).

        This is the registry itself, not a copy: dropping an entry from it
        is how a removed source's live patch stops being served.
        """
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

    def live_entries(self) -> dict[str, dc.Patch]:
        """Return the live patch registry."""
        return self._registry

    def resolve(self, row: Mapping, **trim) -> dc.Patch:
        """Look the patch up; live patches ignore trim hints."""
        path = str(row["source_path"])
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

    def _read(self, path, row: Mapping, trim: dict, source_patch_key: str):
        """
        Read one row's patch through dc.read.

        The recorded format/version are forwarded so dc.read skips format
        probing; it reads the file exactly once (an earlier fast path that
        called the reader directly re-read the file whenever the reader
        returned patches lazily).
        """
        id_kwargs = {"source_patch_key": source_patch_key} if source_patch_key else {}
        kwargs = {"path": path}
        if row.get("source_format"):
            kwargs["file_format"] = row["source_format"]
        if row.get("source_version"):
            kwargs["file_version"] = row["source_version"]
        return dc.read(**kwargs, **id_kwargs, **trim)

    def resolve_path(self, path: str | Path) -> str | Path:
        """
        Resolve a row's source path against the catalog root.

        Relative paths resolve against the root; URIs and absolute paths
        pass through untouched.
        """
        if self._root is not None and "://" not in str(path):
            if not Path(path).is_absolute():
                return self._root / path
        return path

    def resolve(self, row: Mapping, **trim) -> dc.Patch:
        """Read the patch, passing range trims down as read hints."""
        path = self.resolve_path(row["source_path"])
        source_patch_key = _row_source_patch_key(row)
        if source_patch_key.isdigit():
            # Positional (synthesized) ids index the full source read; a
            # trimmed read would shift or drop patches and bind the wrong
            # one, so these rows read the whole source.
            trim = {}
        spool = self._read(path, row, trim, source_patch_key)
        patch = _resolve_read_spool(spool, source_patch_key)
        # Hive-style path attrs override what the file recorded (renaming
        # a path is the user's way of correcting metadata), so loaded
        # patches agree with the index contents.
        raw_path_attrs = row.get("_path_attrs")
        if isinstance(raw_path_attrs, str) and raw_path_attrs:
            patch = patch.update_attrs(**json.loads(raw_path_attrs))
        return patch


class CompositeResolver(PatchResolver):
    """
    Route rows to a live registry, a plan, or the filesystem by scheme.

    Union catalogs mix file-backed rows (absolute paths), in-memory rows
    (memory:// paths), and plan-output rows (plan://token/... paths);
    this resolver dispatches accordingly.
    """

    def __init__(self):
        self.live = LiveResolver()
        self.file = FileResolver(root=None)
        # plan://<token>/ prefix -> the PlanResolver that owns it
        self.plans: dict[str, PatchResolver] = {}

    def live_entries(self) -> dict[str, dc.Patch]:
        """Return the merged live patch registry."""
        return self.live._registry

    def plan_entries(self) -> Mapping[str, PatchResolver]:
        """Return the plan-prefix routing table."""
        return self.plans

    def absorb(self, resolver: PatchResolver, paths=None) -> None:
        """
        Take over another resolver's live and plan entries.

        ``paths`` restricts absorption to the given synthetic paths
        (the entries a transfer actually references); None takes all.
        """
        entries = resolver.live_entries()
        if paths is not None:
            entries = {k: v for k, v in entries.items() if k in paths}
        self.live._registry.update(entries)
        plans = getattr(resolver, "plan_entries", dict)()
        if paths is not None:
            plans = {
                prefix: plan
                for prefix, plan in plans.items()
                if any(str(p).startswith(prefix) for p in paths)
            }
        self.plans.update(plans)

    def resolve(self, row: Mapping, **trim) -> dc.Patch:
        """Dispatch by scheme: live registry, plan, or file reader."""
        path = str(row.get("source_path", ""))
        if is_memory_uri(path):
            return self.live.resolve(row, **trim)
        for prefix, plan in self.plans.items():
            if path.startswith(prefix):
                return plan.resolve(row, **trim)
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


def _membership_resolver(
    resolver: PatchResolver, keep: dict, paths=()
) -> PatchResolver:
    """
    Return a copy of resolver whose live registry holds only `keep`.

    ``paths`` are the synthetic paths the view's rows reference: plan
    routes serving any of them must survive the restriction or mixed
    planned/live views lose their plan-backed rows on serialization.
    """
    if isinstance(resolver, LiveResolver):
        out = LiveResolver()
        out._registry = dict(keep)
        return out
    out = CompositeResolver()
    out.live._registry = dict(keep)
    plans = getattr(resolver, "plan_entries", dict)()
    out.plans = {
        prefix: plan
        for prefix, plan in plans.items()
        if any(str(p).startswith(prefix) for p in paths)
    }
    return out


def _merge_source_records(existing, new):
    """
    Merge two partial records for the same source.

    Union members export only their selected patches, so two members can
    hold disjoint (or overlapping) slices of one multi-patch file. The
    merged record unions the patch lists by source_patch_key: a patch
    keeps its first-occurrence position, a duplicate identity takes the
    last occurrence's metadata (dict-merge semantics, matching the
    ordering contract), and the source-level metadata (mtime, size)
    comes from the last record.
    """
    if existing is None:
        return new
    patches = {p.source_patch_key: p for p in existing.patches}
    patches.update({p.source_patch_key: p for p in new.patches})
    return replace(new, patches=tuple(patches.values()))


@dataclass
class _CatalogRevision:
    """Shared mutation revision, and its lock, for live catalog views."""

    value: int = 0
    # A root and all its views share this; it guards the revision counter
    # and every cache keyed on it.
    lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    def __getstate__(self) -> dict:
        """Pickle the revision without its process-local lock."""
        return {"value": self.value}

    def __setstate__(self, state: dict) -> None:
        """Restore the revision with a fresh process-local lock."""
        self.value = state["value"]
        self.lock = RLock()


@dataclass
class _RevisionCache:
    """
    A value cached against the catalog revision which produced it.

    Reads and writes happen under the revision lock; a value built at an
    older revision is simply not returned.
    """

    value: Any = None
    revision: int = -1

    def get(self, revision: int):
        """Return the cached value, or None when empty or stale."""
        return self.value if self.revision == revision else None

    def set(self, value, revision: int):
        """Store (and return) a value built at the given revision."""
        self.value, self.revision = value, revision
        return value

    def clear(self) -> None:
        """Drop the cached value."""
        self.value, self.revision = None, -1


class _Keep:
    """Sentinel: _view keeps the current order/ids spec unless told otherwise.

    A dedicated class rather than a bare object so that testing against it
    narrows the parameter to the spec type it otherwise holds.
    """


_KEEP = _Keep()


def _rides_the_envelopes(value) -> bool:
    """True when a coord selector is one `adjust_segments` folds in."""
    # A unit-bearing range arrives as a `_CanonicalRange`, which
    # `_adjust_unit_segments` folds in per row unit.
    return getattr(value, "magnitudes", None) is not None or is_range(value)


def _residual_cuts_unmarked_rows(residuals) -> bool:
    """
    True when a residual selection can trim a row nothing marks trimmed.

    A value range rides along with a query whose bounds `to_df` folds
    into the presented envelopes, so `_modified` already names every row
    it cuts. Sample indices have no envelope to fold into, and a
    selector which is not a range never reached `adjust_segments`;
    either trims at load with the row still claiming to be whole.
    """
    return any(
        samples or relative or not all(_rides_the_envelopes(x) for x in coords.values())
        for coords, samples, relative in residuals
    )


# What a row states about the whole of its source patch, which a trim
# leaves untrue. `patch_id` is deliberately not here; see below.
_FORGOTTEN_ON_TRIM = ("_data_size", "processing_id")


def _forget_what_a_trim_invalidates(df: pd.DataFrame, residuals=()) -> pd.DataFrame:
    """
    Blank what a trimmed row states about the whole of its source.

    A trimmed row describes fewer samples than its source patch holds,
    and how many is known only once the trim is applied, so it states no
    size rather than the source's. `processing_id` goes the same way for
    the same reason: a trim is an operation, the patch which comes back
    carries the id that operation leads to, and the stored one names the
    patch on disk. Attribute queries still match the stored source id;
    clearing this presented value does not change SQL candidacy.

    `patch_id` stays. A trim does not change which data this is, so the
    stored id is still the loaded patch's and selecting on it still
    finds the row -- the two ids parting company here is what having two
    of them is for.

    A row a selection leaves whole keeps both: only what a selection
    actually cuts loses what the cut invalidates.
    """
    present = [x for x in _FORGOTTEN_ON_TRIM if x in df.columns]
    if not present:
        return df
    if _residual_cuts_unmarked_rows(residuals):
        trimmed = np.ones(len(df), dtype=bool)
    elif "_modified" in df.columns:
        trimmed = df["_modified"].to_numpy(dtype=bool)
    else:
        return df
    if not trimmed.any():
        return df
    forgotten = {name: df[name].where(~trimmed) for name in present}
    if "_data_size" in forgotten:
        # nullable rather than float: a sample count is a count, and the
        # column is compared and presented as one.
        forgotten["_data_size"] = df["_data_size"].astype("Int64").where(~trimmed)
    return df.assign(**forgotten)


# One bit per residual. A view composed of more selections than this
# states no mask, and its rows fall back to `_modified`.
_MASK_BITS = 63

_CUT_MASK_SUFFIX = "_cut_mask"


def _cut_mask_column(name: str) -> str:
    """The private column saying which residuals cut a coordinate."""
    return f"_{name}{_CUT_MASK_SUFFIX}"


def _is_reader_hintable(value) -> bool:
    """
    Whether a coordinate selector can be pushed into the reader.

    Readers take numbers in their coordinate's own units, so a bound
    converted from other units could narrow the read past what exactness
    can restore. Only a bare range, meaning native units on both sides,
    is safe to send.
    """
    return isinstance(value, tuple) and not any(
        hasattr(bound, "units") for bound in value
    )


def _residual_cut_masks(df: pd.DataFrame, residuals) -> dict[str, np.ndarray]:
    """
    Per row and coordinate, which residuals actually narrow it.

    Bit ``i`` is set where residual ``i`` cuts that row's coordinate.
    The walk mirrors `apply_exact_residuals`: each bound is judged
    against what the bounds before it left, exactly as each `select` is
    applied to what the selects before it returned. A bound which cuts
    nothing is then not credited with the cut of one which does, which
    the row's single `_modified` flag cannot tell apart.

    Only reader-hintable bounds are tracked, because only those can
    reach the patch already applied and so need a bit to be recorded at
    all; a sample-index or unit-bearing bound cuts for itself. What such
    a bound leaves behind is not modelled, so a coordinate one touches
    states no mask rather than a mask read off an envelope which is no
    longer true.
    """
    masks: dict[str, np.ndarray] = {}
    if len(residuals) > _MASK_BITS:
        return masks
    live: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    untracked: set[str] = set()
    for index, (coords, samples, relative) in enumerate(residuals):
        for name, value in coords.items():
            if name in untracked:
                continue
            if samples or relative or not _is_reader_hintable(value):
                untracked.add(name)
                masks.pop(name, None)
                live.pop(name, None)
                continue
            bounds = dict(yield_range_tuple_from_kwargs(df, {name: value}))
            if name not in bounds:
                continue
            if name not in live:
                start, stop = df[f"{name}_min"], df[f"{name}_max"]
                if not _orderable(start) or not _orderable(stop):
                    untracked.add(name)
                    continue
                live[name] = (start.to_numpy(), stop.to_numpy())
                masks[name] = np.zeros(len(df), dtype=np.int64)
            low, high = order_range_tuple(bounds[name])
            start, stop = live[name]
            cut = np.zeros(len(df), dtype=bool)
            if low is not None:
                cut |= start < low
                start = np.maximum(start, low)
            if high is not None:
                cut |= stop > high
                stop = np.minimum(stop, high)
            live[name] = (start, stop)
            masks[name] |= cut.astype(np.int64) << index
    return masks


def _orderable(series: pd.Series) -> bool:
    """Whether an envelope column supports the min/max walk of the mask."""
    return bool(
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
        or pd.api.types.is_timedelta64_dtype(series)
    )


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
        resolver: PatchResolver,
        backend=None,
        syncer=None,
        queries: tuple[Query, ...] = (),
        residuals: tuple[tuple[dict, bool, bool], ...] = (),
        revision: _CatalogRevision | None = None,
        order: tuple | None = None,
        ids: tuple | None = None,
        default_order: tuple | None = None,
    ):
        self._backend = backend
        self.resolver = resolver
        self._syncer = syncer
        self._queries = tuple(queries)
        self._residuals = tuple(residuals)
        # presentation specs (D2): an order override ("attr"|"coord",
        # name, ascending) and/or an ordered patch-id membership
        self._order = order
        # the catalog's own presentation contract when no user order is
        # set (directory archives present in per-patch time order —
        # source ordinals alone cannot interleave multi-patch files).
        # Not view state: a root with a default order still updates.
        self._default_order = default_order
        self._ids = None if ids is None else tuple(int(x) for x in ids)
        self._revision = revision or _CatalogRevision()
        self._df_cache = _RevisionCache()
        self._live_cache = _RevisionCache()
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
        for catalog in catalogs:
            # a view transfers only the rows it presents; a root transfers
            # all of them, which lets the whole table move as-is
            patch_ids = (
                catalog.to_df()["_patch_id"].tolist() if catalog.is_view else None
            )
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
        syncer = DBDirectoryIndexer(path, index_path=index_path)
        return cls(
            backend=syncer._backend,
            resolver=FileResolver(root=syncer.path),
            syncer=syncer,
            default_order=_DIRECTORY_ORDER,
        )

    # --- internals ------------------------------------------------------

    @property
    def backend(self):
        """
        The index backend, bootstrapping lazily on first use.

        Every metadata operation funnels through here, so this is also
        where a brand-new directory index gets its one automatic update.
        Bootstrapping runs under the revision lock so concurrent first
        use cannot build (or ingest into) two backends.

        The one-time directory scan stays under the lock as well: it is
        what serializes the build, and there is nothing for a blocked
        reader to read until it finishes. Once done, ensure_updated is a
        flag check. A later explicit update() re-scans an index which is
        already usable, so that one runs outside the lock.
        """
        with self._revision.lock:
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
        resolver = self.resolver
        if self.is_view and resolver.live_entries():
            df = self.to_df()
            paths = list(dict.fromkeys(df["source_path"].astype(str)))
            entries = resolver.live_entries()
            keep = {k: entries[k] for k in paths if k in entries}
            state["resolver"] = _membership_resolver(resolver, keep, paths)
        return state

    def _view(
        self,
        queries,
        residuals,
        order: tuple | _Keep | None = _KEEP,
        ids: tuple | _Keep | None = _KEEP,
    ) -> PatchCatalog:
        out = PatchCatalog(
            backend=self.backend,
            resolver=self.resolver,
            syncer=self._syncer,
            queries=queries,
            residuals=residuals,
            revision=self._revision,
            order=self._order if isinstance(order, _Keep) else order,
            ids=self._ids if isinstance(ids, _Keep) else ids,
            default_order=self._default_order,
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

    @property
    def _effective_order(self) -> tuple | None:
        """The presentation order: a user order spec, else the default."""
        return self._order if self._order is not None else self._default_order

    def transfer_is_lossy(self) -> bool:
        """
        Whether a table-level transfer of this view would lose state.

        Row membership (attr predicates, windows, id arrays) survives a
        table union as-is, but residual trims and order specs live
        Python-side and would silently vanish. A catalog default order
        (directory time presentation) is lost only when ordinal-grain
        transfer would actually present rows differently — an
        interleaved multi-patch file — so ordinary archives keep
        record-grain transfer and its same-source deduplication.
        """
        if self._residuals or self._order is not None:
            return True
        if self._default_order is None:
            return False
        by_ordinal = tuple(
            self.backend.query_ids(
                list(self._queries) or None,
                order_by=None,
                patch_ids=self._ids,
            )
        )
        return tuple(self.ordered_ids()) != by_ordinal

    @property
    def residuals(self) -> tuple[tuple[dict, bool, bool], ...]:
        """
        The selections applied per patch when it is materialized.

        Each is a ``(coords, samples, relative)`` tuple: the coordinate
        selectors SQL could not answer exactly, whether they index samples,
        and whether their bounds are relative to each patch.
        """
        return self._residuals

    @property
    def syncer(self):
        """The directory syncer keeping this catalog current, or None."""
        return self._syncer

    def ordered_ids(self) -> tuple[int, ...]:
        """
        The view's patch ids in presentation order (ids only, cheap).

        A fixed membership is its own presentation order — an integer
        array may have reordered it — so predicates composed *after* it
        was fixed filter that order rather than replacing it. Returning
        the membership unfiltered would ignore them entirely, and letting
        SQL order the result would undo the arrangement.
        """
        if self._ids is not None and self._order is None:
            if not self._queries:
                return self._ids
            matched = set(
                self.backend.query_ids(list(self._queries), patch_ids=self._ids)
            )
            return tuple(x for x in self._ids if x in matched)
        return tuple(
            self.backend.query_ids(
                list(self._queries) or None,
                order_by=self._effective_order,
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
        ids = self.ordered_ids()[item]
        return self._view(self._queries, self._residuals, ids=tuple(ids))

    def restrict(self, indices, ids=None) -> PatchCatalog:
        """
        Return a view keeping the presented rows an array selects.

        ``indices`` is a boolean mask over rows or an array of integer
        positions (order-preserving; duplicate positions collapse to
        one row, matching the spool's set semantics). ``ids`` is this
        view's presented ids, for a caller which has just read them.
        """
        ids = np.asarray(self.ordered_ids() if ids is None else ids)
        picked = ids[np.asarray(indices)]
        deduped = tuple(dict.fromkeys(int(x) for x in picked))
        return self._view(self._queries, self._residuals, ids=deduped)

    def _invalidate(self) -> None:
        with self._revision.lock:
            self._revision.value += 1
            self._df_cache.clear()
            self._live_cache.clear()

    def _cold_live_values(self) -> tuple | None:
        """
        The patches, in registry (construction) order, when the registry
        alone defines contents — a root live catalog whose backend was
        never realized. None whenever the registry is not authoritative.

        This keeps len/iteration/indexing on freshly-built patch-list
        spools allocation-free: no ingest, no SQL, no flat relation.

        Callers must hold the revision lock.
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
        revision = self._revision.value
        if (live := self._live_cache.get(revision)) is None:
            live = self._live_cache.set(
                tuple(self.resolver.live_entries().values()), revision
            )
        return live

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
        _attrs: namespace_select_type = None,
        _coords: namespace_select_type = None,
        samples: bool = False,
        relative: bool = False,
        **kwargs,
    ) -> PatchCatalog:
        """
        Compose a selection; validation is eager, execution is lazy.

        samples=True selectors are patch-local and never become index
        predicates. Relative coordinate bounds remain patch-local, while an
        existence-only predicate excludes patches without the coordinate.
        """
        query = resolve_query(
            self.backend.attr_names(),
            self.backend.coord_names(),
            _attrs=_attrs,
            _coords=_coords,
            **kwargs,
        )
        if samples:
            if query.attrs:
                msg = (
                    "samples=True selections are coordinate-only; got attrs "
                    f"{sorted(query.attrs)}."
                )
                raise InvalidSpoolQueryError(msg)
            residual = (dict(query.coords), True, relative)
            return self._view(self._queries, (*self._residuals, residual))
        if relative and query.coords:
            _reject_string_coords(self.backend, query.coords)
            residual = (dict(query.coords), False, True)
            exists = {
                name: CoordExists(value if is_range(value) else None)
                for name, value in query.coords.items()
            }
            queries = (*self._queries, Query(attrs=query.attrs, coords=exists))
            return self._view(queries, (*self._residuals, residual))
        # coord range predicates are re-applied exactly at patch load;
        # bare ranges are their own residual (native units on both
        # sides), unit-bearing ones carry canonical quantities so each
        # patch converts them to its own units.
        residuals = self._residuals
        if query.coords:
            si_coords, residual_coords = _canonical_coord_selectors(
                self.backend, query.coords
            )
            query = Query(attrs=query.attrs, coords=si_coords)
            residuals = (*residuals, (residual_coords, False, False))
        return self._view((*self._queries, query), residuals)

    # --- realization ------------------------------------------------------

    def to_df(self) -> pd.DataFrame:
        """
        The spool-facing flat patch-row relation under the selection.

        Unique-per-patch structural columns (patch_id and friends) are
        hidden or renamed private so chunk merge-compatibility (which
        compares all non-private columns) is not spuriously blocked.
        """
        with self._revision.lock:
            if (cached := self._df_cache.get(self._revision.value)) is not None:
                return cached
            df = self.backend.query(
                list(self._queries) or None,
                order_by=self._effective_order,
                patch_ids=self._ids,
            )
            if self._ids is not None and self._order is None:
                # id membership presents in its own (window/array) order
                position = {pid: i for i, pid in enumerate(self._ids)}
                df = df.sort_values(
                    "_patch_id", key=lambda s: s.map(position), kind="stable"
                ).reset_index(drop=True)
            # The early ones are already done; see SPOOL_EARLY_RENAMES.
            df = df.rename(columns=dict(SPOOL_LATE_RENAMES))
            # Masks refer to this view's residuals, not its parent's.
            df = df.drop(
                columns=[
                    c
                    for c in df.columns
                    if str(c).startswith("_") and str(c).endswith(_CUT_MASK_SUFFIX)
                ]
            )
            if masks := _residual_cut_masks(df, self._residuals):
                df = df.assign(**{_cut_mask_column(k): v for k, v in masks.items()})
            # Circular import: chunk planning also uses the catalog.
            from dascore.utils.chunk_plan import (  # noqa: PLC0415
                patch_local_adjusted_envelopes,
            )

            df = patch_local_adjusted_envelopes(df, self._residuals, drop_empty=False)
            df = _forget_what_a_trim_invalidates(df, self._residuals)
            # Re-read the revision: bootstrapping the backend above can
            # bump it, and this frame reflects the state after that.
            return self._df_cache.set(df, self._revision.value)

    def __len__(self) -> int:
        with self._revision.lock:
            if (live := self._cold_live_values()) is not None:
                return len(live)
            if (df := self._df_cache.get(self._revision.value)) is not None:
                return len(df)
            # A range residual *after* a patch-local one can drop rows SQL
            # candidacy kept: the patch-local pass empties the envelope of a
            # patch its window misses, and the range pass then finds nothing
            # left to overlap. Only that order forces us to build the frame.
            patch_local = False
            for _, samples, relative in self._residuals:
                if samples or relative:
                    patch_local = True
                elif patch_local:
                    return len(self.to_df())
            # Otherwise SQL candidacy already accounts for every drop, so the
            # count matches len(to_df()) without projecting or pivoting.
            return self.backend.count(list(self._queries) or None, patch_ids=self._ids)

    def get_patch(self, index: int) -> dc.Patch:
        """Materialize one patch: resolve, then exact two-stage trim."""
        with self._revision.lock:
            if (live := self._cold_live_values()) is not None:
                return live[index]
            row = self.to_df().iloc[index].to_dict()
        # Reading (and trimming) the patch happens outside the lock; only
        # the row it starts from must come from a consistent relation.
        return self.resolve_row(row)

    def resolve_row(self, row: Mapping, extra_trim: Mapping | None = None) -> dc.Patch:
        """
        Resolve one flat-relation row and apply exact residual selects.

        extra_trim carries caller-side read hints (e.g. chunk instruction
        ranges) merged over the view's own residual ranges; like all trim
        hints they only reduce reading, exactness is re-applied above.
        """
        trim_hint = {}
        patch_local: set = set()
        for coords, samples, relative in self._residuals:
            if samples or relative:
                patch_local.update(coords)
                continue
            # Canonical-SI and quantity bounds stay out of reader hints:
            # readers take numbers in their native units, so a
            # converted-narrower hint could drop data exactness cannot
            # restore. A coordinate an earlier patch-local residual owns
            # stays out too -- that residual resolves against the source
            # patch, so narrowing the read first would move its bounds.
            trim_hint.update(
                {
                    k: v
                    for k, v in coords.items()
                    if k not in patch_local and _is_reader_hintable(v)
                }
            )
        trim_hint.update(extra_trim or {})
        for name in patch_local:
            trim_hint.pop(name, None)
        patch = self.resolver.resolve(row, **trim_hint)
        hinted = self._hinted_per_residual(row, set(trim_hint))
        return apply_exact_residuals(patch, self._residuals, hinted=hinted)

    def _hinted_per_residual(self, row: Mapping, hint_names: set[str]):
        """
        Per residual, which of its coordinates the reader already cut.

        A bound the reader applied leaves its `select` nothing to do, so
        whether that selection was a trim or a no-op has to be answered
        from the row. `to_df` answers it per residual in the cut masks;
        a row carrying none -- one from a caller which built it itself,
        or a coordinate the masks do not track -- falls back to the
        row's `_modified` flag, which cannot say which residual did the
        cutting and so credits every hinted one.
        """
        masks = {name: row.get(_cut_mask_column(name)) for name in hint_names}
        modified = bool(row.get("_modified"))
        out = []
        for index, (coords, samples, relative) in enumerate(self._residuals):
            names = set()
            if not samples and not relative:
                for name in coords:
                    if name not in hint_names:
                        continue
                    mask = masks.get(name)
                    if mask is None or pd.isnull(mask):
                        if modified:
                            names.add(name)
                    elif int(mask) >> index & 1:
                        names.add(name)
            out.append(names)
        return tuple(out)

    def __iter__(self):
        """
        Yield every patch under the selection.

        The relation is snapshotted once, then each patch is read and
        trimmed outside the revision lock. A patch which cannot be
        resolved is skipped with a warning rather than ending iteration
        (see #583); indexing with get_patch still raises.
        """
        with self._revision.lock:
            live = self._cold_live_values()
            df = None if live is not None else self.to_df()
        if live is not None:
            yield from live
            return
        assert df is not None  # the frame is fetched when there are no live values
        for index in range(len(df)):
            try:
                yield self.resolve_row(df.iloc[index].to_dict())
            except MissingPatchError as e:
                # Usually a coordinate mismatch trimmed the patch to nothing.
                msg = f"Skipping patch at index {index} (see #583): {e}"
                warnings.warn(msg, UserWarning, stacklevel=2)

    # --- mutation (root only) ----------------------------------------------

    def add(self, patches: Sequence[dc.Patch] | dc.Patch) -> PatchCatalog:
        """Add live patches to the catalog."""
        with self._revision.lock:
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
        # The scan itself is long-running file IO; it manages its own
        # state, so only the cache invalidation needs the revision lock.
        if self._syncer is not None:
            self._syncer.update(progress=progress)
        self._invalidate()
        return self

    def remove(self, source_paths: Sequence[str], base_uri: str = "") -> PatchCatalog:
        """Remove sources (and their patches) from the catalog."""
        with self._revision.lock:
            self._require_root("remove")
            source_paths = list(source_paths)
            self.backend.delete_sources(source_paths, base_uri=base_uri)
            # The live registry is the store for in-memory patches; it must
            # stay in step with the backend rows (pickling rebuilds from it).
            registry = self.resolver.live_entries()
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
        with self._revision.lock:
            if self._backend is not None:
                self._backend.close()
