"""Deferred catalog view for independent absolute coordinate windows."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from dascore.io.index.catalog import PatchCatalog
from dascore.io.index.planned import derived_catalog
from dascore.utils.chunk_plan import (
    _ensure_patch_row,
    _trim_explicit,
    build_subdivision_plan,
    refuse_riders,
)
from dascore.utils.explicit_ranges import ExplicitRanges, known_coordinates


class _SchemaBackend:
    """Expose schema names without building the deferred piece relation."""

    def __init__(self, catalog):
        self.catalog = catalog

    def attr_names(self):
        """Return names from the parent schema."""
        return self.catalog.parent.backend.attr_names()

    def coord_names(self):
        """Return names from the parent schema."""
        return self.catalog.parent.backend.coord_names()

    def __getattr__(self, name):
        return getattr(self.catalog._materialized().backend, name)


@dataclass
class ExplicitSelectCatalog:
    """Build independent source pieces when the parent relation is requested."""

    parent: Any
    ranges: dict[str, ExplicitRanges]  # the first dim is subdivided
    operations: tuple = ()
    _cached: PatchCatalog | None = field(default=None, init=False, repr=False)
    _source_revision: int = field(default=-1, init=False, repr=False)

    @property
    def _revision(self):
        """Share the parent's revision token for live view invalidation."""
        return self.parent._revision

    def _materialized(self):
        """Cache coherent source pieces and replay deferred view operations."""
        # Match PatchCatalog snapshot locking across every candidate query.
        with self.parent._revision.lock:
            revision = self.parent._revision.value
            if self._cached is not None and revision == self._source_revision:
                return self._cached
            source = _ensure_patch_row(self.parent.to_df().reset_index(drop=True))
            assert source["_patch_row"].is_unique, "catalog rows must be unique"
            by_id = source.set_index("_patch_row", drop=False)
            refuse_riders(self.ranges, source)
            first = next(iter(self.ranges))
            boxes = list(zip(*(x.rows for x in self.ranges.values()), strict=True))
            frames = [
                self.parent.select(
                    _coords={x: b for x, b in zip(self.ranges, box) if b != (None,) * 2}
                ).to_df()["_patch_row"]
                for box in boxes
            ]
            selected = by_id.loc[[x for frame in frames for x in frame]]
            selected = selected.reset_index(drop=True)
            spans = selected[[f"{first}_min", f"{first}_max"]].to_numpy()
            plan = build_subdivision_plan(selected, [[tuple(x)] for x in spans], first)
            # Identity is bookkeeping, never a patch attribute.
            plan.outputs["_request_row"] = [n for n, x in enumerate(frames) for _ in x]
            exact = {
                x: known_coordinates(self.parent, selected, x) for x in self.ranges
            }
            args = (selected, self.ranges, exact, True, "ignore", True)
            plan = _trim_explicit(plan, *args)
            catalog = derived_catalog(
                source_rows=source,
                plan=plan,
                parent=self.parent,
                merge_kwargs={},
                mode="chunk",
                lossy=True,
            )
            for method, args, kwargs in self.operations:
                catalog = getattr(catalog, method)(*args, **kwargs)
            self._cached = catalog
            # Realizing the backend may invalidate the parent under this lock.
            self._source_revision = self.parent._revision.value
            return self._cached

    def __len__(self):
        return len(self._materialized())

    def __iter__(self):
        return iter(self._materialized())

    def to_df(self):
        """Return the realized piece relation."""
        return self._materialized().to_df()

    def get_patch(self, index):
        """Resolve one selected source piece."""
        return self._materialized().get_patch(index)

    def _defer(self, method, *args, **kwargs):
        """Return a view with one more operation queued for realization."""
        return replace(self, operations=(*self.operations, (method, args, kwargs)))

    def select(self, **kwargs):
        """Compose a selection without building piece metadata."""
        return self._defer("select", **kwargs)

    def order_by(self, *args, **kwargs):
        """Compose presentation order without realizing pieces."""
        return self._defer("order_by", *args, **kwargs)

    def window(self, *args, **kwargs):
        """Compose a positional window without realizing pieces."""
        return self._defer("window", *args, **kwargs)

    def restrict(self, *args, **kwargs):
        """Compose positional membership without realizing pieces."""
        return self._defer("restrict", *args, **kwargs)

    @property
    def backend(self):
        """Expose schema lazily and operational metadata from the realized view."""
        if (
            self._cached is not None
            and self._source_revision == self.parent._revision.value
        ):
            return self._cached.backend
        return _SchemaBackend(self)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._materialized(), name)
