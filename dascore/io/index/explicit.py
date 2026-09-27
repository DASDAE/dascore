"""Deferred catalog view for independent absolute coordinate windows."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from dascore.exceptions import MissingPatchError
from dascore.io.index.catalog import PatchCatalog
from dascore.io.index.planned import derived_catalog
from dascore.utils.chunk_plan import (
    _ensure_patch_row,
    _window_samples,
    build_subdivision_plan,
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
    # windows per dimension; the first is subdivided, the others trim
    ranges: dict[str, ExplicitRanges]
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
            names = list(self.ranges)
            boxes = list(zip(*(x.rows for x in self.ranges.values()), strict=True))
            candidate_frames = [
                self.parent.select(
                    _coords={
                        x: b
                        for x, b in zip(names, box)
                        if any(v is not None for v in b)
                    }
                ).to_df()
                for box in boxes
            ]
            ids = {x for frame in candidate_frames for x in frame["_patch_row"]}
            candidates = source[source["_patch_row"].isin(ids)]
            known = {x: known_coordinates(self.parent, candidates, x) for x in names}
            rows = []
            pieces: dict[str, list] = {name: [] for name in names}
            requests = []
            for request, (box, candidate) in enumerate(
                zip(boxes, candidate_frames, strict=True)
            ):
                for patch_row in candidate["_patch_row"]:
                    row = by_id.loc[patch_row]
                    actual = []
                    for name, bounds in zip(names, box, strict=True):
                        low, high = row[f"{name}_min"], row[f"{name}_max"]
                        if all(x is None for x in bounds):  # spans the dim
                            actual.append((low, high))
                            continue
                        coord = known[name].get(patch_row)
                        if coord is None:
                            msg = (
                                f"Cannot verify samples for {name!r} in source "
                                f"{row.get('source_path')!r}: coordinate metadata "
                                "is unavailable."
                            )
                            raise MissingPatchError(msg)
                        unit = row.get(f"_{name}_units")
                        unit = None if pd.isnull(unit) or unit == "" else str(unit)
                        # exact labels only, so the grid is not consulted
                        window = (bounds, low, high, np.nan, unit, (coord,))
                        actual.append(_window_samples(*window))
                    if any(x is None for x in actual):
                        continue
                    rows.append(row)
                    requests.append(request)
                    for name, bounds in zip(names, actual, strict=True):
                        pieces[name].append([bounds])
            selected = pd.DataFrame(rows, columns=source.columns).reset_index(drop=True)
            first, *others = names
            trims = {name: pieces[name] for name in others}
            plan = build_subdivision_plan(selected, pieces[first], first, trims)
            if len(plan.outputs):
                # Identity is bookkeeping, never a patch attribute.
                plan.outputs["_request_row"] = requests
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
