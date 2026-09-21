"""Deferred catalog view for independent absolute coordinate windows."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from dascore.exceptions import MissingPatchError
from dascore.io.index.catalog import PatchCatalog
from dascore.io.index.planned import derived_catalog
from dascore.utils.chunk_plan import (
    _ensure_patch_row,
    build_subdivision_plan,
    exact_coordinate_bounds,
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
    name: str
    ranges: ExplicitRanges
    operations: tuple = ()
    _cached: PatchCatalog | None = field(default=None, init=False, repr=False)
    _source_revision: int = field(default=-1, init=False, repr=False)

    @property
    def _revision(self):
        return self.parent._revision

    def _materialized(self):
        # Match PatchCatalog snapshot locking across every candidate query.
        with self.parent._revision.lock:
            revision = self.parent._revision.value
            if self._cached is not None and revision == self._source_revision:
                return self._cached
            source = _ensure_patch_row(self.parent.to_df().reset_index(drop=True))
            assert source["_patch_row"].is_unique, "catalog rows must be unique"
            by_id = source.set_index("_patch_row", drop=False)
            candidate_frames = [
                self.parent.select(_coords={self.name: bounds}).to_df()
                for bounds in self.ranges.rows
            ]
            ids = {x for frame in candidate_frames for x in frame["_patch_row"]}
            known = known_coordinates(
                self.parent, source[source["_patch_row"].isin(ids)], self.name
            )
            rows = []
            pieces = []
            requests = []
            for request, (bounds, candidate) in enumerate(
                zip(self.ranges.rows, candidate_frames, strict=True)
            ):
                for _, projected in candidate.iterrows():
                    row = by_id.loc[projected["_patch_row"]]
                    coord = known.get(row["_patch_row"])
                    if coord is not None:
                        actual = exact_coordinate_bounds(coord, bounds)
                        if actual is None:
                            continue
                    else:
                        msg = (
                            f"Cannot verify samples for {self.name!r} in source "
                            f"{row.get('source_path')!r}: coordinate metadata "
                            "is unavailable."
                        )
                        raise MissingPatchError(msg)
                    rows.append(row)
                    requests.append(request)
                    pieces.append([actual])
            selected = pd.DataFrame(rows, columns=source.columns).reset_index(drop=True)
            plan = build_subdivision_plan(selected, pieces, self.name)
            if len(plan.outputs):
                # Identity is bookkeeping, never a patch attribute.
                plan.outputs["_request_row"] = requests
            catalog = derived_catalog(
                source_rows=source.drop_duplicates("_patch_row"),
                plan=plan,
                parent=self.parent,
                merge_kwargs={},
                mode="chunk",
                lossy=True,
            )
            for method, args, kwargs in self.operations:
                catalog = getattr(catalog, method)(*args, **kwargs)
            self._cached = catalog
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
