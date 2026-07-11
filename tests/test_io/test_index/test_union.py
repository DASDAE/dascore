"""Tests for spool union (spool + spool) and catalog merging."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.core.coords import CoordRange
from dascore.io.index.catalog import CompositeResolver, PatchCatalog


@pytest.fixture(scope="module")
def contiguous_patches():
    """Two contiguous example patches."""
    t0 = np.datetime64("2020-01-01", "ns")
    p1 = dc.get_example_patch(time_min=t0)
    time = p1.get_coord("time")
    p2 = dc.get_example_patch(time_min=time.max() + time.step)
    return p1, p2


@pytest.fixture()
def dir_spool(contiguous_patches, tmp_path):
    """A directory spool holding the first patch."""
    p1, _ = contiguous_patches
    dc.write(p1, tmp_path / "a.h5", "dasdae")
    return dc.spool(tmp_path).update()


class TestMemoryUnion:
    """Unions of in-memory spools."""

    def test_lengths_add(self):
        """The union contains the patches of both."""
        sp1 = dc.get_example_spool("random_das")
        sp2 = dc.get_example_spool("diverse_das")
        combined = sp1 + sp2
        assert len(combined) == len(sp1) + len(sp2)

    def test_patches_shared_not_copied(self, contiguous_patches):
        """In-memory patches resolve to the same objects."""
        p1, p2 = contiguous_patches
        combined = dc.spool([p1]) + dc.spool([p2])
        loaded = list(combined)
        assert any(x is p1 for x in loaded)
        assert any(x is p2 for x in loaded)

    def test_select_on_union(self):
        """Selection works over the merged metadata."""
        sp1 = dc.get_example_spool("random_das")
        sp2 = dc.get_example_spool("diverse_das")
        combined = sp1 + sp2
        selected = combined.select(network="das2")
        assert len(selected)
        for patch in selected:
            assert patch.attrs.network == "das2"

    def test_non_spool_add(self):
        """Adding a non-spool returns NotImplemented semantics."""
        sp1 = dc.get_example_spool("random_das")
        with pytest.raises(TypeError):
            _ = sp1 + 42

    def test_chunk_across_members(self, contiguous_patches):
        """Contiguous patches from different spools merge into one."""
        p1, p2 = contiguous_patches
        combined = dc.spool([p1]) + dc.spool([p2])
        merged = combined.chunk(time=None)
        assert len(merged) == 1
        patch = merged[0]
        time = patch.get_coord("time")
        assert isinstance(time, CoordRange)
        assert time.min() == p1.get_coord("time").min()
        assert time.max() == p2.get_coord("time").max()

    def test_selection_carries_by_membership(self):
        """A selected input contributes only its selected rows."""
        sp2 = dc.get_example_spool("diverse_das")
        sub = sp2.select(network="das2")
        combined = dc.get_example_spool("random_das") + sub
        assert len(combined) == len(dc.get_example_spool("random_das")) + len(sub)


class TestFileUnion:
    """Unions involving file-backed spools."""

    def test_dir_plus_memory(self, dir_spool, contiguous_patches):
        """A directory spool and memory spool combine lazily."""
        _, p2 = contiguous_patches
        combined = dir_spool + dc.spool([p2])
        assert len(combined) == 2
        loaded = list(combined)
        assert all(x.shape for x in loaded)

    def test_chunk_across_file_and_memory(self, dir_spool, contiguous_patches):
        """The union seam merges: file patch + memory patch -> one patch."""
        p1, p2 = contiguous_patches
        combined = dir_spool + dc.spool([p2])
        merged = combined.chunk(time=None)
        assert len(merged) == 1
        patch = merged[0]
        assert patch.shape[patch.get_axis("time")] == (
            p1.shape[p1.get_axis("time")] + p2.shape[p2.get_axis("time")]
        )

    def test_same_source_dedups(self, dir_spool):
        """The same source in both members keeps a single entry."""
        combined = dir_spool + dir_spool
        assert len(combined) == len(dir_spool)

    def test_union_preserves_def_keys(self, dir_spool, contiguous_patches):
        """Coord definitions deduplicate by def key across members."""
        _, p2 = contiguous_patches
        combined = dir_spool + dc.spool([p2])
        df = combined._catalog.to_df()
        # both patches share the same distance coord identity
        assert df["_distance_def_key"].nunique() == 1


class TestCompositeResolver:
    """Resolver dispatch for union catalogs."""

    def test_routes_memory_rows(self, contiguous_patches):
        """memory:// rows go to the live registry."""
        p1, _ = contiguous_patches
        cat = PatchCatalog.union([PatchCatalog.from_patches([p1])])
        assert isinstance(cat.resolver, CompositeResolver)
        row = cat.to_df().iloc[0].to_dict()
        assert cat.resolve_row(row) is p1

    def test_union_of_union(self, contiguous_patches):
        """Unions compose (a union catalog can be a member)."""
        p1, p2 = contiguous_patches
        first = PatchCatalog.union([PatchCatalog.from_patches([p1])])
        second = PatchCatalog.union([first, PatchCatalog.from_patches([p2])])
        assert len(second.to_df()) == 2
        patches = [second.resolve_row(x) for _, x in second.to_df().iterrows()]
        assert {id(x) for x in patches} == {id(p1), id(p2)}
