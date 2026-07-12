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

    def test_union_of_materialized_member(self):
        """A sorted (materialized but catalog-backed) member unions by ids."""
        sp = dc.get_example_spool("random_das")
        other = dc.get_example_spool("diverse_das")
        materialized = sp.sort("time")  # dataframe path, keeps its catalog
        assert not materialized._catalog_native
        combined = materialized + other
        assert len(combined) == len(sp) + len(other)

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

    def test_constructor_select_kwargs_restrict_union(self, tmp_path):
        """A select_kwargs-restricted directory spool unions only its rows."""
        base = dc.get_example_spool("random_das")
        dc.examples.spool_to_directory(base, path=tmp_path)
        full = dc.spool(tmp_path).update()
        df = full.get_contents().sort_values("time_min")
        window = (df["time_min"].iloc[0], df["time_max"].iloc[0])  # first patch
        restricted = dc.spool(tmp_path, select_kwargs={"time": window})
        assert 0 < len(restricted) < len(full)
        combined = restricted + dc.spool([dc.get_example_patch(tag="mem")])
        # the union must not reintroduce the rows the constructor excluded
        assert len(combined) == len(restricted) + 1

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


class TestPatchIdentity:
    """Set semantics by patch instance identity (lineage)."""

    def test_duplicate_instances_collapse(self):
        """The same patch instance twice is one spool entry."""
        patch = dc.get_example_patch()
        assert len(dc.spool([patch, patch])) == 1

    def test_deepcopy_shares_identity(self):
        """Copies of an immutable patch share its identity.

        Identity is minted eagerly at construction, so copies share it
        regardless of when they are made (no access-order dependence).
        """
        import copy

        patch = dc.get_example_patch()
        clone = copy.deepcopy(patch)
        assert clone._instance_id == patch._instance_id
        assert len(dc.spool([patch, clone])) == 1

    def test_new_instance_distinct(self):
        """patch.new() (and any patch op) mints a distinct identity."""
        patch = dc.get_example_patch()
        assert len(dc.spool([patch, patch.new()])) == 2

    def test_ops_mint_new_identity(self):
        """Operations produce instances with their own identity."""
        patch = dc.get_example_patch()
        other = patch.update_attrs(tag="x")
        assert patch._instance_id != other._instance_id

    def test_pickle_round_trip_preserves_content(self):
        """Spools of live patches pickle and rebuild their backend."""
        import pickle

        patch = dc.get_example_patch()
        spool = dc.spool([patch])
        _ = len(spool)  # realize the catalog
        loaded = pickle.loads(pickle.dumps(spool))
        assert len(loaded) == 1
        assert loaded[0] == patch

    def test_union_pickles(self):
        """Union spools survive pickling (rows ride along as records)."""
        import pickle

        p1 = dc.get_example_patch()
        p2 = p1.new()
        combined = dc.spool([p1]) + dc.spool([p2])
        loaded = pickle.loads(pickle.dumps(combined))
        assert len(loaded) == 2

    def test_remove_updates_live_registry(self):
        """Removing a live source removes it from the store as well."""
        import pickle

        patch = dc.get_example_patch()
        catalog = PatchCatalog.from_patches([patch])
        path = catalog.to_df().iloc[0]["path"]
        catalog.remove([path])
        assert len(catalog.to_df()) == 0
        # A pickled catalog rebuilds from the registry; the removed patch
        # must not resurrect.
        loaded = pickle.loads(pickle.dumps(catalog))
        loaded.attr_names()  # bootstrap the backend
        loaded._invalidate()
        assert len(loaded.to_df()) == 0


class TestExportPushdown:
    """Selected-membership export must not scan the whole archive."""

    def test_export_one_of_many_is_narrow(self):
        """Exporting one patch fetches only its own rows, not all sources."""
        patches = [dc.get_example_patch().update_attrs(tag=f"t{i}") for i in range(40)]
        catalog = PatchCatalog.from_patches(patches)
        catalog.to_df()  # bootstrap the backend
        backend = catalog.backend
        con = backend._con

        fetched_patches = []

        def _trace(sql):
            # Count how many patch rows any SELECT against patches pulls.
            if "from patches" in sql.lower() and sql.lower().lstrip().startswith(
                "select"
            ):
                fetched_patches.append(sql)

        target = int(catalog.to_df()["_patch_id"].iloc[0])
        con.set_trace_callback(_trace)
        try:
            records = backend.export_records(patch_ids=[target])
        finally:
            con.set_trace_callback(None)

        # exactly one source/patch comes back...
        assert sum(len(r.patches) for r in records) == 1
        # ...and every patches query was id-filtered (no full-table scan).
        assert fetched_patches
        assert all("patch_id in" in sql.lower() for sql in fetched_patches)

    def test_export_all_matches_full(self):
        """export_records() with no ids returns every source, unchanged."""
        patches = [dc.get_example_patch().update_attrs(tag=f"t{i}") for i in range(5)]
        catalog = PatchCatalog.from_patches(patches)
        catalog.to_df()
        records = catalog.backend.export_records()
        assert sum(len(r.patches) for r in records) == 5

    def test_export_empty_patch_ids(self):
        """Exporting an empty id set returns no records without querying."""
        catalog = PatchCatalog.from_patches([dc.get_example_patch()])
        catalog.to_df()
        assert catalog.backend.export_records(patch_ids=[]) == []

    def test_absolutize_record_passthrough(self, tmp_path):
        """A record already carrying an absolute/URI path is returned as-is."""
        from pathlib import Path

        from dascore.io.index.catalog import _absolutize_record
        from dascore.io.index.ingest import SourceRecord

        # an OS-native absolute path (drive-qualified on Windows)
        abs_path = str((tmp_path / "a.h5").resolve())
        assert Path(abs_path).is_absolute()
        rec = SourceRecord(source_path=abs_path, source_format="X", format_version="1")
        assert _absolutize_record(rec, str(tmp_path)) is rec
        uri = SourceRecord(
            source_path="s3://bucket/a.h5", source_format="X", format_version="1"
        )
        assert _absolutize_record(uri, str(tmp_path)) is uri

    def test_dir_union_absolutizes_relative_paths(self, tmp_path):
        """A directory member's relative source path is absolutized on union."""
        dc.get_example_patch().io.write(tmp_path / "a.h5", "dasdae")
        dir_spool = dc.spool(tmp_path).update(progress=None)
        combined = dir_spool + dc.spool([dc.get_example_patch(tag="mem")])
        assert len(combined) == 2
        # the file-backed member still loads (its path was made absolute)
        contents = combined.get_contents()
        file_row = contents[contents["path"].str.endswith("a.h5")]
        assert len(file_row) == 1
        loaded = [p for p in combined]
        assert len(loaded) == 2
