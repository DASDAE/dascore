"""Tests for spool union (spool + spool) and catalog merging."""

from __future__ import annotations

import copy
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import dascore as dc
from dascore.core.coords import CoordRange
from dascore.io.index.catalog import CompositeResolver, PatchCatalog, _absolutize_record
from dascore.io.index.ingest import SourceRecord


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
        """A planned (materialized but catalog-backed) member unions by ids."""
        sp = dc.get_example_spool("random_das")
        other = dc.get_example_spool("diverse_das")
        # a content-preserving derived catalog (concat groups of one)
        materialized = sp.concatenate(time=1)
        combined = materialized + other
        assert len(combined) == len(sp) + len(other)

    def test_union_of_sorted_member(self):
        """A lazily sorted member unions by its ordered membership."""
        sp = dc.get_example_spool("random_das")
        other = dc.get_example_spool("diverse_das")
        combined = sp.sort("time") + other
        assert len(combined) == len(sp) + len(other)

    def test_select_on_union(self):
        """Selection works over the merged metadata."""
        sp1 = dc.get_example_spool("random_das")
        sp2 = dc.get_example_spool("diverse_das")
        combined = sp1 + sp2
        selected = combined.select(acquisition_key="DAS2.*")
        assert len(selected)
        for patch in selected:
            assert patch.attrs.acquisition_key == "DAS2.R2D1..RAW"

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
        sub = sp2.select(acquisition_key="DAS2.*")
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
        """A selection-restricted directory spool unions only its rows."""
        base = dc.get_example_spool("random_das")
        dc.examples.spool_to_directory(base, path=tmp_path)
        full = dc.spool(tmp_path).update()
        df = full.get_contents().sort_values("time_min")
        window = (df["time_min"].iloc[0], df["time_max"].iloc[0])  # first patch
        restricted = full.select(time=window)
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
        patch = dc.get_example_patch()
        spool = dc.spool([patch])
        _ = len(spool)  # realize the catalog
        loaded = pickle.loads(pickle.dumps(spool))
        assert len(loaded) == 1
        assert loaded[0] == patch

    def test_union_pickles(self):
        """Union spools survive pickling (rows ride along as records)."""
        p1 = dc.get_example_patch()
        p2 = p1.new()
        combined = dc.spool([p1]) + dc.spool([p2])
        loaded = pickle.loads(pickle.dumps(combined))
        assert len(loaded) == 2

    def test_remove_updates_live_registry(self):
        """Removing a live source removes it from the store as well."""
        patch = dc.get_example_patch()
        catalog = PatchCatalog.from_patches([patch])
        path = catalog.to_df().iloc[0]["source_path"]
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
        file_row = contents[contents["source_path"].str.endswith("a.h5")]
        assert len(file_row) == 1
        loaded = [p for p in combined]
        assert len(loaded) == 2


class TestSameFileUnion:
    """Unions of members selecting patches from the same multi-patch file."""

    @pytest.fixture()
    def two_patch_file_spool(self, contiguous_patches, tmp_path):
        """A file spool over one file holding two patches."""
        p1, p2 = (x.update_attrs(history=[]) for x in contiguous_patches)
        path = tmp_path / "two_patch.h5"
        dc.write(dc.spool([p1, p2]), path, "dasdae")
        return dc.spool(path)

    def test_disjoint_selections_union(self, two_patch_file_spool):
        """Two members holding different patches of one file both survive."""
        sp = two_patch_file_spool
        combined = sp[:1] + sp[1:]
        assert len(combined) == 2
        for patch in combined:
            assert isinstance(patch, dc.Patch)

    def test_overlapping_selections_dedup(self, two_patch_file_spool):
        """A patch present in both members appears once (dict-merge)."""
        sp = two_patch_file_spool
        combined = sp[:2] + sp[1:]
        assert len(combined) == 2

    def test_union_absorbs_only_member_registry_entries(self, contiguous_patches):
        """Live entries outside a member's rows don't ride into the union."""
        p1, p2 = contiguous_patches
        t1 = p1.get_coord("time")
        narrowed = dc.spool([p1, p2]).select(time=(None, t1.max()))
        assert len(narrowed) == 1
        other = dc.get_example_patch(time_min="2030-01-01")
        combined = narrowed + dc.spool([other])
        # the trimmed operand materializes into a plan, so its live
        # member rides inside the plan's loader, not the top registry
        registry = combined._catalog.resolver.live_entries()
        assert len(registry) == 1  # other
        plans = combined._catalog.resolver.plan_entries()
        nested_live = {k for p in plans.values() for k in p.live_entries()}
        assert len(nested_live) == 1  # p1; p2 stayed home
        assert len(combined) == 2


class TestLossyStateUnion:
    """Residual trims and order specs survive combining (2026-07-18 F1)."""

    def test_value_trim_survives_union(self):
        """A coordinate-range selection's trim is baked in, not dropped."""
        p = dc.get_example_patch()
        t = p.get_coord("time")
        lo, hi = t.min() + 10 * t.step, t.min() + 20 * t.step
        selected = dc.spool([p]).select(time=(lo, hi))
        combined = selected + dc.spool([])
        assert len(combined) == 1
        got, want = combined[0], selected[0]
        assert got.shape == want.shape
        assert got.get_coord("time").min() == want.get_coord("time").min()
        assert got.get_coord("time").max() == want.get_coord("time").max()

    def test_samples_trim_survives_union(self):
        """A samples window's trim is baked in, not dropped."""
        p = dc.get_example_patch()
        selected = dc.spool([p]).select(time=(0, 10), samples=True)
        combined = selected + dc.spool([])
        assert combined[0].shape == selected[0].shape

    def test_file_backed_value_trim_survives_union(self, dir_spool):
        """The same guarantee holds for file-backed catalogs."""
        patch = dir_spool[0]
        t = patch.get_coord("time")
        lo, hi = t.min() + 10 * t.step, t.min() + 20 * t.step
        selected = dir_spool.select(time=(lo, hi))
        combined = selected + dc.spool([])
        assert combined[0].shape == selected[0].shape

    def test_sort_order_survives_union(self):
        """A sort spec bakes into ordinals instead of silently reverting."""
        p = dc.get_example_patch()
        t = p.get_coord("time")
        early = p.update_attrs(tag="early")
        late = p.update_coords(time_min=t.max() + t.step).update_attrs(tag="late")
        srt = dc.spool([late, early]).sort("time")
        combined = srt + dc.spool([])
        assert [x.attrs.tag for x in combined] == [x.attrs.tag for x in srt]

    def test_membership_selections_still_dedup(self):
        """Row-membership state unions as rows: identity dedup preserved."""
        p = dc.get_example_patch()
        spool = dc.spool([p.update_attrs(tag=f"t{i}") for i in range(4)])
        combined = spool.select(tag="t2") + spool
        assert len(combined) == 4

    def test_windows_still_union_by_rows(self):
        """Slice windows survive as membership without materializing."""
        p = dc.get_example_patch()
        spool = dc.spool([p.update_attrs(tag=f"t{i}") for i in range(4)])
        combined = spool[1:3] + dc.spool([])
        assert [x.attrs.tag for x in combined] == ["t1", "t2"]

    def test_two_selected_operands(self):
        """Different trims on each operand both survive as new contents."""
        p = dc.get_example_patch()
        t = p.get_coord("time")
        lo = t.min() + 10 * t.step
        a = dc.spool([p]).select(time=(lo, t.min() + 20 * t.step))
        b = dc.spool([p]).select(time=(t.min(), lo))
        combined = a + b
        assert len(combined) == 2
        assert {x.shape for x in combined} == {a[0].shape, b[0].shape}

    def test_sorted_derived_union(self):
        """A derived (chunked) catalog with an order spec also survives."""
        p = dc.get_example_patch()
        t = p.get_coord("time")
        early = p.update_attrs(tag="z_early")
        # the gap keeps two outputs; distinct tags survive unmerged
        late = p.update_coords(time_min=t.max() + 10 * t.step).update_attrs(
            tag="a_late"
        )
        chunked = dc.spool([early, late]).chunk(time=None)
        srt = chunked.sort("tag")
        want = [x.attrs.tag for x in srt]
        assert want == ["a_late", "z_early"]  # tag order != time order
        combined = srt + dc.spool([])
        assert [x.attrs.tag for x in combined] == want

    def test_combined_pickles(self):
        """A union holding a materialized operand round-trips pickling."""
        p = dc.get_example_patch()
        t = p.get_coord("time")
        lo, hi = t.min() + 10 * t.step, t.min() + 20 * t.step
        combined = dc.spool([p]).select(time=(lo, hi)) + dc.spool([])
        loaded = pickle.loads(pickle.dumps(combined))
        assert len(loaded) == 1
        assert loaded[0].shape == combined[0].shape


def _patch_shape(patch):
    """Module-level shape getter (process pools need a picklable callable)."""
    return patch.shape


class TestMixedViewPickle:
    """Serialization keeps plan routes in mixed views (round-4 F2)."""

    def test_sliced_mixed_union_pickles(self):
        """A sliced union of planned and live rows loads all rows back."""
        p = dc.get_example_patch()
        t = p.get_coord("time")
        trimmed = dc.spool([p]).select(
            time=(t.min() + 10 * t.step, t.min() + 20 * t.step)
        )
        other = p.new().update_attrs(tag="other")
        view = (trimmed + dc.spool([other]))[:]
        loaded = pickle.loads(pickle.dumps(view))
        shapes = {loaded[i].shape for i in range(len(loaded))}
        assert shapes == {(300, 11), (300, 2000)}

    @pytest.mark.concurrency
    def test_mixed_union_map_processes(self):
        """Process-backed map ships plan routes with each task."""
        p = dc.get_example_patch()
        t = p.get_coord("time")
        trimmed = dc.spool([p]).select(
            time=(t.min() + 10 * t.step, t.min() + 20 * t.step)
        )
        combined = trimmed + dc.spool([p.new().update_attrs(tag="other")])
        with ProcessPoolExecutor(2) as executor:
            shapes = set(combined.map(_patch_shape, client=executor))
        assert shapes == {(300, 11), (300, 2000)}
