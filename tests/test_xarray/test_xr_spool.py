"""Tests for converting a spool to a dask-backed xarray DataTree."""

from __future__ import annotations

import os
import tracemalloc
import typing
from contextlib import contextmanager

import fsspec
import numpy as np
import pytest
from fsspec.implementations.memory import MemoryFileSystem
from upath import UPath

import dascore as dc
import dascore.io.core as io_core
from dascore.config import config_context
from dascore.examples import spool_to_directory
from dascore.exceptions import PatchConversionError
from dascore.io.index.planned import PlanResolver
from dascore.units import get_quantity_str
from dascore.utils.patch_assembly import PatchAssembler
from dascore.xarray.patch import _cf_unit_str
from tests.conftest import join_patches


@pytest.fixture
def array_reads(monkeypatch):
    """Record every stored window a read loads, as (path, windows)."""
    reads = []
    original = io_core._open_array_reader

    @contextmanager
    def _recording(source, *path):
        with original(source, *path) as load:
            yield lambda src: reads.append((src.path, src.windows)) or load(src)

    monkeypatch.setattr(io_core, "_open_array_reader", _recording)
    return reads


class TestSpoolToXarray:
    """Tests for converting a spool to a dask-backed xarray DataTree."""

    @pytest.fixture(autouse=True)
    def _require_libs(self):
        """These tests need both optional libraries."""
        pytest.importorskip("xarray")
        pytest.importorskip("dask")

    @pytest.fixture
    def diverse_tree(self, diverse_spool):
        """Convert the diverse spool, skipping without xarray or dask."""
        return diverse_spool.io.to_xarray()

    def _leaves(self, tree):
        """Return the datasets holding a data variable."""
        return [node for node in tree.subtree if "data" in node.dataset]

    def test_tree_structure(self, diverse_tree):
        """Each leaf holds one lazy data variable with dim coordinates."""
        import dask.array as da  # noqa: PLC0415
        import xarray as xr  # noqa: PLC0415

        assert isinstance(diverse_tree, xr.DataTree)
        leaves = self._leaves(diverse_tree)
        assert leaves
        for leaf in leaves:
            data = leaf.dataset["data"]
            assert isinstance(data.data, da.Array)
            assert set(data.dims) <= set(data.coords)

    def test_matches_chunk(self, diverse_spool, diverse_tree):
        """Every leaf's values equal the equivalent chunk output patch."""
        expected = {}
        for patch in diverse_spool.chunk(time=None):
            coord = patch.get_coord("time")
            key = (
                patch.attrs.tag,
                patch.attrs.acquisition_key or "",
                np.datetime64(coord.min(), "ns"),
                patch.shape,
            )
            expected[key] = patch
        leaves = self._leaves(diverse_tree)
        assert len(leaves) == len(expected)
        for leaf in leaves:
            data = leaf.dataset["data"]
            key = (
                data.attrs["tag"],
                data.attrs.get("acquisition_key") or "",
                np.datetime64(data["time"].values.min(), "ns"),
                data.shape,
            )
            patch = expected.pop(key)
            np.testing.assert_array_equal(data.values, patch.data)
            for dim in patch.dims:
                np.testing.assert_array_equal(
                    data[dim].values, patch.get_coord(dim).values
                )
        assert not expected

    def test_combines_independent_conversions(self, random_patch):
        """Combining trees must keep each conversion's source data distinct."""
        opposite = random_patch.new(data=-random_patch.data)
        first = self._leaves(dc.spool([random_patch]).io.to_xarray())[0].dataset["data"]
        second = self._leaves(dc.spool([opposite]).io.to_xarray())[0].dataset["data"]

        actual = (first + second).compute().values

        np.testing.assert_array_equal(actual, np.zeros_like(random_patch.data))

    def test_builds_without_reading(self, diverse_spool_directory, monkeypatch):
        """Constructing the tree must not read any patch data."""
        from dascore.io.index.catalog import FileResolver, PatchCatalog  # noqa: PLC0415

        spool = dc.spool(diverse_spool_directory).update()

        def _fail(*args, **kwargs):
            raise AssertionError("tree construction read patch data")

        monkeypatch.setattr(PatchCatalog, "resolve_row", _fail)
        monkeypatch.setattr(FileResolver, "resolve", _fail)
        monkeypatch.setattr(PlanResolver, "_load_member", _fail)
        monkeypatch.setattr(io_core, "_open_array_reader", _fail)
        tree = spool.io.to_xarray()
        assert len(self._leaves(tree))

    def test_compute_reads_only_needed_blocks(
        self, diverse_spool_directory, monkeypatch, array_reads
    ):
        """A small selection loads only the member blocks it touches."""
        spool = dc.spool(diverse_spool_directory).update()
        tree = spool.io.to_xarray()
        calls = []
        # A block reads through whichever path its format offers, so both
        # are counted: what the test pins is how many members are read.
        original = PlanResolver._load_member

        def _counting(self, kwargs):
            calls.append("patch")
            return original(self, kwargs)

        monkeypatch.setattr(PlanResolver, "_load_member", _counting)
        # The DAS2.R2D1..RAW random segment merges three source patches;
        # slicing inside the first must load exactly one of the three,
        # and the loaded values must match the eagerly chunked patch.
        leaf = next(
            x
            for x in self._leaves(tree)
            if x.dataset["data"].attrs.get("acquisition_key") == "DAS2.R2D1..RAW"
        )
        data = leaf.dataset["data"]
        assert data.data.npartitions == 3
        small = data.isel(time=slice(0, 5)).compute()
        assert len(calls) + len(array_reads) == 1
        merged = spool.select(acquisition_key="DAS2.R2D1..RAW").chunk(time=None)[0]
        expected = merged.data[:, :5] if merged.dims[0] != "time" else merged.data[:5]
        np.testing.assert_array_equal(small.values, expected)

    def test_plan_backed_spool(self, random_spool):
        """A chunked spool converts and computes like its merged self."""
        tree = random_spool.chunk(time=2).io.to_xarray()
        leaves = self._leaves(tree)
        assert len(leaves) == 1
        merged = random_spool.chunk(time=None)[0]
        np.testing.assert_array_equal(leaves[0].dataset["data"].values, merged.data)

    @pytest.mark.parametrize("bad_dtype", [None, ""])
    def test_missing_dtype_raises(self, random_spool, monkeypatch, bad_dtype):
        """An index without a dtype cannot size the arrays; say so."""
        original = PatchAssembler._df_to_dict_list

        def _null_dtype(self, df):
            return [{**row, "_dtype": bad_dtype} for row in original(self, df)]

        monkeypatch.setattr(PatchAssembler, "_df_to_dict_list", _null_dtype)
        with pytest.raises(PatchConversionError, match=r"dtype.*spool.update\(\)"):
            random_spool.io.to_xarray()

    def test_tolerance_argument(self, diverse_spool):
        """A looser tolerance merges gaps the default keeps as segments."""
        sub = diverse_spool.select(tag="big_gaps")
        default = len(self._leaves(sub.io.to_xarray()))
        loose = len(self._leaves(sub.io.to_xarray(tolerance=10_000)))
        assert loose < default

    def test_stale_index_shape_raises(self, random_spool, monkeypatch):
        """A block whose loaded shape breaks its promise raises clearly."""
        original = PlanResolver._load_member

        def _truncated(self, kwargs):
            patch = original(self, kwargs)
            return patch.select(time=(0, 5), samples=True)

        monkeypatch.setattr(PlanResolver, "_load_member", _truncated)
        tree = random_spool.io.to_xarray()  # building reads nothing
        leaf = self._leaves(tree)[0]
        with pytest.raises(PatchConversionError, match="promised"):
            leaf.dataset["data"].compute()

    def test_segment_names_follow_dim_order(self, diverse_spool):
        """segment_0..n are ordered along the merged dimension."""
        tree = diverse_spool.io.to_xarray()
        for node in tree.children.values():
            starts = [
                child.dataset["data"]["time"].values.min()
                for _, child in sorted(node.children.items())
            ]
            assert starts == sorted(starts)

    def test_sampling_jitter_steps(self, random_patch):
        """Members merged under sampling tolerance keep their own grids."""
        first = random_patch
        coord = first.get_coord("time")
        step = coord.step * 1.04  # within the 5% sampling tolerance
        second = first.update_coords(time_min=coord.max() + coord.step, time_step=step)
        spool = dc.spool([first, second])
        merged = spool.chunk(time=None)[0]
        leaf = self._leaves(spool.io.to_xarray())[0]
        data = leaf.dataset["data"]
        assert data.shape == merged.data.shape
        np.testing.assert_array_equal(data.values, merged.data)
        np.testing.assert_array_equal(
            data["time"].values, merged.get_coord("time").values
        )

    def test_off_grid_overlap(self, random_patch):
        """An overlap whose grids misalign still sizes blocks exactly."""
        coord = random_patch.get_coord("time")
        shifted = random_patch.update_coords(time_min=coord.max() - 9.3 * coord.step)
        spool = dc.spool([random_patch, shifted])
        merged = spool.chunk(time=None)[0]
        leaf = self._leaves(spool.io.to_xarray())[0]
        data = leaf.dataset["data"]
        assert data.shape == merged.data.shape
        np.testing.assert_array_equal(data.values, merged.data)
        np.testing.assert_array_equal(
            data["time"].values, merged.get_coord("time").values
        )

    def test_trim_of_a_sample_selection_keeps_its_grid(self, random_patch):
        """A member cut from a sample-selected patch stays on that patch's grid.

        The selection withholds the source's own range, so the cut is
        sized from its row alone; an off-grid overlap would show a cut
        off its grid as labels a fraction of a step off.
        """
        coord = random_patch.get_coord("time")
        shifted = random_patch.update_coords(time_min=coord.max() - 9.3 * coord.step)
        spool = dc.spool([random_patch, shifted]).select(time=(1, -1), samples=True)
        merged = spool.chunk(time=None)[0]
        data = self._leaves(spool.io.to_xarray())[0].dataset["data"]
        np.testing.assert_array_equal(data.values, merged.data)
        np.testing.assert_array_equal(
            data["time"].values, merged.get_coord("time").values
        )

    def test_mixed_dtypes_round_as_chunk_does(self, random_patch):
        """Members loading as patches pass through the merge's promotion chain."""
        coord = random_patch.get_coord("time")
        dtypes = (np.int16, np.float32, np.float64)
        patches = [
            random_patch.new(
                data=(random_patch.data * 1000).astype(dtype)
            ).update_coords(time_min=coord.min() + num * len(coord) * coord.step)
            for num, dtype in enumerate(dtypes)
        ]
        spool = dc.spool(patches)
        merged = spool.chunk(time=None)[0]
        data = self._leaves(spool.io.to_xarray())[0].dataset["data"]
        assert data.dtype == merged.data.dtype == np.float64
        np.testing.assert_array_equal(data.values, merged.data)

    def test_single_sample_non_dim(self, random_patch):
        """A one-sample non-merge dimension has no step yet converts."""
        thin = random_patch.select(distance=(0, 1), samples=True)
        thin = thin.update_coords(distance=np.array([5.0]))
        leaf = self._leaves(dc.spool([thin]).io.to_xarray())[0]
        assert leaf.dataset["data"].shape == thin.shape
        np.testing.assert_array_equal(leaf.dataset["data"].values, thin.data)
        np.testing.assert_array_equal(leaf.dataset["data"]["distance"].values, [5.0])

    def test_irregular_dim_raises(self, random_patch):
        """A multi-sample coordinate with no step cannot be sized."""
        time = random_patch.get_coord("time").values.copy()
        time[1] += np.timedelta64(1, "ms")
        wobbly = random_patch.update_coords(time=time)
        with pytest.raises(PatchConversionError, match="no sampling step"):
            dc.spool([wobbly]).io.to_xarray()

    def test_irregular_non_dim_raises(self, random_patch):
        """A stepless non-merge dimension cannot be sized either."""
        dist = random_patch.get_coord("distance").values.copy().astype(float)
        dist[1] += 0.5
        wobbly = random_patch.update_coords(distance=dist)
        with pytest.raises(PatchConversionError, match="no sampling step"):
            dc.spool([wobbly]).io.to_xarray()

    def test_no_group_attrs(self, random_spool):
        """With no grouping attributes the whole spool is one group."""
        with config_context(patch_kind_attrs=()):
            tree = random_spool.io.to_xarray()
        assert len(tree.children) == 1
        assert len(self._leaves(tree)) == 1

    def test_quantity_tolerance(self, diverse_spool):
        """A unit-bearing tolerance is handed to simplify as it stands."""
        sub = diverse_spool.select(tag="big_gaps")
        default = len(self._leaves(sub.io.to_xarray()))
        loose = len(self._leaves(sub.io.to_xarray(tolerance=dc.get_quantity("1 hour"))))
        assert loose < default

    def test_duplicate_node_names_raise(self, random_patch, monkeypatch):
        """Two groups resolving to one node name must not overwrite arrays."""
        import dascore.utils.display as display_module  # noqa: PLC0415

        patches = [
            random_patch.update_attrs(cable_id="a"),
            random_patch.update_attrs(cable_id="b"),
        ]
        monkeypatch.setattr(
            display_module, "group_names", lambda *args, **kwargs: ["same", "same"]
        )
        with pytest.raises(PatchConversionError, match="more than one group"):
            dc.spool(patches).io.to_xarray(group="cable_id")

    def test_assoc_coord_samples_select_refused(self, random_patch):
        """A samples selection on an associated coordinate cannot be sized."""
        n = len(random_patch.get_coord("distance"))
        patch = random_patch.update_coords(zone=("distance", np.arange(n)))
        sub = dc.spool([patch]).select(zone=(0, 5), samples=True)
        with pytest.raises(PatchConversionError, match="associated"):
            sub.io.to_xarray()

    def test_enriched_spool_refused(self, random_spool):
        """Pending inventory enrichment would be dropped; refuse instead."""
        sub = random_spool[0:2]
        sub._enrich_kwargs = {"coords": True}
        with pytest.raises(PatchConversionError, match="enrichment"):
            sub.io.to_xarray()

    def test_quantity_tolerance_with_units(self, random_patch):
        """A unit-bearing tolerance reads against the coordinate's units."""
        d = random_patch.get_coord("distance")
        gap = random_patch.update_coords(
            distance_min=d.max() + 5 * d.step  # a 5-step gap along distance
        )
        spool = dc.spool([random_patch, gap])
        tol = dc.get_quantity("10 m")
        merged = join_patches([random_patch, gap], "distance")
        tree = spool.io.to_xarray(dim="distance", tolerance=tol)
        leaves = self._leaves(tree)
        assert len(leaves) == 1
        data = leaves[0].dataset["data"]
        assert data.shape == merged.data.shape
        np.testing.assert_array_equal(data.values, merged.data)
        np.testing.assert_array_equal(
            data["distance"].values, merged.get_coord("distance").values
        )

    def test_value_select_refused(self, random_spool):
        """A pending value-range selection cannot be sized; it raises."""
        coord = random_spool[0].get_coord("time")
        sub = random_spool.select(time=(coord.min() + coord.step // 2, None))
        with pytest.raises(PatchConversionError, match="value selections"):
            sub.io.to_xarray()

    def test_relative_select_refused(self, random_spool):
        """A relative bound resolves per patch, so it cannot size blocks."""
        sub = random_spool.select(time=(1, None), relative=True)
        with pytest.raises(PatchConversionError, match="relative selections"):
            sub.io.to_xarray()

    def test_samples_select_supported(self, random_spool):
        """A samples-based selection stays exact and converts."""
        sub = random_spool.select(time=(10, -10), samples=True)
        merged = sub.chunk(time=None)[0]
        leaf = self._leaves(sub.io.to_xarray())[0]
        data = leaf.dataset["data"]
        assert data.shape == merged.data.shape
        np.testing.assert_array_equal(data.values, merged.data)
        np.testing.assert_array_equal(
            data["time"].values, merged.get_coord("time").values
        )

    def test_descending_dim_raises(self, random_patch):
        """A descending merge dimension is refused with a clear message."""
        flipped = random_patch.update_coords(
            distance=random_patch.get_coord("distance").values[::-1]
        )
        with pytest.raises(PatchConversionError, match="descending"):
            dc.spool([flipped]).io.to_xarray(dim="distance")

    def test_descending_non_dim_coord(self, random_patch):
        """A descending non-merge coordinate keeps its order and values."""
        flipped = random_patch.update_coords(
            distance=random_patch.get_coord("distance").values[::-1]
        )
        leaf = self._leaves(dc.spool([flipped]).io.to_xarray())[0]
        data = leaf.dataset["data"]
        np.testing.assert_array_equal(
            data["distance"].values, flipped.get_coord("distance").values
        )
        np.testing.assert_array_equal(data.values, flipped.data)

    def test_descending_non_dim_coord_without_a_grid(self, random_patch):
        """A descending coordinate no stored grid describes is sized by envelope."""
        # float values, so the row states no exact grid to rebuild from
        # and the envelope is all there is to size the lazy array with
        values = random_patch.get_coord("distance").values[::-1] * 1.5
        flipped = random_patch.update_coords(distance=values)
        leaf = self._leaves(dc.spool([flipped]).io.to_xarray())[0]
        data = leaf.dataset["data"]
        np.testing.assert_array_equal(data["distance"].values, values)
        np.testing.assert_array_equal(data.values, flipped.data)

    def test_mixed_dtype_upcasts(self, random_patch):
        """Blocks narrower than the combined dtype upcast at load."""
        coord = random_patch.get_coord("time")
        narrow = random_patch.new(data=random_patch.data.astype(np.float32))
        narrow = narrow.update_coords(time_min=coord.max() + coord.step)
        spool = dc.spool([random_patch, narrow])
        leaf = self._leaves(spool.io.to_xarray())[0]
        data = leaf.dataset["data"]
        assert data.dtype == np.float64
        # A slice touching only the narrow member must upcast in the
        # loader itself, not by concatenation with a wider block.
        assert data.isel(time=slice(-3, None)).compute().dtype == np.float64

    def test_single_group_spool(self, random_spool):
        """A homogeneous spool merges into one segment of one group."""
        tree = random_spool.io.to_xarray()
        leaves = self._leaves(tree)
        assert len(leaves) == 1
        merged = random_spool.chunk(time=None)[0]
        np.testing.assert_array_equal(leaves[0].dataset["data"].values, merged.data)

    def test_group_argument(self, diverse_spool):
        """An explicit group partitions the tree by that attribute."""
        tree = diverse_spool.io.to_xarray(group="tag", conflict="drop")
        tags = {x.dataset["data"].attrs.get("tag") for x in self._leaves(tree)}
        contents_tags = set(diverse_spool.get_contents()["tag"])
        assert tags == contents_tags
        # Grouping by tag alone merges kinds the default grouping keeps
        # apart (e.g. differing acquisition keys), so the node count must
        # equal the tag count, not the finer default partition.
        assert len(tree.children) == len(contents_tags)

    def test_bad_group_raises(self, diverse_spool):
        """A group attribute no patch has raises the standard query error."""
        from dascore.exceptions import InvalidSpoolQueryError  # noqa: PLC0415

        with pytest.raises(InvalidSpoolQueryError, match="do not exist"):
            diverse_spool.io.to_xarray(group="not_an_attr")

    def test_empty_spool(self):
        """An empty spool converts to an empty tree."""
        import xarray as xr  # noqa: PLC0415

        tree = dc.spool([]).io.to_xarray()
        assert isinstance(tree, xr.DataTree)
        assert not self._leaves(tree)

    def test_slash_in_group_name_raises(self, random_patch):
        """A group value which cannot name a tree node raises."""
        # Two values are needed: a lone group is named by its ordinal.
        patches = [
            random_patch.update_attrs(cable_id="a/b"),
            random_patch.update_attrs(cable_id="c/d"),
        ]
        with pytest.raises(PatchConversionError, match="cannot name"):
            dc.spool(patches).io.to_xarray(group="cable_id")


class TestToXarrayReadArray:
    """Tests wiring the read_array fast path into to_xarray blocks."""

    @pytest.fixture(autouse=True)
    def _require_libs(self):
        """These tests need both optional libraries."""
        pytest.importorskip("xarray")
        pytest.importorskip("dask")

    @pytest.fixture(scope="class")
    def dasdae_directory(self, tmp_path_factory):
        """A directory of single-patch DASDAE files with distinct data.

        Distinct arrays per file, or a block reading the wrong member
        would still pass the parity assertions.
        """
        path = tmp_path_factory.mktemp("to_xarray_read_array")
        for num, patch in enumerate(dc.get_example_spool()):
            patch.new(data=patch.data + num).io.write(
                path / f"patch_{num}.h5", "dasdae"
            )
        return path

    @pytest.fixture
    def override_calls(self, array_reads):
        """The windows of stored arrays reads load, in each source's order."""
        return array_reads

    def _leaf(self, tree):
        """The first dataset holding a data variable."""
        return next(node for node in tree.subtree if "data" in node.dataset)

    def test_mixed_unit_members_merge(self, tmp_path):
        """Members spelling one distance in metres and feet join as chunk does.

        Their rows state an integer and a float coordinate; the lazy tree
        builds both in the plan's units so the segments share a dtype.
        """
        metres = dc.get_example_patch().set_units(distance="m")
        dist = metres.get_coord("distance")
        span = float(dist.max() - dist.min() + dist.step)
        feet = metres.update_coords(distance=(dist.values + span) / 0.3048)
        feet = feet.set_units(distance="ft")
        dc.write(metres, tmp_path / "m.h5", "dasdae")
        dc.write(feet, tmp_path / "ft.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        out = self._leaf(spool.io.to_xarray(dim="distance"))["data"].data.compute()
        assert np.array_equal(out, spool.chunk(distance=None)[0].data)

    def test_fast_path_loads_blocks(
        self, dasdae_directory, override_calls, monkeypatch
    ):
        """With an override, computing never builds a member Patch."""
        spool = dc.spool(dasdae_directory).update()
        eager = spool.chunk(time=None)[0].data
        # The eager reference can use dev's array fast path too; count
        # only reads made by constructing and computing the xarray tree.
        override_calls.clear()
        tree = spool.io.to_xarray()
        assert not override_calls

        def _fail(*args, **kwargs):
            raise AssertionError("fast path fell back to patch loading")

        monkeypatch.setattr(PlanResolver, "_load_member", _fail)
        out = self._leaf(tree)["data"].data.compute()
        assert np.array_equal(out, eager)
        assert len(override_calls) == len(spool)

    def test_residual_spool_falls_back(self, dasdae_directory, override_calls):
        """A samples-selected spool loads through the exact patch path."""
        spool = dc.spool(dasdae_directory).update()
        sub = spool.select(time=(2, 100), samples=True)
        out = self._leaf(sub.io.to_xarray())["data"].data.compute()
        assert np.array_equal(out, sub.chunk(time=None)[0].data)
        assert override_calls == []

    def test_chunked_spool_reads_file_windows(self, dasdae_directory, override_calls):
        """A plan-backed spool's trimmed rows read windows of their files.

        Its collapsed member rows state trimmed envelopes beside each
        file's own range, so each window is placed on the file's grid;
        one placed on the trimmed envelope would read the wrong samples.
        """
        spool = dc.spool(dasdae_directory).update().chunk(time=3)
        eager = spool.chunk(time=None)[0].data
        override_calls.clear()
        out = self._leaf(spool.io.to_xarray())["data"].data.compute()
        assert np.array_equal(out, eager)
        assert len(override_calls) > len(dc.spool(dasdae_directory))

    def test_interior_window_fast_path(self, tmp_path, override_calls):
        """An overlap-trimmed member reads an interior file window.

        Two half-overlapping files merge into one segment, so the second
        member's window starts mid-file — the case where a wrong window
        anchor would silently read the wrong samples.
        """
        first = dc.get_example_patch()
        time = first.get_coord("time")
        half = time.values[len(time) // 2]
        # distinct data so reading the wrong file cannot pass parity
        second = first.update_coords(time_min=half).new(data=first.data + 1)
        for num, patch in enumerate((first, second)):
            patch.update_attrs(history=[]).io.write(tmp_path / f"p{num}.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        eager = spool.chunk(time=None)[0].data
        override_calls.clear()
        out = self._leaf(spool.io.to_xarray())["data"].data.compute()
        assert np.array_equal(out, eager)
        # the trimmed member's window must not be anchored at the start
        axis = first.dims.index("time")
        starts = sorted(windows[axis][0] for _, windows in override_calls)
        assert len(override_calls) == 2
        assert starts[0] == 0 and starts[1] > 0

    def test_mixed_data_units_convert_as_chunk(self, tmp_path, override_calls):
        """A file in other data units loads as a patch, converted to the first's."""
        first = dc.get_example_patch()
        time = first.get_coord("time")
        second = first.update_coords(time_min=time.max() + time.step)
        first.set_units("m/s").io.write(tmp_path / "a.h5", "dasdae")
        second.new(data=first.data * 100).set_units("cm/s").io.write(
            tmp_path / "b.h5", "dasdae"
        )
        spool = dc.spool(tmp_path).update()
        kwargs = dict(group=[], conflict="keep_first")
        (merged,) = spool.chunk(time=None, **kwargs)
        override_calls.clear()
        out = self._leaf(spool.io.to_xarray(**kwargs))["data"].values
        np.testing.assert_allclose(out, merged.data)
        # the first is in the merged units already, so reads as a window
        assert [path for path, _ in override_calls] == [str(tmp_path / "a.h5")]

    def test_a_file_rewritten_after_building_raises(self, tmp_path):
        """A window read which is not the shape indexed says to update."""
        patch = dc.get_example_patch()
        path = tmp_path / "p.h5"
        patch.io.write(path, "dasdae")
        tree = dc.spool(tmp_path).update().io.to_xarray()
        path.unlink()
        patch.select(distance=(0, 10), samples=True).io.write(path, "dasdae")
        with pytest.raises(PatchConversionError, match="promised"):
            self._leaf(tree)["data"].compute()

    def test_a_changed_local_file_is_refused(self, tmp_path):
        """A file changed since indexing is refused when the tree is built."""
        for num, patch in enumerate(dc.get_example_spool()):
            patch.io.write(tmp_path / f"p{num}.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        assert self._leaf(spool.io.to_xarray())  # untouched files convert
        path = tmp_path / "p1.h5"
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        with pytest.raises(PatchConversionError, match=r"spool.update\(\)"):
            spool.io.to_xarray()

    @staticmethod
    def _touch(path):
        """Move a file's mtime on a second, as a rewrite would."""
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))

    @pytest.mark.parametrize(
        "derive",
        [
            lambda x: x.chunk(distance=100),
            lambda x: x.chunk(time=None).select(time=(0, 10), samples=True),
        ],
        ids=["other dim", "selected"],
    )
    def test_a_changed_file_under_a_nested_plan_is_refused(self, tmp_path, derive):
        """A plan loading another plan's outputs is checked down to its files."""
        for num, patch in enumerate(dc.get_example_spool()):
            patch.io.write(tmp_path / f"p{num}.h5", "dasdae")
        spool = derive(dc.spool(tmp_path).update())
        assert self._leaf(spool.io.to_xarray())  # untouched files convert
        self._touch(tmp_path / "p1.h5")
        with pytest.raises(PatchConversionError, match=r"spool.update\(\)"):
            spool.io.to_xarray()

    def test_an_unmeasurable_member_checks_the_rest(self, tmp_path, random_patch):
        """A patch held in memory beside the files does not excuse a changed one."""
        for num, patch in enumerate(dc.get_example_spool()):
            patch.io.write(tmp_path / f"p{num}.h5", "dasdae")
        later = random_patch.update_coords(time_min=np.datetime64("2030-01-01"))
        spool = dc.spool(tmp_path).update() + dc.spool([later])
        assert len(_segments(spool.io.to_xarray())) == 2
        self._touch(tmp_path / "p1.h5")
        with pytest.raises(PatchConversionError, match=r"spool.update\(\)"):
            spool.io.to_xarray()

    def test_a_remote_file_keeps_its_storage_options(self, tmp_path, token_store):
        """A window read opens a remote file with the options its path carries."""
        dc.get_example_patch().io.write(tmp_path / "p.h5", "dasdae")
        path = UPath("tokenmem://bucket/p.h5", token="secret")
        path.write_bytes((tmp_path / "p.h5").read_bytes())
        spool = dc.spool(path)
        out = self._leaf(spool.io.to_xarray())["data"].values
        assert np.array_equal(out, spool.chunk(time=None)[0].data)

    @pytest.mark.parametrize("engine", ["h5netcdf", "zarr"])
    def test_the_tree_writes_with_its_units(self, tmp_path, random_spool, engine):
        """A store holds the segment's attrs, data units included."""
        xr = pytest.importorskip("xarray")
        pytest.importorskip(engine)
        spool = dc.spool([x.update_attrs(data_units="m/s") for x in random_spool])
        tree = spool.io.to_xarray()
        path = tmp_path / "tree"
        if engine == "zarr":
            tree.to_zarr(path)
        else:
            tree.to_netcdf(path, engine=engine)
        back = xr.open_datatree(path, engine=engine)
        assert self._leaf(back)["data"].attrs["data_units"] == "m / s"


class _TokenStore(MemoryFileSystem):
    """An in-memory store which opens nothing without its token."""

    protocol = "tokenmem"
    store: typing.ClassVar[dict] = {}
    pseudo_dirs: typing.ClassVar[list] = [""]

    def __init__(self, token=None, **kwargs):
        super().__init__(**kwargs)
        self.token = token

    @classmethod
    def _strip_protocol(cls, path):
        """Name paths as the memory store does."""
        path = str(path).replace(f"{cls.protocol}://", "memory://")
        return MemoryFileSystem._strip_protocol(path)

    def _open(self, path, *args, **kwargs):
        if self.token != "secret":
            raise PermissionError("no token")
        return super()._open(path, *args, **kwargs)


@pytest.fixture
def token_store():
    """Register the token store for one test."""
    fsspec.register_implementation(_TokenStore.protocol, _TokenStore, clobber=True)
    yield
    _TokenStore.store.clear()


class TestToXarrayLazyCoords:
    """The tree's evenly sampled time coordinates are served lazily."""

    @pytest.fixture(autouse=True)
    def _require_libs(self):
        """These tests need both optional libraries."""
        pytest.importorskip("xarray")
        pytest.importorskip("dask")

    def _leaf(self, tree):
        """The first dataset holding a data variable."""
        return next(node for node in tree.subtree if "data" in node.dataset)

    def test_time_coordinate_is_lazy(self, random_spool):
        """An evenly sampled merged time coordinate gets the lazy index."""
        from dascore.xarray.index import CoordIndex  # noqa: PLC0415

        data = self._leaf(random_spool.io.to_xarray())["data"]
        assert isinstance(data.xindexes["time"], CoordIndex)
        # the labels are served on demand, not stored as an array
        assert not isinstance(data["time"].variable._data, np.ndarray)
        merged = random_spool.chunk(time=None)[0]
        np.testing.assert_array_equal(
            data["time"].values, merged.get_coord("time").values
        )

    def test_only_the_merged_dimension_is_lazy(self, random_spool):
        """Distance keeps an ordinary index, so per-channel arrays combine."""
        import xarray as xr  # noqa: PLC0415

        from dascore.xarray.index import CoordIndex  # noqa: PLC0415

        data = self._leaf(random_spool.io.to_xarray())["data"]
        assert type(data.xindexes["distance"]).__name__ == "PandasIndex"
        distance = data["distance"].values
        gains = xr.DataArray(np.ones(len(distance)), coords={"distance": distance})
        assert (data * gains).sizes == data.sizes
        # along the merged dimension, an ordinary index is one swap away
        times = xr.DataArray(
            np.ones(data.sizes["time"]), coords={"time": data["time"].values}
        )
        assert isinstance(data.xindexes["time"], CoordIndex)
        eager = data.drop_indexes("time").set_xindex("time")
        assert (eager * times).sizes == data.sizes

    def test_sel_matches_patch_select(self, random_spool):
        """Label selection on the tree equals dascore's own select."""
        data = self._leaf(random_spool.io.to_xarray())["data"]
        merged = random_spool.chunk(time=None)[0]
        t = merged.get_coord("time").values
        sub = data.sel(time=slice(t[100], t[300]))
        expected = merged.select(time=(t[100], t[300]))
        assert sub.sizes["time"] == expected.shape[expected.dims.index("time")]
        np.testing.assert_array_equal(sub.compute().values, expected.data)

    def test_segmented_time_stays_lazy(self, random_patch):
        """A jittered merge is not one range; it is served as its segments."""
        from dascore.xarray.index import CoordIndex  # noqa: PLC0415

        coord = random_patch.get_coord("time")
        step = coord.step * 1.04  # within sampling tolerance, off-grid
        second = random_patch.update_coords(
            time_min=coord.max() + coord.step, time_step=step
        )
        spool = dc.spool([random_patch, second])
        data = self._leaf(spool.io.to_xarray())["data"]
        index = data.xindexes["time"]
        assert isinstance(index, CoordIndex)
        assert index.coordinate.runs_count > 1
        merged = spool.chunk(time=None)[0]
        np.testing.assert_array_equal(
            data["time"].values, merged.get_coord("time").values
        )


class TestToXarrayBlockSize:
    """A source patch larger than `block_size` is read in several windows."""

    @pytest.fixture(autouse=True)
    def _require_libs(self):
        """These tests need both optional libraries."""
        pytest.importorskip("xarray")
        pytest.importorskip("dask")

    @pytest.fixture(scope="class")
    def dasdae_directory(self, tmp_path_factory):
        """Three adjacent DASDAE files with distinct data per file."""
        path = tmp_path_factory.mktemp("to_xarray_block_size")
        spool = dc.get_example_spool("random_das", length=3, time_gap=0)
        for num, patch in enumerate(spool):
            patch.new(data=patch.data + num).io.write(path / f"p{num}.h5", "dasdae")
        return path

    @pytest.fixture(scope="class")
    def file_spool(self, dasdae_directory):
        """The indexed spool over those files."""
        return dc.spool(dasdae_directory).update()

    @staticmethod
    def _leaf(tree):
        """The one data variable such a single-group tree holds."""
        leaves = [node for node in tree.subtree if "data" in node.dataset]
        assert len(leaves) == 1
        return leaves[0].dataset["data"]

    def test_a_member_splits_into_several_blocks(self, file_spool):
        """A block ceiling below a member's size cuts it into pieces."""
        quarter = file_spool[0].data.nbytes // 4
        whole = self._leaf(file_spool.io.to_xarray(block_size=0))
        split = self._leaf(file_spool.io.to_xarray(block_size=quarter))
        assert whole.data.npartitions == len(file_spool)
        assert split.data.npartitions == 4 * len(file_spool)
        # and the pieces say the same thing the one block said
        assert np.array_equal(split.compute().values, whole.compute().values)

    def test_pieces_hold_every_sample_once(self, file_spool):
        """The split values equal the eagerly chunked patch, in order."""
        eighth = file_spool[0].data.nbytes // 8
        data = self._leaf(file_spool.io.to_xarray(block_size=eighth))
        merged = file_spool.chunk(time=None)[0]
        expected = merged.transpose(*data.dims).data
        assert np.array_equal(data.compute().values, expected)
        assert np.array_equal(np.asarray(data["time"].values), merged.get_array("time"))

    @staticmethod
    def _time_windows(spool, reads):
        """The time window of each recorded read."""
        axis = spool[0].dims.index("time")
        return [windows[axis] for _, windows in reads]

    @pytest.mark.parametrize("quarters", [0, 1])
    def test_a_selection_reads_only_its_samples(
        self, file_spool, array_reads, quarters
    ):
        """A window reaches the reader as itself, whatever the blocks are.

        The selection fuses into the segment source's own read, so the
        blocks bound a bulk read and say nothing about this one.
        """
        block_size = quarters * file_spool[0].data.nbytes // 4
        data = self._leaf(file_spool.io.to_xarray(block_size=block_size))
        piece = data.isel(time=slice(3, 11)).compute()
        assert self._time_windows(file_spool, array_reads) == [(3, 11)]
        assert piece.sizes["time"] == 8

    def test_a_selection_reads_only_the_members_it_covers(
        self, file_spool, array_reads
    ):
        """A window inside one member never opens the others."""
        samples = len(file_spool[0].get_coord("time"))
        data = self._leaf(file_spool.io.to_xarray())
        # a window straddling the first seam covers two of three members
        straddle = data.isel(time=slice(samples - 2, samples + 2)).compute()
        assert straddle.sizes["time"] == 4
        assert len({path for path, _ in array_reads}) == 2

    def test_a_selection_on_another_dimension_reads_less(self, file_spool, array_reads):
        """A distance window is pushed into the read, not applied after."""
        data = self._leaf(file_spool.io.to_xarray())
        axis = file_spool[0].dims.index("distance")
        narrow = data.isel(distance=slice(0, 3)).compute()
        assert narrow.sizes["distance"] == 3
        assert {windows[axis] for _, windows in array_reads} == {(0, 3)}

    @pytest.mark.parametrize("block_size", [0, 1_000_000])
    def test_every_index_form_matches_the_merged_patch(self, file_spool, block_size):
        """Reading is contiguous, so other index forms go through a window.

        An integer drops its dimension, a stride and a reversal pick
        from the span they cover, and an empty selection reads nothing;
        each must still answer what the merged patch holds.
        """
        data = self._leaf(file_spool.io.to_xarray(block_size=block_size))
        merged = file_spool.chunk(time=None)[0]
        whole = merged.transpose(*data.dims).data
        samples = data.sizes["time"]
        seam = len(file_spool[0].get_coord("time"))
        cases = {
            "head": (slice(0, 5), whole[:, :5]),
            "across a seam": (slice(seam - 5, seam + 5), whole[:, seam - 5 : seam + 5]),
            "integer": (3, whole[:, 3]),
            "negative integer": (-2, whole[:, -2]),
            "stride": (slice(0, 20, 3), whole[:, 0:20:3]),
            "reversed": (slice(None, None, -1), whole[:, ::-1]),
            "positions": ([1, 5, samples - 1], whole[:, [1, 5, samples - 1]]),
            "empty": (slice(0, 0), whole[:, 0:0]),
            "no positions": ([], whole[:, []]),
        }
        for name, (index, expected) in cases.items():
            got = np.asarray(data.isel(time=index).compute().values)
            assert np.array_equal(got, expected), name
        # and an index on both dimensions at once
        pair = data.isel(distance=[0, 3], time=slice(0, 7)).compute().values
        assert np.array_equal(pair, whole[[0, 3]][:, :7])

    @pytest.mark.parametrize(
        "index",
        [
            dict(distance=5, time=slice(0, 1500, 999)),
            dict(distance=slice(0, 6, 2), time=slice(0, 1500, 999)),
            dict(distance=0, time=slice(5, 5)),
        ],
    )
    def test_a_block_the_selection_misses_keeps_its_shape(self, random_patch, index):
        """A block holding none of a selection is empty in the selection's shape."""
        halves = (
            random_patch.select(time=(0, 1000), samples=True),
            random_patch.select(time=(1000, None), samples=True),
        )
        data = self._leaf(dc.spool(halves).io.to_xarray())
        expected = random_patch.transpose(*data.dims).isel(**index)
        assert np.array_equal(data.isel(**index).compute().values, expected.data)

    def test_a_window_ending_on_a_seam_reads_one_member(self, file_spool, array_reads):
        """The member after a window's end is not opened for no samples."""
        samples = len(file_spool[0].get_coord("time"))
        data = self._leaf(file_spool.io.to_xarray())
        data.isel(time=slice(0, samples)).compute()
        assert self._time_windows(file_spool, array_reads) == [(0, samples)]
        array_reads.clear()
        self._leaf(file_spool.io.to_xarray(block_size=0)).compute()
        windows = self._time_windows(file_spool, array_reads)
        assert len(windows) == len(file_spool)
        assert all(stop > start for start, stop in windows)

    def test_a_selection_missing_every_member_reads_nothing(
        self, file_spool, array_reads
    ):
        """An empty window has no member to ask, and is empty rather than absent."""
        data = self._leaf(file_spool.io.to_xarray())
        out = data.isel(time=slice(0, 0)).compute()
        assert out.sizes["time"] == 0
        assert array_reads == []

    def test_a_patch_read_still_narrows_other_dimensions(self, file_spool, monkeypatch):
        """A member loading as a patch applies the other dimensions' windows itself."""
        monkeypatch.setattr(PlanResolver, "_member_array_source", lambda *args: None)
        data = self._leaf(file_spool.io.to_xarray())
        merged = file_spool.chunk(time=None)[0]
        whole = merged.transpose(*data.dims).data
        got = data.isel(distance=slice(0, 4), time=slice(0, 6)).compute().values
        assert np.array_equal(got, whole[:4, :6])

    def test_a_block_reads_only_its_own_window(self, file_spool, array_reads):
        """Computing one piece reads that piece's samples, not the file."""
        quarter = file_spool[0].data.nbytes // 4
        data = self._leaf(file_spool.io.to_xarray(block_size=quarter))
        samples = len(file_spool[0].get_coord("time"))
        piece = data.isel(time=slice(0, samples // 4)).compute()
        assert self._time_windows(file_spool, array_reads) == [(0, samples // 4)]
        assert piece.sizes["time"] == samples // 4

    def test_a_member_the_index_cannot_window_stays_whole(self, random_spool):
        """An in-memory member reads as a patch, so splitting it would cost."""
        tree = random_spool.io.to_xarray(block_size=1)
        assert self._leaf(tree).data.npartitions == len(random_spool)

    def test_block_size_accepts_a_string(self, file_spool):
        """A dask byte string sizes blocks like the count it names."""
        quarter = file_spool[0].data.nbytes // 4
        named = self._leaf(file_spool.io.to_xarray(block_size=f"{quarter}B"))
        counted = self._leaf(file_spool.io.to_xarray(block_size=quarter))
        assert named.data.chunks == counted.data.chunks

    def test_the_default_leaves_ordinary_files_whole(self, file_spool):
        """A file well under the default ceiling is still one block."""
        data = self._leaf(file_spool.io.to_xarray())
        assert data.data.npartitions == len(file_spool)

    @pytest.mark.parametrize("block_size", [-1, "-1MiB"])
    def test_a_negative_ceiling_is_refused(self, file_spool, block_size):
        """No size fits in a negative budget, so it is an error, not one sample."""
        with pytest.raises(PatchConversionError, match="block_size"):
            file_spool.io.to_xarray(block_size=block_size)

    def test_the_config_sets_the_default(self, file_spool):
        """An unset `block_size` takes the configured ceiling."""
        quarter = file_spool[0].data.nbytes // 4
        with config_context(xarray_block_size=quarter):
            configured = self._leaf(file_spool.io.to_xarray())
        passed = self._leaf(file_spool.io.to_xarray(block_size=quarter))
        assert configured.data.chunks == passed.data.chunks
        # and an argument still wins over the configured value
        with config_context(xarray_block_size=quarter):
            override = self._leaf(file_spool.io.to_xarray(block_size=0))
        assert override.data.npartitions == len(file_spool)

    def test_a_file_member_loading_as_a_patch_stays_whole(
        self, file_spool, monkeypatch
    ):
        """A file member which cannot be windowed is one block, read once.

        Splitting it would load the whole file once per piece.
        """
        monkeypatch.setattr(PlanResolver, "_member_array_source", lambda *args: None)
        quarter = file_spool[0].data.nbytes // 4
        data = self._leaf(file_spool.io.to_xarray(block_size=quarter))
        assert data.data.npartitions == len(file_spool)
        merged = file_spool.chunk(time=None)[0]
        assert np.array_equal(data.compute().values, merged.transpose(*data.dims).data)

    def test_pieces_of_an_interior_window_stay_anchored(self, tmp_path):
        """A member trimmed by an overlap splits from where it starts.

        Two half-overlapping files merge into one segment, so the second
        member's window starts mid-file. Its pieces are offsets from that
        start, not from the file's; anchoring them at zero would read the
        wrong samples and still fill every block.
        """
        first = dc.get_example_patch()
        time = first.get_coord("time")
        half = time.values[len(time) // 2]
        # distinct data, or reading the wrong samples would still match
        second = first.update_coords(time_min=half).new(data=first.data + 1)
        for num, patch in enumerate((first, second)):
            patch.update_attrs(history=[]).io.write(tmp_path / f"p{num}.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        merged = spool.chunk(time=None)[0]
        data = self._leaf(spool.io.to_xarray(block_size=merged.data.nbytes // 8))
        assert data.data.npartitions > 2
        assert np.array_equal(data.compute().values, merged.transpose(*data.dims).data)


class TestBlockPieces:
    """The sample cut behind `block_size`."""

    def test_a_count_under_the_limit_is_one_piece(self):
        """Nothing is cut when the whole thing fits."""
        from dascore.xarray.spool import _block_pieces  # noqa: PLC0415

        assert _block_pieces(10, 10) == ((0, 10),)
        assert _block_pieces(10, None) == ((0, 10),)

    def test_pieces_tile_the_count(self):
        """Every sample lands in exactly one piece, in order."""
        from dascore.xarray.spool import _block_pieces  # noqa: PLC0415

        for count, limit in ((10, 3), (10, 4), (1000, 7), (5, 1)):
            pieces = _block_pieces(count, limit)
            assert pieces[0][0] == 0 and pieces[-1][1] == count
            assert all(b == pieces[i + 1][0] for i, (_, b) in enumerate(pieces[:-1]))
            assert all(0 < b - a <= limit for a, b in pieces)

    @pytest.mark.parametrize(
        "count,limit", [(100, 30), (101, 30), (10, 3), (1000, 7), (37, 5)]
    )
    def test_pieces_are_even(self, count, limit):
        """No piece is more than one sample shorter than another.

        The counts which do not divide are the point: a size repeated
        until the samples run out leaves the whole deficit in the last
        piece, which the evenly divisible cases cannot show.
        """
        from dascore.xarray.spool import _block_pieces  # noqa: PLC0415

        sizes = [b - a for a, b in _block_pieces(count, limit)]
        assert max(sizes) - min(sizes) <= 1

    def test_one_sample_rows_still_split(self):
        """A row costing more than the ceiling still yields whole samples."""
        from dascore.xarray.spool import _samples_per_block  # noqa: PLC0415

        limit = _samples_per_block(10, np.dtype("float64"), {"x": 100, "t": 5}, "t")
        assert limit == 1

    def test_a_zero_budget_is_no_limit(self):
        """Zero says one block per patch, not a block holding no samples."""
        from dascore.xarray.spool import _samples_per_block  # noqa: PLC0415

        assert _samples_per_block(0, np.dtype("float64"), {"t": 5}, "t") is None
        assert _samples_per_block(None, np.dtype("float64"), {"t": 5}, "t") is None


class TestToXarrayExactGrid:
    """A spool whose coordinates carry an exact grid converts like any other."""

    @pytest.fixture(autouse=True)
    def _require_libs(self):
        """These tests need both optional libraries."""
        pytest.importorskip("xarray")
        pytest.importorskip("dask")

    @pytest.fixture(scope="class")
    def third_second_spool(self, tmp_path_factory):
        """Five adjacent files sampled on an exact one third second grid."""
        path = tmp_path_factory.mktemp("exact_grid_tree")
        start = np.datetime64("2020-01-01")
        for num in range(5):
            time = dc.core.get_coord(start=start, step=(1, 3), shape=(20,))
            distance = dc.core.get_coord(start=0.0, step=1.0, shape=(4,))
            patch = dc.Patch(
                data=np.random.default_rng(num).random((20, 4)),
                coords={"time": time, "distance": distance},
                dims=("time", "distance"),
            )
            patch.io.write(path / f"g{num}.h5", "dasdae")
            start = time.max() + time.step
        return dc.spool(path).update()

    def test_a_sample_selection_converts(self, third_second_spool):
        """A window of an exact grid sizes its blocks as it reads them."""
        tree = third_second_spool.select(time=(1, -1), samples=True).io.to_xarray()
        leaves = [x for x in tree.subtree if "data" in x.dataset]
        assert len(leaves) == 5
        for leaf in leaves:
            array = leaf.dataset["data"]
            assert np.asarray(array.values).shape == array.shape


def _segments(tree):
    """The data variables of a tree, in node order."""
    return [node.dataset["data"] for node in tree.subtree if "data" in node.dataset]


def _assert_attrs_match(data, patch):
    """
    A segment's attrs are the patch's, as a store can hold them.

    Data units appear as a string under ``data_units`` and in CF form under
    ``units``; history is absent; an id appears only where the rows know
    chunk's, and then equals it.
    """
    expected = {k: v for k, v in dict(patch.attrs).items() if v is not None}
    expected.pop("history", None)
    if (units := expected.get("data_units")) is not None:
        expected["data_units"] = get_quantity_str(units)
        expected["units"] = _cf_unit_str(units)
    got = dict(data.attrs)
    for name in ("data_id", "origin_id"):
        if name not in got:
            expected.pop(name, None)
    assert got == expected


@pytest.fixture(scope="module")
def requires_xarray_dask():
    """Skip without both optional libraries."""
    pytest.importorskip("xarray")
    pytest.importorskip("dask")


@pytest.fixture(scope="module")
def patches():
    """Three abutting patches, then a fourth past a gap; distinct data."""
    spool = dc.get_example_spool("random_das", length=4, shape=(20, 500))
    out = [
        patch.new(data=patch.data + num).set_units("m/s")
        for num, patch in enumerate(spool)
    ]
    gap = np.timedelta64(60, "s")
    out[3] = out[3].update_coords(time_min=out[3].get_coord("time").min() + gap)
    return out


@pytest.mark.usefixtures("requires_xarray_dask")
class TestTreeMatchesChunk:
    """
    Every segment equals the patch `chunk(time=None)` merges, in order.

    Pins the tree against the eager reference for in-memory and
    file-backed members, multi-member segments, mixed dtypes, transposed
    members, sample selections and members trimmed mid-file.
    """

    @pytest.fixture(params=["memory", "dasdae"])
    def to_spool(self, request, tmp_path):
        """Build a spool from patches, held in memory or in DASDAE files."""
        if request.param == "memory":
            return dc.spool
        return lambda patches: dc.spool(spool_to_directory(patches, tmp_path)).update()

    @staticmethod
    def _assert_matches(tree, expected):
        """Each segment's data, dims, dtype, coords and units equal the patch's."""
        segments = _segments(tree)
        assert len(segments) == len(expected)
        for data, patch in zip(segments, expected, strict=True):
            assert data.dims == patch.dims
            assert data.dtype == patch.data.dtype
            values = data.values
            # the computed blocks, not just the declared dtype
            assert values.dtype == patch.data.dtype
            np.testing.assert_array_equal(values, patch.data)
            out = data.dc.to_patch()
            for dim in patch.dims:
                np.testing.assert_array_equal(
                    out.get_coord(dim).values, patch.get_coord(dim).values
                )
            # Only the merged dimension's units survive today; the others
            # are pinned by TestKnownDivergences.test_non_merged_dim_units.
            assert out.get_coord("time").units == patch.get_coord("time").units
            assert out.attrs.data_units == patch.attrs.data_units
            _assert_attrs_match(data, patch)

    def test_segments(self, patches, to_spool):
        """A gap splits two segments; the first merges three members."""
        spool = to_spool(patches)
        tree = spool.io.to_xarray()
        assert len(_segments(tree)) == 2
        self._assert_matches(tree, spool.chunk(time=None))
        # whole members state the ids chunk gives, merged or not
        assert all("data_id" in x.attrs for x in _segments(tree))

    @pytest.mark.parametrize("conflict", ["keep_first", "drop"])
    @pytest.mark.parametrize("units, dtype", [("cm", np.float64), (None, np.int16)])
    def test_members_in_other_data_units(
        self, patches, to_spool, conflict, units, dtype
    ):
        """
        int16 members in m and cm merge as chunk merges them.

        chunk converts the second to the first's units, which makes it
        float64, and promotes the merge to that; casting back to int16
        would turn every 0.01 into 0. A member stating no units takes
        the first's and keeps its values.
        """
        ones = np.ones(patches[0].shape, dtype=np.int16)
        spool = to_spool(
            [
                patches[0].new(data=ones).set_units("m"),
                patches[1].new(data=ones).set_units(units),
            ]
        )
        kwargs = dict(conflict=conflict, group=[])
        tree = spool.io.to_xarray(**kwargs)
        assert _segments(tree)[0].dtype == dtype
        assert _segments(tree)[0].attrs["data_units"] == "m"
        self._assert_matches(tree, spool.chunk(time=None, **kwargs))

    @pytest.mark.parametrize("read", ["window", "patch"])
    def test_lossy_promotion_chain(self, patches, tmp_path, monkeypatch, read):
        """
        A member rounds where the merge's buffer rounded it, on either read.

        int64 2**53 + 1 goes through float64 on its way to longdouble,
        which keeps 2**53; a direct cast to longdouble would keep the 1.
        """
        specs = (("int64", 2**53 + 1), ("float64", 1), ("longdouble", 2))
        members = [
            patch.new(data=np.full(patch.shape, value, dtype=dtype))
            for patch, (dtype, value) in zip(patches, specs)
        ]
        spool = dc.spool(spool_to_directory(members, tmp_path)).update()
        expected = spool.chunk(time=None)
        if read == "patch":
            monkeypatch.setattr(
                PlanResolver, "_member_array_source", lambda *args: None
            )
        tree = spool.io.to_xarray()
        assert _segments(tree)[0].values[0, 0] == np.longdouble(2**53)
        self._assert_matches(tree, expected)

    def test_associated_coords(self, patches, to_spool):
        """Associated coordinates leave the attrs as chunk's patch states them."""
        size = len(patches[0].get_coord("distance"))
        latitude = ("distance", np.linspace(10, 11, size))
        spool = to_spool([x.update_coords(latitude=latitude) for x in patches[:2]])
        self._assert_matches(spool.io.to_xarray(), spool.chunk(time=None))

    def test_merge_dim_coord_differs(self, patches, to_spool):
        """
        A coordinate on time which differs per member does not conflict.

        The tree does not carry associated coordinates, so only the data
        and dimension coordinates are compared with chunk's patch.
        """
        members = [
            x.update_coords(
                power=("time", np.full(len(x.get_coord("time")), float(num)))
            ).update_attrs(history=[])
            for num, x in enumerate(patches[:2])
        ]
        spool = to_spool(members)
        self._assert_matches(spool.io.to_xarray(), spool.chunk(time=None))

    @pytest.mark.parametrize(
        "select, window",
        [
            (dict(distance=(1, None)), None),
            (dict(distance=(1, None)), 100),
            (dict(time=(1, -1)), None),
            (dict(time=(1, -1)), 600),
        ],
        ids=["off", "off-cut", "on", "on-wide"],
    )
    def test_rechunked_sample_selection_refused(
        self, patches, to_spool, select, window
    ):
        """
        A same-dimension chunk of a sample selection is refused when built.

        Re-planning it collapses onto the source rows and loses the
        selection wherever a member is a whole source or the selection
        is on another dimension, so its members would load unselected:
        wrong samples from files, a false "changed" from memory.
        """
        spool = to_spool(patches[:2]).select(samples=True, **select)
        if window is None:
            spool = spool.chunk(time=None)
        else:
            step = patches[0].get_coord("time").step
            spool = spool.chunk(time=step * window, keep_partial=True)
        with pytest.raises(PatchConversionError, match="sample selection"):
            spool.io.to_xarray()

    @pytest.mark.parametrize("select", [dict(time=(1, -1)), dict(time=(5, None))])
    def test_rechunked_sample_selection(self, patches, to_spool, select):
        """
        A sample selection chunked into windows of its members converts.

        Every member is a window inside its selected source, which the
        re-plan loads as chunk does.
        """
        spool = to_spool(patches[:2]).select(samples=True, **select)
        step = patches[0].get_coord("time").step
        spool = spool.chunk(time=step * 100, keep_partial=True)
        self._assert_matches(spool.io.to_xarray(), spool.chunk(time=None))

    @staticmethod
    def _selected_chunk(starts):
        """Chunk 20-second patches starting at ``starts``, then drop a channel."""
        members = [
            dc.Patch(
                data=np.random.default_rng(num).random((3, 20)),
                coords={
                    "distance": np.arange(3),
                    "time": dc.get_coord(
                        start=dc.to_datetime64("2020-01-01") + dc.to_timedelta64(x),
                        step=dc.to_timedelta64(1),
                        shape=(20,),
                    ),
                },
                dims=("distance", "time"),
            )
            for num, x in enumerate(starts)
        ]
        spool = dc.spool(members).chunk(time=None)
        return spool.select(distance=(1, None), samples=True)

    @pytest.mark.parametrize("starts", [(0, 20), (0, 10)])
    def test_selection_of_an_exact_chunk(self, starts):
        """A selected chunk whose envelope is its merged grid converts as chunk."""
        spool = self._selected_chunk(starts)
        self._assert_matches(spool.io.to_xarray(), spool.chunk(time=None))

    @pytest.mark.parametrize("starts", [(0, 15.2), (0, 20.5, 41)])
    def test_selection_of_a_snapped_chunk_refused(self, starts):
        """
        A selected chunk which snapped its members is refused when built.

        Its envelope ends where its last member did, not where the merged
        grid does (an off-grid end, or one drifted a whole sample), so it
        cannot size the patch it loads.
        """
        spool = self._selected_chunk(starts)
        with pytest.raises(PatchConversionError, match="snapped"):
            spool.io.to_xarray()

    @pytest.mark.parametrize("select", [None, dict(time=(1, None))])
    @pytest.mark.parametrize("source", ["memory", "dasdae"])
    @pytest.mark.parametrize(
        "grid", [(550.0, 99.0, "cm"), (5.5, 0.99, "m"), (1000.0, 100.0, "cm")]
    )
    def test_trimmed_member_off_grid(self, tmp_path, source, select, grid):
        """
        A member trimmed between two of its samples keeps its own labels.

        The second member overlaps the first and is trimmed at a bound
        which is not one of its samples (counted in cm or in m); its
        labels are its own samples past that bound, as chunk gives, not
        ones counted from the bound. One which abuts it in cm is only
        converted, which changes its id as chunk's load changes it.
        """
        time = dc.get_coord(
            start=np.datetime64("2020-01-01"), step=np.timedelta64(1, "s"), shape=(4,)
        )
        members = [
            dc.Patch(
                data=np.arange(40.0).reshape(10, 4) + 100 * num,
                coords={
                    "distance": dc.get_coord(
                        start=start, step=step, shape=(10,), units=units
                    ),
                    "time": time,
                },
                dims=("distance", "time"),
            )
            for num, (start, step, units) in enumerate([(0.0, 1.0, "m"), grid])
        ]
        spool = dc.spool(members)
        if source == "dasdae":
            # members sharing a time range need a directory each
            for num, patch in enumerate(members):
                spool_to_directory([patch], tmp_path / str(num))
            spool = dc.spool(tmp_path).update()
        if select is not None:
            spool = spool.select(samples=True, **select)
        tree = spool.io.to_xarray(dim="distance", group=[])
        self._assert_matches(tree, spool.chunk(distance=None, group=[]))

    def test_mixed_dtypes(self, patches, to_spool):
        """int16 and float32 members merge to float32 as chunk merges them."""
        mixed = [
            patches[0].new(data=(patches[0].data * 100).astype(np.int16)),
            patches[1].new(data=patches[1].data.astype(np.float32)),
            patches[2].new(data=(patches[2].data * 100).astype(np.int16)),
        ]
        spool = to_spool(mixed)
        tree = spool.io.to_xarray()
        assert _segments(tree)[0].dtype == np.float32
        self._assert_matches(tree, spool.chunk(time=None))

    def test_transposed_member(self, patches, to_spool):
        """A member stored (time, distance) is its own segment, as in chunk."""
        spool = to_spool([patches[0], patches[1].transpose(), patches[2]])
        tree = spool.io.to_xarray()
        native, flipped = ("distance", "time"), ("time", "distance")
        assert [x.dims for x in _segments(tree)] == [native, flipped, native]
        self._assert_matches(tree, spool.chunk(time=None))

    def test_samples_select(self, patches, to_spool):
        """Trimming each member by samples opens gaps, so every one splits."""
        spool = to_spool(patches).select(time=(10, -10), samples=True)
        tree = spool.io.to_xarray()
        assert len(_segments(tree)) == 4
        self._assert_matches(tree, spool.chunk(time=None))

    def test_overlap_trim(self, patches, to_spool):
        """
        A member overlapping its predecessor is read from mid-file.

        The second patch starts halfway through the first, so the merge
        keeps only its tail; reading it from its file's start would
        splice in the wrong (distinct) samples.
        """
        first = patches[0].update_attrs(history=[])
        time = first.get_coord("time").values
        second = first.update_coords(time_min=time[len(time) // 2])
        second = second.new(data=first.data + 1).update_attrs(history=[])
        spool = to_spool([first, second])
        self._assert_matches(spool.io.to_xarray(), spool.chunk(time=None))

    def test_selection_across_seam(self, patches, to_spool):
        """Isel and sel windows straddling a member seam equal Patch.select."""
        spool = to_spool(patches[:3])
        (data,) = _segments(spool.io.to_xarray())
        (merged,) = spool.chunk(time=None)
        seam = len(patches[0].get_coord("time"))
        sub = data.isel(time=slice(seam - 5, seam + 5), distance=slice(2, 9))
        expected = merged.select(
            time=(seam - 5, seam + 5), distance=(2, 9), samples=True
        )
        self._assert_matches_patch(sub, expected)
        times = merged.get_coord("time").values
        sub = data.sel(time=slice(times[seam - 3], times[seam + 3]))
        expected = merged.select(time=(times[seam - 3], times[seam + 3]))
        assert sub.sizes["time"] == 7
        self._assert_matches_patch(sub, expected)

    @staticmethod
    def _assert_matches_patch(data, patch):
        """One selected array's dims, values and coordinates equal the patch's."""
        assert data.dims == patch.dims
        np.testing.assert_array_equal(data.compute().values, patch.data)
        for dim in patch.dims:
            np.testing.assert_array_equal(data[dim].values, patch.get_coord(dim).values)


@pytest.fixture(scope="module")
def long_segment(requires_xarray_dask):
    """One channel of 2e6 int8 samples: tiny data, a long time axis."""
    samples = 2_000_000
    time = dc.core.get_coord(
        start=np.datetime64("2020-01-01"),
        step=np.timedelta64(1, "ms"),
        shape=(samples,),
    )
    distance = dc.core.get_coord(start=0.0, step=1.0, shape=(1,))
    patch = dc.Patch(
        data=np.zeros((1, samples), dtype=np.int8),
        coords={"distance": distance, "time": time},
        dims=("distance", "time"),
    )
    (data,) = _segments(dc.spool([patch]).io.to_xarray())
    return data


@pytest.mark.usefixtures("requires_xarray_dask")
class TestKnownDivergences:
    """Where the tree once disagreed with chunk or wasted memory."""

    @staticmethod
    def _peak_bytes(func):
        """Peak bytes traced while ``func`` runs, after one warm-up call."""
        func()
        started = not tracemalloc.is_tracing()
        if started:
            tracemalloc.start()
        try:
            # a caller's own tracing keeps its history; measure from here
            tracemalloc.reset_peak()
            base, _ = tracemalloc.get_traced_memory()
            func()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            if started:
                tracemalloc.stop()
        return peak - base

    def test_contiguous_window_memory(self, long_segment):
        """
        Control for the memory tests: a step-one window stays small.

        If this fails the measurement, not the indexing, is at fault.
        """
        samples = long_segment.sizes["time"]
        window = long_segment.isel(time=slice(5, 6))
        peak = self._peak_bytes(lambda: window.compute(scheduler="synchronous"))
        assert peak < samples * 8 // 4

    @pytest.mark.parametrize("index", [5, slice(10, 20, 2)], ids=["point", "stride"])
    def test_point_and_stride_memory(self, long_segment, index):
        """
        A scalar or short strided isel allocates nothing as long as the time axis.

        Peak traced memory must stay far below the 8 bytes per sample a
        positional array over the whole axis costs.
        """
        samples = long_segment.sizes["time"]
        peak = self._peak_bytes(
            lambda: long_segment.isel(time=index).compute(scheduler="synchronous")
        )
        assert peak < samples * 8 // 4

    def test_stride_on_two_dims(self, random_spool):
        """Striding both dimensions at once equals the same isel of chunk."""
        (data,) = _segments(random_spool.io.to_xarray())
        (merged,) = random_spool.chunk(time=None)
        index = dict(distance=slice(None, None, 2), time=slice(None, None, 2))
        expected = merged.isel(**index)
        np.testing.assert_array_equal(
            data.isel(**index).compute().values, expected.data
        )

    def test_keep_first_data_units(self, random_patch):
        """
        A member in other data units is converted to the merged units.

        chunk keeps the first member's m/s and converts the second's
        100 cm/s to 1 m/s, so every sample is one; the tree must agree
        rather than splice in the raw hundreds.
        """
        coord = random_patch.get_coord("time")
        first = random_patch.new(data=np.ones_like(random_patch.data))
        second = random_patch.new(data=np.full_like(random_patch.data, 100.0))
        second = second.update_coords(time_min=coord.max() + coord.step)
        spool = dc.spool([first.set_units("m/s"), second.set_units("cm/s")])
        kwargs = dict(group=[], conflict="keep_first")
        (merged,) = spool.chunk(time=None, **kwargs)
        (data,) = _segments(spool.io.to_xarray(**kwargs))
        np.testing.assert_array_equal(data.values, merged.data)

    def test_non_merged_dim_units(self, random_spool):
        """The distance coordinate keeps its units through a round trip."""
        (data,) = _segments(random_spool.io.to_xarray())
        (merged,) = random_spool.chunk(time=None)
        expected = merged.get_coord("distance").units
        assert expected is not None
        assert data.dc.to_patch().get_coord("distance").units == expected

    def test_associated_coord_attrs(self, random_patch):
        """An associated coordinate's summary (latitude_min etc.) stays out of attrs."""
        n = len(random_patch.get_coord("distance"))
        patch = random_patch.update_coords(
            latitude=("distance", np.linspace(10, 11, n))
        )
        (data,) = _segments(dc.spool([patch]).io.to_xarray())
        leaked = {x for x in data.attrs if x.startswith("latitude")}
        assert not leaked
