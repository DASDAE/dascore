"""Tests for converting a spool to a dask-backed xarray DataTree."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.config import config_context
from dascore.exceptions import PatchConversionError


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

    def test_builds_without_reading(self, diverse_spool_directory, monkeypatch):
        """Constructing the tree must not read any patch data."""
        from dascore.io.index.catalog import FileResolver, PatchCatalog  # noqa: PLC0415
        from dascore.io.index.planned import PlanResolver  # noqa: PLC0415

        spool = dc.spool(diverse_spool_directory).update()

        def _fail(*args, **kwargs):
            raise AssertionError("tree construction read patch data")

        monkeypatch.setattr(PatchCatalog, "resolve_row", _fail)
        monkeypatch.setattr(FileResolver, "resolve", _fail)
        monkeypatch.setattr(PlanResolver, "_load_member", _fail)
        monkeypatch.setattr(PlanResolver, "_load_member_array", _fail)
        tree = spool.io.to_xarray()
        assert len(self._leaves(tree))

    def test_compute_reads_only_needed_blocks(
        self, diverse_spool_directory, monkeypatch
    ):
        """A small selection loads only the member blocks it touches."""
        from dascore.io.index.planned import PlanResolver  # noqa: PLC0415

        spool = dc.spool(diverse_spool_directory).update()
        tree = spool.io.to_xarray()
        calls = []
        original = PlanResolver._load_member

        def _counting(self, kwargs):
            calls.append(1)
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
        assert len(calls) == 1
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
        import dascore.utils.chunk_plan as chunk_plan_module  # noqa: PLC0415

        original = chunk_plan_module.build_chunk_plan

        def _null_dtype(*args, **kwargs):
            plan = original(*args, **kwargs)
            plan.outputs["_dtype"] = bad_dtype
            return plan

        monkeypatch.setattr(chunk_plan_module, "build_chunk_plan", _null_dtype)
        with pytest.raises(PatchConversionError, match="dtype"):
            random_spool.io.to_xarray()

    def test_tolerance_argument(self, diverse_spool):
        """A looser tolerance merges gaps the default keeps as segments."""
        sub = diverse_spool.select(tag="big_gaps")
        default = len(self._leaves(sub.io.to_xarray()))
        loose = len(self._leaves(sub.io.to_xarray(tolerance=10_000)))
        assert loose < default

    def test_stale_index_shape_raises(self, random_spool, monkeypatch):
        """A block whose loaded shape breaks its promise raises clearly."""
        from dascore.io.index.planned import PlanResolver  # noqa: PLC0415

        tree = random_spool.io.to_xarray()
        original = PlanResolver._load_member

        def _truncated(self, kwargs):
            patch = original(self, kwargs)
            return patch.select(time=(0, 5), samples=True)

        monkeypatch.setattr(PlanResolver, "_load_member", _truncated)
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

    def test_transposed_member_load(self, random_spool, monkeypatch):
        """A member loading in another dim order is transposed to match."""
        from dascore.io.index.planned import PlanResolver  # noqa: PLC0415

        tree = random_spool.io.to_xarray()
        merged = random_spool.chunk(time=None)[0]
        original = PlanResolver._load_member

        def _transposed(self, kwargs):
            return original(self, kwargs).transpose()

        monkeypatch.setattr(PlanResolver, "_load_member", _transposed)
        leaf = self._leaves(tree)[0]
        np.testing.assert_array_equal(leaf.dataset["data"].values, merged.data)

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
        merged = spool.chunk(distance=None, tolerance=tol)[0]
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
    def override_calls(self):
        """Give DASDAE a counting read_array override."""
        from dascore.io.core import FiberIO  # noqa: PLC0415
        from dascore.io.dasdae.core import DASDAEV1  # noqa: PLC0415

        calls = []

        def read_array(self, resource, windows, **kwargs):
            # a real override's caster wrapper consumes _pre_cast; this
            # raw function sees it and must not forward it to read
            kwargs.pop("_pre_cast", None)
            calls.append(windows)
            return FiberIO.read_array(self, resource, windows, **kwargs)

        # set and delete by hand: monkeypatch would restore the inherited
        # method as an own class attribute rather than remove it
        DASDAEV1.read_array = read_array
        yield calls
        del DASDAEV1.read_array

    def _leaf(self, tree):
        """The first dataset holding a data variable."""
        return next(node for node in tree.subtree if "data" in node.dataset)

    def test_fast_path_loads_blocks(
        self, dasdae_directory, override_calls, monkeypatch
    ):
        """With an override, computing never builds a member Patch."""
        from dascore.io.index.planned import PlanResolver  # noqa: PLC0415

        spool = dc.spool(dasdae_directory).update()
        eager = spool.chunk(time=None)[0].data
        tree = spool.io.to_xarray()

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

    def test_chunked_spool_falls_back(self, dasdae_directory, override_calls):
        """A plan-backed spool's trimmed rows never take the fast path.

        Its collapsed member rows state trimmed envelopes, so a sample
        window computed against them is not a window on the file grid;
        the fast path must refuse or it reads the wrong samples.
        """
        spool = dc.spool(dasdae_directory).update().chunk(time=3)
        eager = spool.chunk(time=None)[0].data
        out = self._leaf(spool.io.to_xarray())["data"].data.compute()
        assert np.array_equal(out, eager)
        assert override_calls == []

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
        out = self._leaf(spool.io.to_xarray())["data"].data.compute()
        assert np.array_equal(out, eager)
        # the trimmed member's window must not be anchored at the start
        starts = sorted(window["time"][0] for window in override_calls)
        assert len(override_calls) == 2
        assert starts[0] == 0 and starts[1] > 0

    def test_transposes_source_order(self):
        """A native-order array is transposed to the tree's dims.

        Three dimensions with a cyclic permutation, so the permutation
        differs from its inverse and a reversed mapping cannot pass.
        """
        from dascore.xarray.spool import _load_xarray_block  # noqa: PLC0415

        native = np.arange(24).reshape(2, 3, 4)

        class _Fake:
            def _load_member_array(self, row, windows):
                return native

        row = {"dims": "time,distance,depth", "source_path": "x"}
        out = _load_xarray_block(
            _Fake(),
            row,
            "time",
            (0, 1),
            ("distance", "depth", "time"),
            (3, 4, 2),
            native.dtype,
            (0, 2),
        )
        assert np.array_equal(out, native.transpose(1, 2, 0))

    def test_mismatched_dims_fall_back(self, random_patch):
        """A row stating different dims than the tree takes the patch path."""
        from dascore.xarray.spool import _load_xarray_block  # noqa: PLC0415

        patch = random_patch

        class _Fake:
            def _load_member_array(self, row, windows):
                raise AssertionError("fast path consulted with foreign dims")

            def _load_member(self, row):
                return patch

        coord = patch.get_coord("time")
        out = _load_xarray_block(
            _Fake(),
            {"dims": "depth,time", "source_path": "x"},
            "time",
            (coord.min(), coord.max()),
            patch.dims,
            patch.shape,
            patch.data.dtype,
            (0, len(coord)),
        )
        assert np.array_equal(out, patch.data)

    def test_stale_shape_raises(self):
        """An array which breaks the index's promise raises."""
        from dascore.exceptions import PatchConversionError  # noqa: PLC0415
        from dascore.xarray.spool import _load_xarray_block  # noqa: PLC0415

        class _Fake:
            def _load_member_array(self, row, windows):
                return np.zeros((2, 2))

        row = {"dims": "time,distance", "source_path": "x"}
        with pytest.raises(PatchConversionError, match="promised"):
            _load_xarray_block(
                _Fake(), row, "time", (0, 2), ("time", "distance"), (3, 4), "f8", (0, 3)
            )

    def test_row_without_dims_falls_back(self, random_patch):
        """A row which cannot state its dimension order takes the patch path."""
        from dascore.xarray.spool import _load_xarray_block  # noqa: PLC0415

        patch = random_patch

        class _Fake:
            def _load_member_array(self, row, windows):
                raise AssertionError("fast path consulted without dims")

            def _load_member(self, row):
                return patch

        coord = patch.get_coord("time")
        lims = (coord.min(), coord.max())
        out = _load_xarray_block(
            _Fake(),
            {"source_path": "x"},
            "time",
            lims,
            patch.dims,
            patch.shape,
            patch.data.dtype,
            (0, len(coord)),
        )
        assert np.array_equal(out, patch.data)


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
        from dascore.xarray.index import TemporalRangeIndex  # noqa: PLC0415

        data = self._leaf(random_spool.io.to_xarray())["data"]
        assert isinstance(data.xindexes["time"], TemporalRangeIndex)
        # the labels are served on demand, not stored as an array
        assert not isinstance(data["time"].variable._data, np.ndarray)
        merged = random_spool.chunk(time=None)[0]
        np.testing.assert_array_equal(
            data["time"].values, merged.get_coord("time").values
        )

    def test_sel_matches_patch_select(self, random_spool):
        """Label selection on the tree equals dascore's own select."""
        data = self._leaf(random_spool.io.to_xarray())["data"]
        merged = random_spool.chunk(time=None)[0]
        t = merged.get_coord("time").values
        sub = data.sel(time=slice(t[100], t[300]))
        expected = merged.select(time=(t[100], t[300]))
        assert sub.sizes["time"] == expected.shape[expected.dims.index("time")]
        np.testing.assert_array_equal(sub.compute().values, expected.data)

    def test_segmented_time_materializes(self, random_patch):
        """A jittered merge is not one range; its labels spell out."""
        from dascore.xarray.index import TemporalRangeIndex  # noqa: PLC0415

        coord = random_patch.get_coord("time")
        step = coord.step * 1.04  # within sampling tolerance, off-grid
        second = random_patch.update_coords(
            time_min=coord.max() + coord.step, time_step=step
        )
        spool = dc.spool([random_patch, second])
        data = self._leaf(spool.io.to_xarray())["data"]
        assert not isinstance(data.xindexes["time"], TemporalRangeIndex)
        merged = spool.chunk(time=None)[0]
        np.testing.assert_array_equal(
            data["time"].values, merged.get_coord("time").values
        )
