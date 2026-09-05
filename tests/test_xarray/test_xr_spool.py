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
        # A block reads through whichever path its format offers, so both
        # are counted: what the test pins is how many members are read.
        original = PlanResolver._load_member
        original_array = PlanResolver._load_member_array

        def _counting(self, kwargs):
            calls.append("patch")
            return original(self, kwargs)

        def _counting_array(self, row, windows, **kwargs):
            calls.append("array")
            return original_array(self, row, windows, **kwargs)

        monkeypatch.setattr(PlanResolver, "_load_member", _counting)
        monkeypatch.setattr(PlanResolver, "_load_member_array", _counting_array)
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

        # set and restore by hand: monkeypatch would put the inherited
        # method back as an own class attribute rather than remove it,
        # and DASDAE has an override of its own to hand back afterwards.
        missing = object()
        stored = DASDAEV1.__dict__.get("read_array", missing)
        DASDAEV1.read_array = read_array
        yield calls
        if stored is missing:
            del DASDAEV1.read_array
        else:
            DASDAEV1.read_array = stored

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
        whole = self._leaf(file_spool.io.to_xarray(block_size=None))
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

    def test_a_block_reads_only_its_own_window(self, file_spool, monkeypatch):
        """Computing one piece reads that piece's samples, not the file."""
        from dascore.io.index.planned import PlanResolver  # noqa: PLC0415

        windows = []
        original = PlanResolver._load_member_array

        def _counting(self, row, member_windows, **kwargs):
            windows.append(dict(member_windows))
            return original(self, row, member_windows, **kwargs)

        monkeypatch.setattr(PlanResolver, "_load_member_array", _counting)
        quarter = file_spool[0].data.nbytes // 4
        data = self._leaf(file_spool.io.to_xarray(block_size=quarter))
        samples = len(file_spool[0].get_coord("time"))
        piece = data.isel(time=slice(0, samples // 4)).compute()
        assert windows == [{"time": (0, samples // 4)}]
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

    def test_a_piece_falling_back_reads_its_own_bounds(self, file_spool, monkeypatch):
        """A piece whose window read fails still loads its own samples.

        The array path can decline at load (a file which changed since
        indexing), and the piece then loads as a patch trimmed by value.
        Those bounds are the piece's own, not its member's, or the block
        would come back the size of the whole member.
        """
        from dascore.io.index.planned import PlanResolver  # noqa: PLC0415

        quarter = file_spool[0].data.nbytes // 4
        data = self._leaf(file_spool.io.to_xarray(block_size=quarter))
        merged = file_spool.chunk(time=None)[0]
        monkeypatch.setattr(
            PlanResolver, "_load_member_array", lambda self, row, w, **k: None
        )
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

    def test_pieces_are_even(self):
        """No piece is more than one sample shorter than another."""
        from dascore.xarray.spool import _block_pieces  # noqa: PLC0415

        sizes = [b - a for a, b in _block_pieces(100, 30)]
        assert max(sizes) - min(sizes) <= 1

    def test_one_sample_rows_still_split(self):
        """A row costing more than the ceiling still yields whole samples."""
        from dascore.xarray.spool import _samples_per_block  # noqa: PLC0415

        limit = _samples_per_block(10, np.dtype("float64"), {"x": 100, "t": 5}, "t")
        assert limit == 1
