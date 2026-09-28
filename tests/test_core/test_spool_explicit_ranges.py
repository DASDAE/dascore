"""Independent bounded windows for spool selection and chunking."""

import pickle
import re
import threading
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

import dascore as dc
from dascore.core.coords import concat_coords, get_coord
from dascore.examples import inventory_patch_pair
from dascore.exceptions import ChunkError, MissingPatchError, ParameterError, UnitError
from dascore.io.dasdae.core import DASDAEV1, DASDAEV2
from dascore.io.index.catalog import PatchCatalog
from dascore.units import get_quantity, m, s
from dascore.utils.chunk_plan import _ensure_patch_row
from dascore.utils.explicit_ranges import known_coordinates


def _ends(patch, dim="distance"):
    """A loaded patch's first and last samples along `dim`."""
    return patch.coords[dim].min(), patch.coords[dim].max()


def _listed(frame):
    """The distance envelopes a contents or plan-outputs frame lists."""
    return frame[["distance_min", "distance_max"]].to_numpy().tolist()


def _patch(values):
    """A small one-dimensional patch with known coordinate values."""
    return dc.Patch(
        data=np.arange(len(values)),
        coords={"distance": np.asarray(values)},
        dims=("distance",),
    )


@pytest.fixture
def filled_2d_gap():
    """A fresh all-fill output with an unchunked associated time coordinate."""
    time = np.arange(3).astype("datetime64[s]")

    def make(values):
        return dc.Patch(
            data=np.ones((len(values), 3)),
            coords={
                "distance": values,
                "time": time,
                "reference_time": ("time", time),
            },
            dims=("distance", "time"),
        )

    source = dc.spool([make(np.arange(5)), make(np.arange(7, 12))])
    filled = source.chunk(distance=np.array([[5, 6]]), tolerance=4, fill_value=-1)
    return filled, time


@pytest.fixture
def legacy_residual_source(tmp_path):
    """Make real files whose second persisted key lacks exact metadata."""

    def make(second_start):
        for index, start in enumerate((0, second_start)):
            patch = _patch(np.arange(start, start + 5))
            dc.write(patch, tmp_path / f"{index}.h5", file_format="DASDAE")
        source = dc.spool(tmp_path).update(progress=None)
        records = source._catalog.backend.export_records()
        assert all(record.patches[0].source_patch_key for record in records)
        legacy = replace(
            records[1],
            patches=tuple(
                replace(patch, source_patch_key="") for patch in records[1].patches
            ),
        )
        source._catalog.backend.write_sources([records[0], legacy])
        source._catalog._invalidate()
        # Replaying this shared-dimension residual requires source metadata:
        # the native-key first file has it, the blank-key second file does not.
        return source.select(distance=(0, second_start + 4))

    return make


def _assert_window_matches_plan(source, bounds, expected):
    """Assert that a planned window, its catalog, and its data agree."""
    windows = np.array([bounds])
    plan = source.chunk_plan(distance=windows)
    chunked = source.chunk(distance=windows)
    assert len(plan.outputs) == len(chunked) == 1
    assert chunked[0].coords["distance"].values.tolist() == expected
    assert plan.outputs.iloc[0]["distance_min"] == expected[0]
    assert plan.outputs.iloc[0]["distance_max"] == expected[-1]
    assert chunked.get_contents().iloc[0]["distance_min"] == expected[0]
    assert chunked.get_contents().iloc[0]["distance_max"] == expected[-1]


class TestExplicitSelect:
    """Each range trims source pieces independently."""

    def test_overlap_and_duplicates_keep_request_order(self):
        """Shared samples appear once per requested row."""
        spool = dc.spool(_patch(np.arange(10)))
        windows = np.array([[5, 7], [2, 4], [5, 7], [4, 5]])
        out = spool.select(distance=windows)
        windows[0] = [0, 1]
        assert [_ends(x) for x in out] == [(5, 7), (2, 4), (5, 7), (4, 5)]

    def test_exact_regular_and_uneven_envelopes(self):
        """Metadata states the samples the patches actually contain."""
        for values, bounds, expected in (
            (np.arange(10), [[1.2, 4.8]], (2, 4)),
            ([0.0, 1.0, 3.0, 6.0, 9.0], [[2.0, 7.0]], (3, 6)),
        ):
            out = dc.spool(_patch(values)).select(distance=np.array(bounds))
            row = out.get_contents().iloc[0]
            patch = out[0]
            assert (row["distance_min"], row["distance_max"]) == expected
            assert _ends(patch) == expected

    def test_uneven_empty_range_is_omitted(self):
        """A gap inside an uneven envelope has no selected piece."""
        spool = dc.spool(_patch([0.0, 1.0, 3.0, 6.0, 9.0]))
        assert len(spool.select(distance=np.array([[4.0, 5.0]]))) == 0

    def test_chained_selection_is_deferred_and_respects_parent(self, monkeypatch):
        """Composing a view does not realize its source and retains residuals."""
        spool = dc.spool(_patch(np.arange(10)))
        source = spool.select(distance=(3, 8), samples=True)
        original = PatchCatalog.to_df

        def forbidden(_self):
            raise AssertionError("parent relation was realized during construction")

        monkeypatch.setattr(PatchCatalog, "to_df", forbidden)
        out = source.select(distance=np.array([[0, 5]])).select(
            _coords={"distance": (0, 5)}
        )
        monkeypatch.setattr(PatchCatalog, "to_df", original)
        row = out.get_contents().iloc[0]
        assert (row["distance_min"], row["distance_max"]) == (3, 5)
        assert _ends(out[0]) == (
            3,
            5,
        )

    def test_array_with_inventory_attribute(self):
        """Inventory-only filters apply before independent coordinate windows."""
        patch, inventory = inventory_patch_pair()
        spool = dc.spool(patch).attach_inventory(inventory)
        time = patch.get_coord("time").values
        windows = np.array([[time[1], time[3]], [time[6], time[8]]])
        out = spool.select(time=windows, gauge_length=10.0)
        assert len(out) == 2
        assert [x.get_coord("time").values.tolist() for x in out] == [
            time[1:4].tolist(),
            time[6:9].tolist(),
        ]
        assert len(spool.select(time=windows, gauge_length=20.0)) == 0
        assert len(spool.select(time=windows).select(gauge_length=10.0)) == 2
        assert len(spool.select(time=windows).select(gauge_length=20.0)) == 0

    def test_deferred_order_and_window(self, monkeypatch):
        """Ordering and slicing a window view remain deferred until inspected."""
        spool = dc.spool(_patch(np.arange(10)))
        original = PatchCatalog.to_df

        def forbidden(_self):
            raise AssertionError("parent relation was realized during construction")

        monkeypatch.setattr(PatchCatalog, "to_df", forbidden)
        out = spool.select(distance=np.array([[5, 7], [1, 3]])).sort("distance")[:1]
        monkeypatch.setattr(PatchCatalog, "to_df", original)
        assert len(out) == 1
        assert _ends(out[0]) == (
            1,
            3,
        )

    def test_nested_lists_and_pickle(self):
        """Nested-list windows retain their order through serialization."""
        spool = dc.spool(_patch(np.arange(10)))
        selected = spool.select(distance=[[5, 7], [1, 3]])
        restored = pickle.loads(pickle.dumps(selected))
        assert [x.coords["distance"].min() for x in restored] == [5, 1]
        assert [
            x.coords["distance"].min() for x in spool.chunk(distance=[[5, 7], [1, 3]])
        ] == [5, 1]

    def test_operational_backend_uses_realized_window_view(self):
        """Backend operations beyond schema names describe the selected view."""
        selected = dc.spool(_patch(np.arange(10))).select(distance=np.array([[2, 4]]))
        backend = selected._catalog.backend
        assert backend.coord_dims_map() == {"distance": "distance"}
        assert _listed(selected.get_contents()) == [[2, 4]]
        assert selected[0].coords["distance"].values.tolist() == [2, 3, 4]

    def test_mismatched_window_counts_raise(self):
        """Windows on several dimensions need one row each per request."""
        spool = dc.spool(dc.get_example_patch())
        one, two = np.array([[0, 1]]), np.array([[0, 1], [2, 3]])
        for method in (spool.select, spool.chunk):
            with pytest.raises(ParameterError, match="same number"):
                method(distance=one, time=two)

    @pytest.mark.concurrency
    def test_concurrent_add_keeps_a_coherent_parent_snapshot(self, monkeypatch):
        """An update during realization keeps source and candidates coherent."""
        source = dc.spool(_patch(np.arange(10)))
        selected = source.select(distance=np.array([[5, 15]]))
        original = PatchCatalog.to_df
        started = threading.Event()
        finished = threading.Event()
        worker = None

        def add_patch():
            started.set()
            source._catalog.add(_patch(np.arange(10, 20)))
            finished.set()

        def snapshot(catalog):
            nonlocal worker
            frame = original(catalog)
            if catalog is selected._catalog.parent and worker is None:
                worker = threading.Thread(target=add_patch)
                worker.start()
                assert started.wait(2)
                finished.wait(0.05)
            return frame

        monkeypatch.setattr(PatchCatalog, "to_df", snapshot)
        assert _listed(selected.get_contents()) == [[5, 9]]
        assert worker is not None
        worker.join(timeout=2)
        assert finished.is_set()
        assert _listed(selected.get_contents()) == [[5, 9], [10, 15]]

    @pytest.mark.parametrize("flag", ["samples", "relative"])
    def test_array_flags_raise(self, flag):
        """Deferred array sample and relative forms have no ambiguous meaning."""
        spool = dc.spool(_patch(np.arange(10)))
        with pytest.raises(ParameterError, match="Explicit ranges"):
            spool.select(distance=np.array([[1, 3]]), **{flag: True})


class TestExplicitChunk:
    """Each requested output is checked against actual coverage."""

    def test_plan_and_loaded_windows_agree(self):
        """Repeated and unsorted requests retain individual outputs."""
        spool = dc.spool(_patch(np.arange(10)))
        windows = np.array([[5, 7], [1, 3], [5, 7]])
        plan = spool.chunk_plan(distance=windows)
        out = spool.chunk(distance=windows)
        assert list(plan.outputs["_request_row"]) == [0, 1, 2]
        assert [_ends(x) for x in out] == [(5, 7), (1, 3), (5, 7)]

    @pytest.mark.parametrize("policy", ["raise", "warn", "ignore"])
    def test_incomplete_policy_and_partial(self, policy):
        """Strict requests fail or skip; keep_partial retains available samples."""
        spool = dc.spool(_patch(np.arange(10)))
        windows = np.array([[2, 4], [8, 10]])
        if policy == "raise":
            with pytest.raises(ChunkError, match="row 1"):
                spool.chunk_plan(distance=windows, on_incomplete=policy)
            with pytest.raises(ChunkError, match="row 1"):
                spool.chunk(distance=windows, on_incomplete=policy)
        else:
            if policy == "warn":
                with pytest.warns(UserWarning, match="row 1"):
                    plan = spool.chunk_plan(distance=windows, on_incomplete=policy)
            else:
                plan = spool.chunk_plan(distance=windows, on_incomplete=policy)
            assert len(plan.outputs) == 1
        partial = spool.chunk(distance=windows, keep_partial=True, on_incomplete=policy)
        assert [_ends(x) for x in partial] == [(2, 4), (8, 9)]

    def test_uneven_coordinates_use_exact_samples(self):
        """Uneven ranges succeed when samples exist and skip empty windows."""
        spool = dc.spool(_patch([0.0, 1.0, 3.0, 6.0, 9.0]))
        plan = spool.chunk_plan(distance=np.array([[2.0, 7.0]]))
        out = spool.chunk(distance=np.array([[2.0, 7.0]]))
        assert (
            plan.outputs.iloc[0]["distance_min"],
            plan.outputs.iloc[0]["distance_max"],
        ) == (3, 6)
        assert _ends(out[0]) == (
            3,
            6,
        )
        with pytest.raises(ChunkError, match="contains no source samples"):
            spool.chunk_plan(distance=np.array([[4.0, 5.0]]))
        assert (
            len(spool.chunk(distance=np.array([[4.0, 5.0]]), on_incomplete="ignore"))
            == 0
        )

    @pytest.mark.parametrize("keep_partial", [False, True])
    def test_uneven_fill_reports_actual_samples(self, keep_partial):
        """A fill value cannot invent an uneven grid or its requested edges."""
        spool = dc.spool(_patch([0.0, 1.0, 3.0, 4.0, 6.0]))
        windows = np.array([[1.2, 3.8]])
        plan = spool.chunk_plan(
            distance=windows,
            fill_value=-1,
            tolerance=10,
            keep_partial=keep_partial,
        )
        out = spool.chunk(
            distance=windows,
            fill_value=-1,
            tolerance=10,
            keep_partial=keep_partial,
        )
        assert _listed(plan.outputs) == [[3.0, 3.0]]
        assert _listed(out.get_contents()) == [[3.0, 3.0]]
        assert out[0].coords["distance"].values.tolist() == [3.0]
        assert out[0].data.tolist() == [2]

    @pytest.mark.parametrize("keep_partial", [False, True])
    def test_uneven_fill_empty_request_follows_policy(self, keep_partial):
        """A requested interval with no uneven samples fails during planning."""
        spool = dc.spool(_patch([0.0, 1.0, 3.0, 4.0, 6.0]))
        windows = np.array([[2.0, 2.5]])
        kwargs = dict(distance=windows, fill_value=-1, tolerance=10)
        with pytest.raises(ChunkError, match="contains no source samples"):
            spool.chunk_plan(keep_partial=keep_partial, **kwargs)
        with pytest.raises(ChunkError, match="contains no source samples"):
            spool.chunk(keep_partial=keep_partial, **kwargs)
        with pytest.warns(UserWarning, match="contains no source samples"):
            warned = spool.chunk_plan(
                keep_partial=keep_partial, on_incomplete="warn", **kwargs
            )
        assert warned.outputs.empty
        ignored = spool.chunk(
            keep_partial=keep_partial, on_incomplete="ignore", **kwargs
        )
        assert len(ignored) == 0
        assert ignored.get_contents().empty

    @pytest.mark.parametrize(
        ("bounds", "reason"),
        [
            ([20.0, 30.0], "outside source coverage"),
            ([1.1, 1.9], "contains no sampled position"),
        ],
    )
    @pytest.mark.parametrize("method", ["chunk", "chunk_plan"])
    def test_empty_filled_window_follows_incomplete_policy(
        self, bounds, reason, method
    ):
        """Fill cannot make a request outside coverage or without grid samples."""
        spool = dc.spool(_patch(np.arange(10)))
        kwargs = {"distance": np.array([bounds]), "fill_value": -1}
        call = getattr(spool, method)
        with pytest.raises(ChunkError, match=reason):
            call(**kwargs)
        with pytest.warns(UserWarning, match=reason):
            warned = call(on_incomplete="warn", **kwargs)
        ignored = call(on_incomplete="ignore", **kwargs)
        for result in (warned, ignored):
            assert len(result.outputs if method == "chunk_plan" else result) == 0

    def test_descending_fill_only_projection_fails_loudly(self):
        """An unreconstructable derived fill grid cannot borrow its anchor."""

        def descending(start):
            return _patch(np.arange(start, start + 5, dtype=float)[::-1])

        filled = dc.spool([descending(0), descending(7)]).chunk(
            distance=np.array([[5, 6]]), tolerance=4, fill_value=-1
        )
        assert filled[0].coords["distance"].values.tolist() == [6.0, 5.0]
        with pytest.raises(ChunkError, match="no members or reconstructable grid"):
            filled.select(distance=np.array([[5, 6]])).get_contents()
        with pytest.raises(ChunkError, match="no members or reconstructable grid"):
            filled.chunk_plan(distance=np.array([[5, 6]]))

    def test_tolerated_gap_needs_requested_edge_or_fill(self):
        """Tolerance bridges internal gaps but cannot invent an edge sample."""
        patch = _patch(np.arange(10))
        spool = dc.spool([patch.select(distance=(0, 4)), patch.select(distance=(7, 9))])
        with pytest.raises(ChunkError, match="sampled bounds"):
            spool.chunk_plan(distance=np.array([[5, 9]]), tolerance=4)
        partial = spool.chunk(
            distance=np.array([[5, 9]]), tolerance=4, keep_partial=True
        )
        assert _ends(partial[0]) == (7, 9)
        filled = spool.chunk(distance=np.array([[5, 9]]), tolerance=4, fill_value=0)
        assert _ends(filled[0]) == (5, 9)
        fill_only = spool.chunk(distance=np.array([[5, 6]]), tolerance=4, fill_value=0)
        assert _ends(fill_only[0]) == (5, 6)

    def test_datetime_string_array_matches_scalar_selection(self):
        """NumPy string endpoints retain datetime meaning in array form."""
        values = np.arange(10).astype("datetime64[D]")
        patch = dc.Patch(data=np.arange(10), coords={"time": values}, dims=("time",))
        spool = dc.spool(patch)
        bounds = ("1970-01-02", "1970-01-05")
        scalar = spool.select(time=bounds)
        array = spool.select(time=np.array([bounds]))
        assert len(array) == len(scalar) == 1
        assert np.array_equal(
            array[0].coords["time"].values, scalar[0].coords["time"].values
        )
        chunked = spool.chunk(time=np.array([bounds]))
        assert np.array_equal(
            chunked[0].coords["time"].values, scalar[0].coords["time"].values
        )
        mixed = spool.select(
            time=np.array([[np.datetime64(bounds[0]), bounds[1]]], dtype=object)
        )
        assert np.array_equal(
            mixed[0].coords["time"].values, scalar[0].coords["time"].values
        )
        with pytest.raises(ParameterError, match="incomparable"):
            spool.select(time=np.array([[1, "later"]], dtype=object))

    @pytest.mark.parametrize("kind", ["datetime64[s]", "timedelta64[s]"])
    def test_time_family_absolute_windows(self, kind):
        """Absolute time and duration windows retain their native endpoints."""
        values = np.arange(10).astype(kind)
        patch = dc.Patch(data=np.arange(10), coords={"time": values}, dims=("time",))
        spool = dc.spool(patch)
        requested = np.array([[values[2], values[4]]])
        selected = spool.select(time=requested)
        chunked = spool.chunk(time=requested)
        for out in (selected, chunked):
            coord = out[0].coords["time"]
            assert (coord.min(), coord.max()) == (values[2], values[4])
            row = out.get_contents().iloc[0]
            assert (row["time_min"], row["time_max"]) == (values[2], values[4])

    @pytest.mark.parametrize("kind", ["datetime64[ns]", "timedelta64[ns]"])
    def test_time_family_final_nanosecond_sample(self, kind):
        """A singleton window includes the final sample at nanosecond precision."""
        values = (np.arange(5) * 1_000_000 + 123).astype(kind)
        patch = dc.Patch(data=np.arange(5), coords={"time": values}, dims=("time",))
        spool = dc.spool(patch)
        requested = np.array([[values[-1], values[-1]]])
        plan = spool.chunk_plan(time=requested)
        assert len(plan.outputs) == 1
        row = plan.outputs.iloc[0]
        assert (row["time_min"], row["time_max"]) == (values[-1], values[-1])
        chunked = spool.chunk(time=requested)
        assert len(chunked) == 1
        np.testing.assert_array_equal(chunked[0].coords["time"].values, values[-1:])
        np.testing.assert_array_equal(chunked[0].data, [4])

    def test_missing_dimension_drop_and_between_sample_window(self):
        """A dropped dimension and a window between samples produce no output."""
        patch = dc.Patch(
            data=np.arange(10),
            coords={"time": np.arange(10).astype("datetime64[s]")},
            dims=("time",),
        )
        with pytest.raises(ChunkError, match="source is empty"):
            dc.spool(patch).chunk_plan(distance=np.array([[1, 3]]), missing_dim="drop")
        spool = dc.spool(_patch(np.arange(10)))
        with pytest.raises(ChunkError, match="no sampled position"):
            spool.chunk_plan(distance=np.array([[1.1, 1.9]]))
        assert (
            len(spool.chunk(distance=np.array([[1.1, 1.9]]), on_incomplete="ignore"))
            == 0
        )

    def test_datetime_rejects_quantity_and_mixed_bounds_reject_reverse(self):
        """Only instants can bound datetimes; mixed native/unit points stay ordered."""
        values = np.arange(10).astype("datetime64[s]")
        time_spool = dc.spool(
            dc.Patch(data=np.arange(10), coords={"time": values}, dims=("time",))
        )
        with pytest.raises(ParameterError, match="absolute instants"):
            time_spool.chunk_plan(time=np.array([[2 * s, 4 * s]], dtype=object))
        distance = dc.spool(_patch(np.arange(10)).set_units(distance="m"))
        with pytest.raises(ParameterError, match="lower bound above"):
            distance.chunk_plan(distance=np.array([[5, 2 * m]], dtype=object))

    def test_uneven_request_past_envelope_needs_partial(self):
        """An uneven source cannot supply an unsampled outer endpoint."""
        spool = dc.spool(_patch([0.0, 1.0, 3.0, 6.0, 9.0]))
        with pytest.raises(ChunkError, match="sampled bounds"):
            spool.chunk_plan(distance=np.array([[-1.0, 6.0]]))
        partial = spool.chunk(distance=np.array([[-1.0, 6.0]]), keep_partial=True)
        assert _ends(partial[0]) == (0, 6)

    def test_unitless_coordinate_rejects_quantity_points(self):
        """A unit-bearing absolute request cannot select a unitless grid."""
        spool = dc.spool(_patch(np.arange(10)))
        with pytest.raises(UnitError, match="unitless coordinate"):
            spool.chunk_plan(distance=np.array([[2 * m, 4 * m]], dtype=object))

    def test_explicit_ranges_on_empty_and_disjoint_sources(self):
        """Absent requests follow policy, including on a filtered empty source."""
        spool = dc.spool(_patch(np.arange(10)))
        empty = spool.select(distance=(20, 30))
        with pytest.raises(ChunkError, match="source is empty"):
            empty.chunk_plan(distance=np.array([[1, 3]]))
        assert (
            len(empty.chunk(distance=np.array([[1, 3]]), on_incomplete="ignore")) == 0
        )
        with pytest.raises(ChunkError, match="outside source coverage"):
            spool.chunk_plan(distance=np.array([[20, 30]]))
        assert (
            len(spool.chunk(distance=np.array([[20, 30]]), on_incomplete="ignore")) == 0
        )

    def test_unparsable_string_bounds_are_rejected(self):
        """String endpoints in explicit windows must name actual instants."""
        spool = dc.spool(_patch(np.arange(10)))
        for bounds in (["not-a-date", "later"], [None, "later"]):
            with pytest.raises(ParameterError, match="datetime"):
                spool.chunk_plan(distance=np.array([bounds], dtype=object))

    @pytest.mark.parametrize(
        "bounds",
        [
            (True, 2),
            ([1], 2),
            (np.inf, 2),
            (1, -np.inf),
            (3, 1),
            (3 * m, 1 * s),
        ],
    )
    def test_invalid_absolute_bounds(self, bounds):
        """A bounded window needs finite scalar ordered compatible endpoints."""
        spool = dc.spool(_patch(np.arange(10)))
        with pytest.raises(ParameterError):
            spool.chunk_plan(distance=np.array([bounds], dtype=object))

    def test_argument_validation_and_empty_requests(self):
        """Malformed windows and policy names fail before planning data."""
        spool = dc.spool(_patch(np.arange(10)))
        assert len(spool.chunk(distance=np.empty((0, 2)))) == 0
        assert len(spool.select(distance=np.empty((0, 2)))) == 0
        open_low = spool.chunk(distance=np.array([[np.nan, 2]]))
        assert open_low[0].coords["distance"].values.tolist() == [0, 1, 2]
        for value in (np.array([1, 2]), np.ones((2, 3))):
            with pytest.raises(ParameterError):
                spool.chunk(distance=value)
        with pytest.raises(ParameterError, match="on_incomplete"):
            spool.chunk_plan(distance=np.array([[1, 2]]), on_incomplete="invalid")
        with pytest.raises(ParameterError, match="overlap"):
            spool.chunk(distance=np.array([[1, 2]]), overlap=1)


class TestExplicitMetadataSources:
    """File and in-memory sources report the same exact windows."""

    @pytest.mark.parametrize("method", ["select", "chunk"])
    @pytest.mark.parametrize("directory", [False, True])
    @pytest.mark.parametrize(
        "start,step,indices",
        [
            (0.1, 0.1, (44, 44)),
            (-3.3, 1.0209, (3, 12)),
            (1e9, 0.03, (44, 44)),
            (np.float32(0.1), 0.1, (3, 12)),
        ],
    )
    def test_file_float_window_preserves_labels(
        self, tmp_path, method, directory, start, step, indices
    ):
        """An indexed float envelope cannot reconstruct a sliced grid's labels."""
        coord = get_coord(start=start, step=step, shape=(100,))[3:93:2]
        patch = dc.Patch(
            data=np.arange(len(coord)),
            coords={"distance": coord},
            dims=("distance",),
        )
        path = tmp_path / "float.h5"
        dc.write(patch, path, file_format="DASDAE")
        source = (
            dc.spool(tmp_path).update(progress=None) if directory else dc.spool(path)
        )
        bounds = tuple(coord.values[list(indices)])
        expected = patch.select(distance=bounds)
        windows = np.array([bounds])
        result = getattr(source, method)(distance=windows)
        assert len(result) == 1
        contents = result.get_contents()
        if method == "chunk":
            plan = source.chunk_plan(distance=windows)
            assert (
                plan.outputs["distance_min"].iloc[0] == contents["distance_min"].iloc[0]
            )
            assert (
                plan.outputs["distance_max"].iloc[0] == contents["distance_max"].iloc[0]
            )
        loaded = result[0]
        expected_coord = expected.get_coord("distance")
        assert (
            loaded.get_coord("distance").values.tobytes()
            == expected_coord.values.tobytes()
        )
        assert np.array_equal(loaded.data, expected.data)
        assert contents["distance_min"].iloc[0] == expected_coord.min()
        assert contents["distance_max"].iloc[0] == expected_coord.max()

    def test_regular_file_derived_window_uses_indexed_grid(self, tmp_path, monkeypatch):
        """A nested regular file window needs no payload coordinate scan."""
        path = tmp_path / "regular.h5"
        dc.write(_patch(np.arange(10)), path, file_format="DASDAE")
        source = dc.spool(path)
        derived = source.chunk(distance=np.array([[2, 8]]))
        scanner = dc.scan_payloads

        def forbidden(*args, **kwargs):
            raise AssertionError("regular indexed grid scanned its file payload")

        monkeypatch.setattr(dc, "scan_payloads", forbidden)
        selected = derived.select(distance=np.array([[3, 5]]))
        plan = derived.chunk_plan(distance=np.array([[3, 5]]))
        assert _listed(selected.get_contents()) == [[3, 5]]
        assert _listed(plan.outputs) == [[3, 5]]
        monkeypatch.setattr(dc, "scan_payloads", scanner)
        assert selected[0].coords["distance"].values.tolist() == [3, 4, 5]

    @pytest.mark.parametrize("directory", [False, True])
    def test_file_uneven_coordinates_without_array_reads(
        self, tmp_path, monkeypatch, directory
    ):
        """Planning reads full coordinate payloads but no measurement arrays."""
        patch = _patch([0.0, 1.0, 3.0, 6.0, 9.0])
        path = tmp_path / "uneven.h5"
        dc.write(patch, path, file_format="DASDAE")
        spool = dc.spool(tmp_path).update() if directory else dc.spool(path)
        read_array = DASDAEV1.read_array

        def forbidden(*args, **kwargs):
            raise AssertionError("measurement array loaded during metadata planning")

        monkeypatch.setattr(DASDAEV1, "read_array", forbidden)
        selected = spool.select(distance=np.array([[2.0, 7.0], [4.0, 5.0]]))
        assert _listed(selected.get_contents()) == [[3.0, 6.0]]
        plan = spool.chunk_plan(
            distance=np.array([[2.0, 7.0], [4.0, 5.0]]), on_incomplete="ignore"
        )
        chunked = spool.chunk(
            distance=np.array([[2.0, 7.0], [4.0, 5.0]]), on_incomplete="ignore"
        )
        assert _listed(plan.outputs) == [[3.0, 6.0]]
        assert _listed(chunked.get_contents()) == [[3.0, 6.0]]
        monkeypatch.setattr(DASDAEV1, "read_array", read_array)
        assert _ends(selected[0]) == (3, 6)
        assert _ends(chunked[0]) == (3, 6)

    def test_fill_only_file_window_uses_coordinate_metadata(
        self, tmp_path, monkeypatch
    ):
        """Filling a tiny file-backed hole does not read a whole source array."""
        time = np.arange(3).astype("datetime64[s]")

        def make_patch(values):
            return dc.Patch(
                data=np.ones((len(values), 3)),
                coords={
                    "distance": values,
                    "time": time,
                    "reference_time": ("time", time),
                },
                dims=("distance", "time"),
            )

        first = make_patch(np.arange(10_000))
        second = make_patch(np.arange(10_009, 20_009))
        path = tmp_path / "gapped.h5"
        dc.write(
            dc.spool([first, second]), path, file_format="DASDAE", file_version="2"
        )
        source = dc.spool(path)
        out = source.chunk(
            distance=np.array([[10_000, 10_008]]), tolerance=11, fill_value=-1
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("fill-only anchor read a measurement array")

        monkeypatch.setattr(DASDAEV2, "read_array", forbidden)
        patch = out[0]
        assert patch.shape == (9, 3)
        assert np.all(patch.data == -1)
        assert np.array_equal(patch.coords["reference_time"].values, time)
        assert _ends(patch) == (
            10_000,
            10_008,
        )

    def test_fill_only_legacy_blank_key_loads_anchor(self, tmp_path):
        """A persisted blank key can load a sole patch despite a native scan key."""
        time = np.arange(3).astype("datetime64[s]")
        for index, values in enumerate((np.arange(5), np.arange(7, 12))):
            patch = dc.Patch(
                data=np.ones((5, 3)),
                coords={
                    "distance": values,
                    "time": time,
                    "reference_time": ("time", time),
                },
                dims=("distance", "time"),
            )
            dc.write(patch, tmp_path / f"{index}.h5", file_format="DASDAE")
        source = dc.spool(tmp_path).update(progress=None)
        records = source._catalog.backend.export_records()
        assert all(record.patches[0].source_patch_key for record in records)
        legacy = [
            replace(
                record,
                patches=tuple(
                    replace(patch, source_patch_key="") for patch in record.patches
                ),
            )
            for record in records
        ]
        source._catalog.backend.write_sources(legacy)
        source._catalog._invalidate()
        filled = source.chunk(distance=np.array([[5, 6]]), tolerance=4, fill_value=-1)
        contents = filled.get_contents()
        assert len(contents) == 1
        assert (contents.iloc[0]["distance_min"], contents.iloc[0]["distance_max"]) == (
            5,
            6,
        )
        patch = filled[0]
        assert patch.shape == (2, 3)
        assert np.all(patch.data == -1)
        assert patch.coords["distance"].values.tolist() == [5.0, 6.0]
        assert np.array_equal(patch.coords["reference_time"].values, time)

    @pytest.mark.parametrize("policy", ["raise", "warn", "ignore"])
    def test_distant_unavailable_partition_does_not_veto_window(
        self, legacy_residual_source, policy
    ):
        """A disconnected legacy file cannot veto a known source's samples."""
        source = legacy_residual_source(100)
        window = np.array([[0, 3]])
        plan = source.chunk_plan(distance=window, on_incomplete=policy)
        chunked = source.chunk(distance=window, on_incomplete=policy)
        assert len(plan.outputs) == len(chunked) == 1
        assert plan.outputs.iloc[0]["distance_min"] == 0
        assert plan.outputs.iloc[0]["distance_max"] == 3
        assert chunked[0].coords["distance"].values.tolist() == [0, 1, 2, 3]
        assert chunked.get_contents().iloc[0]["distance_max"] == 3
        # The gap between these disconnected partitions supplies no sample.
        assert not len(
            source.chunk(distance=np.array([[5, 99]]), on_incomplete="ignore")
        )

    @pytest.mark.parametrize("method", ["chunk", "chunk_plan"])
    def test_unavailable_partition_still_follows_policy(
        self, legacy_residual_source, method
    ):
        """A request intersecting the blank-key partition stays incomplete."""
        source = legacy_residual_source(100)
        call = getattr(source, method)
        window = np.array([[100, 103]])
        with pytest.raises(
            ChunkError, match="exact source coordinates are unavailable"
        ):
            call(distance=window)
        with pytest.warns(
            UserWarning, match="exact source coordinates are unavailable"
        ):
            warned = call(distance=window, on_incomplete="warn")
        ignored = call(distance=window, on_incomplete="ignore")
        for result in (warned, ignored):
            assert len(result.outputs if method == "chunk_plan" else result) == 0

    def test_fill_only_gap_checks_both_supporting_sources(self, legacy_residual_source):
        """A bridged gap checks its unavailable supporting source."""
        source = legacy_residual_source(7)
        window = np.array([[5, 6]])
        with pytest.raises(
            ChunkError, match="exact source coordinates are unavailable"
        ):
            source.chunk_plan(distance=window, tolerance=4, fill_value=-1)
        with pytest.warns(
            UserWarning, match="exact source coordinates are unavailable"
        ):
            warned = source.chunk(
                distance=window, tolerance=4, fill_value=-1, on_incomplete="warn"
            )
        assert len(warned) == 0
        assert not len(
            source.chunk(
                distance=window,
                tolerance=4,
                fill_value=-1,
                on_incomplete="ignore",
            )
        )

    def test_nested_plan_fill_only_anchor(self):
        """A derived source still supplies structure to an all-fill window."""
        time = np.arange(3).astype("datetime64[s]")

        def make(values):
            return dc.Patch(
                data=np.ones((len(values), 3)),
                coords={"distance": values, "time": time},
                dims=("distance", "time"),
            )

        source = dc.spool([make(np.arange(5)), make(np.arange(7, 12))])
        derived = source.chunk(time=None)
        filled = derived.chunk(distance=np.array([[5, 6]]), tolerance=4, fill_value=-1)
        assert filled[0].shape == (2, 3)
        assert np.all(filled[0].data == -1)

    def test_fill_only_anchor_applies_parent_sample_residual(self):
        """Metadata-only anchor coordinates reflect prior sample selections."""
        time = np.arange(5).astype("datetime64[s]")

        def make(values):
            return dc.Patch(
                data=np.ones((len(values), 5)),
                coords={"distance": values, "time": time},
                dims=("distance", "time"),
            )

        source = dc.spool([make(np.arange(5)), make(np.arange(7, 12))])
        selected = source.select(time=(1, 4), samples=True)
        filled = selected.chunk(distance=np.array([[5, 6]]), tolerance=4, fill_value=-1)
        assert filled[0].shape == (2, 3)
        assert np.array_equal(filled[0].coords["time"].values, time[1:4])

    def test_mixed_unit_fill_anchor_uses_plan_units(self):
        """A fill-only output anchored by feet still uses the planned metres."""

        def metre_patch(values):
            return _patch(values).set_units(distance="m")

        left = metre_patch(np.arange(5))
        right = metre_patch(np.arange(7, 12)).convert_units(distance="ft")
        out = dc.spool([left, right]).chunk(
            distance=np.array([[5 * m, 6 * m]], dtype=object),
            tolerance=4,
            fill_value=-1,
        )
        assert _ends(out[0]) == (
            5,
            6,
        )
        assert str(out[0].coords["distance"].units) == str(
            left.coords["distance"].units
        )

    def test_missing_nested_member_metadata_is_not_assumed_complete(self):
        """A derived view cannot invent a lost irregular member coordinate."""
        source = dc.spool(_patch([0.0, 1.0, 3.0, 6.0, 9.0]))
        derived = source.chunk(distance=None)
        derived._catalog.resolver.loader.live._registry.clear()
        with pytest.raises(
            MissingPatchError, match="coordinate metadata is unavailable"
        ):
            derived.select(distance=np.array([[0.0, 6.0]])).get_contents()

    @pytest.mark.parametrize("backend", ["memory", "file", "legacy"])
    def test_fill_partition_anchor_converts_units(self, tmp_path, backend):
        """A later fill partition can start in units unlike the plan's unit."""
        patches = [
            _patch(np.arange(start, start + 5)).set_units(distance="m")
            for start in (0, 10, 17)
        ]
        patches[1] = patches[1].convert_units(distance="ft")
        source = dc.spool(patches)
        if backend != "memory":
            for index, patch in enumerate(patches):
                dc.write(patch, tmp_path / f"{index}.h5", file_format="DASDAE")
            source = dc.spool(tmp_path).update(progress=None)
            if backend == "legacy":
                records = source._catalog.backend.export_records()
                source._catalog.backend.write_sources(
                    [
                        replace(
                            record,
                            patches=tuple(
                                replace(patch, source_patch_key="")
                                for patch in record.patches
                            ),
                        )
                        for record in records
                    ]
                )
                source._catalog._invalidate()
        windows = np.array([[15 * m, 16 * m]], dtype=object)
        result = source.chunk(distance=windows, tolerance=4, fill_value=-1)
        assert np.allclose(
            result.get_contents()[["distance_min", "distance_max"]].to_numpy(),
            [[15, 16]],
        )
        loaded = result[0]
        assert np.allclose(loaded.get_coord("distance").values, [15, 16])
        assert loaded.get_coord("distance").units == m
        assert loaded.data.tolist() == [-1, -1]

    @pytest.mark.parametrize("values", [np.arange(10.0), [0.0, 1.0, 3.0, 6.0, 9.0]])
    def test_collapsed_file_members_recover_coordinates(self, tmp_path, values):
        """Re-planning file members reaches the plan's file loader."""
        patch = _patch(values)
        path = tmp_path / "source.h5"
        dc.write(patch, path, file_format="DASDAE")
        merged = dc.spool(path).chunk(distance=None)
        windows = np.array([[1.0, 7.0]])
        plan = merged.chunk_plan(distance=windows)
        result = merged.chunk(distance=windows)
        expected = patch.select(distance=(1, 7))
        assert len(plan.outputs) == len(result) == 1
        assert np.array_equal(result[0].data, expected.data)
        assert np.array_equal(
            result[0].get_coord("distance").values,
            expected.get_coord("distance").values,
        )
        assert (
            plan.outputs["distance_min"].iloc[0] == expected.get_coord("distance").min()
        )
        assert (
            plan.outputs["distance_max"].iloc[0] == expected.get_coord("distance").max()
        )

    def test_collapsed_nested_members_use_owning_plan(self):
        """A plan on a second dimension must recover its parent's full axis."""
        time = dc.to_datetime64(np.arange(3))
        patch = dc.Patch(
            data=np.arange(27).reshape(9, 3),
            coords={"distance": np.arange(9), "time": time},
            dims=("distance", "time"),
        )
        first = dc.spool(patch).chunk(time=1, keep_partial=True)
        expected = [part.select(distance=(1, 7)) for part in first]
        nested = first.chunk(distance=3, keep_partial=True)
        windows = np.array([[1, 7]])
        plan = nested.chunk_plan(distance=windows)
        result = nested.chunk(distance=windows)
        assert len(plan.outputs) == len(result) == len(expected) > 1
        assert _listed(plan.outputs) == [[1, 7]] * len(expected)
        for loaded, wanted in zip(result, expected, strict=True):
            assert loaded.get_coord("distance").values.tolist() == list(range(1, 8))
            assert np.array_equal(
                loaded.get_coord("time").values, wanted.get_coord("time").values
            )
            assert np.array_equal(loaded.data, wanted.data)

    def test_missing_live_coordinate_metadata_is_not_assumed_complete(self):
        """A stale live index cannot advertise unverified uneven samples."""
        source = dc.spool(_patch([0.0, 1.0, 3.0, 6.0, 9.0]))
        source.get_contents()
        source._catalog.resolver._registry.clear()
        window = np.array([[0.0, 6.0]])
        with pytest.raises(
            MissingPatchError, match="coordinate metadata is unavailable"
        ):
            source.select(distance=window).get_contents()
        with pytest.raises(
            ChunkError, match="exact source coordinates are unavailable"
        ):
            source.chunk_plan(distance=window)
        assert len(source.chunk(distance=window, on_incomplete="ignore")) == 0
        regular = dc.spool(_patch(np.arange(10)))
        regular.get_contents()
        regular._catalog.resolver._registry.clear()
        with pytest.raises(
            MissingPatchError, match="coordinate metadata is unavailable"
        ):
            regular.select(distance=np.array([[0, 3]])).get_contents()
        # A regular indexed grid still states its sample positions without
        # the live payload; planning may proceed even though loading cannot.
        assert len(regular.chunk_plan(distance=np.array([[0, 3]])).outputs) == 1

    def test_filled_output_rechunks_and_union_metadata(self):
        """Derived and union-backed windows retain exact planned bounds."""
        patch = _patch(np.arange(10))
        filled = dc.spool(patch).chunk(distance=np.array([[2, 4]]), fill_value=0)
        again = filled.chunk(distance=np.array([[2, 4]]))
        assert _ends(again[0]) == (2, 4)
        union = dc.spool(patch.select(distance=(0, 4))) + dc.spool(
            patch.select(distance=(5, 9))
        )
        joined = union.chunk(distance=np.array([[0, 9]]))
        assert _ends(joined[0]) == (0, 9)

    def test_union_of_planned_sources_uses_source_metadata(self):
        """A union routes each plan-backed row to its own coordinate source."""
        patch = _patch(np.arange(10))
        left = dc.spool(patch.select(distance=(0, 4))).chunk(
            distance=np.array([[0, 4]])
        )
        right = dc.spool(patch.select(distance=(5, 9))).chunk(
            distance=np.array([[5, 9]])
        )
        union = left + right
        selected = union.select(distance=np.array([[0, 9]]))
        assert [_ends(x) for x in selected] == [(0, 4), (5, 9)]
        joined = union.chunk(distance=np.array([[0, 9]]))
        assert _ends(joined[0]) == (0, 9)

    def test_residual_bearing_plan_stays_clipped(self):
        """Nested windows cannot recover samples removed by parent residuals."""
        source = dc.spool(_patch(np.arange(10)))
        derived = source.select(distance=(1, 8), samples=True).chunk(distance=None)
        selected = derived.select(distance=np.array([[0, 5]]))
        assert _ends(selected[0]) == (1, 5)

    @pytest.mark.parametrize(
        "unit,low,high", [("degC", 2, 4), ("kelvin", 275.15, 277.15)]
    )
    def test_affine_quantity_bounds_match_samples(self, unit, low, high):
        """Celsius coordinates accept equivalent absolute temperature points."""
        patch = _patch(np.arange(10)).set_units(distance="degC")
        spool = dc.spool(patch)
        quantity = get_quantity(unit)
        windows = np.array([[low * quantity, high * quantity]], dtype=object)
        selected = spool.select(distance=windows)
        chunked = spool.chunk(distance=windows)
        plan = spool.chunk_plan(distance=windows)
        for out in (selected, chunked):
            assert len(out) == 1
            assert _listed(out.get_contents()) == [[2.0, 4.0]]
            assert out[0].coords["distance"].values.tolist() == [2, 3, 4]
            assert out[0].data.tolist() == [2, 3, 4]
        assert _listed(plan.outputs) == [[2.0, 4.0]]

    def test_quantity_points_across_compatible_units(self):
        """An absolute metre window keeps its physical bounds in mixed units."""
        patch = dc.get_example_patch().set_units(distance="m")
        spool = dc.spool([patch, patch.convert_units(distance="ft")])
        windows = np.array([[20 * m, 25 * m]], dtype=object)
        selected = spool.select(distance=windows)
        assert len(selected) == 2
        chunked = spool.chunk(distance=windows)
        plan = spool.chunk_plan(distance=windows)
        assert len(chunked) == len(plan.outputs) == 1
        assert _ends(chunked[0]) == (20, 25)
        narrower = np.array([[21 * m, 24 * m]], dtype=object)
        for derived in (
            chunked.select(distance=narrower),
            chunked.chunk(distance=narrower),
        ):
            assert len(derived) == 1
            assert _ends(derived[0]) == (21, 24)

    def test_conflicts_only_between_contributors(self):
        """Separate windows carry different attrs; a combined one conflicts."""
        patch = _patch(np.arange(10))
        left = patch.select(distance=(0, 4)).update_attrs(tag="left")
        right = patch.select(distance=(5, 9)).update_attrs(tag="right")
        spool = dc.spool([left, right])
        separate = spool.chunk(distance=np.array([[0, 4], [5, 9]]), group=[])
        assert [x.attrs.tag for x in separate] == ["left", "right"]
        with pytest.raises(dc.exceptions.CoordMergeError, match="tag"):
            spool.chunk_plan(distance=np.array([[0, 9]]), group=[])

    def test_inward_aligned_bounds_near_envelope(self):
        """Off-grid outer edges need only their inward grid positions."""
        spool = dc.spool(_patch(np.arange(10)))
        windows = np.array([[-0.2, 3.8], [1.2, 9.2]])
        out = spool.chunk(distance=windows)
        assert [_ends(x) for x in out] == [(0, 3), (2, 9)]


class TestExplicitDerivedViews:
    """Derived outputs retain exact coordinate and revision semantics."""

    def test_second_explicit_selection_keeps_exact_bounds(self):
        """A second off-grid window does not advertise unsampled edges."""
        source = dc.spool(_patch(np.arange(10)))
        first = source.select(distance=np.array([[1.2, 4.8]]))
        second = first.select(distance=np.array([[2.2, 3.8]]))
        assert _listed(second.get_contents()) == [[3, 3]]
        assert _ends(second[0]) == (3, 3)

    def test_multi_member_irregular_rechunk_checks_actual_samples(self):
        """Rechunking a merged uneven output cannot publish a gap-only row."""
        patch = _patch([0.0, 1.0, 3.0, 6.0, 8.0, 11.0])
        source = dc.spool(
            [
                patch.select(distance=(0, 3)),
                patch.select(distance=(6, 11)),
            ]
        )
        joined = source.chunk(distance=np.array([[0.0, 11.0]]), tolerance=10)
        with pytest.raises(ChunkError, match="contains no source samples"):
            joined.chunk(distance=np.array([[4.0, 5.0]]))
        assert (
            len(joined.chunk(distance=np.array([[4.0, 5.0]]), on_incomplete="ignore"))
            == 0
        )
        piece = joined.chunk(distance=np.array([[6.0, 9.0]]))
        assert _listed(piece.get_contents()) == [[6, 8]]

    def test_live_parent_revision_updates_selected_pieces(self):
        """The view rebuilds its piece plan after a live parent changes."""
        source = dc.spool(_patch(np.arange(10)))
        selected = source.select(distance=np.array([[5, 15]]))
        assert len(selected) == 1
        source._catalog.add(_patch(np.arange(10, 20)))
        assert _listed(selected.get_contents()) == [[5, 9], [10, 15]]


class TestReviewRegressions:
    """Explicit windows agree with the samples and coordinates they yield."""

    def test_shifted_disconnected_partition_uses_its_own_grid(self):
        """A later compatible acquisition may have a different origin."""
        source = dc.spool([_patch(np.arange(5.0)), _patch(np.arange(10.5, 15.5))])
        for bounds, expected in (
            ([11, 14], [11.5, 12.5, 13.5]),
            ([10.5, 10.5], [10.5]),
        ):
            _assert_window_matches_plan(source, bounds, expected)

    @pytest.mark.parametrize("file_backed", [False, True])
    def test_associated_selector_replays_on_shared_dimension(
        self, file_backed, tmp_path, monkeypatch
    ):
        """Exact planning sees ordinary selectors through another coordinate."""
        patch = dc.Patch(
            data=np.arange(10),
            coords={
                "distance": np.arange(10),
                "depth": ("distance", 2 * np.arange(10)),
            },
            dims=("distance",),
        )
        if file_backed:
            path = tmp_path / "associated.h5"
            dc.write(patch, path, file_format="DASDAE")
            source = dc.spool(path)
            read_array = DASDAEV1.read_array

            def forbidden(*args, **kwargs):
                raise AssertionError("planning loaded the measurement array")

            monkeypatch.setattr(DASDAEV1, "read_array", forbidden)
        else:
            source = dc.spool(patch)
        empty = source.select(distance=np.array([[0, 1]]), depth=(4, 8))
        assert empty.get_contents().empty
        selected = source.select(distance=np.array([[0, 3]]), depth=(4, 8))
        assert _listed(selected.get_contents()) == [[2, 3]]
        with pytest.raises(ChunkError, match="sampled"):
            source.select(depth=(4, 8)).chunk_plan(distance=np.array([[0, 1]]))
        if file_backed:
            monkeypatch.setattr(DASDAEV1, "read_array", read_array)
        assert selected[0].coords["distance"].values.tolist() == [2, 3]
        derived = source.select(depth=np.array([[4, 8]]))
        rechunked = derived.chunk(distance=np.array([[2, 3]]))
        assert rechunked[0].coords["distance"].values.tolist() == [2, 3]

    def test_fill_only_output_advertises_unchunked_associated_coord(
        self, filled_2d_gap
    ):
        """An all-fill view remains selectable by an anchor's rider."""
        filled, time = filled_2d_gap
        assert "reference_time" in filled._catalog.backend.coord_names()
        filtered = filled.select(reference_time=(time[0], time[1]))
        assert len(filtered) == 1
        assert filtered[0].coords["reference_time"].values.tolist() == time[:2].tolist()
        assert filtered[0].coords["distance"].values.tolist() == [5, 6]

    @pytest.mark.parametrize(
        "bounds,expected",
        [
            ([5.25, 8.25], [5.25, 6.25, 7.25, 8.25]),
            ([5.25, 5.25], [5.25]),
            ([5.1, 5.4], [5.25]),
        ],
    )
    def test_snap_joined_jitter_keeps_exact_window(self, bounds, expected):
        """Joined patches can begin a fraction of a sample off-grid."""
        source = dc.spool([_patch(np.arange(5.0)), _patch(np.arange(5.25, 10.25))])
        _assert_window_matches_plan(source, bounds, expected)

    def test_uneven_three_samples_with_fill_load_without_grid(self):
        """A fill option does not force an uneven but complete piece to fill."""
        source = dc.spool(_patch([0.0, 1.0, 3.0, 4.0, 6.0]))
        windows = np.array([[0.5, 4.5]])
        plan = source.chunk_plan(distance=windows, tolerance=10, fill_value=-1)
        chunked = source.chunk(distance=windows, tolerance=10, fill_value=-1)
        assert _listed(plan.outputs) == [[1, 4]]
        assert _listed(chunked.get_contents()) == [[1, 4]]
        assert chunked[0].coords["distance"].values.tolist() == [1, 3, 4]
        assert chunked[0].data.tolist() == [1, 2, 3]

    def test_float_grid_missing_edge_obeys_incomplete_policy(self):
        """Floating arithmetic cannot hide a requested missing edge sample."""
        source = dc.spool(_patch(np.arange(0.3, 0.6, 0.1)))
        windows = np.array([[0.2, 0.4]])
        with pytest.raises(ChunkError, match="sampled bounds"):
            source.chunk_plan(distance=windows)
        with pytest.warns(UserWarning, match="sampled bounds"):
            warned = source.chunk(distance=windows, on_incomplete="warn")
        assert len(warned) == 0
        assert len(source.chunk(distance=windows, on_incomplete="ignore")) == 0
        partial = source.chunk(distance=windows, keep_partial=True)
        assert np.allclose(partial[0].coords["distance"].values, [0.3, 0.4])

    def test_segmented_source_preserves_gap_partitions(self):
        """A full request retains independent pieces of one segmented source."""
        coord = concat_coords(
            get_coord(start=0, step=1, shape=3),
            get_coord(start=10, step=1, shape=3),
        )
        patch = dc.Patch(
            data=np.arange(6), coords={"distance": coord}, dims=("distance",)
        )
        source = dc.spool(patch)
        out = source.chunk(distance=np.array([[0, 12]]), keep_partial=True)
        assert len(out) == 2
        assert [patch.coords["distance"].values.tolist() for patch in out] == [
            [0, 1, 2],
            [10, 11, 12],
        ]
        assert _listed(out.get_contents()) == [[0, 2], [10, 12]]

    def test_exact_edges_preserve_internal_overlap_cuts(self):
        """Recovering source labels cannot reintroduce clipped overlap samples."""
        source = dc.spool([_patch(np.arange(6)), _patch(np.arange(3, 9))])
        out = source.chunk(distance=np.array([[0, 8]]))
        assert len(out) == 1
        assert out[0].coords["distance"].values.tolist() == list(range(9))
        assert out[0].data.tolist() == [0, 1, 2, 3, 4, 5, 3, 4, 5]

    def test_multi_member_associated_filter_rechunks_exact_samples(self):
        """A merged plan replays a shared-dimension selector on all members."""

        def make(values):
            return dc.Patch(
                data=np.arange(len(values)),
                coords={"distance": values, "depth": ("distance", 2 * values)},
                dims=("distance",),
            )

        source = dc.spool([make(np.arange(5)), make(np.arange(5, 10))])
        merged = source.chunk(distance=np.array([[0, 9]]), conflict="drop")
        selected = merged.select(depth=(4, 12))
        windows = np.array([[2, 6]])
        plan = selected.chunk_plan(distance=windows)
        out = selected.chunk(distance=windows)
        assert len(plan.outputs) == len(out) == 1
        assert _listed(plan.outputs) == [[2, 6]]
        assert _listed(out.get_contents()) == [[2, 6]]
        assert out[0].coords["distance"].values.tolist() == [2, 3, 4, 5, 6]
        assert out[0].coords["depth"].values.tolist() == [4, 6, 8, 10, 12]

    def test_nested_union_plan_replays_associated_selector(self):
        """A union resolves plan sources without reading measurement arrays."""
        patch = dc.Patch(
            data=np.arange(10),
            coords={
                "distance": np.arange(10),
                "depth": ("distance", 2 * np.arange(10)),
            },
            dims=("distance",),
        )
        source = dc.spool(patch)
        first = source.chunk(distance=np.array([[0, 4]]))
        second = source.chunk(distance=np.array([[5, 9]]))
        union = first + second
        selected = union.select(depth=(4, 12))
        out = selected.chunk(distance=np.array([[2, 6]]), keep_partial=True)
        assert _listed(out.get_contents()) == [[2, 6]]
        assert out[0].coords["distance"].values.tolist() == [2, 3, 4, 5, 6]

    def test_mixed_unit_plan_recovers_associated_selector(self):
        """Metadata normalizes member distance units before combining a plan."""

        def make(values):
            return dc.Patch(
                data=np.arange(len(values)),
                coords={"distance": values, "depth": ("distance", 2 * values)},
                dims=("distance",),
            ).set_units(distance="m")

        left = make(np.arange(5))
        right = make(np.arange(5, 10)).convert_units(distance="ft")
        source = dc.spool([left, right])
        with pytest.warns(UserWarning, match="histories differ"):
            merged = source.chunk(
                distance=np.array([[2 * m, 8 * m]], dtype=object),
                conflict="drop",
            )
            # Load once while the warning context captures merge history.
            assert merged[0].coords["distance"].values.tolist() == list(range(2, 9))
        selected = merged.select(depth=(8, 16))
        windows = np.array([[5 * m, 8 * m]], dtype=object)
        plan = selected.chunk_plan(distance=windows)
        out = selected.chunk(distance=windows)
        assert _listed(plan.outputs) == [[5, 8]]
        assert _listed(out.get_contents()) == [[5, 8]]
        with pytest.warns(UserWarning, match="histories differ"):
            assert out[0].coords["distance"].values.tolist() == [5, 6, 7, 8]

    def test_filled_plan_recovery_uses_output_grid_and_rider(self, filled_2d_gap):
        """A fill-only derived source keeps its own grid and time rider."""
        filled, time = filled_2d_gap
        selected = filled.select(reference_time=(time[0], time[1]))
        windows = np.array([[time[0], time[1]]])
        plan = selected.chunk_plan(time=windows)
        out = selected.chunk(time=windows)
        assert len(plan.outputs) == len(out) == 1
        assert _listed(out.get_contents()) == [[5, 6]]
        assert out[0].coords["distance"].values.tolist() == [5, 6]
        assert np.array_equal(out[0].coords["reference_time"].values, time[:2])
        assert np.all(out[0].data == -1)

    def test_missing_nested_shared_metadata_obeys_policy(self):
        """A regular index cannot prove an associated residual after source loss."""
        patch = dc.Patch(
            data=np.arange(10),
            coords={
                "distance": np.arange(10),
                "depth": ("distance", 2 * np.arange(10)),
            },
            dims=("distance",),
        )
        derived = dc.spool(patch).chunk(distance=np.array([[0, 9]]))
        derived._catalog.resolver.loader.live._registry.clear()
        selected = derived.select(depth=(4, 8))
        windows = np.array([[2, 4]])
        with pytest.raises(ChunkError, match="coordinates are unavailable"):
            selected.chunk_plan(distance=windows)
        with pytest.warns(UserWarning, match="coordinates are unavailable"):
            warned = selected.chunk(distance=windows, on_incomplete="warn")
        assert len(warned) == 0
        assert not len(selected.chunk(distance=windows, on_incomplete="ignore"))

    def test_filled_derived_grid_matches_metadata_when_no_fill_needed(self):
        """A filled-capable plan with existing samples keeps its rider."""
        values = np.arange(10)
        patch = dc.Patch(
            data=values,
            coords={"distance": values, "depth": ("distance", 2 * values)},
            dims=("distance",),
        )
        derived = dc.spool(patch).chunk(distance=np.array([[2, 4]]), fill_value=-1)
        out = derived.select(depth=(4, 6)).chunk(distance=np.array([[2, 3]]))
        assert _listed(out.get_contents()) == [[2, 3]]
        assert out[0].coords["distance"].values.tolist() == [2, 3]
        assert out[0].coords["depth"].values.tolist() == [4, 6]
        narrowed = derived.select(depth=(6, 6))
        with pytest.raises(ChunkError, match="sampled"):
            narrowed.chunk_plan(distance=np.array([[2, 4]]))
        partial = narrowed.chunk(distance=np.array([[2, 4]]), keep_partial=True)
        assert _listed(partial.get_contents()) == [[3, 3]]
        assert partial[0].coords["depth"].values.tolist() == [6]

    def test_missing_fill_anchor_metadata_obeys_policy(self, filled_2d_gap):
        """An unresolvable all-fill anchor cannot prove a chained window."""
        filled, time = filled_2d_gap
        filled._catalog.resolver.loader.live._registry.clear()
        selected = filled.select(reference_time=(time[0], time[1]))
        windows = np.array([[time[0], time[1]]])
        with pytest.raises(ChunkError, match="coordinates are unavailable"):
            selected.chunk_plan(time=windows)
        assert not len(selected.chunk(time=windows, on_incomplete="ignore"))

    def test_zero_dimensional_numpy_length_remains_scalar(self):
        """NumPy scalar arrays still name a chunk length."""
        source = dc.get_example_spool()
        assert len(source.chunk(time=np.array(4.0))) == len(source.chunk(time=4.0))


T0 = np.datetime64("2020-01-03T00:00:00", "ns")  # the example spool's start


def _t(*secs):
    """Time windows (n, 2) at offsets from `T0`; None stays open."""
    ns = [
        None if s is None else T0 + np.timedelta64(round(s * 1e3), "ms") for s in secs
    ]
    return np.array(ns, dtype=object).reshape(-1, 2)


def _d(*values):
    """Distance windows (n, 2)."""
    return np.array(values, dtype=object).reshape(-1, 2)


def _s(v):
    """An instant as seconds from `T0`; anything else unchanged."""
    return (v - T0) / np.timedelta64(1, "s") if isinstance(v, np.datetime64) else v


def _boxes(spool, dims=("time", "distance")):
    """Each output's (min, max) per dim, checked against the loaded data."""
    rows = spool.get_contents().to_dict("records")
    listed = [tuple((x[f"{d}_min"], x[f"{d}_max"]) for d in dims) for x in rows]
    ends = [[(p.get_coord(d).min(), p.get_coord(d).max()) for d in dims] for p in spool]
    assert listed == [tuple(x) for x in ends]
    return [tuple(tuple(map(_s, pair)) for pair in box) for box in ends]


class TestExplicitBoxes:
    """Row i of every explicit dimension is one rectangular window."""

    @pytest.fixture(scope="class")
    def spool(self):
        """The example spool: three patches adjacent in time."""
        return dc.get_example_spool()

    def test_boxes_and_open_ends(self, spool):
        """Windows keep request order; open ends take the data's edge."""
        time, distance = _t(1, 2, 7, 9, 1, 2), _d(10, 20, 100, 150, 10, 20)
        selected = _boxes(spool.select(time=time, distance=distance))
        box = ((1, 2), (10, 20))
        assert selected == [box, ((7, 7.996), (100, 150)), ((8, 9), (100, 150)), box]
        # None, NaN, NaT, inf and Ellipsis are open; a fully open row spans
        time = _t(None, 1, 23, None, None, None)
        distance = _d(None, None, 290, np.nan, -np.inf, 3)
        boxes = [((0, 1), (0, 299)), ((23, 23.996), (290, 299)), ((0, 23.996), (0, 3))]
        assert _boxes(spool.chunk(time=time, distance=distance)) == boxes
        selected = _boxes(spool.select(time=time, distance=distance))
        assert selected[:2] == boxes[:2] and len(selected) == 5  # a piece per source
        narrow = spool.select(distance=(50, 299))
        out = narrow.chunk(time=_t(0, 1), distance=_d(..., 60))
        assert _boxes(out) == [((0, 1), (50, 60))]
        # an open row on the chunked dim merges each segment whole
        gapped = dc.spool([spool[0], spool[2]])
        assert len(gapped.chunk(time=_t(None, None))) == 2
        assert len(gapped.chunk(time=_t(None, 1))) == 1
        with pytest.raises(ChunkError, match="split by a gap"):
            gapped.chunk(time=_t(None, 23.996))

    def test_trim_matches_main_dim(self, spool):
        """Completeness does not depend on keyword order; missed groups pass."""
        time, distance = _t(0, 7.998), _d(10, 299.5)  # both short of a sample
        expected = [((0, 7.996), (10, 299))]
        assert _boxes(spool.chunk(time=time, distance=distance)) == expected
        assert _boxes(spool.chunk(distance=distance, time=time)) == expected
        split = spool.chunk(distance=150).chunk(time=_t(1, 2), distance=_d(10, 20))
        assert _boxes(split) == [((1, 2), (10, 20))]
        # a group the box misses on another dim needs no main-dim coverage
        coords = {"x": np.arange(10), "y": np.arange(5)}
        first = dc.Patch(np.ones((10, 5)), coords=coords, dims=("x", "y"))
        rest = first.select(x=(0, 2)).update_coords(y=coords["y"] + 10)
        pair = dc.spool([first, rest])
        box = [((0, 8), (0, 2))]
        assert _boxes(pair.chunk(x=_d(0, 8), y=_d(0, 2)), "xy") == box
        assert _boxes(pair.chunk(y=_d(0, 2), x=_d(0, 8)), "xy") == box
        # trimmed dims never merge, so a box across files needs time first
        with pytest.raises(ChunkError, match="only the first dim merges"):
            spool.chunk(distance=distance, time=_t(7, 9))

    def test_chunk_boxes_across_files(self, spool, tmp_path, monkeypatch):
        """Each window merges across files, trims both dims, and loads lazily."""
        for number, patch in enumerate(spool):
            patch.io.write(tmp_path / f"{number}.h5", "DASDAE")
        files = dc.spool(tmp_path).update(progress=None)
        time, distance = _t(7, 9, 15, 17), _d(10, 20, 200, 299)
        with monkeypatch.context() as patched:  # no data is read while planning
            for reader in (DASDAEV1, DASDAEV2):
                patched.setattr(reader, "read_array", pytest.fail)
                patched.setattr(reader, "read", pytest.fail)
            out = files.chunk(time=time, distance=distance)
            assert len(out.get_contents()) == 2
            assert len(files.select(time=time, distance=distance).get_contents()) == 4
        assert _boxes(out) == [((7, 9), (10, 20)), ((15, 17), (200, 299))]
        whole = spool.chunk(time=None)[0]
        expected = whole.select(time=tuple(time[0]), distance=(10, 20))
        np.testing.assert_array_equal(out[0].data, expected.data)
        # chained selections narrow the trimmed outputs further
        assert _boxes(out.select(distance=(12, 15)))[0][1] == (12, 15)
        again = out.select(distance=_d(15, 25, 250, 400))
        assert [x[1] for x in _boxes(again)] == [(15, 20), (250, 299)]

    @pytest.mark.parametrize(
        "distance, reason, partial",
        [
            ((10.2, 10.8), "dim distance: contains no source", []),
            ((250, 400), "sampled bounds are incomplete", [(250, 299)]),
            ((400, 500), "outside source coverage", []),
        ],
    )
    def test_unmet_trim_follows_policy(self, spool, distance, reason, partial):
        """An unmet trim fails its request unless the policy skips it."""
        kwargs = dict(time=_t(0, 1, 0, 1), distance=_d(0, 5, *distance))
        with pytest.raises(ChunkError, match=f"row 1.*{reason}"):
            spool.chunk(**kwargs)
        with pytest.warns(UserWarning, match="row 1"):
            spool.chunk(on_incomplete="warn", **kwargs)
        ignored = spool.chunk(on_incomplete="ignore", **kwargs)
        assert [x[1] for x in _boxes(ignored)] == [(0, 5)]
        kept = spool.chunk(keep_partial=True, on_incomplete="ignore", **kwargs)
        assert [x[1] for x in _boxes(kept)] == [(0, 5), *partial]

    def test_unmet_trim_fails_its_group(self, spool):
        """A trim incomplete in one compatible group fails that group alone."""
        narrow = spool[0].update_attrs(tag="b").select(distance=(0, 100))
        tagged = dc.spool([spool[0].update_attrs(tag="a"), narrow])
        windows = dict(time=_t(0, 1), distance=_d(50, 150))
        with pytest.raises(ChunkError) as error:
            tagged.chunk(**windows)
        assert re.findall(r"group \d+_\d+, dim distance", str(error.value)) != []
        assert str(error.value).count("group") == 1
        out = tagged.chunk(on_incomplete="ignore", **windows)
        assert [x[1] for x in _boxes(out)] == [(50, 150)] and out[0].attrs.tag == "a"

    def test_trim_scans_only_touched_files(self, tmp_path, monkeypatch):
        """A trim reads coordinates only from the files a box touches."""
        for number, patch in enumerate(dc.get_example_spool("random_das", length=6)):
            distance = patch.get_coord("distance").values * 1.02  # float: scanned
            patch = patch.update_coords(distance=distance)
            patch.io.write(tmp_path / f"{number}.h5", "DASDAE")
        files = dc.spool(tmp_path).update(progress=None)
        monkeypatch.setattr(dc, "scan_payloads", scan := Mock(wraps=dc.scan_payloads))
        assert len(files.chunk(time=_t(1, 2), distance=_d(10.2, 20.4))) == 1
        assert scan.call_count == 1

    @pytest.mark.parametrize("case", ["first", "mixed", "size", "fill", "depth"])
    def test_refusals(self, case):
        """Refused combinations of windows, whether selected or chunked."""
        patch = dc.get_example_patch("random_patch_with_lat_lon")
        time = np.array([patch.get_coord("time").values[[10, 100]]])
        pair, bad = dict(time=time, distance=_d(10, 20)), "non-dimensional"
        error, match, windows = {
            "first": (ParameterError, bad, dict(latitude=_d(-109.85, -109.84))),
            "mixed": (ParameterError, bad, {}),
            "size": (ParameterError, "chunk size", dict(distance=10)),
            "fill": (ParameterError, "fill_value", dict(fill_value=0)),
            "depth": (ChunkError, "'depth' dimension", dict(depth=_d(10, 20))),
        }[case]
        if case == "mixed":  # distance is a dimension on one patch only
            rider = patch.rename_coords(distance="channel")
            values = ("channel", patch.get_coord("distance").values)
            patch = [patch, rider.update_coords(distance=values)]
        spool = dc.spool(patch)
        for method in (spool.chunk, spool.select)[: 1 + (match == bad)]:
            with pytest.raises(error, match=match):
                len(method(**{**pair, **windows}))

    def test_open_trim(self, spool):
        """An open trim keeps patches lacking the dim, and even the patch's id."""
        lone = spool[0].select(distance=(0, 0)).squeeze("distance")
        mixed = dc.spool([spool[0], lone])
        for distance, count in ((_d(None, None), 2), (_d(10, 20), 1)):
            for method in (mixed.select, mixed.chunk):
                out = method(time=_t(0, 1), distance=distance)
                assert len({p.dims for p in out}) == len(out) == count
        # patches lacking an open first dim load; an untouched one keeps its size
        mixed = dc.spool([spool[0], spool[0].select(time=(None, T0)).squeeze("time")])
        t, d = _t(None, None), _d(1, 3)
        outs = mixed.select(time=t, distance=d), mixed.chunk(distance=d, time=t)
        for out in (*outs, mixed.select(distance=d, time=t)):
            assert [p.shape for p in out] == [(3, 2000), (3,)]
        opened = mixed.select(distance=_d(None, None), time=t)
        sizes = [x.get_contents()["data_size"].tolist() for x in (opened, mixed)]
        assert sizes[0] == sizes[1]
        spanned = spool.chunk(time=_t(0, 1), distance=_d(None, None))
        assert spanned[0].attrs.data_id == spool.chunk(time=_t(0, 1))[0].attrs.data_id
        # NaT leaves either end open on a duration coordinate too
        lags = np.arange(5).astype("timedelta64[s]")
        lag = dc.spool(dc.Patch(data=lags, coords={"time": lags}, dims=("time",)))
        span = np.array([[1, "NaT"], ["NaT", 2]], dtype="timedelta64[s]")
        got = [x[0] for x in _boxes(lag.chunk(time=span), ["time"])]
        assert got == [(lags[1], lags[4]), (lags[0], lags[2])]

    def test_trim_metadata(self):
        """Trim metadata: recovered coords, rider identities, missing coords."""
        patch = dc.get_example_patch("random_patch_with_lat_lon")
        time = np.array([patch.get_coord("time").values[[10, 100]]])
        out = dc.spool([patch]).chunk(time=time, distance=_d(10, 20))
        rows = _ensure_patch_row(out._catalog.to_df())
        assert rows["_latitude_def_key"].isna().all()
        (coord,) = known_coordinates(out._catalog, rows, "distance").values()
        assert (coord.min(), coord.max()) == (10, 20)

        def make(distance, clear=True):
            coords = {"distance": distance, "time": np.arange(4)}
            patch = dc.Patch(np.ones((5, 4)), coords=coords, dims=("distance", "time"))
            out = dc.spool([patch])
            out.get_contents()
            if clear:  # a regular grid plans from the index alone
                out._catalog.resolver._registry.clear()
            return out

        time, uneven, inner = (
            np.array([[1, 2]]),
            np.array([0.0, 1, 3, 6, 9]),
            _d(0.5, 2.5),
        )
        rows = make(np.arange(5.0)).chunk_plan(time=time, distance=inner).outputs
        assert _listed(rows) == [[1, 2]]
        spanned = make(uneven).chunk_plan(time=time, distance=_d(None, None))
        assert len(spanned.outputs) == 1
        with pytest.raises(ChunkError, match="coordinates are unavailable"):
            make(uneven).chunk_plan(time=time, distance=inner)
        known = make(uneven, clear=False)
        assert _boxes(known.chunk(time=time, distance=inner))[0][1] == (1, 1)
        past = dict(time=time, distance=_d(3.0, 20.0))
        with pytest.raises(ChunkError, match="incomplete"):
            known.chunk(**past)
        partial = known.chunk(keep_partial=True, **past)[0]
        assert partial.get_coord("distance").values.tolist() == [3.0, 6.0, 9.0]

    def test_three_dims(self):
        """Three explicit dims select and merge one box from a 3-D spool."""
        data, dims = np.arange(4 * 5 * 12).reshape(4, 5, 12), ("a", "b", "c")
        coords = {"a": np.arange(4), "b": np.arange(5), "c": np.arange(12)}
        whole = dc.Patch(data=data, coords=coords, dims=dims)
        spool = dc.spool([whole.select(c=(0, 5)), whole.select(c=(6, 11))])
        windows = dict(c=_d(1, 7), a=_d(1, 2), b=_d(0, 3))
        chunked = spool.chunk(**windows)
        assert _boxes(chunked, dims) == [((1, 2), (0, 3), (1, 7))]
        expected = whole.select(a=(1, 2), b=(0, 3), c=(1, 7))
        np.testing.assert_array_equal(chunked[0].data, expected.data)
        pieces = [((1, 2), (0, 3), (1, 5)), ((1, 2), (0, 3), (6, 7))]
        assert _boxes(spool.select(**windows), dims) == pieces
