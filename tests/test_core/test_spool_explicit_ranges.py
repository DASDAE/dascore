"""Independent bounded windows for spool selection and cutting."""

import pickle
import threading
import warnings
from dataclasses import replace
from datetime import timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.core.coords import concat_coords, get_coord
from dascore.examples import inventory_patch_pair
from dascore.exceptions import (
    InvalidSpoolQueryError,
    MissingPatchError,
    ParameterError,
    UnitError,
)
from dascore.io.dasdae.core import DASDAEV1, DASDAEV2
from dascore.io.index.catalog import PatchCatalog
from dascore.units import ft, get_quantity, m, s
from dascore.utils.chunk_plan import _ensure_patch_row
from dascore.utils.explicit_ranges import known_coordinates
from dascore.utils.patch_assembly import coord_from_row
from dascore.warnings import DASCoreWarning
from tests.test_core.test_patch_chunk import (
    _T0,
    _assert_labels,
    _cut_spools,
    _exact_values,
    _time_values,
)


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


def _assert_window_matches_contents(source, bounds, expected):
    """Assert that a cut window's contents and its data agree."""
    cut = source.cut(distance=np.array([bounds]))
    assert len(cut) == 1
    assert cut[0].coords["distance"].values.tolist() == expected
    assert cut.get_contents().iloc[0]["distance_min"] == expected[0]
    assert cut.get_contents().iloc[0]["distance_max"] == expected[-1]


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
            x.coords["distance"].min() for x in spool.cut(distance=[[5, 7], [1, 3]])
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
        for method in (spool.select, spool.cut):
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


class TestExplicitCut:
    """Each window keeps the samples its sources hold inside it."""

    def test_contents_and_loaded_windows_agree(self):
        """Repeated and unsorted requests retain individual outputs."""
        spool = dc.spool(_patch(np.arange(10)))
        windows = np.array([[5, 7], [1, 3], [5, 7]])
        out = spool.cut(distance=windows)
        assert list(out.get_contents()["window"]) == [0, 1, 2]
        assert _listed(out.get_contents()) == [[5, 7], [1, 3], [5, 7]]
        assert [_ends(x) for x in out] == [(5, 7), (1, 3), (5, 7)]

    def test_incomplete_window_keeps_its_samples(self):
        """A window past its sources keeps what they hold (rule 9)."""
        spool = dc.spool(_patch(np.arange(10)))
        out = spool.cut(distance=np.array([[2, 4], [8, 10]]))
        assert [_ends(x) for x in out] == [(2, 4), (8, 9)]
        assert _listed(out.get_contents()) == [[2, 4], [8, 9]]

    @pytest.mark.parametrize("bounds", [[20.0, 30.0], [1.1, 1.9], [-3, -1]])
    def test_window_without_samples_gives_nothing(self, bounds):
        """A window outside coverage or between samples gives no output."""
        spool = dc.spool(_patch(np.arange(10)))
        out = spool.cut(distance=np.array([bounds, [2, 3]]))
        assert out.get_contents()["window"].tolist() == [1]

    def test_tolerance_joins_but_invents_no_edge(self):
        """Tolerance bridges internal gaps but cannot invent an edge sample."""
        patch = _patch(np.arange(10))
        spool = dc.spool([patch.select(distance=(0, 4)), patch.select(distance=(7, 9))])
        (partial,) = spool.cut(distance=np.array([[5, 9]]), tolerance=4)
        assert _ends(partial) == (7, 9)
        assert len(spool.cut(distance=np.array([[2, 9]]))) == 2
        with pytest.warns(UserWarning, match="tolerance"):
            (joined,) = spool.cut(distance=np.array([[2, 9]]), tolerance=4)
        assert _ends(joined) == (2, 9)
        assert len(spool.cut(distance=np.array([[5, 6]]), tolerance=4)) == 0

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
        cut = spool.cut(time=np.array([bounds]))
        assert np.array_equal(
            cut[0].coords["time"].values, scalar[0].coords["time"].values
        )
        mixed = spool.select(
            time=np.array([[np.datetime64(bounds[0]), bounds[1]]], dtype=object)
        )
        assert np.array_equal(
            mixed[0].coords["time"].values, scalar[0].coords["time"].values
        )
        with pytest.raises(ParameterError, match="not datetimes"):
            len(spool.select(time=np.array([[1, "later"]], dtype=object)))

    @pytest.mark.parametrize("kind", ["datetime64[s]", "timedelta64[s]"])
    def test_time_family_absolute_windows(self, kind):
        """Absolute time and duration windows retain their native endpoints."""
        values = np.arange(10).astype(kind)
        patch = dc.Patch(data=np.arange(10), coords={"time": values}, dims=("time",))
        spool = dc.spool(patch)
        requested = np.array([[values[2], values[4]]])
        selected = spool.select(time=requested)
        cut = spool.cut(time=requested)
        for out in (selected, cut):
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
        cut = spool.cut(time=requested)
        assert len(cut) == 1
        row = cut.get_contents().iloc[0]
        assert (row["time_min"], row["time_max"]) == (values[-1], values[-1])
        np.testing.assert_array_equal(cut[0].coords["time"].values, values[-1:])
        np.testing.assert_array_equal(cut[0].data, [4])

    def test_datetime_rejects_quantity_and_mixed_bounds_swap(self):
        """Only instants bound datetimes; mixed native/unit points swap (rule 8)."""
        values = np.arange(10).astype("datetime64[s]")
        time_spool = dc.spool(
            dc.Patch(data=np.arange(10), coords={"time": values}, dims=("time",))
        )
        with pytest.raises(ParameterError, match="absolute instants"):
            time_spool.cut(time=np.array([[2 * s, 4 * s]], dtype=object))
        distance = dc.spool(_patch(np.arange(10)).set_units(distance="m"))
        swapped = distance.cut(distance=np.array([[5, 2 * m]], dtype=object))
        assert [_ends(x) for x in swapped] == [(2, 5)]
        with pytest.raises(ParameterError, match="incompatible units"):
            distance.cut(distance=np.array([[3 * m, 1 * s]], dtype=object))

    def test_unitless_coordinate_rejects_quantity_points(self):
        """A unit-bearing absolute request cannot select a unitless grid."""
        spool = dc.spool(_patch(np.arange(10)))
        with pytest.raises(UnitError, match="unitless coordinate"):
            spool.cut(distance=np.array([[2 * m, 4 * m]], dtype=object))

    def test_explicit_ranges_on_empty_and_disjoint_sources(self):
        """Absent windows give nothing, including on a filtered empty source."""
        spool = dc.spool(_patch(np.arange(10)))
        empty = spool.select(distance=(20, 30))
        assert len(empty.cut(distance=np.array([[1, 3]]))) == 0
        assert len(spool.cut(distance=np.array([[20, 30]]))) == 0

    def test_unparsable_string_bounds_are_rejected(self):
        """String endpoints in explicit windows must name actual instants."""
        spool = dc.spool(_patch(np.arange(10)))
        for bounds in (["not-a-date", "later"], [None, "later"]):
            with pytest.raises(ParameterError, match="datetime"):
                spool.cut(distance=np.array([bounds], dtype=object))

    @pytest.mark.parametrize(
        "bounds",
        [
            (True, 2),
            ([1], 2),
            ("1", 2),
        ],
    )
    def test_invalid_absolute_bounds(self, bounds):
        """A bounded window needs scalar, comparable endpoints."""
        spool = dc.spool(_patch(np.arange(10)))
        with pytest.raises(ParameterError):
            spool.cut(distance=np.array([bounds], dtype=object))

    @pytest.mark.parametrize("method", ["select", "cut"])
    def test_text_dimension_refused(self, method):
        """Labels with no order to window are refused, naming the dimension."""
        coords = {"station": np.array(["a", "b", "c"])}
        spool = dc.spool(dc.Patch(data=np.arange(3), coords=coords, dims=("station",)))
        with pytest.raises(ParameterError, match="'station' need it to hold"):
            len(getattr(spool, method)(station=np.array([[0, 1]])))

    def test_argument_validation_and_empty_requests(self):
        """Malformed windows and policy names fail before planning data."""
        spool = dc.spool(_patch(np.arange(10)))
        empty = spool.cut(distance=np.empty((0, 2)))
        assert len(empty) == 0
        assert empty.get_contents()["window"].dtype == "Int64"
        assert len(spool.select(distance=np.empty((0, 2)))) == 0
        open_low = spool.cut(distance=np.array([[np.nan, 2]]))
        assert open_low[0].coords["distance"].values.tolist() == [0, 1, 2]
        for value in (np.array([1, 2]), np.ones((2, 3))):
            with pytest.raises(ParameterError):
                spool.cut(distance=value)
        with pytest.raises(ParameterError, match="overlap"):
            spool.cut(distance=np.array([[1, 2]]), overlap=1)


class TestExplicitMetadataSources:
    """File and in-memory sources report the same exact windows."""

    @pytest.mark.parametrize("method", ["select", "cut"])
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
        derived = source.cut(distance=np.array([[2, 8]]))
        scanner = dc.scan_payloads

        def forbidden(*args, **kwargs):
            raise AssertionError("regular indexed grid scanned its file payload")

        monkeypatch.setattr(dc, "scan_payloads", forbidden)
        selected = derived.select(distance=np.array([[3, 5]]))
        cut = derived.cut(distance=np.array([[3, 5]]))
        assert _listed(selected.get_contents()) == [[3, 5]]
        assert _listed(cut.get_contents()) == [[3, 5]]
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
        chunked = spool.cut(distance=np.array([[2.0, 7.0], [4.0, 5.0]]))
        assert _listed(chunked.get_contents()) == [[3.0, 6.0]]
        monkeypatch.setattr(DASDAEV1, "read_array", read_array)
        assert _ends(selected[0]) == (3, 6)
        assert _ends(chunked[0]) == (3, 6)

    def test_distant_unavailable_partition_does_not_veto_window(
        self, legacy_residual_source
    ):
        """A disconnected legacy file cannot veto a known source's samples."""
        source = legacy_residual_source(100)
        cut = source.cut(distance=np.array([[0, 3]]))
        assert len(cut) == 1
        assert cut[0].coords["distance"].values.tolist() == [0, 1, 2, 3]
        assert _listed(cut.get_contents()) == [[0, 3]]
        # The gap between these disconnected partitions supplies no sample.
        assert not len(source.cut(distance=np.array([[5, 99]])))

    @pytest.mark.parametrize("method", ["cut", "select"])
    def test_unavailable_partition_is_refused(self, legacy_residual_source, method):
        """A window over the blank-key source cannot name its samples."""
        source = legacy_residual_source(100)
        window = np.array([[100, 103]])
        with pytest.raises(MissingPatchError, match="metadata is unavailable"):
            len(getattr(source, method)(distance=window))

    def test_missing_nested_member_metadata_is_not_assumed_complete(self):
        """A derived view cannot invent a lost irregular member coordinate."""
        source = dc.spool(_patch([0.0, 1.0, 3.0, 6.0, 9.0]))
        derived = source.chunk(distance=None)
        derived._catalog.resolver.loader.live._registry.clear()
        with pytest.raises(
            MissingPatchError, match="coordinate metadata is unavailable"
        ):
            derived.select(distance=np.array([[0.0, 6.0]])).get_contents()

    @pytest.mark.parametrize("values", [np.arange(10.0), [0.0, 1.0, 3.0, 6.0, 9.0]])
    def test_collapsed_file_members_recover_coordinates(self, tmp_path, values):
        """Re-planning file members reaches the plan's file loader."""
        patch = _patch(values)
        path = tmp_path / "source.h5"
        dc.write(patch, path, file_format="DASDAE")
        merged = dc.spool(path).chunk(distance=None)
        windows = np.array([[1.0, 7.0]])
        result = merged.cut(distance=windows)
        expected = patch.select(distance=(1, 7))
        assert len(result) == 1
        assert np.array_equal(result[0].data, expected.data)
        assert np.array_equal(
            result[0].get_coord("distance").values,
            expected.get_coord("distance").values,
        )
        contents = result.get_contents()
        assert contents["distance_min"].iloc[0] == expected.get_coord("distance").min()
        assert contents["distance_max"].iloc[0] == expected.get_coord("distance").max()

    def test_collapsed_nested_members_use_owning_plan(self):
        """A window over a plan on a second dimension rejoins its full axis."""
        time = dc.to_datetime64(np.arange(3))
        patch = dc.Patch(
            data=np.arange(27).reshape(9, 3),
            coords={"distance": np.arange(9), "time": time},
            dims=("distance", "time"),
        )
        first = dc.spool(patch).chunk(time=1, keep_partial=True)
        assert len(first) > 1
        nested = first.chunk(distance=3, keep_partial=True)
        # cut joins the window along every dimension, so the time pieces
        # chunk made rejoin into one patch
        (loaded,) = nested.cut(distance=np.array([[1, 7]]))
        wanted = patch.select(distance=(1, 7))
        assert loaded.get_coord("distance").values.tolist() == list(range(1, 8))
        assert np.array_equal(
            loaded.get_coord("time").values, wanted.get_coord("time").values
        )
        assert np.array_equal(loaded.data, wanted.data)

    @pytest.mark.parametrize("conflict", ["keep_first", "drop"])
    def test_planned_members_follow_conflict(self, conflict, channel_conflict_pair):
        """
        A merged output's index and rebuilt coordinates apply its conflict
        policy, warning only when the output loads.
        """
        with warnings.catch_warnings(action="error"):
            merged = dc.spool(list(channel_conflict_pair)).chunk(
                time=None, conflict=conflict
            )
            out = merged.select(distance=np.array([[0, 10]]))
            if conflict == "drop":
                with pytest.raises(InvalidSpoolQueryError, match="channel"):
                    merged.select(channel=(0, 10))
            else:  # the index holds only the channels the merge keeps
                assert not len(merged.select(channel=(1000, 1010)))
                assert len(merged.select(channel=(0, 10))) == 1
        with pytest.warns(DASCoreWarning):
            out = out[0]
        assert ("channel" in out.coords.coord_map) == (conflict == "keep_first")
        assert _ends(out) == (0, 10)

    def test_keep_first_indexes_first_units(self, channel_conflict_pair):
        """keep_first indexes the first member's coordinate with its units."""
        first, second = channel_conflict_pair
        pair = [first.set_units(channel="m"), second.set_units(channel="km")]
        merged = dc.spool(pair).chunk(time=None, conflict="keep_first")
        assert merged.get_contents()["channel_units"].tolist() == ["m"]

    def test_missing_live_coordinate_metadata_is_not_assumed_complete(self):
        """A stale live index cannot advertise unverified uneven samples."""
        source = dc.spool(_patch([0.0, 1.0, 3.0, 6.0, 9.0]))
        source.get_contents()
        source._catalog.resolver._registry.clear()
        window = np.array([[0.0, 6.0]])
        for method in ("select", "cut"):
            with pytest.raises(
                MissingPatchError, match="coordinate metadata is unavailable"
            ):
                getattr(source, method)(distance=window).get_contents()
        regular = dc.spool(_patch(np.arange(10)))
        regular.get_contents()
        regular._catalog.resolver._registry.clear()
        for method in ("select", "cut"):
            with pytest.raises(
                MissingPatchError, match="coordinate metadata is unavailable"
            ):
                getattr(regular, method)(distance=np.array([[0, 3]])).get_contents()

    def test_union_metadata(self):
        """Union-backed windows retain exact planned bounds."""
        patch = _patch(np.arange(10))
        union = dc.spool(patch.select(distance=(0, 4))) + dc.spool(
            patch.select(distance=(5, 9))
        )
        joined = union.cut(distance=np.array([[0, 9]]))
        assert _ends(joined[0]) == (0, 9)

    def test_union_of_planned_sources_uses_source_metadata(self):
        """A union routes each plan-backed row to its own coordinate source."""
        patch = _patch(np.arange(10))
        left = dc.spool(patch.select(distance=(0, 4))).cut(distance=np.array([[0, 4]]))
        right = dc.spool(patch.select(distance=(5, 9))).cut(distance=np.array([[5, 9]]))
        union = left + right
        selected = union.select(distance=np.array([[0, 9]]))
        assert [_ends(x) for x in selected] == [(0, 4), (5, 9)]
        joined = union.cut(distance=np.array([[0, 9]]))
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
        cut = spool.cut(distance=windows)
        for out in (selected, cut):
            assert len(out) == 1
            assert _listed(out.get_contents()) == [[2.0, 4.0]]
            assert out[0].coords["distance"].values.tolist() == [2, 3, 4]
            assert out[0].data.tolist() == [2, 3, 4]

    def test_quantity_points_across_compatible_units(self):
        """An absolute metre window keeps its physical bounds in mixed units."""
        patch = dc.get_example_patch().set_units(distance="m")
        spool = dc.spool([patch, patch.convert_units(distance="ft")])
        windows = np.array([[20 * m, 25 * m]], dtype=object)
        selected = spool.select(distance=windows)
        assert len(selected) == 2
        cut = spool.cut(distance=windows)
        # the overlap is kept once, from the earlier source, in its metres
        assert len(cut) == 1
        assert _ends(cut[0]) == (20, 25)
        narrower = np.array([[21 * m, 24 * m]], dtype=object)
        for derived in (
            cut.select(distance=narrower),
            cut.cut(distance=narrower),
        ):
            assert len(derived) == 1
            assert _ends(derived[0]) == (21, 24)

    def test_conflicts_only_between_contributors(self):
        """Separate windows carry different attrs; a combined one conflicts."""
        patch = _patch(np.arange(10))
        left = patch.select(distance=(0, 4)).update_attrs(instrument_id="left")
        right = patch.select(distance=(5, 9)).update_attrs(instrument_id="right")
        spool = dc.spool([left, right])
        separate = spool.cut(distance=np.array([[0, 4], [5, 9]]))
        assert [x.attrs.instrument_id for x in separate] == ["left", "right"]
        with pytest.raises(dc.exceptions.CoordMergeError, match="instrument_id"):
            spool.cut(distance=np.array([[0, 9]]))

    def test_inward_aligned_bounds_near_envelope(self):
        """Off-grid outer edges need only their inward grid positions."""
        spool = dc.spool(_patch(np.arange(10)))
        windows = np.array([[-0.2, 3.8], [1.2, 9.2]])
        out = spool.cut(distance=windows)
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
        patch = _patch([0.0, 1.0, 3.0, 6.0, 8.0, 11.0]).set_units(distance="m")
        source = dc.spool(
            [
                patch.select(distance=(0, 3)),
                patch.select(distance=(6, 11)),
            ]
        )
        # labels with no step join only by a distance
        window = np.array([[0.0, 11.0]])
        assert len(source.cut(distance=window)) == 2
        joined = source.cut(distance=window, tolerance=dc.get_quantity("10 m"))
        assert len(joined) == 1
        assert len(joined.cut(distance=np.array([[4.0, 5.0]]))) == 0
        piece = joined.cut(distance=np.array([[6.0, 9.0]]))
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
            _assert_window_matches_contents(source, bounds, expected)

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
        assert not len(source.select(depth=(4, 8)).cut(distance=np.array([[0, 1]])))
        if file_backed:
            monkeypatch.setattr(DASDAEV1, "read_array", read_array)
        assert selected[0].coords["distance"].values.tolist() == [2, 3]
        derived = source.select(depth=np.array([[4, 8]]))
        recut = derived.cut(distance=np.array([[2, 3]]))
        assert recut[0].coords["distance"].values.tolist() == [2, 3]

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
        _assert_window_matches_contents(source, bounds, expected)

    def test_float_grid_missing_edge_keeps_held_samples(self):
        """Floating arithmetic invents no sample at a missing window edge."""
        source = dc.spool(_patch(np.arange(0.3, 0.6, 0.1)))
        (partial,) = source.cut(distance=np.array([[0.2, 0.4]]))
        assert np.allclose(partial.coords["distance"].values, [0.3, 0.4])

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
        out = source.cut(distance=np.array([[0, 12]]))
        assert len(out) == 2
        assert out.get_contents()["window"].tolist() == [0, 0]
        assert [patch.coords["distance"].values.tolist() for patch in out] == [
            [0, 1, 2],
            [10, 11, 12],
        ]
        assert _listed(out.get_contents()) == [[0, 2], [10, 12]]

    def test_exact_edges_preserve_internal_overlap_cuts(self):
        """Recovering source labels cannot reintroduce clipped overlap samples."""
        source = dc.spool([_patch(np.arange(6)), _patch(np.arange(3, 9))])
        out = source.cut(distance=np.array([[0, 8]]))
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
        merged = source.cut(distance=np.array([[0, 9]]))
        selected = merged.select(depth=(4, 12))
        out = selected.cut(distance=np.array([[2, 6]]))
        assert len(out) == 1
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
        first = source.cut(distance=np.array([[0, 4]]))
        second = source.cut(distance=np.array([[5, 9]]))
        union = first + second
        selected = union.select(depth=(4, 12))
        out = selected.cut(distance=np.array([[2, 6]]))
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
            merged = source.cut(distance=np.array([[2 * m, 8 * m]], dtype=object))
            # Load once while the warning context captures merge history.
            assert merged[0].coords["distance"].values.tolist() == list(range(2, 9))
        selected = merged.select(depth=(8, 16))
        windows = np.array([[5 * m, 8 * m]], dtype=object)
        out = selected.cut(distance=windows)
        # the window lies in the feet patch alone, its labels read in metres
        assert np.allclose(_listed(out.get_contents()), [[5, 8]])
        assert np.allclose(out[0].coords["distance"].values, [5, 6, 7, 8])

    def test_missing_nested_shared_metadata_is_refused(self):
        """A regular index cannot prove an associated residual after source loss."""
        patch = dc.Patch(
            data=np.arange(10),
            coords={
                "distance": np.arange(10),
                "depth": ("distance", 2 * np.arange(10)),
            },
            dims=("distance",),
        )
        derived = dc.spool(patch).cut(distance=np.array([[0, 9]]))
        derived._catalog.resolver.loader.live._registry.clear()
        selected = derived.select(depth=(4, 8))
        windows = np.array([[2, 4]])
        with pytest.raises(MissingPatchError, match="metadata is unavailable"):
            selected.cut(distance=windows)

    def test_derived_window_keeps_its_rider(self):
        """A window over a cut keeps its rider, and the samples it holds."""
        values = np.arange(10)
        patch = dc.Patch(
            data=values,
            coords={"distance": values, "depth": ("distance", 2 * values)},
            dims=("distance",),
        )
        derived = dc.spool(patch).cut(distance=np.array([[2, 4]]))
        out = derived.select(depth=(4, 6)).cut(distance=np.array([[2, 3]]))
        assert _listed(out.get_contents()) == [[2, 3]]
        assert out[0].coords["distance"].values.tolist() == [2, 3]
        assert out[0].coords["depth"].values.tolist() == [4, 6]
        narrowed = derived.select(depth=(6, 6))
        partial = narrowed.cut(distance=np.array([[2, 4]]))
        assert _listed(partial.get_contents()) == [[3, 3]]
        assert partial[0].coords["depth"].values.tolist() == [6]

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
        assert _boxes(spool.cut(time=time, distance=distance)) == boxes
        selected = _boxes(spool.select(time=time, distance=distance))
        assert selected[:2] == boxes[:2] and len(selected) == 5  # a piece per source
        narrow = spool.select(distance=(50, 299))
        out = narrow.cut(time=_t(0, 1), distance=_d(..., 60))
        assert _boxes(out) == [((0, 1), (50, 60))]
        # an open row merges each segment whole; a gap splits a window
        gapped = dc.spool([spool[0], spool[2]])
        assert len(gapped.cut(time=_t(None, None))) == 2
        assert len(gapped.cut(time=_t(None, 1))) == 1
        split = gapped.cut(time=_t(None, 23.996))
        assert [x[0] for x in _boxes(split)] == [(0, 7.996), (16, 23.996)]

    def test_keyword_order_and_missed_groups(self, spool):
        """Keyword order does not matter; a group a box misses gives nothing."""
        time, distance = _t(0, 7.998), _d(10, 299.5)  # both short of a sample
        expected = [((0, 7.996), (10, 299))]
        assert _boxes(spool.cut(time=time, distance=distance)) == expected
        assert _boxes(spool.cut(distance=distance, time=time)) == expected
        split = spool.chunk(distance=150).cut(time=_t(1, 2), distance=_d(10, 20))
        assert _boxes(split) == [((1, 2), (10, 20))]
        # a group the box misses on another dim needs no main-dim coverage
        coords = {"x": np.arange(10), "y": np.arange(5)}
        first = dc.Patch(np.ones((10, 5)), coords=coords, dims=("x", "y"))
        rest = first.select(x=(0, 2)).update_coords(y=coords["y"] + 10)
        pair = dc.spool([first, rest])
        box = [((0, 8), (0, 2))]
        assert _boxes(pair.cut(x=_d(0, 8), y=_d(0, 2)), "xy") == box
        assert _boxes(pair.cut(y=_d(0, 2), x=_d(0, 8)), "xy") == box
        # a box across files joins whichever dim is named first
        across = [((7, 9), (10, 299))]
        assert _boxes(spool.cut(distance=distance, time=_t(7, 9))) == across

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
            out = files.cut(time=time, distance=distance)
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
        "distance, partial",
        [((10.2, 10.8), []), ((250, 400), [(250, 299)]), ((400, 500), [])],
    )
    def test_short_trim_keeps_its_samples(self, spool, distance, partial):
        """A box short on its second dim keeps what it holds (rule 9)."""
        kwargs = dict(time=_t(0, 1, 0, 1), distance=_d(0, 5, *distance))
        kept = spool.cut(**kwargs)
        assert [x[1] for x in _boxes(kept)] == [(0, 5), *partial]

    def test_short_trim_in_one_group(self, spool):
        """A box short in one compatible group still keeps that group's part."""
        narrow = spool[0].update_attrs(tag="b").select(distance=(0, 100))
        tagged = dc.spool([spool[0].update_attrs(tag="a"), narrow])
        out = tagged.cut(time=_t(0, 1), distance=_d(50, 150))
        assert [x[1] for x in _boxes(out)] == [(50, 150), (50, 100)]
        assert [x.attrs.tag for x in out] == ["a", "b"]

    def test_trim_scans_only_touched_files(self, tmp_path, monkeypatch):
        """A trim reads coordinates only from the files a box touches."""
        for number, patch in enumerate(dc.get_example_spool("random_das", length=6)):
            distance = patch.get_coord("distance").values * 1.02  # float: scanned
            patch = patch.update_coords(distance=distance)
            patch.io.write(tmp_path / f"{number}.h5", "DASDAE")
        files = dc.spool(tmp_path).update(progress=None)
        monkeypatch.setattr(dc, "scan_payloads", scan := Mock(wraps=dc.scan_payloads))
        assert len(files.cut(time=_t(1, 2), distance=_d(10.2, 20.4))) == 1
        assert scan.call_count == 1

    @pytest.mark.parametrize("case", ["first", "mixed", "size", "fill", "depth"])
    def test_refusals(self, case):
        """Refused combinations of windows, whether selected or cut."""
        patch = dc.get_example_patch("random_patch_with_lat_lon")
        time = np.array([patch.get_coord("time").values[[10, 100]]])
        pair, bad = dict(time=time, distance=_d(10, 20)), "non-dimensional"
        # select refuses a non-dimensional coordinate beside another window;
        # cut windows only dimensions, and skips patches lacking them
        select, cut, windows = {
            "first": (bad, "dimensions only", dict(latitude=_d(-109.85, -109.84))),
            "mixed": (bad, None, {}),
            "size": (None, r"\(n, 2\) windows", dict(distance=10)),
            "fill": (None, "fill_value", dict(fill_value=0)),
            "depth": (None, "dimensions only", dict(depth=_d(10, 20))),
        }[case]
        if case == "mixed":  # distance is a dimension on one patch only
            rider = patch.rename_coords(distance="channel")
            values = ("channel", patch.get_coord("distance").values)
            patch = [patch, rider.update_coords(distance=values)]
        spool = dc.spool(patch)
        for method, match in ((spool.cut, cut), (spool.select, select)):
            if match is not None:
                with pytest.raises(ParameterError, match=match):
                    len(method(**{**pair, **windows}))
        if case == "mixed":  # only the patch holding the dimension is cut
            (out,) = spool.cut(**pair)
            assert out.dims == patch[0].dims and _ends(out) == (10, 20)

    def test_open_trim(self, spool):
        """An open trim keeps patches lacking the dim, and even the patch's id."""
        lone = spool[0].select(distance=(0, 0)).squeeze("distance")
        mixed = dc.spool([spool[0], lone])
        for distance, count in ((_d(None, None), 2), (_d(10, 20), 1)):
            for method in (mixed.select, mixed.cut):
                out = method(time=_t(0, 1), distance=distance)
                assert len({p.dims for p in out}) == len(out) == count
        # patches lacking an open first dim load; an untouched one keeps its size
        mixed = dc.spool([spool[0], spool[0].select(time=(None, T0)).squeeze("time")])
        t, d = _t(None, None), _d(1, 3)
        outs = mixed.select(time=t, distance=d), mixed.cut(distance=d, time=t)
        for out in (*outs, mixed.select(distance=d, time=t)):
            assert [p.shape for p in out] == [(3, 2000), (3,)]
        opened = mixed.select(distance=_d(None, None), time=t)
        sizes = [x.get_contents()["data_size"].tolist() for x in (opened, mixed)]
        assert sizes[0] == sizes[1]
        spanned = spool.cut(time=_t(0, 1), distance=_d(None, None))
        assert spanned[0].attrs.data_id == spool.cut(time=_t(0, 1))[0].attrs.data_id
        # NaT leaves either end open on a duration coordinate too
        lags = np.arange(5).astype("timedelta64[s]")
        lag = dc.spool(dc.Patch(data=lags, coords={"time": lags}, dims=("time",)))
        span = np.array([[1, "NaT"], ["NaT", 2]], dtype="timedelta64[s]")
        got = [x[0] for x in _boxes(lag.cut(time=span), ["time"])]
        assert got == [(lags[1], lags[4]), (lags[0], lags[2])]

    def test_trim_metadata(self):
        """Trim metadata: recovered coords, rider identities, missing coords."""
        patch = dc.get_example_patch("random_patch_with_lat_lon")
        time = np.array([patch.get_coord("time").values[[10, 100]]])
        out = dc.spool([patch]).cut(time=time, distance=_d(10, 20))
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
        # without its live patch, a source's samples are unknown
        for distance in (inner, _d(None, None)):
            with pytest.raises(MissingPatchError, match="metadata is unavailable"):
                make(uneven).cut(time=time, distance=distance)
        known = make(uneven, clear=False)
        assert _boxes(known.cut(time=time, distance=inner))[0][1] == (1, 1)
        past = dict(time=time, distance=_d(3.0, 20.0))
        partial = known.cut(**past)[0]
        assert partial.get_coord("distance").values.tolist() == [3.0, 6.0, 9.0]

    def test_three_dims(self):
        """Three explicit dims select and merge one box from a 3-D spool."""
        data, dims = np.arange(4 * 5 * 12).reshape(4, 5, 12), ("a", "b", "c")
        coords = {"a": np.arange(4), "b": np.arange(5), "c": np.arange(12)}
        whole = dc.Patch(data=data, coords=coords, dims=dims)
        spool = dc.spool([whole.select(c=(0, 5)), whole.select(c=(6, 11))])
        windows = dict(c=_d(1, 7), a=_d(1, 2), b=_d(0, 3))
        chunked = spool.cut(**windows)
        assert _boxes(chunked, dims) == [((1, 2), (0, 3), (1, 7))]
        expected = whole.select(a=(1, 2), b=(0, 3), c=(1, 7))
        np.testing.assert_array_equal(chunked[0].data, expected.data)
        pieces = [((1, 2), (0, 3), (1, 5)), ((1, 2), (0, 3), (6, 7))]
        assert _boxes(spool.select(**windows), dims) == pieces


def _assert_selects(spool, whole, window):
    """One output: the merged patch selected to the window, labels and data."""
    expected = whole.select(x=tuple(window))
    assert len(spool) == 1
    _assert_labels(spool[0].get_coord("x").values, expected.get_coord("x").values)
    assert np.array_equal(spool[0].data, expected.data)


# name: (labels, piece sizes); index steps truncate 1/1024 and 1/3000 s
_CHAINS = {
    "1024": (_time_values(1024, 768), [256] * 3),
    "3000": (_time_values(3000, 2250), [750] * 3),
    "exact": (_exact_values(768), [250, 270, 248]),
    "exact_far": (_exact_values(768, offset=3_686_401), [251, 269, 248]),
    "float_desc": (10 - np.arange(70) / 3, [20, 27, 23]),
    "float_tenth": (5 + np.arange(120) * 0.1, [60, 60]),
    "int_desc": (10 - 3 * np.arange(70), [23, 20, 27]),
}


# the whole-tick step of an exact 1/1024 s grid
_STEP = np.timedelta64(976562, "ns")


def _assert_chain_loads(path, pieces, first, count=4, method="chunk"):
    """A chunk (or cut), then a regular chunk, equal the same on loaded patches."""
    for num, piece in enumerate(pieces):
        piece.io.write(path / f"{num}.h5", "dasdae")
    second = dict(x=50 * _STEP + _STEP // 2, keep_partial=True)
    made = getattr(dc.spool(pieces), method)(**first)
    eager = dc.spool(list(made)).chunk(**second)
    expected = [x.get_coord("x").values for x in eager]
    for spool in (dc.spool(pieces), dc.spool(path).update(progress=None)):
        got = getattr(spool, method)(**first).chunk(**second)
        assert len(got) == len(expected) == count
        for patch, labels in zip(got, expected, strict=True):
            assert np.array_equal(patch.get_coord("x").values, labels)


class TestChainedExplicitCut:
    """Chunks chained with cut windows keep every sample they held."""

    @pytest.fixture(scope="class")
    def sources(self, tmp_path_factory):
        """Each chain's merged patch, memory spool and directory spool."""
        return {
            name: _cut_spools(tmp_path_factory.mktemp(name), *args)
            for name, args in _CHAINS.items()
        }

    @pytest.mark.parametrize(
        "case, length, window",
        [
            ("1024", 0.1, _T0 + np.array([30, 690], dtype="m8[ms]")),
            ("3000", 0.1, _T0 + np.array([30, 690], dtype="m8[ms]")),
            # float labels a regular chunk computes meet the file's own labels
            ("float_tenth", 0.73, np.array([7.23, 15.33])),
        ],
    )
    def test_window(self, sources, case, length, window):
        """Disk and memory both equal the merged patch selected to the window."""
        whole, mem, disk = sources[case]
        for spool in (mem, disk):
            first = spool.chunk(x=length, keep_partial=True)
            _assert_selects(first.cut(x=window[None]), whole, window)

    @pytest.mark.parametrize("sign, shift", [(1, 0), (-1, 0), (1, 300)])
    def test_windows_then_regular_chunk(self, tmp_path, sign, shift):
        """
        Windows then a regular chunk on a 1/1024 s grid drop no sample.

        A middle piece shifted off the others' lattice (by ``shift`` ns)
        joins no exact grid, and still matches the loaded patches.
        """
        values = _exact_values(300, sign=sign)
        pieces = list(_cut_spools(None, values, [100, 100, 100])[1])
        coord = pieces[1].get_coord("x")
        moved = coord.update(min=coord.min() + np.timedelta64(shift, "ns"))
        pieces[1] = pieces[1].update_coords(x=moved)
        window = values.min() + np.array([2 * _STEP + _STEP // 3, 250 * _STEP])
        _assert_chain_loads(tmp_path, pieces, dict(x=window[None]), 5, "cut")

    def test_filled_gaps_then_regular_chunk(self, tmp_path):
        """Filling a hole then a regular chunk drop no sample."""
        whole = _cut_spools(None, _exact_values(300), [300])[0]
        pieces = [whole.select(x=x, samples=True) for x in [(0, 140), (160, None)]]
        filled = dict(x=None, fill_value=0)
        _assert_chain_loads(tmp_path, pieces, filled, count=6, method="fill_gaps")

    def test_window_past_the_members_is_padded(self):
        """A window running past the members it reaches is padded on the grid."""
        whole = _cut_spools(None, _exact_values(30), [30])[0]
        pieces = [whole.select(x=x, samples=True) for x in [(0, 10), (20, 30)]]
        filled = dc.spool(pieces).fill_gaps("x", fill_value=-1)
        eager = dc.spool(list(filled))
        second = dict(x=5 * _STEP, keep_partial=True)
        got, expected = filled.chunk(**second), eager.chunk(**second)
        assert [x.data[0].tolist() for x in got] == [
            x.data[0].tolist() for x in expected
        ]

    def test_trimmed_source_grid_is_not_reused(self):
        """A selected row keeps its file's grid, which no longer fits its bounds."""
        values = _exact_values(300)
        mem = _cut_spools(None, values, [300])[1]
        selected = mem.select(x=(values.values[1], values.values[-2]))
        out = selected.cut(x=values.values[[2, 200]][None, :])
        row = out._df.iloc[0]
        assert len(coord_from_row(row, "x")) == len(out[0].get_coord("x"))

    def test_subdivided_grid_is_not_trusted(self):
        """Subdivided rows copy the source grid; each output states its own."""
        coord = _exact_values(300)
        patch = dc.Patch(
            np.zeros((2, 300)), coords={"d": [0, 1], "x": coord}, dims=("d", "x")
        )
        spool = dc.spool([patch, patch.update_coords(d=[2, 3])])
        sources, rows = spool._plan_frames()
        cuts = [[(coord.values[0], coord.values[n])] for n in (99, 199)]
        out = spool._subdivided(sources, rows, cuts, "x")
        assert [x.shape for x in out.chunk(d=None)] == [(2, 100), (2, 200)]

    @pytest.mark.parametrize("seed", range(len(_CHAINS)))
    def test_random_plans(self, sources, seed):
        """Random lengths and windows: disk equals memory equals truth."""
        rng = np.random.default_rng(seed)
        whole, mem, disk = sources[list(_CHAINS)[seed]]
        values = whole.get_coord("x").values
        length = abs(values[-1] - values[0]) * float(rng.uniform(0.07, 0.31))
        length = int(np.ceil(length)) if values.dtype.kind == "i" else length
        half = len(values) // 2
        window = np.sort([rng.choice(values[:half]), rng.choice(values[half:])])
        for spool in (mem, disk):
            first = spool.chunk(x=length, keep_partial=True)
            _assert_selects(first.cut(x=window[None]), whole, window)


def _gapped_2d(*starts, depth=False):
    """Patches of five distances from each start, with riders on time (and depth)."""
    time = np.arange(3).astype("datetime64[s]")

    def make(start):
        values = np.arange(start, start + 5, dtype=np.int64)
        coords = {"distance": values, "time": time, "reference_time": ("time", time)}
        coords |= {"depth": ("distance", 2 * values)} if depth else {}
        data = np.ones((5, 3))
        return dc.Patch(data=data, coords=coords, dims=("distance", "time"))

    return dc.spool([make(x) for x in starts]), time


class TestWindowsOverFilledGaps:
    """Windows over a spool which filled its gaps keep the filled grid."""

    @pytest.mark.parametrize("method", ["select", "cut"])
    def test_window_uses_the_filled_grid(self, method):
        """A window's samples include the fill, labelled on the output's grid."""
        source, _ = _gapped_2d(0, 7)
        filled = source.fill_gaps(distance=3, fill_value=-1)
        (patch,) = getattr(filled, method)(distance=np.array([[3, 8]]))
        assert patch.coords["distance"].values.tolist() == [3, 4, 5, 6, 7, 8]
        assert patch.data[:, 0].tolist() == [1, 1, -1, -1, 1, 1]

    def test_rider_selection_then_window(self):
        """A selection on a rider, then a window, still finds the fill."""
        source, time = _gapped_2d(0, 7)
        filled = source.fill_gaps(distance=3, fill_value=-1)
        narrowed = filled.select(reference_time=(time[0], time[1]))
        (patch,) = narrowed.cut(distance=np.array([[3, 8]]))
        assert patch.shape == (6, 2)
        assert patch.coords["distance"].values.tolist() == [3, 4, 5, 6, 7, 8]
        assert patch.data[:, 0].tolist() == [1, 1, -1, -1, 1, 1]

    def test_nothing_to_fill_keeps_riders(self):
        """Contiguous patches fill nothing, so a rider window keeps its riders."""
        source, time = _gapped_2d(0, 5)
        narrowed = source.select(reference_time=(time[0], time[1]))
        filled = narrowed.fill_gaps(distance=3, fill_value=-1)
        (patch,) = filled.cut(distance=np.array([[3, 6]]))
        assert (patch.data == 1).all() and "reference_time" in patch.coords.coord_map

    @pytest.mark.parametrize("unit", ["s", "ns"])
    @pytest.mark.parametrize("length", [None, 5])
    def test_fill_rows_state_the_riders_that_load(self, unit, length):
        """Rows of a fill state a rider where, and as, the loaded patch has it."""
        times = lambda a, b: np.arange(a, b).astype(f"datetime64[{unit}]")  # noqa

        def make(start, stop, tag, gain=False):
            coords = {"time": times(start, stop)}
            if gain:
                values = np.arange(start, stop, dtype=np.float64)
                coords["gain"] = ("time", values)
            data = np.ones(stop - start)
            return dc.Patch(
                data=data, coords=coords, dims=("time",), attrs={"tag": tag}
            )

        parts = [make(0, 5, "a", True), make(5, 10, "a", True), make(14, 16, "a", True)]
        parts += [make(100, 105, "b"), make(115, 120, "b")]
        filled = dc.spool(parts).fill_gaps(time=np.timedelta64(20, unit))
        if length is not None:
            filled = filled.chunk(time=np.timedelta64(length, unit), keep_partial=True)
        rows = filled.get_contents()
        loaded = list(filled)
        assert rows["gain_min"].notna().tolist() == [
            "gain" in x.coords.coord_map for x in loaded
        ]
        for (_, row), patch in zip(rows.iterrows(), loaded, strict=True):
            if "gain" in patch.coords.coord_map:
                gain = patch.get_array("gain")
                # a rider's range is stated to float32 precision
                assert np.isclose(np.nanmin(gain), row["gain_min"], rtol=1e-6)
                assert np.isclose(np.nanmax(gain), row["gain_max"], rtol=1e-6)
        tags = [x.attrs.tag for x in filled.select(gain=(0, 100))]
        assert set(tags) == {"a"} and len(tags) == (1 if length is None else 4)
        gain = np.concatenate([x.get_array("gain") for x in loaded[: len(tags)]])
        expected = np.r_[np.arange(10.0), np.nan * np.ones(4), 14.0, 15.0]
        np.testing.assert_array_equal(gain, expected)

    @pytest.mark.parametrize("length", [None, 2])
    def test_fill_rows_leave_out_riders_that_cannot_fill(self, length):
        """A rider no fill value fits is gone exactly where the fill writes."""
        times = lambda a, b: np.arange(a, b).astype("datetime64[s]")  # noqa

        def make(start, stop):
            gain = ("time", np.arange(start, stop))
            coords = {"time": times(start, stop), "gain": gain}
            return dc.Patch(data=np.ones(stop - start), coords=coords, dims=("time",))

        spool = dc.spool([make(0, 4), make(10, 14)]).fill_gaps("time")
        if length is not None:
            spool = spool.chunk(time=length)
        with pytest.warns(DASCoreWarning, match="dropped associated"):
            held = ["gain" in x.coords.coord_map for x in spool]
        rows = spool.get_contents()
        stated = rows.get("gain_min", pd.Series(np.nan, index=rows.index))
        assert stated.notna().tolist() == held
        expected = [False] if length is None else [True] * 2 + [False] * 3 + [True] * 2
        assert held == expected

    def test_window_inside_the_gap(self):
        """A window wholly inside a filled gap is all fill, reading one sample."""
        source, _ = _gapped_2d(0, 10, depth=True)
        filled = source.fill_gaps(distance=6, fill_value=-1)
        (fill,) = filled.cut(distance=np.array([[5.5, 7]]))
        assert fill.coords["distance"].values.tolist() == [6, 7]
        assert (fill.data == -1).all()
        # an associated coordinate along the gap cannot be filled
        assert "depth" not in fill.coords.coord_map

    def test_wider_gap_stays(self):
        """A gap wider than the size is a boundary, filled with nothing."""
        source, _ = _gapped_2d(0, 10)
        assert len(source.fill_gaps(distance=4, fill_value=-1)) == 2


@pytest.fixture(scope="module", params=["memory", "directory"])
def forms_spool(request, tmp_path_factory):
    """The example spool, in memory or as a directory of files."""
    spool = dc.get_example_spool()
    if request.param == "memory":
        return spool
    path = tmp_path_factory.mktemp("forms")
    dc.examples.spool_to_directory(spool, path=path)
    return dc.spool(path).update(progress=None)


class TestWindowForms:
    """The forms a window's bounds take, alike in select and cut."""

    @staticmethod
    def _spans(spool, dim):
        """Each output's ends along dim, from contents and from the data."""
        rows = spool.get_contents()[[f"{dim}_min", f"{dim}_max"]]
        loaded = [(x.get_coord(dim).min(), x.get_coord(dim).max()) for x in spool]
        assert list(rows.itertuples(index=False, name=None)) == loaded
        return loaded

    @pytest.mark.parametrize("method", ["select", "cut"])
    def test_reversed_bounds_swap(self, forms_spool, method):
        """A row's reversed bounds are swapped, per row and dimension (rule 8)."""
        start = forms_spool.get_contents()["time_min"].min()
        times = np.array([start + np.timedelta64(x, "s") for x in (1, 9, 3, 12)])
        call = getattr(forms_spool, method)
        forward = call(time=times.reshape(2, 2), distance=np.array([[10, 20], [5, 9]]))
        swapped = call(
            time=times.reshape(2, 2)[:, ::-1], distance=np.array([[20, 10], [5, 9]])
        )
        for dim in ("time", "distance"):
            assert self._spans(swapped, dim) == self._spans(forward, dim)

    @pytest.mark.parametrize("method", ["select", "cut"])
    def test_infinite_end_is_open(self, forms_spool, method):
        """An infinity opens its own side, whichever column holds it."""
        call = getattr(forms_spool, method)
        expected = self._spans(
            call(distance=np.array([[290, None], [None, 3]])), "distance"
        )
        got = call(distance=np.array([[np.inf, 290], [3, -np.inf]]))
        assert self._spans(got, "distance") == expected

    @pytest.mark.parametrize("method", ["select", "cut"])
    def test_quantity_array(self, forms_spool, method):
        """A quantity holding an (n, 2) array is windows, as quantities are."""
        call = getattr(forms_spool, method)
        each = np.array([[10 * m, 20 * m], [5 * m, 9 * m]], dtype=object)
        whole = np.array([[10, 20], [5, 9]]) * m
        expected = self._spans(call(distance=each), "distance")
        assert self._spans(call(distance=whole), "distance") == expected
        assert expected[0] == (10, 20)
        # a quantity array's units are its own, here feet on a metre axis
        feet = self._spans(call(distance=np.array([[10, 20]]) * ft), "distance")
        assert set(feet) == {(4, 6)}

    @pytest.mark.parametrize("method", ["select", "cut"])
    def test_timedelta_on_datetime_refused(self, forms_spool, method):
        """An offset is not an instant: say to add it to a start time."""
        offsets = np.array([[1, 3]]).astype("timedelta64[s]")
        for bound in (offsets, pd.Timedelta("1s"), timedelta(seconds=1)):
            bounds = (
                np.array([[bound, bound]], dtype=object)
                if offsets is not bound
                else bound
            )
            with pytest.raises(ParameterError, match=r"absolute instants.*start"):
                len(getattr(forms_spool, method)(time=bounds))

    @pytest.mark.parametrize("method", ["select", "cut"])
    def test_numbers_on_datetime_are_epoch_seconds(self, forms_spool, method):
        """A number bounding a datetime is seconds since 1970, as in Patch.select."""
        start = forms_spool.get_contents()["time_min"].min()
        seconds = (start - np.datetime64(0, "s")) / np.timedelta64(1, "s")
        call = getattr(forms_spool, method)
        expected = call(time=np.array([[start, start + np.timedelta64(2, "s")]]))
        got = call(time=np.array([[seconds, seconds + 2.0]]))
        (got_ends,) = self._spans(got, "time")
        assert self._spans(expected, "time") == [got_ends] and got_ends[0] == start
        patch = forms_spool[0].select(time=(seconds, seconds + 2.0))
        assert (
            patch.get_coord("time").min(),
            patch.get_coord("time").max(),
        ) == got_ends


class TestWindowEdgeCases:
    """Windows the review found mishandled."""

    @pytest.mark.parametrize("bounds", [(np.inf, np.inf), (-np.inf, -np.inf)])
    def test_same_sign_infinities_refused(self, bounds):
        """A row bounded on one side by both infinities holds nothing sensible."""
        spool = dc.spool(_patch(np.arange(10)))
        with pytest.raises(ParameterError, match="infinit"):
            spool.cut(distance=np.array([bounds]))

    def test_quantity_windows_with_patches_lacking_the_dim(self):
        """Patches lacking the dim are skipped, not read in no unit."""
        patch = dc.get_example_patch().set_units(distance="m")
        lone = patch.select(distance=(0, 0)).squeeze("distance")
        spool = dc.spool([patch, lone])
        assert (
            len(spool.select(distance=np.array([[10 * m, 20 * m]], dtype=object))) == 1
        )
        open_row = np.array([[None, None], [10 * m, 20 * m]], dtype=object)
        assert len(spool.select(distance=open_row)) == 3

    @pytest.mark.parametrize("method", ["select", "cut"])
    def test_empty_recovered_coordinate(self, method):
        """A source whose selected coordinate holds no labels gives nothing."""
        patch = dc.Patch(
            data=np.arange(4.0),
            coords={"x": np.arange(4.0), "depth": ("x", np.arange(4.0) * 10)},
            dims=("x",),
        )
        narrowed = dc.spool(patch).select(depth=(1, 2))
        assert not len(getattr(narrowed, method)(x=np.array([[0, 3]])))


class TestReviewThreads:
    """Windows the PR's review threads found mishandled."""

    def test_select_keeps_spool_order(self):
        """Select returns each window's pieces in the spool's order (as on dev)."""
        later = _patch(np.arange(10.0, 20.0))
        spool = dc.spool([later, _patch(np.arange(10.0))])
        order = spool.get_contents()["distance_min"].tolist()
        pieces = spool.select(distance=np.array([[0, 19]]))
        assert pieces.get_contents()["distance_min"].tolist() == order
        assert [_ends(x)[0] for x in spool.cut(distance=np.array([[0, 19]]))] == [0]

    @pytest.mark.parametrize("method", ["select", "cut"])
    def test_grid_windows_never_expand_labels(self, method, monkeypatch):
        """A grid's window samples come from index arithmetic, not its labels."""
        coord = get_coord(start=0.0, step=0.5, shape=(20_000,))
        patch = dc.Patch(
            data=np.arange(20_000.0), coords={"distance": coord}, dims=("distance",)
        )
        spool = dc.spool([patch, patch.update_coords(distance_min=9000.0)])
        windows = np.array([[10.2, 20.0], [8995.0, 9005.0], [30000.0, 40000.0]])
        expected = getattr(spool, method)(distance=windows).get_contents()
        values = type(coord).values

        def forbidden(self):
            if self.evenly_sampled and len(self) > 10_000:
                raise AssertionError("a grid's labels were expanded")
            return values.fget(self)

        monkeypatch.setattr(type(coord), "values", property(forbidden))
        got = getattr(spool, method)(distance=windows).get_contents()
        assert _listed(got) == _listed(expected)
        assert _listed(got)[0] == [10.5, 20.0]

    def test_fill_only_output_takes_a_sibling_coordinate(self):
        """A window on another dim over an output holding only fill finds it."""
        time = np.arange(5).astype("datetime64[s]")

        def make(start):
            coords = {"distance": np.arange(3, dtype=np.int64), "time": time + start}
            return dc.Patch(
                data=np.ones((3, 5)), coords=coords, dims=("distance", "time")
            )

        filled = dc.spool([make(0), make(15)]).fill_gaps(time=20, fill_value=0)
        filled = filled.chunk(time=5)
        assert len(filled) == 4
        pieces = filled.select(distance=np.array([[0, 1]]))
        assert [x.shape for x in pieces] == [(2, 5)] * 4
        assert np.all(pieces[1].data == 0) and np.all(pieces[0].data == 1)
        (joined,) = filled.cut(distance=np.array([[0, 1]]))
        assert joined.shape == (2, 20) and np.all(joined.data[:, 5:15] == 0)

    def test_mixed_unit_integers_keep_float_envelopes(self):
        """Integer labels normalised to fractions are stated as they load."""

        def make(values, units):
            coord = get_coord(data=np.asarray(values, dtype=np.int64), units=units)
            return dc.Patch(data=np.ones(len(values)), coords={"x": coord}, dims=("x",))

        spool = dc.spool(
            [make(range(5), "m"), make(range(5, 10), "m"), make(range(40, 45), "ft")]
        )
        joined = spool.chunk(x=None)
        rows = joined.get_contents()[["x_min", "x_max"]].to_numpy().tolist()
        loaded = [[x.get_coord("x").min(), x.get_coord("x").max()] for x in joined]
        np.testing.assert_allclose(rows, loaded)
