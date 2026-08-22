"""
Tests for derived catalogs (plan-as-catalog) and coverage of their edges.
"""

from __future__ import annotations

import pickle
import re

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.core.coords import get_coord
from dascore.exceptions import CoordMergeError, MissingPatchError, ParameterError
from dascore.io.index.catalog import PatchCatalog
from dascore.io.index.planned import (
    PlanResolver,
    _apply_predictions,
    _aux_coord_info,
    _coord_record_from_row,
    _cut_rider,
    _describe,
    _extrema,
    _ns,
    _null_like,
    _stated_units,
    collapse_working_df,
    derived_catalog,
    predicted_coords,
)
from dascore.units import m
from dascore.utils.chunk_plan import (
    ChunkPlan,
    build_concat_plan,
    samples_adjusted_envelopes,
)
from dascore.utils.patch import concatenate_patches


@pytest.fixture(scope="module")
def patches():
    """Three contiguous example patches."""
    return list(dc.get_example_spool("random_das"))


class TestHelpers:
    """Unit coverage for the conversion helpers."""

    def test_ns_forms(self):
        """All datetime/timedelta forms convert to the same ns."""
        ts = pd.Timestamp("2020-01-01")
        assert _ns(ts) == _ns(ts.to_datetime64()) == ts.value
        td = pd.Timedelta(seconds=1)
        td_ns = _ns(td)
        assert td_ns is not None
        assert td_ns == _ns(td.to_timedelta64()) == td.value
        assert _ns(None) is None

    def test_coord_record_numpy_datetimes(self):
        """np.datetime64 envelope values build the same record."""
        lo = np.datetime64("2020-01-01", "ns")
        hi = np.datetime64("2020-01-02", "ns")
        row = {"time_min": lo, "time_max": hi, "time_step": np.timedelta64(1, "s")}
        record = _coord_record_from_row(row, "time")
        assert record is not None
        assert record.value_kind == "time"
        assert record.min_ns == _ns(lo)

    def test_coord_record_without_values(self):
        """A null envelope is a coordinate only with a fingerprinted identity."""
        row = {"rank_min": None, "rank_max": None}
        assert _coord_record_from_row(row, "rank") is None
        row["_rank_def_key"] = "sum:abc"
        assert _coord_record_from_row(row, "rank") is None
        row["_rank_def_key"] = "fp:" + "a" * 32
        record = _coord_record_from_row(row, "rank")
        assert record is not None
        assert record.coord_hash == "a" * 32
        assert record.min_num is None and record.length is None

    def test_coord_record_half_null_timedelta(self):
        """A one-sided timedelta envelope keeps NaT rather than raising."""
        row = {"time_min": pd.Timedelta(seconds=1), "time_max": pd.NaT}
        record = _coord_record_from_row(row, "time")
        assert record is not None
        assert record.min_ns == pd.Timedelta(seconds=1).value
        assert pd.isnull(np.timedelta64(record.max_ns, "ns"))

    def test_coord_record_zero_step_length(self):
        """A degenerate step leaves length unknown instead of raising."""
        row = {"time_min": 0.0, "time_max": 1.0, "time_step": 0.0}
        record = _coord_record_from_row(row, "time")
        assert record is not None
        assert record.length is None

    def test_coord_record_empty_units_dropped(self):
        """An empty-string units cell normalizes to None (a real step keeps length)."""
        row = {
            "distance_min": 0.0,
            "distance_max": 10.0,
            "distance_step": 1.0,
            "_distance_units": "",
        }
        record = _coord_record_from_row(row, "distance")
        assert record is not None
        assert record.units is None
        assert record.length == 11

    def test_plan_resolver_requires_output_id(self):
        """member_rows without output_id is a construction error."""
        with pytest.raises(ValueError, match="output_id"):
            PlanResolver(
                token="x",
                dim="time",
                member_rows=pd.DataFrame({"source_path": []}),
                loader=None,
                merge_kwargs={},
            )

    def test_derived_catalog_adds_patch_ids(self, patches):
        """source_rows without _patch_id get positional ids."""
        spool = dc.spool(patches)
        rows = spool._df.drop(columns=["_patch_id"]).reset_index(drop=True)
        members = pd.DataFrame(
            {"output_id": [0], "_patch_id": [0], "_modified": [False]}
        )
        outputs = rows.iloc[:1].assign(output_id=0)
        plan = ChunkPlan(outputs, members, "time", None, {})
        catalog = derived_catalog(
            source_rows=rows,
            plan=plan,
            parent=spool._catalog,
            merge_kwargs={},
            mode="concat",
        )
        assert len(catalog) == 1


class TestDerivedComposition:
    """Operation-order coverage over derived catalogs."""

    def test_collapse_with_value_residual(self, patches):
        """Chunk of a selected chunked spool re-plans from trimmed members."""
        t0 = patches[0].get_coord("time").min()
        t1 = patches[1].get_coord("time").max()
        chunked = dc.spool(patches).chunk(time=2)
        selected = chunked.select(time=(t0, t1))
        merged = selected.chunk(time=None)
        assert len(merged) >= 1
        out = merged[0]
        assert out.get_coord("time").min() >= t0

    def test_sort_by_attr_on_windowed_regex_view(self, patches):
        """Regex selection + window + attr sort compose through SQL."""
        tagged = [
            p.update_attrs(tag=f"t{num}", history=[]) for num, p in enumerate(patches)
        ]
        spool = dc.spool(tagged)
        view = spool.select(tag=re.compile("t[0-9]"))[1:]
        out = view.sort("tag")
        tags = [p.attrs["tag"] for p in out]
        assert tags == sorted(tags)

    def test_attr_membership_array(self, patches):
        """Attr membership with a numpy array of values selects rows."""
        tagged = [
            p.update_attrs(tag=f"t{num}", history=[]) for num, p in enumerate(patches)
        ]
        spool = dc.spool(tagged)
        out = spool.select(tag=np.array(["t0", "t2"]))
        assert len(out) == 2

    def test_sort_by_envelope_column_name(self, patches):
        """Sort accepts the explicit `{dim}_min` column form."""
        spool = dc.spool(list(reversed(patches)))
        out = spool.sort("time_min")
        assert out.get_contents()["time_min"].is_monotonic_increasing

    def test_concatenate_requires_one_kwarg(self, patches):
        """Concatenate validates its dimension keyword."""
        with pytest.raises(ParameterError, match="exactly one dimension"):
            dc.spool(patches).concatenate(time=None, distance=None)

    def test_union_view_of_live_spools_pickles_composite(self, patches):
        """A selected union pickles a membership-restricted composite."""
        t0 = patches[0].get_coord("time")
        combined = dc.spool(patches[:2]) + dc.spool(patches[2:])
        view = combined.select(time=(None, t0.max()))
        assert len(view) == 1
        loaded = pickle.loads(pickle.dumps(view))
        assert len(loaded) == 1
        assert isinstance(loaded[0], dc.Patch)

    def test_missing_live_patch_getitem(self, patches):
        """A missing registry entry surfaces as MissingPatchError, not
        out-of-bounds.
        """
        spool = dc.spool(patches[:1])
        _ = spool.get_contents()  # realize rows
        spool._catalog.resolver._registry.clear()
        with pytest.raises(MissingPatchError, match="not available"):
            spool[0]

    def test_samples_negative_index_skips_envelope_adjust(self, patches):
        """Negative samples windows stay candidacy supersets (no crash)."""
        spool = dc.spool(patches[:1]).select(time=(0, -10), samples=True)
        merged = spool.chunk(time=None)
        assert len(merged) == 1

    def test_complete_overlap_merge(self):
        """Two identical-envelope patches merge by keeping the first."""
        patch = dc.get_example_patch()
        twin = patch.new()
        merged = dc.spool([patch, twin]).chunk(time=None)
        assert len(merged) == 1
        assert isinstance(merged[0], dc.Patch)


class TestRemainingEdges:
    """Direct coverage of defensive/rare branches."""

    def test_collapse_with_quantity_residual(self, patches):
        """A quantity-selected chunked view re-chunks without applying
        unit-bearing bounds to envelopes (they stay load residuals).
        """
        chunked = dc.spool(patches).chunk(time=2)
        selected = chunked.select(_coords={"distance": (0 * m, 10 * m)})
        merged = selected.chunk(time=None)
        assert len(merged) == 1
        coord = merged[0].get_coord("distance")
        assert float(coord.max()) <= 10

    def test_samples_adjust_skips_missing_columns(self):
        """Residuals naming absent envelope columns pass through."""
        df = pd.DataFrame({"time_min": [0.0], "time_max": [1.0]})
        residuals = (({"depth": (0, 5)}, True),)
        out = samples_adjusted_envelopes(df, residuals)
        assert out.equals(df)


class TestAuxiliaryCoords:
    """Derived catalogs keep non-dimension coords (2026-07-18 F2)."""

    @pytest.fixture()
    def sensor_spool(self):
        """Two contiguous patches carrying an aux coord on distance."""
        p = dc.get_example_patch()
        sensor = np.arange(p.shape[p.get_axis("distance")], dtype=float)
        p = p.update_coords(sensor=("distance", sensor))
        t = p.get_coord("time")
        p2 = p.update_coords(time_min=t.max() + t.step)
        return dc.spool([p, p2])

    @pytest.mark.parametrize("op", ["chunk", "concatenate"])
    def test_aux_coord_survives(self, sensor_spool, op):
        """Chunk and concat outputs keep describing the aux coord."""
        if op == "chunk":
            derived = sensor_spool.chunk(time=None, conflict="drop")
        else:
            derived = sensor_spool.concatenate(time=None)
        contents = derived.get_contents()
        assert "sensor_min" in contents.columns
        assert "sensor_max" in contents.columns
        # and it stays selectable
        out = derived.select(sensor=(10, 20))
        assert len(out) == 1
        coord = out[0].get_coord("sensor")
        assert coord.min() == 10.0
        assert coord.max() == 20.0

    def test_aux_identity_preserved_when_unchanged(self, sensor_spool):
        """Members sharing one def key off the planned dim keep identity."""
        derived = sensor_spool.chunk(time=None, conflict="drop")
        source_key = sensor_spool._catalog.to_df()["_sensor_def_key"].iloc[0]
        derived_key = derived._catalog.to_df()["_sensor_def_key"].iloc[0]
        assert derived_key == source_key
        assert str(derived_key).startswith("fp:")

    def test_aux_identity_dropped_when_riding_trimmed_dim(self, sensor_spool):
        """A residual trim on distance voids sensor's identity claim."""
        d = sensor_spool[0].get_coord("distance")
        lo, hi = d.min() + 5 * d.step, d.min() + 50 * d.step
        selected = sensor_spool.select(distance=(lo, hi))
        derived = selected.chunk(time=None, conflict="drop")
        key = derived._catalog.to_df()["_sensor_def_key"].iloc[0]
        assert not str(key).startswith("fp:")
        # loading still yields the trimmed coord
        assert derived[0].get_coord("sensor").min() == 5.0

    def test_string_aux_coord(self):
        """String-valued aux coords survive with a lexicographic envelope."""
        p = dc.get_example_patch()
        n = p.shape[p.get_axis("distance")]
        labels = np.array([f"s{i:03d}" for i in range(n)])
        p = p.update_coords(station=("distance", labels))
        derived = dc.spool([p]).chunk(time=None)
        contents = derived.get_contents()
        assert contents["station_min"].iloc[0] == "s000"
        assert contents["station_max"].iloc[0] == f"s{n - 1:03d}"
        assert "station" in derived[0].coords.coord_map


class TestAuxInfoEdges:
    """Edge branches of the aux-coord aggregation helpers."""

    def test_coord_record_missing_envelope_returns_none(self):
        """A row without envelope values yields no coord record."""
        assert _coord_record_from_row({}, "time") is None

    def test_absent_envelope_columns_skipped(self):
        """A mapped coord with no envelope columns contributes nothing."""
        members = pd.DataFrame(
            {"output_id": [0], "_patch_id": [1], "_modified": [False]}
        )
        sources = pd.DataFrame({"_patch_id": [1]})
        assert _aux_coord_info(sources, members, "time", {"ghost": "distance"}) == {}

    def test_all_null_group_skipped(self):
        """An output whose members carry no values for a coord is skipped."""
        members = pd.DataFrame(
            {"output_id": [0], "_patch_id": [1], "_modified": [False]}
        )
        sources = pd.DataFrame(
            {"_patch_id": [1], "sensor_min": [np.nan], "sensor_max": [np.nan]}
        )
        assert _aux_coord_info(sources, members, "time", {"sensor": "distance"}) == {}


class TestCollapseGuard:
    """collapse_working_df only applies to plan-backed catalogs."""

    def test_non_plan_catalog_returns_none(self):
        """A live catalog has no plan to collapse."""
        catalog = dc.spool([dc.get_example_patch()])._catalog
        assert collapse_working_df(catalog) is None

    def test_collapse_keeps_modified(self):
        """
        The flag says a member is a trim rather than a whole source.

        Losing it makes the next plan mark such a member "load whole",
        and the loader then reads the entire file it was a slice of.
        """
        chunked = dc.spool(dc.get_example_patch()).chunk(time=2)
        collapsed = collapse_working_df(chunked._catalog)
        assert "_modified" in collapsed.columns
        assert collapsed["_modified"].all()


class TestRePlanKeepsTheTrim:
    """Re-planning a dimension must not load back what was trimmed off."""

    def test_nested_chunk_matches_a_direct_one(self):
        """
        Chunking twice gives what chunking once does, data included.

        The example patch spans 8 s, so `chunk(time=3)` keeps two full
        chunks and drops the 2 s remainder; the nested form used to
        report 4500 samples from a 2000 sample patch instead.
        """
        patch = dc.get_example_patch()
        direct = dc.spool(patch).chunk(time=3)
        nested = dc.spool(patch).chunk(time=2).chunk(time=3)
        assert len(nested) == len(direct)
        assert sum(x.shape[1] for x in nested) == sum(x.shape[1] for x in direct)
        for one, two in zip(direct, nested, strict=True):
            assert np.array_equal(one.data, two.data)
            assert np.array_equal(one.get_array("time"), two.get_array("time"))

    def test_a_re_chunked_selection_keeps_its_range(self):
        """
        The reported case: the second chunk ignored the selection.

        A member reloaded whole overlaps its neighbour, so the merge
        repeated the overlap and the coordinate stopped increasing.
        """
        patch = dc.get_example_patch().set_units(distance="m")
        coord = patch.get_coord("distance")
        span = float(coord.max() - coord.min() + coord.step)
        moved = patch.update_coords(distance=coord.data + span).set_units(distance="m")
        chunked = dc.spool([patch, moved]).chunk(
            distance=200, conflict="keep_first", keep_partial=True
        )
        selected = chunked.select(distance=(250 * m, 450 * m))
        wanted = sum(x.shape[x.get_axis("distance")] for x in selected)
        merged = selected.chunk(distance=None, conflict="keep_first")
        values = np.asarray(merged[0].get_coord("distance").values)
        assert len(values) == wanted
        assert len(np.unique(values)) == len(values)

    def test_merging_chunks_of_one_patch_rebuilds_it(self):
        """Every piece is a trim, so none of them may load whole."""
        patch = dc.get_example_patch()
        merged = dc.spool(patch).chunk(distance=100).chunk(distance=None)
        assert len(merged) == 1
        assert merged[0].shape == patch.shape
        assert np.array_equal(merged[0].data, patch.data)


class TestStatedUnits:
    """Row values arrive from dataframes, so absence is NaN."""

    @pytest.mark.parametrize("value", [None, "", np.nan, pd.NaT])
    def test_absent_units_read_as_none(self, value):
        """NaN never equals itself, so it must not look like a mismatch."""
        assert _stated_units(value) is None

    def test_stated_units_pass_through(self):
        """A real spelling survives as a string."""
        assert _stated_units("ft") == "ft"


class TestPredictedCoords:
    """What a plan claims about an output, decided by the real join."""

    @pytest.fixture()
    def pair(self):
        """Two patches which meet end to end along time."""
        first = dc.get_example_patch()
        time = first.get_coord("time")
        second = dc.get_example_patch(time_min=time.max() + time.step)
        return first, second

    def _plan_and_backend(self, patches, **kwargs):
        """A concat plan over the patches, and the spool's backend."""
        spool = dc.spool(list(patches))
        frame = PatchCatalog.from_patches(list(patches)).to_df()
        plan = build_concat_plan(frame, **kwargs)
        return plan, spool._catalog.backend

    def test_joined_dimension_is_the_real_join(self, pair):
        """The concatenated dimension is described as assembly builds it."""
        plan, backend = self._plan_and_backend(pair, time=None)
        described = predicted_coords(backend, plan.members, "time")[0]
        time = described["time"]
        whole = concatenate_patches(list(pair), time=None)[0].get_coord("time")
        assert time.min == whole.min()
        assert time.max == whole.max()
        assert time.len == len(whole)
        assert time.step == whole.step
        assert time.fingerprint == whole.fingerprint()

    def test_other_coordinates_keep_their_identity(self, pair):
        """A coordinate every member shares is described as it stands."""
        plan, backend = self._plan_and_backend(pair, time=None)
        described = predicted_coords(backend, plan.members, "time")[0]
        distance = pair[0].get_coord("distance")
        assert described["distance"].fingerprint == distance.fingerprint()
        assert described["distance"].len == len(distance)

    def test_a_trimmed_dimension_claims_no_identity(self, pair):
        """Values a residual trims at load are not vouched for."""
        plan, backend = self._plan_and_backend(pair, time=None)
        described = predicted_coords(
            backend, plan.members, "time", trimmed_dims=frozenset({"distance"})
        )[0]
        assert described["distance"].fingerprint is None

    def test_members_which_cannot_be_joined_span_the_envelope(self):
        """Labels have no step, so only the envelope is claimed."""
        base = dc.get_example_patch()
        n = base.shape[base.get_axis("distance")]
        labels = np.array([f"s{i:03d}" for i in range(n)])
        renamed = base.rename_coords(distance="range")
        first = renamed.update_coords(range=labels)
        second = renamed.update_coords(range=np.array([f"t{i:03d}" for i in range(n)]))
        plan, backend = self._plan_and_backend([first, second], range=None)
        described = predicted_coords(backend, plan.members, "range")[0]
        assert described["range"].step is None
        assert described["range"].fingerprint is None
        assert described["range"].min == "s000"
        assert described["range"].max == f"t{n - 1:03d}"

    def test_a_trim_of_the_joined_dimension_is_already_counted(self, pair):
        """The planned dimension's own trim rides in the rows being joined."""
        plan, backend = self._plan_and_backend(pair, time=None)
        described = predicted_coords(
            backend, plan.members, "time", trimmed_dims=frozenset({"time"})
        )[0]
        # the members state what will be loaded, so the join still holds
        assert described["time"].fingerprint is not None
        assert described["time"].len is not None

    def test_a_rider_of_a_trimmed_dimension_is_voided(self, pair):
        """A coordinate joined along a dimension the load trims says less."""
        first, second = pair
        nt = first.shape[first.get_axis("time")]
        patches = [
            x.update_coords(clock=("time", np.arange(nt) * 1.0 + i * nt))
            for i, x in enumerate((first, second))
        ]
        plan, backend = self._plan_and_backend(patches, time=None)
        described = predicted_coords(
            backend, plan.members, "time", trimmed_dims=frozenset({"time"})
        )[0]
        assert described["clock"].fingerprint is None
        assert described["clock"].step is None
        # the envelope still bounds where the rider lies
        assert described["clock"].min == 0.0

    def test_a_trim_of_another_dimension_voids_what_rides_it(self, pair):
        """A coordinate cut at load claims neither step nor identity."""
        first, second = pair
        n = first.shape[first.get_axis("distance")]
        patches = [
            x.update_coords(latitude=("distance", np.arange(n) * 1.0))
            for x in (first, second)
        ]
        plan, backend = self._plan_and_backend(patches, time=None)
        described = predicted_coords(
            backend, plan.members, "time", trimmed_dims=frozenset({"distance"})
        )[0]
        assert described["latitude"].fingerprint is None
        assert described["latitude"].step is None

    def test_a_trimmed_member_is_described_by_its_trim(self, pair):
        """A member which loads part of its patch states that part."""
        first, second = pair
        plan, backend = self._plan_and_backend(pair, time=None)
        members = plan.members.copy()
        time = first.get_coord("time")
        cut = time.min() + (time.max() - time.min()) / 2
        members["_modified"] = [True, False]
        members.loc[members.index[0], "time_max"] = cut
        described = predicted_coords(backend, members, "time")[0]
        # the trimmed member's own range bounds the join, not its source
        assert described["time"].min == time.min()
        assert described["time"].max == second.get_coord("time").max()

    def test_a_trim_which_states_no_range_is_left_alone(self, pair):
        """A member marked modified but stating no range keeps its summary."""
        plan, backend = self._plan_and_backend(pair, time=None)
        members = plan.members.copy()
        members["_modified"] = True
        members["time_min"] = pd.NaT
        members["time_max"] = pd.NaT
        described = predicted_coords(backend, members, "time")[0]
        whole = concatenate_patches(list(pair), time=None)[0].get_coord("time")
        assert described["time"].max == whole.max()

    def test_a_coordinate_one_member_lacks(self, pair):
        """What a partly-held coordinate becomes depends on the assembly."""
        first, second = pair
        n = first.shape[first.get_axis("distance")]
        lat = second.update_coords(latitude=("distance", np.arange(n) * 1.0))
        plan, backend = self._plan_and_backend([first, lat], time=None)
        # a concatenation carries it over from the member which has it
        described = predicted_coords(backend, plan.members, "time", mode="concat")[0]
        assert "latitude" in described
        assert described["latitude"].len == n
        # a merge drops what its members do not all share, so nothing is said
        merged = predicted_coords(backend, plan.members, "time", mode="chunk")[0]
        # named, but stated as nothing: the merge will not carry it
        assert merged["latitude"] is None

    def test_a_merge_drops_what_its_members_disagree_about(self, pair):
        """Under drop, a coordinate the members differ on is not described."""
        first, second = pair
        n = first.shape[first.get_axis("distance")]
        patches = [
            first.update_coords(latitude=("distance", np.arange(n) * 1.0)),
            second.update_coords(latitude=("distance", np.ones(n))),
        ]
        plan, backend = self._plan_and_backend(patches, time=None)
        dropped = predicted_coords(
            backend, plan.members, "time", mode="chunk", drop_conflicting=True
        )[0]
        assert dropped["latitude"] is None
        # refusing instead of dropping, the output either matches or raises
        kept = predicted_coords(backend, plan.members, "time", mode="chunk")[0]
        assert "latitude" in kept

    def test_a_coordinate_nobody_states_is_left_to_the_row(self, pair):
        """A blank dimension is described by the plan, not predicted."""
        first, _ = pair
        blanks = [first.mean("time"), first.new().mean("time")]
        plan, backend = self._plan_and_backend(blanks, time=None)
        described = predicted_coords(backend, plan.members, "time")[0]
        assert described["time"] is None  # the row states its identity
        assert "distance" in described  # everything else is still described

    def test_null_of_each_kind(self):
        """The missing value of a kind is of that kind."""
        assert pd.isnull(_null_like(np.datetime64("2020-01-01", "ns")))
        assert isinstance(_null_like(np.timedelta64(1, "s")), np.timedelta64)
        assert isinstance(_null_like(pd.Timedelta(1, "s")), np.timedelta64)
        assert pd.isnull(_null_like(1.0))

    def test_predictions_skip_columns_the_frame_lacks(self, pair):
        """Restating an envelope touches only columns the frame has."""
        plan, backend = self._plan_and_backend(pair, time=None)
        described = predicted_coords(backend, plan.members, "time")
        outputs = plan.outputs.drop(columns=["time_step"])
        applied = _apply_predictions(outputs, described, "time")
        assert "time_step" not in applied.columns
        assert applied["time_max"].iloc[0] == described[0]["time"].max

    def test_a_replanned_view_falls_back_to_its_rows(self):
        """Members this index does not know are described from the plan."""
        spool = dc.get_example_spool("random_das")
        patches = [
            x.update_coords(
                sensor=("distance", np.arange(x.shape[x.get_axis("distance")]) * 1.0)
            )
            for x in spool
        ]
        joined = dc.spool(patches).concatenate(time=None)
        # the re-plan collapses to members of the *original* spool, whose
        # ids this derived index does not use
        again = joined.chunk(time=None)
        row = again.get_contents().iloc[0]
        assert row["sensor_min"] == 0.0
        assert row["sensor_max"] == patches[0].get_coord("sensor").max()
        assert "sensor" in again[0].coords.coord_map

    def test_a_label_coordinate_falls_back_to_its_row(self):
        """A string coordinate is described from the row when predicting cannot."""
        row = {
            "station_min": "a000",
            "station_max": "a299",
            "_station_def_key": "fp:" + "b" * 32,
        }
        record = _coord_record_from_row(row, "station", dims=("distance",))
        assert record is not None
        assert record.value_kind == "str"
        assert record.min_str == "a000"
        assert record.coord_hash == "b" * 32

    def test_a_rider_falls_back_without_its_identity(self):
        """In the fallback a rider keeps identity only when alone and whole."""
        spool = dc.get_example_spool("random_das")
        patches = [
            x.update_coords(
                clock=("time", np.arange(x.shape[x.get_axis("time")]) * 1.0)
            )
            for x in spool
        ]
        joined = dc.spool(patches).concatenate(time=None)
        again = joined.chunk(time=None)  # re-plan: members are unknown here
        frame = again._catalog.to_df()
        assert not str(frame["_clock_def_key"].iloc[0]).startswith("fp:")
        assert "clock" in again[0].coords.coord_map

    def test_extrema_of_values_which_do_not_compare(self):
        """A group holding two kinds of value has no envelope."""
        frame = pd.DataFrame({"code": [0, 1], "value": ["a", 2.0]})
        grouped = frame.groupby("code")["value"]
        assert list(_extrema(grouped, "min")) == ["a", 2.0]
        mixed = pd.DataFrame({"code": [0, 0], "value": ["a", 2.0]})
        assert list(_extrema(mixed.groupby("code")["value"], "min")) == [None]

    def test_no_members_describes_nothing(self, pair):
        """An empty member table claims nothing."""
        plan, backend = self._plan_and_backend(pair, time=None)
        empty = plan.members.iloc[:0]
        assert predicted_coords(backend, empty, "time") == {}
        assert predicted_coords(None, plan.members, "time") == {}


class TestWhatAMergeWillNotCarry:
    """A row states a coordinate only where the patch will hold it."""

    @pytest.fixture()
    def pair(self):
        """Two patches meeting end to end along time."""
        first = dc.get_example_patch()
        time = first.get_coord("time")
        second = dc.get_example_patch(time_min=time.max() + time.step)
        return first, second

    def test_rider_only_one_member_states(self, pair, assert_contents_match):
        """A merge drops a coordinate its members do not all hold."""
        first, second = pair
        samples = first.shape[first.get_axis("time")]
        held = first.update_coords(bar=("time", np.arange(float(samples))))
        spool = dc.spool([held, second])
        # a value beside no value is a conflict, so a plain merge refuses
        with pytest.raises(CoordMergeError):
            spool.chunk(time=None)
        merged = spool.chunk(time=None, conflict="drop")
        assert "bar" not in merged[0].coords.coord_map
        assert "bar_min" not in merged.get_contents().columns
        assert_contents_match(merged)

    def test_conflicting_rider_is_dropped_in_the_fallback(
        self, pair, assert_contents_match
    ):
        """A re-plan describes no coordinate the merge drops for conflicting."""
        first, second = pair
        axis = first.get_axis("distance")
        values = np.arange(float(first.shape[axis]))
        left = first.update_coords(
            depth=("distance", get_coord(values=values, units="m"))
        )
        right = second.update_coords(
            depth=("distance", get_coord(values=values, units="ft"))
        )
        time = first.get_coord("time")
        step = (time.max() - time.min()) / 3
        spool = dc.spool([left, right]).chunk(time=step, conflict="drop")
        # the subdivision keeps it wherever an output has one member; the
        # output spanning the seam merges two spellings and drops it
        assert "depth" in spool[0].coords.coord_map
        stated = spool.get_contents()["depth_min"]
        assert stated.notnull().any() and stated.isnull().any()
        # merging them back drops it, and the row must not still state it
        again = spool.chunk(time=None, conflict="drop")
        assert "depth" not in again[0].coords.coord_map
        assert "depth_min" not in again.get_contents().columns


class TestRidersKeepMemberOrder:
    """A concatenation lays members end to end; the join sorts them."""

    def _pair(self, low, high):
        """Two contiguous patches carrying a rider on time."""
        first = dc.get_example_patch()
        time = first.get_coord("time")
        second = dc.get_example_patch(time_min=time.max() + time.step)
        samples = first.shape[first.get_axis("time")]
        return (
            first.update_coords(foo=("time", np.arange(*low, dtype=float))),
            second.update_coords(foo=("time", np.arange(*high, dtype=float))),
        ), samples

    def test_reversed_rider_claims_no_structure(self, assert_contents_match):
        """Blocks running backwards concatenate into an array, not a range."""
        first = dc.get_example_patch()
        samples = first.shape[first.get_axis("time")]
        (pair, _) = self._pair((samples, 2 * samples), (0, samples))
        spool = dc.spool(list(pair)).concatenate(time=None)
        row = spool.get_contents().iloc[0]
        assert pd.isnull(row["foo_step"])
        frame = spool._catalog.to_df()
        assert not str(frame["_foo_def_key"].iloc[0]).startswith("fp:")
        assert_contents_match(spool)

    def test_ordered_rider_keeps_its_identity(self, assert_contents_match):
        """Blocks already in order do join into one range."""
        first = dc.get_example_patch()
        samples = first.shape[first.get_axis("time")]
        (pair, _) = self._pair((0, samples), (samples, 2 * samples))
        spool = dc.spool(list(pair)).concatenate(time=None)
        assert spool.get_contents().iloc[0]["foo_step"] == 1.0
        assert_contents_match(spool)


class TestMomentsAndDurations:
    """A datetime and a timedelta are not two spellings of one kind."""

    def test_mixed_time_kinds_state_no_envelope(self):
        """Neither bounds the other, so the row states neither."""
        first = dc.get_example_patch()
        time = first.get_coord("time")
        second = dc.get_example_patch(time_min=time.max() + time.step)
        samples = first.shape[first.get_axis("time")]
        moment = get_coord(values=time.values.copy(), units="s")
        duration = get_coord(
            values=np.arange(samples).astype("timedelta64[s]"), units="s"
        )
        spool = dc.spool(
            [
                first.update_coords(stamp=("time", moment)),
                second.update_coords(stamp=("time", duration)),
            ]
        ).concatenate(time=None)
        assert pd.isnull(spool.get_contents().iloc[0]["stamp_min"])


class TestAgreementNeedsIdentity:
    """Two coordinates nobody can identify are unknown, not equal."""

    def _summary(self, values, fingerprint):
        """A distance-riding summary stating (or not) an identity."""
        summary = get_coord(values=values).to_summary()
        return summary.model_copy(
            update=dict(dims=("distance",), fingerprint=fingerprint)
        )

    def test_unidentified_members_do_not_agree(self):
        """A merge told to drop conflicts drops what it cannot compare."""
        left = self._summary(np.arange(4.0), None)
        right = self._summary(np.arange(4.0) + 10, None)
        described = _describe(
            "rough",
            [left, right],
            "time",
            frozenset(),
            None,
            mode="chunk",
            drop_conflicting=True,
        )
        assert described is None

    def test_a_lone_member_keeps_what_it_says(self):
        """With nothing to agree with, an unidentified member still counts."""
        only = self._summary(np.arange(4.0), None)
        described = _describe(
            "rough",
            [only],
            "time",
            frozenset(),
            None,
            mode="chunk",
            drop_conflicting=True,
        )
        assert described is not None
        assert described.min == only.min

    def test_dims_must_match_to_survive_a_merge(self, assert_contents_match):
        """A coordinate hung on different dimensions is dropped, not merged."""
        first = dc.get_example_patch()
        time = first.get_coord("time")
        second = dc.get_example_patch(time_min=time.max() + time.step)
        values = np.arange(float(first.shape[first.get_axis("distance")]))
        # matching values, so no envelope conflict raises first
        left = first.update_coords(baz=("distance", values))
        right = second.update_coords(
            baz=("time", np.resize(values, first.shape[first.get_axis("time")]))
        )
        merged = dc.spool([left, right]).chunk(time=None, conflict="drop")
        assert "baz" not in merged[0].coords.coord_map
        assert "baz_min" not in merged.get_contents().columns
        assert_contents_match(merged)

    def test_one_fingerprint_two_spellings_do_not_agree(self, assert_contents_match):
        """A fingerprint is normalized; what a merge compares is not."""
        first = dc.get_example_patch()
        time = first.get_coord("time")
        second = dc.get_example_patch(time_min=time.max() + time.step)
        values = np.arange(float(first.shape[first.get_axis("distance")]))
        metres = get_coord(values=values, units="m")
        centimetres = get_coord(values=values * 100.0, units="cm")
        assert metres.to_summary().fingerprint == centimetres.to_summary().fingerprint
        left = first.update_coords(depth=("distance", metres))
        right = second.update_coords(depth=("distance", centimetres))
        merged = dc.spool([left, right]).chunk(time=None, conflict="drop")
        assert "depth" not in merged[0].coords.coord_map
        assert "depth_min" not in merged.get_contents().columns
        assert_contents_match(merged)

    def test_one_spelling_still_agrees(self, assert_contents_match):
        """Members stating one coordinate the same way keep it."""
        first = dc.get_example_patch()
        time = first.get_coord("time")
        second = dc.get_example_patch(time_min=time.max() + time.step)
        values = np.arange(float(first.shape[first.get_axis("distance")]))
        coord = get_coord(values=values, units="m")
        merged = dc.spool(
            [
                first.update_coords(depth=("distance", coord)),
                second.update_coords(depth=("distance", coord)),
            ]
        ).chunk(time=None)
        assert "depth" in merged[0].coords.coord_map
        assert merged.get_contents().iloc[0]["depth_min"] == 0.0
        assert_contents_match(merged)


class TestTrimmedCoordsStayTrimmed:
    """A residual trims at load; the record must not outrun it."""

    def test_a_samples_residual_survives_a_union(self):
        """Candidacy is answered from the record, so it holds the trim."""
        first = dc.get_example_patch()
        time = first.get_coord("time")
        second = dc.get_example_patch(time_min=time.max() + time.step)
        selected = dc.spool([first]).select(distance=(5, 50), samples=True)
        union = selected + dc.spool([second])
        # the trimmed patch holds distance 5..49 and must not be a
        # candidate for values only the untrimmed source ever held
        elsewhere = union.select(distance=(60, 80))
        assert len(elsewhere) == len(elsewhere.get_contents()) == 1
        assert len(list(elsewhere)) == 1
        loaded = elsewhere[0].get_coord("distance")
        assert loaded.min() == 60 and loaded.max() == 80

    def test_the_record_says_what_the_patch_holds(self):
        """The stored envelope matches the trimmed patch, not its source."""
        first = dc.get_example_patch()
        time = first.get_coord("time")
        second = dc.get_example_patch(time_min=time.max() + time.step)
        selected = dc.spool([first]).select(distance=(5, 50), samples=True)
        union = selected + dc.spool([second])
        frame = union._catalog.backend.coord_frame([1, 2])
        distance = frame[frame["coord_name"] == "distance"]
        stated = distance[distance["patch_id"] == 1].iloc[0]
        held = union[0].get_coord("distance")
        assert stated["min_num"] == held.min()
        assert stated["max_num"] == held.max()
        assert stated["length"] == len(held)


class TestRidersSurviveTheirMerge:
    """A rider is joined, not compared, and never snapped."""

    @pytest.fixture()
    def riding(self):
        """Two contiguous patches carrying a clock on time."""
        first = dc.get_example_patch()
        time = first.get_coord("time")
        second = dc.get_example_patch(time_min=time.max() + time.step)
        samples = first.shape[first.get_axis("time")]
        return first, second, samples

    def test_a_replan_keeps_the_rider_it_holds(self, riding):
        """Differing definitions are how a rider works, not a conflict."""
        first, second, samples = riding
        time = first.get_coord("time")
        left = first.update_coords(clock=("time", np.arange(float(samples))))
        right = second.update_coords(
            clock=("time", np.arange(float(samples), 2.0 * samples))
        )
        step = (time.max() - time.min()) / 2
        spool = dc.spool([left, right]).chunk(time=step, conflict="drop")
        again = spool.chunk(time=None, conflict="drop")
        assert "clock" in again[0].coords.coord_map
        assert again.get_contents().iloc[0]["clock_min"] == 0.0

    def test_an_irregular_rider_is_not_snapped(self, riding, assert_contents_match):
        """Assembly simplifies the merged dimension and nothing else."""
        first, second, samples = riding
        left = first.update_coords(clock=("time", np.arange(float(samples))))
        # the second block starts half a step late: a seam a tolerant
        # snap would absorb on the merged dimension, but not here
        right = second.update_coords(
            clock=("time", np.arange(float(samples)) + samples + 0.5)
        )
        merged = dc.spool([left, right]).chunk(time=None, conflict="keep_first")
        assert pd.isnull(merged.get_contents().iloc[0]["clock_step"])
        assert_contents_match(merged)

    def test_a_contiguous_rider_keeps_its_step(self, riding, assert_contents_match):
        """Not snapping is not the same as claiming nothing."""
        first, second, samples = riding
        left = first.update_coords(clock=("time", np.arange(float(samples))))
        right = second.update_coords(
            clock=("time", np.arange(float(samples)) + samples)
        )
        merged = dc.spool([left, right]).chunk(time=None, conflict="keep_first")
        assert merged.get_contents().iloc[0]["clock_step"] == 1.0
        assert_contents_match(merged)

    def test_a_restated_member_is_not_a_trimmed_one(self, assert_contents_match):
        """The planner's unit and the index's spelling describe one member."""
        first = dc.get_example_patch()
        size = first.shape[first.get_axis("distance")]
        values = np.arange(float(size))
        left = first.update_coords(
            distance=get_coord(values=values, units="m"),
            rider=("distance", values),
        )
        right = first.update_coords(
            distance=get_coord(values=(values + size) * 100.0, units="cm"),
            rider=("distance", values + size),
        )
        merged = dc.spool([left, right]).chunk(distance=None, conflict="keep_first")
        row = merged.get_contents().iloc[0]
        assert row["rider_min"] == 0.0 and row["rider_max"] == 2 * size - 1
        assert_contents_match(merged)


class TestFallbackAgreement:
    """The fallback mirrors the rules prediction applies."""

    def _frames(self, keys):
        """Source rows and members for one output holding `keys`."""
        count = len(keys)
        sources = pd.DataFrame(
            {
                "_patch_id": list(range(1, count + 1)),
                "depth_min": [0.0] * count,
                "depth_max": [9.0] * count,
                "_depth_def_key": keys,
            }
        )
        members = pd.DataFrame(
            {
                "output_id": [0] * count,
                "_patch_id": list(range(1, count + 1)),
                "_modified": [False] * count,
            }
        )
        return sources, members

    def test_a_lone_unidentified_member_is_still_described(self):
        """`nunique` counts no nulls, and one member disagrees with nobody."""
        sources, members = self._frames([None])
        described = _aux_coord_info(
            sources,
            members,
            "time",
            {"depth": "distance"},
            mode="chunk",
            drop_conflicting=True,
        )
        assert "depth" in described[0]

    def test_members_which_disagree_are_still_dropped(self):
        """The exemption is for having nobody to disagree with."""
        sources, members = self._frames(["fp:a", "fp:b"])
        described = _aux_coord_info(
            sources,
            members,
            "time",
            {"depth": "distance"},
            mode="chunk",
            drop_conflicting=True,
        )
        assert "depth" not in described.get(0, {})


class TestWhatTheUnionCarriesOver:
    """The fallback describes what raw concatenation will produce."""

    def test_mixed_numeric_dtypes_promote(self):
        """Numpy promotes what it concatenates, so the record must too."""
        first = dc.get_example_patch()
        time = first.get_coord("time")
        second = dc.get_example_patch(time_min=time.max() + time.step)
        samples = first.shape[first.get_axis("time")]
        rng = np.random.default_rng(0)
        left = first.update_coords(
            rider=("time", np.sort(rng.integers(0, 100, samples)).astype("int32"))
        )
        right = second.update_coords(
            rider=("time", np.sort(rng.uniform(200, 300, samples)).astype("float64"))
        )
        spool = dc.spool([left, right]).concatenate(time=None)
        frame = spool._catalog.backend.coord_frame([0, 1, 2])
        stated = frame[frame["coord_name"] == "rider"]["dtype"].iloc[0]
        assert stated == str(spool[0].get_coord("rider").dtype) == "float64"

    def test_one_dtype_is_left_alone(self):
        """Promotion is not an excuse to restate what already agrees."""
        first = dc.get_example_patch()
        time = first.get_coord("time")
        second = dc.get_example_patch(time_min=time.max() + time.step)
        samples = first.shape[first.get_axis("time")]
        rng = np.random.default_rng(0)
        values = np.sort(rng.integers(0, 100, samples)).astype("int32")
        spool = dc.spool(
            [
                first.update_coords(rider=("time", values)),
                second.update_coords(rider=("time", values + 100)),
            ]
        ).concatenate(time=None)
        frame = spool._catalog.backend.coord_frame([0, 1, 2])
        assert frame[frame["coord_name"] == "rider"]["dtype"].iloc[0] == "int32"

    def test_zero_in_two_units_is_not_one_bound(self, assert_contents_match):
        """Equal numbers in different spellings are not the same bounds."""
        first = dc.get_example_patch()
        time = first.get_coord("time")
        data = np.random.default_rng(0).random((1, len(time)))

        def sample(units):
            """A one-sample patch whose distance sits at zero."""
            distance = get_coord(values=np.array([0.0]), units=units)
            return dc.Patch(
                data=data,
                coords={"distance": distance, "time": time},
                dims=("distance", "time"),
            )

        spool = dc.spool([sample("m"), sample("cm")]).concatenate(distance=None)
        row = spool.get_contents().iloc[0]
        assert row["distance_min"] == 0.0 and row["distance_max"] == 0.0
        # the output is real, so a selection over it must keep it
        assert len(spool.select(distance=(-1, 1))) == 1
        assert_contents_match(spool)


class TestRawJoinsUseRawValues:
    """A concatenation lays values end to end; a range regenerates them."""

    def _pieces(self, lengths, start=-10.0, step=0.1):
        """Patches whose distance ranges meet but drift inside their spans."""
        time = dc.get_example_patch().get_coord("time")
        patches = []
        for length in lengths:
            distance = get_coord(start=start, step=step, stop=start + step * length)
            data = np.random.default_rng(0).random((length, len(time)))
            patches.append(
                dc.Patch(
                    data=data,
                    coords={"distance": distance, "time": time},
                    dims=("distance", "time"),
                )
            )
            start = start + step * length
        return patches

    def test_drifting_floats_state_the_step_they_will_have(self, assert_contents_match):
        """Boundaries can match while interior samples do not."""
        spool = dc.spool(self._pieces((2, 3, 4))).concatenate(distance=None)
        loaded = spool[0].get_coord("distance")
        assert spool.get_contents().iloc[0]["distance_step"] == loaded.step
        frame = spool._catalog.backend.coord_frame([0, 1, 2, 3])
        stated = frame[frame["coord_name"] == "distance"]["fingerprint"].iloc[0]
        assert stated == loaded.fingerprint()
        assert_contents_match(spool)

    def test_floats_which_do_not_drift_keep_the_fused_range(
        self, assert_contents_match
    ):
        """Rebuilding from values is a correction, not a policy."""
        spool = dc.spool(self._pieces((2, 3, 4), start=0.0, step=1.0)).concatenate(
            distance=None
        )
        assert spool.get_contents().iloc[0]["distance_step"] == 1.0
        assert_contents_match(spool)

    def test_integer_widths_promote(self):
        """A fused range takes the first dtype; concatenation promotes."""
        time = dc.get_example_patch().get_coord("time")

        def piece(values):
            """A patch whose distance holds exactly these values."""
            data = np.random.default_rng(0).random((len(values), len(time)))
            return dc.Patch(
                data=data,
                coords={"distance": get_coord(values=values), "time": time},
                dims=("distance", "time"),
            )

        spool = dc.spool(
            [
                piece(np.arange(0, 5, dtype="int32")),
                piece(np.arange(5, 10, dtype="int64")),
            ]
        ).concatenate(distance=None)
        loaded = spool[0].get_coord("distance")
        frame = spool._catalog.backend.coord_frame([0, 1, 2])
        stated = frame[frame["coord_name"] == "distance"]
        assert stated["dtype"].iloc[0] == str(loaded.dtype) == "int64"
        assert stated["fingerprint"].iloc[0] == loaded.fingerprint()


class TestCutRiders:
    """A coordinate riding a cut dimension is sliced along with it."""

    def _summaries(self, samples=100):
        """A whole dimension and a rider of the same length."""
        start = np.datetime64("2020-01-01")
        step = np.timedelta64(1, "s")
        whole = get_coord(start=start, step=step, stop=start + step * samples)
        rider = get_coord(values=np.arange(float(samples)))
        return whole.to_summary(), rider.to_summary()

    def test_the_slice_is_exact(self):
        """Evenly sampled members give the cut exactly."""
        whole, rider = self._summaries()
        start = np.datetime64("2020-01-01")
        row = {
            "time_min": start + np.timedelta64(10, "s"),
            "time_max": start + np.timedelta64(20, "s"),
        }
        sliced = _cut_rider(rider, whole, row, "time")
        assert sliced.min == 10.0 and sliced.max == 20.0 and sliced.len == 11

    def test_an_unmeasured_dimension_says_nothing(self):
        """Without a grid on both sides the slice cannot be worked out."""
        whole, rider = self._summaries()
        row = {"time_min": np.datetime64("2020-01-01"), "time_max": None}
        assert _cut_rider(rider, None, row, "time") is None
        assert _cut_rider(rider, whole, row, "time") is None
        stepless = rider.model_copy(update=dict(step=None))
        row = {
            "time_min": np.datetime64("2020-01-01"),
            "time_max": np.datetime64("2020-01-01") + np.timedelta64(5, "s"),
        }
        assert _cut_rider(stepless, whole, row, "time") is None

    def test_an_array_rider_keeps_no_envelope(self, assert_contents_match):
        """What cannot be sliced is not guessed at."""
        first = dc.get_example_patch()
        time = first.get_coord("time")
        samples = first.shape[first.get_axis("time")]
        rng = np.random.default_rng(0)
        rough = first.update_coords(rough=("time", np.sort(rng.uniform(0, 1, samples))))
        spool = dc.spool([rough]).chunk(
            time=(time.max() - time.min()) / 2, conflict="keep_first"
        )
        assert spool.get_contents()["rough_min"].isnull().all()
        assert "rough" in spool[0].coords.coord_map
