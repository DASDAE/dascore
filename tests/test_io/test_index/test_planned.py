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
from dascore.exceptions import MissingPatchError, ParameterError
from dascore.io.index.catalog import PatchCatalog
from dascore.io.index.planned import (
    PlanResolver,
    _aux_coord_info,
    _coord_record_from_row,
    _ns,
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


class TestNumericAttrUnits:
    """The attr units a plan resolves for stamping."""

    def test_no_parent_knows_nothing(self):
        """Without a parent index there are no attr units to resolve."""
        assert _plan_attr_units(None, pd.DataFrame({"foo": [1.0]})) == {}


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

    def test_a_trimmed_join_claims_no_identity(self, pair):
        """A residual trimming the joined dimension voids its identity."""
        plan, backend = self._plan_and_backend(pair, time=None)
        described = predicted_coords(
            backend, plan.members, "time", trimmed_dims=frozenset({"time"})
        )[0]
        assert described["time"].fingerprint is None
        assert described["time"].len is None
        # the envelope still says where the output lies
        assert described["time"].min == pair[0].get_coord("time").min()

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
        """A coordinate only some members hold is still described."""
        first, second = pair
        n = first.shape[first.get_axis("distance")]
        lat = second.update_coords(latitude=("distance", np.arange(n) * 1.0))
        plan, backend = self._plan_and_backend([first, lat], time=None)
        described = predicted_coords(backend, plan.members, "time")[0]
        assert "latitude" in described
        assert described["latitude"].len == n

    def test_no_members_describes_nothing(self, pair):
        """An empty member table claims nothing."""
        plan, backend = self._plan_and_backend(pair, time=None)
        empty = plan.members.iloc[:0]
        assert predicted_coords(backend, empty, "time") == {}
        assert predicted_coords(None, plan.members, "time") == {}
