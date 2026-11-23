"""
Tests for the unified Spool's equality and update contracts.

Equality is over rows, never backends: order-sensitive, comparing what
each row loads, with pending residual selections included. update() is case-based:
directory spools sync, file spools rescan, in-memory spools are
trivially current, and combined spools with file rows raise.
"""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.compat import UPath
from dascore.exceptions import InvalidSpoolError
from dascore.io.index.catalog import FileResolver
from dascore.io.index.planned import PlanResolver
from dascore.utils.identity import window_data_id
from dascore.utils.patch_assembly import stored_window_id


@pytest.fixture(scope="module")
def patches():
    """Three contiguous example patches in time order."""
    return list(dc.get_example_spool("random_das"))


class TestEqualityContract:
    """Equality: rows not backends; order-sensitive; residuals count."""

    def test_residual_breaks_equality(self, patches):
        """A samples residual changes loaded patches, so spools differ."""
        plain = dc.spool(patches)
        trimmed = dc.spool(patches).select(distance=(0, 10), samples=True)
        # same visible rows (samples never excludes patches) ...
        assert len(plain) == len(trimmed)
        # ... but not equal spools.
        assert plain != trimmed

    def test_equal_selections_compare_equal(self, patches):
        """The same selection on equal spools yields equal spools."""
        sel_1 = dc.spool(patches).select(distance=(0, 10))
        sel_2 = dc.spool(patches).select(distance=(0, 10))
        assert sel_1 == sel_2

    @pytest.mark.parametrize(
        "name", ["sensor_coord_dtype", "sensor_def_key", "sensor_cut_mask"]
    )
    def test_an_attr_spelled_like_a_private_column_still_counts(self, patches, name):
        """Only the generated `_<coord>...` column is stripped, not an attr."""
        one = dc.spool([p.update_attrs(**{name: "a"}) for p in patches])
        same = dc.spool([p.update_attrs(**{name: "a"}) for p in patches])
        other = dc.spool([p.update_attrs(**{name: "b"}) for p in patches])
        assert one == same
        assert one != other

    def test_plan_states_the_dtype_a_coordinate_it_passes_through_has(self):
        """A plan equals its materialization, and agrees on the dtype."""
        patch = dc.get_example_patch()
        patch = patch.update_coords(distance=np.arange(patch.shape[0], dtype=np.int32))
        planned = dc.spool(patch).chunk(time=2)
        materialized = dc.spool(list(planned))
        # The chunked dimension's envelope is a window the plan computed,
        # so it keeps the float placeholder; distance passes through
        # untouched and is stated as the integer coordinate it is, which
        # is what an output rebuilt from its row alone has to go on.
        assert set(planned._df["_distance_coord_dtype"]) == {"int32"}
        assert set(materialized._df["_distance_coord_dtype"]) == {"int32"}
        assert planned[0].get_coord("distance").dtype == np.int32
        assert planned == materialized

    def test_order_matters(self, patches):
        """Same patches in a different order are not equal."""
        assert dc.spool(patches) != dc.spool(list(reversed(patches)))

    def test_live_vs_directory_equal(self, patches, tmp_path):
        """Identical contents compare equal across backings (rows, not
        backends): a live spool equals a directory spool over the same
        patches written to disk.
        """
        directory = dc.examples.spool_to_directory(
            dc.spool(patches), path=tmp_path / "spool_dir"
        )
        dir_spool = dc.spool(directory).update(progress=None)
        live_spool = dc.spool(patches)
        assert live_spool == dir_spool
        assert dir_spool == live_spool


class TestEqualityComparesLoads:
    """Equality compares what each spool would load, reading no file (#1642)."""

    @pytest.fixture()
    def ones(self):
        """A patch built from an array, so it has a random data_id."""
        return dc.Patch(
            data=np.ones((2, 6)),
            coords={"distance": np.arange(2.0), "time": np.arange(6.0)},
            dims=("distance", "time"),
        )

    @pytest.fixture()
    def directories(self, ones, tmp_path):
        """Spools of files holding `ones`, and of files holding it doubled."""
        out = []
        for name, patch in (("ones", ones), ("twos", ones * 2)):
            dc.write(patch, tmp_path / name / "patch.h5", "dasdae")
            out.append(dc.spool(tmp_path / name).update(progress=None))
        return out

    @pytest.fixture()
    def no_reads(self, monkeypatch):
        """Fail any load of a file row."""

        def _fail(*args, **kwargs):
            raise AssertionError("equality read a file")

        monkeypatch.setattr(FileResolver, "resolve", _fail)

    def test_equal_arrays_with_different_ids(self, ones):
        """Patches in memory compare as patches, whatever their ids."""
        twin = dc.Patch(data=ones.data, coords=ones.coords, dims=ones.dims)
        assert twin.attrs.data_id != ones.attrs.data_id
        assert dc.spool([ones]) == dc.spool([twin])
        assert dc.spool([ones]).chunk(time=3) == dc.spool([twin]).chunk(time=3)

    def test_different_data(self, ones):
        """Patches with equal metadata and different data differ."""
        assert dc.spool([ones]) != dc.spool([ones * 2])
        assert dc.spool([ones]).chunk(time=3) != dc.spool([ones * 2]).chunk(time=3)
        window = {"time": (1, 4)}
        assert dc.spool([ones]).select(**window) != dc.spool([ones * 2]).select(
            **window
        )

    @pytest.fixture(params=[6.0, 6.25])
    def adjacent(self, ones, tmp_path, request):
        """Two files, the second starting a step after the first ends or near it."""
        patches = [ones, ones.update_coords(time_min=request.param) * 2]
        path = dc.examples.spool_to_directory(dc.spool(patches), path=tmp_path / "a")
        return dc.spool(path).update(progress=None)

    @pytest.mark.parametrize("on_disk", [True, False])
    def test_view_equals_its_loaded_result(self, ones, directories, on_disk):
        """A selection or plan equals the patches it loads."""
        source = directories[0] if on_disk else dc.spool([ones])
        selected = source.select(time=(1, 4))
        for view in (selected, source.chunk(time=3), selected.chunk(time=2)):
            assert view == view.load()
            assert view.load() == view
            assert view == dc.spool(list(view))

    def test_merged_files_equal_their_loaded_result(self, adjacent):
        """What the rows say a merge of files loads is what it loads."""
        # the wider max_shift puts the late file on the first one's grid
        kwargs = {"max_shift": 0.3}
        for view in (adjacent.chunk(time=None, **kwargs), adjacent.chunk(time=4)):
            assert view == dc.spool(list(view))
            assert view != dc.spool([x * 2 for x in view])

    def test_a_selection_after_a_plan_of_files(self, adjacent):
        """A selection trims a plan's output as it loads, so the two compare alike."""
        merged = adjacent.chunk(time=None, max_shift=0.3)
        for view in (
            merged.select(time=(2, 8)),
            merged.select(time=(2, 8), samples=True),
            merged.select(distance=(0, 0)),
        ):
            assert view == dc.spool(list(view))
            combined = view + dc.spool([])
            assert combined == dc.spool(list(combined))

    def test_files_against_patches_of_other_data(self, ones, directories, no_reads):
        """A selection or plan of files names what it loads, as a file row does."""
        on_disk, _ = directories
        window = {"time": (1, 4)}
        assert on_disk.select(**window) != dc.spool([ones.select(**window) * 2])
        doubled = [x * 2 for x in dc.spool([ones]).chunk(time=3)]
        assert on_disk.chunk(time=3) != dc.spool(doubled)

    def test_plans_loading_alike(self, directories, no_reads):
        """Plans of files which load the same samples the same way are equal."""
        ones, twos = directories
        window = {"time": (1, 4)}
        selected = ones.select(**window)
        assert selected == selected.chunk(time=None)
        assert selected.chunk(time=None) == selected
        assert selected != twos.select(**window).chunk(time=None)

    @pytest.mark.parametrize("combined", [False, True])
    def test_plan_of_selected_files(self, directories, no_reads, combined):
        """A member a selection trimmed is its file's id and window."""
        # combined with nothing, the files are indexed in memory
        ones, twos = (x + dc.spool([]) if combined else x for x in directories)
        window = {"time": (1, 4)}
        planned = ones.select(**window).chunk(time=2)
        assert planned == ones.select(**window).chunk(time=2)
        assert planned != twos.select(**window).chunk(time=2)

    def test_spools_of_same_files(self, directories, tmp_path, no_reads):
        """Spools over the same files compare their data ids."""
        ones, twos = directories
        assert ones == dc.spool(tmp_path / "ones").update(progress=None)
        assert ones != twos
        window = {"time": (1, 4)}
        assert ones.select(**window) == ones.select(**window)
        assert ones.select(**window) != twos.select(**window)
        assert ones.chunk(time=3) == ones.chunk(time=3)
        assert ones.chunk(time=3) != twos.chunk(time=3)

    def test_memory_vs_files_written_from_it(self, ones, directories):
        """A written patch keeps its id, so the files equal the memory spool."""
        on_disk, doubled = directories
        assert dc.spool([ones]) == on_disk
        assert on_disk == dc.spool([ones])
        assert dc.spool([ones]) != doubled

    def test_file_named_by_a_uri(self, ones):
        """A file named by a URI is a file row, not a plan's output."""
        path = UPath("memory://dascore/equality/ones.h5")
        dc.write(ones, path, "dasdae")
        assert dc.spool(path) == dc.spool([ones])
        assert dc.spool(path) != dc.spool([ones * 2])

    def test_comparing_reads_no_file(self, ones, directories, no_reads):
        """Neither files nor plans over them are read to compare."""
        on_disk, _ = directories
        assert on_disk == dc.spool([ones])
        view = on_disk.select(time=(1, 4))
        # the window of the file, which the same selection in memory names
        assert view == dc.spool([ones.select(time=(1, 4))])
        assert view == dc.spool([ones]).select(time=(1, 4))
        assert on_disk.chunk(time=3) == on_disk.chunk(time=3)

    @pytest.mark.parametrize("on_disk", [True, False])
    def test_plans_filling_differently_differ(self, ones, tmp_path, on_disk):
        """Plans with equal rows which fill a gap differently differ."""
        patches = [ones, ones.update_coords(time_min=8.0)]
        source = dc.spool(patches)
        if on_disk:
            path = dc.examples.spool_to_directory(source, path=tmp_path)
            source = dc.spool(path).update(progress=None)
        zeros, nans = source.fill_gaps(time=20, fill_value=0), source.fill_gaps(time=20)
        rows = [x.get_contents().drop(columns="source_path") for x in (zeros, nans)]
        assert rows[0].equals(rows[1])
        assert zeros != nans

    def test_selection_and_plan_loading_alike(self, ones):
        """In memory, a selection equals a plan which loads the same patch."""
        selected = dc.spool([ones]).select(time=(1, 4))
        assert selected == selected.chunk(time=None)

    def test_plan_of_selection_in_memory(self, ones):
        """A plan of a selection compares what it loads with a plan of patches."""
        planned = dc.spool([ones]).select(time=(1, 4)).chunk(time=2)
        part = ones.select(time=(1, 4))
        assert planned == dc.spool([part]).chunk(time=2)
        assert planned != dc.spool([part * 2]).chunk(time=2)

    def test_snapped_spool_equals_its_loaded_result(self, ones):
        """A snap applied at load is part of what the spool loads."""
        numbered = ones.new(data=np.arange(12.0).reshape(2, 6))
        snapped = dc.spool([numbered]).snap_coords("time", reverse=True)
        assert snapped == snapped.load()
        assert snapped != dc.spool([numbered])

    def test_route_which_is_not_a_plan(self, ones, monkeypatch):
        """Rows another library's route loads compare by metadata alone."""
        chunked = dc.spool([ones]).chunk(time=3)
        resolver = chunked._catalog.resolver
        routes = {prefix: object() for prefix in resolver.plan_entries()}
        monkeypatch.setattr(resolver, "plan_entries", lambda: routes)
        assert chunked == dc.spool([ones * 2]).chunk(time=3)

    @pytest.mark.parametrize("case", ["data_units", "uneven", "time_units"])
    def test_files_the_rows_cannot_place(self, ones, tmp_path, case):
        """Where the rows cannot name what a plan of files loads, it equals it."""
        later, kwargs = ones.update_coords(time_min=6.0), {}
        if case == "data_units":  # loading converts the later member
            members = [
                ones.update_attrs(data_units="m"),
                later.update_attrs(data_units="mm"),
            ]
            kwargs = {"conflict": "keep_first"}
        elif case == "uneven":  # no step places the samples
            uneven = ones.update_coords(time=np.array([0.0, 1, 2, 4, 5, 7]))
            members = [uneven, uneven.update_coords(time_min=8.0)]
        else:  # the plan counts both in one unit
            milli = later.update_coords(time=later.get_array("time") * 1000)
            members = [ones.set_units(time="s"), milli.set_units(time="ms")]
        path = dc.examples.spool_to_directory(dc.spool(members), path=tmp_path / case)
        files = dc.spool(path).update(progress=None)
        view = files.chunk(time=None, **kwargs)
        assert view == dc.spool(list(view))

    def test_a_selection_on_a_rider(self, ones, tmp_path, no_reads):
        """Selecting on a coordinate beside the dimensions is a select, not a window."""
        rider = ones.update_coords(lat=("distance", np.array([0.5, 1.5])))
        spools = []
        for name, patch in (("ones", rider), ("twos", rider * 2)):
            dc.write(patch, tmp_path / name / "patch.h5", "dasdae")
            spools.append(dc.spool(tmp_path / name).update(progress=None))
        ones, twos, again = (
            x.select(lat=(0, 1))
            for x in (*spools, dc.spool(tmp_path / "ones").update(progress=None))
        )
        # a selection compares its source; a plan over one, its metadata
        assert ones == again and ones != twos
        assert ones.chunk(time=3) == again.chunk(time=3)


class TestStoredWindowId:
    """The data id of the samples a trimmed row keeps, from the index rows alone."""

    @pytest.fixture()
    def stored(self):
        """The index's row for a ten-sample array."""
        return {
            "source_path": "a.h5",
            "source_patch_key": "",
            "dims": "time",
            "data_id": "x" * 32,
            "time_min": 0.0,
            "time_max": 9.0,
            "time_step": 1.0,
        }

    def test_a_window(self, stored):
        """A window is the array's id and the samples kept; all of it is the array."""
        row = {**stored, "source_path": "/root/a.h5", "time_min": 2.0, "time_max": 4.0}
        assert stored_window_id(row, stored) == window_data_id("x" * 32, [(2, 5)], [10])
        assert stored_window_id(stored, stored) == "x" * 32

    @pytest.mark.parametrize(
        ("row", "whole"),
        [
            ({}, {"data_id": None}),  # a moved source has its id cleared
            ({"source_path": "b.h5"}, {}),  # another file
            ({"_time_units": "ms"}, {}),  # another unit
            ({}, {"time_step": np.nan}),  # no step places the samples
            ({"time_min": 4.5, "time_max": 4.6}, {}),  # no sample kept
        ],
    )
    def test_none_where_the_rows_cannot_say(self, stored, row, whole):
        """The rows name no window rather than a wrong one."""
        assert stored_window_id({**stored, **row}, {**stored, **whole}) is None


class TestUpdateContract:
    """update() means sync-with-source; the cases differ by source."""

    def test_live_only_is_noop(self, patches):
        """A purely in-memory spool is trivially current."""
        spool = dc.spool(patches)
        assert spool.update() is spool

    def test_union_raises(self, patches):
        """Combining spools is a computation; the result cannot update."""
        combined = dc.spool(patches[:1]) + dc.spool(patches[1:])
        with pytest.raises(InvalidSpoolError, match="root spool"):
            combined.update()

    def test_union_with_file_rows_raises(self, patches, tmp_path):
        """A combined spool with file rows cannot update either."""
        directory = dc.examples.spool_to_directory(
            dc.spool(patches), path=tmp_path / "dir_a"
        )
        combined = dc.spool(directory).update(progress=None) + dc.spool(patches[:1])
        with pytest.raises(InvalidSpoolError, match="root spool"):
            combined.update()

    def test_selected_spool_raises(self, patches):
        """Any operation severs update: a selected spool refuses it."""
        selected = dc.spool(patches).select(tag="random")
        with pytest.raises(InvalidSpoolError, match="root spool"):
            selected.update()

    def test_selected_file_spool_raises(self, patches, tmp_path):
        """A selected single-file spool refuses update instead of
        silently widening back to the whole file (review P1).
        """
        path = tmp_path / "sel_file.h5"
        dc.write(patches[0], path, "dasdae")
        selected = dc.spool(path).select(distance=(0, 10), samples=True)
        with pytest.raises(InvalidSpoolError, match="root spool"):
            selected.update()

    def test_directory_update_picks_up_new_files(self, patches, tmp_path):
        """The syncer case: new files appear after update()."""
        directory = tmp_path / "dir_b"
        dc.write(patches[0], directory / "a.h5", "dasdae")
        spool = dc.spool(directory).update(progress=None)
        assert len(spool) == 1
        dc.write(patches[1], directory / "b.h5", "dasdae")
        assert len(spool.update(progress=None)) == 2

    def test_file_spool_update_refreshes(self, patches, tmp_path):
        """The single-file case: update rescans the file."""
        path = tmp_path / "single.h5"
        dc.write(patches[0], path, "dasdae")
        spool = dc.spool(path)
        new = spool.update()
        assert len(new) == len(spool)
        assert new == spool


class TestChunkPlanContract:
    """Planned views: collapse semantics and derived state."""

    def test_replan_collapses(self, patches):
        """Re-chunking a planned spool re-plans from members (no nesting)."""
        spool = dc.spool(patches)
        hourly = spool.chunk(time=2)
        merged = hourly.chunk(time=None)
        assert len(merged) == 1
        time = merged[0].get_coord("time")
        mins = [p.get_coord("time").min() for p in patches]
        maxs = [p.get_coord("time").max() for p in patches]
        assert time.min() == min(mins)
        assert time.max() == max(maxs)

    @pytest.mark.parametrize("on_disk", [True, False])
    def test_replan_keeps_other_selection(self, numbered, tmp_path, on_disk):
        """Re-planning after a selection on another dimension keeps it (#1637)."""
        source = numbered
        if on_disk:
            directory = dc.examples.spool_to_directory(numbered, path=tmp_path)
            source = dc.spool(directory).update(progress=None)
        distance = {"distance": (1, 2)}
        expected = dc.spool(numbered).chunk(time=None)[0].select(**distance)
        step = expected.get_coord("time").step
        # one window per patch
        once = dc.spool(source).select(**distance).chunk(time=8 * step)
        twice = once.chunk(time=None)
        for replanned in (twice, twice.chunk(time=None), once.concatenate(time=None)):
            (patch,) = replanned
            assert patch.coords == expected.coords
            assert np.array_equal(patch.data, expected.data)

    @pytest.fixture(scope="class")
    def numbered(self):
        """Two contiguous 4x8 patches whose values are their sample numbers."""
        base = dc.get_example_patch().select(distance=(0, 4), time=(0, 8), samples=True)
        first = base.new(data=np.arange(32.0).reshape(4, 8))
        time = first.get_coord("time")
        second = first.new(data=first.data + 100)
        return [first, second.update_coords(time_min=time.max() + time.step)]

    @pytest.mark.parametrize("on_disk", [True, False])
    @pytest.mark.parametrize("kwargs", [{"distance": (1, None)}, {"time": (1, -1)}])
    def test_replan_keeps_sample_selection(self, numbered, tmp_path, on_disk, kwargs):
        """Re-chunking after a samples selection loads the selected samples."""
        source = numbered
        if on_disk:
            directory = dc.examples.spool_to_directory(numbered, path=tmp_path)
            source = dc.spool(directory).update(progress=None)
        selected = dc.spool(source).select(**kwargs, samples=True)
        expected = [p.select(**kwargs, samples=True).data for p in numbered]
        once = selected.chunk(time=None)
        twice = once.chunk(time=None)
        thrice = twice.chunk(time=None)
        for chunked in (once, twice, thrice):
            data = np.concatenate([p.data for p in chunked], axis=1)
            assert np.array_equal(data, np.concatenate(expected, axis=1))
            assert [p.coords for p in chunked] == [p.coords for p in once]

    @pytest.mark.parametrize("on_disk", [True, False])
    def test_replan_after_other_selection_keeps_samples(
        self, numbered, tmp_path, on_disk
    ):
        """Re-chunking after a selection on another dimension keeps every sample."""
        source = numbered
        if on_disk:
            directory = dc.examples.spool_to_directory(numbered, path=tmp_path)
            source = dc.spool(directory).update(progress=None)
        spool, distance = dc.spool(source), {"distance": (1, 2)}
        expected = dc.spool(numbered).chunk(time=None)[0].select(**distance)
        step = expected.get_coord("time").step

        def chunk(spool, *lengths):
            for length in lengths:
                spool = spool.chunk(time=length * step, keep_partial=True)
            return spool.chunk(time=None)

        selected = spool.select(**distance)
        windows = spool.chunk(time=3 * step, keep_partial=True)
        chains = (
            chunk(selected, 3),
            chunk(selected, 3, 5),
            chunk(windows.select(**distance), 5, 2),
        )
        for chunked in chains:
            (patch,) = chunked
            # a memory spool's re-chunk also loads the other channels (#1637)
            patch = patch.select(**distance)
            assert patch.get_coord("time") == expected.get_coord("time")
            assert np.array_equal(patch.data, expected.data)

    def test_replan_after_other_selection_mixed_units(self):
        """Patches in metres and feet keep every sample when re-chunked."""
        metres = dc.Patch(
            data=np.ones((6, 2)),
            coords={"distance": np.arange(6.0), "time": np.arange(2.0)},
            dims=("distance", "time"),
        ).set_units(distance="m")
        feet = metres.update_coords(distance_min=6.0).convert_units(distance="ft")
        selected = dc.spool([metres, feet]).select(time=(1, 1))
        chunked = selected.chunk(distance=3, keep_partial=True)
        assert [p.shape[0] for p in chunked] == [3, 3, 3, 3]
        (patch,) = chunked.chunk(distance=None)
        assert np.allclose(patch.get_array("distance"), np.arange(12.0))

    def test_replan_after_other_selection_of_planned_rows(self):
        """Re-chunking a selected plan's outputs keeps the samples they hold."""
        patch = dc.Patch(
            data=np.arange(24.0).reshape(2, 12),
            coords={"distance": np.arange(2.0), "time": np.arange(12.0)},
            dims=("distance", "time"),
        )
        planned = dc.spool([patch]).select(time=(2.5, 10.5)).chunk(distance=None)
        chunked = planned.select(distance=(1, 1)).chunk(time=3, keep_partial=True)
        (out,) = chunked.chunk(time=None)
        assert np.array_equal(out.get_array("time"), np.arange(3.0, 11.0))

    def test_replan_after_other_selection_integer_labels(self):
        """Integer labels chunk when only some rows state where their grid starts."""
        patch = dc.Patch(
            data=np.arange(24.0).reshape(2, 12),
            coords={"distance": np.arange(2), "time": np.arange(12)},
            dims=("distance", "time"),
        )
        planned = dc.spool([patch]).chunk(distance=None)
        later = dc.spool([patch.update_coords(time_min=12)])
        for spool, count in ((planned, 12), (planned + later, 24)):
            chunked = spool.select(distance=(1, 1)).chunk(time=3, keep_partial=True)
            times = np.concatenate([p.get_array("time") for p in chunked])
            assert np.array_equal(times, np.arange(count))

    def test_planned_state_is_derived(self, patches):
        """A chunked spool is a fresh derived catalog, not a mode flag."""
        spool = dc.spool(patches)
        chunked = spool.chunk(time=2)
        assert chunked._catalog is not spool._catalog
        assert isinstance(chunked._catalog.resolver, PlanResolver)


class TestTypeSurface:
    """The collapsed hierarchy: one concrete Spool class."""

    def test_every_spool_is_spool(self, patches, tmp_path):
        """All construction paths yield the same concrete class."""
        live = dc.spool(patches)
        directory = dc.examples.spool_to_directory(
            dc.spool(patches), path=tmp_path / "types_dir"
        )
        dir_spool = dc.spool(directory).update(progress=None)
        file_path = tmp_path / "one.h5"
        dc.write(patches[0], file_path, "dasdae")
        file_spool = dc.spool(file_path)
        for spool in (live, dir_spool, file_spool, live.chunk(time=2)):
            assert type(spool) is dc.Spool


class TestEqualityOverEffectiveRows:
    """Equality compares contents, not representation (2026-07-18)."""

    @pytest.fixture()
    def patch(self):
        """The example patch."""
        return dc.get_example_patch()

    def test_value_trimmed_view_equals_materialized(self, patch):
        """A coordinate-trimmed view equals its union-materialized twin."""
        t = patch.get_coord("time")
        sel = dc.spool([patch]).select(
            time=(t.min() + 10 * t.step, t.min() + 20 * t.step)
        )
        assert sel == sel + dc.spool([])

    @pytest.mark.parametrize("window", [(0, 10), (-10, None)])
    def test_samples_trimmed_view_equals_materialized(self, patch, window):
        """Samples-trimmed views (negative included) equal their twins."""
        sel = dc.spool([patch]).select(time=window, samples=True)
        assert sel == sel + dc.spool([])

    def test_differing_trims_stay_unequal(self, patch):
        """Different windows fold to different envelopes and stay unequal."""
        a = dc.spool([patch]).select(time=(0, 10), samples=True)
        b = dc.spool([patch]).select(time=(0, 11), samples=True)
        assert a != b
        assert a != dc.spool([patch])
