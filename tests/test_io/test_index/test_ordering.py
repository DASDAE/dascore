"""
Tests for the catalog ordering contract (source ordinals).

Patch rows present in (ordinal, patch_id) order: live spools keep
construction order, unions concatenate (dedup keeps first-occurrence
position), and directory archives present in time order (the syncer
renumbers after each sync).
"""

from __future__ import annotations

import copy
import sqlite3

import numpy as np
import pytest

import dascore as dc
from dascore.io.index.indexer import DBDirectoryIndexer


@pytest.fixture(scope="module")
def three_patches():
    """Three time-contiguous example patches."""
    t0 = np.datetime64("2020-01-01", "ns")
    p1 = dc.get_example_patch(time_min=t0)
    time = p1.get_coord("time")
    p2 = dc.get_example_patch(time_min=time.max() + time.step)
    p3 = dc.get_example_patch(time_min=p2.get_coord("time").max() + time.step)
    return p1, p2, p3


class TestLiveSpoolOrder:
    """Patch-list spools keep construction order on every path."""

    def test_out_of_time_order_kept(self, three_patches):
        """Construction order wins even when it disagrees with time order."""
        p1, p2, p3 = three_patches
        spool = dc.spool([p3, p1, p2])
        # the tuple fast path
        loaded = list(spool)
        assert loaded[0] is p3 and loaded[1] is p1 and loaded[2] is p2
        # the catalog relation presents the same order
        df = spool.get_contents()
        expected = [p.get_coord("time").min() for p in (p3, p1, p2)]
        assert list(df["time_min"]) == expected
        # and indexing after realization still agrees
        assert spool[0] is p3

    def test_selection_preserves_relative_order(self, three_patches):
        """A narrowed view keeps the surviving rows in spool order."""
        p1, p2, p3 = three_patches
        spool = dc.spool([p3, p1, p2])
        t3 = p3.get_coord("time").min()
        t1 = p1.get_coord("time").min()
        selected = spool.select(time=(min(t1, t3), None))
        df = selected.get_contents()
        assert list(df["time_min"])[:2] == [t3, t1]


class TestUnionOrder:
    """Combined spools are list concatenation, deduped dict-merge style."""

    def test_concatenation_order(self, three_patches):
        """(a + b) presents a's rows then b's."""
        p1, p2, p3 = three_patches
        combined = dc.spool([p3]) + dc.spool([p1, p2])
        loaded = list(combined)
        assert loaded[0] is p3 and loaded[1] is p1 and loaded[2] is p2

    def test_dedup_keeps_first_position(self, three_patches):
        """A patch in both members keeps its first position, appears once."""
        p1, p2, p3 = three_patches
        combined = dc.spool([p1, p2]) + dc.spool([p2, p3])
        assert len(combined) == 3
        loaded = list(combined)
        assert loaded[0] is p1 and loaded[1] is p2 and loaded[2] is p3

    def test_union_of_union_order(self, three_patches):
        """Order survives a second union (export/re-ingest round trip)."""
        p1, p2, p3 = three_patches
        combined = (dc.spool([p3]) + dc.spool([p2])) + dc.spool([p1])
        loaded = list(combined)
        assert loaded[0] is p3 and loaded[1] is p2 and loaded[2] is p1


class TestDirectoryOrder:
    """File archives present in time order, maintained across syncs."""

    def test_time_order_disagrees_with_name_order(self, tmp_path):
        """Presentation follows patch time, not file names or walk order."""
        t0 = np.datetime64("2020-01-01", "ns")
        early = dc.get_example_patch(time_min=t0)
        late = dc.get_example_patch(time_min=t0 + np.timedelta64(3600, "s"))
        dc.write(late, tmp_path / "a_late.h5", "dasdae")
        dc.write(early, tmp_path / "z_early.h5", "dasdae")
        spool = dc.spool(tmp_path).update(progress=None)
        df = spool.get_contents()
        assert df["time_min"].is_monotonic_increasing

    def test_update_interleaves_new_files_by_time(self, tmp_path):
        """A later-indexed but earlier-in-time file sorts into place."""
        t0 = np.datetime64("2020-01-01", "ns")
        mid = dc.get_example_patch(time_min=t0 + np.timedelta64(1800, "s"))
        late = dc.get_example_patch(time_min=t0 + np.timedelta64(3600, "s"))
        dc.write(mid, tmp_path / "mid.h5", "dasdae")
        dc.write(late, tmp_path / "late.h5", "dasdae")
        dc.spool(tmp_path).update(progress=None)  # build the index
        early = dc.get_example_patch(time_min=t0)
        dc.write(early, tmp_path / "early.h5", "dasdae")
        updated = dc.spool(tmp_path).update(progress=None)
        df = updated.get_contents()
        assert len(df) == 3
        assert df["time_min"].is_monotonic_increasing
        assert df["time_min"].iloc[0] == early.get_coord("time").min()


class TestIndexVersionRebuild:
    """Old-version index files rebuild automatically (disposable cache)."""

    def test_version_mismatch_rebuilds(self, tmp_path):
        """An index of another schema version is replaced, not fatal."""
        patch = dc.get_example_patch()
        dc.write(patch, tmp_path / "a.h5", "dasdae")
        spool = dc.spool(tmp_path).update(progress=None)
        index_path = spool.indexer.index_path
        spool.indexer.close()
        # simulate an index written by another (older/newer) schema version;
        # close the connection explicitly (the sqlite3 context manager only
        # manages transactions) or Windows cannot unlink the file below.
        con = sqlite3.connect(index_path)
        try:
            con.execute("UPDATE meta_data SET index_version = 1")
            con.commit()
        finally:
            con.close()
        reopened = dc.spool(tmp_path).update(progress=None)
        assert len(reopened) == 1
        reopened.indexer.close()

    def test_indexer_deepcopy_shares_instance(self, tmp_path):
        """Derived spools share the indexer (and its live DB connection)."""
        dc.write(dc.get_example_patch(), tmp_path / "a.h5", "dasdae")
        spool = dc.spool(tmp_path).update(progress=None)
        assert copy.deepcopy(spool.indexer) is spool.indexer
        spool.indexer.close()


class TestSortNonHotCoords:
    """Sorting by coords without patches-table columns (2026-07-18 F3)."""

    @pytest.fixture()
    def renamed_spool(self):
        """Two patches whose time coord is renamed (not a hot column)."""
        p = dc.get_example_patch().rename_coords(time="event_time")
        t = p.get_coord("event_time")
        span = t.max() - t.min() + t.step
        p2 = p.update_coords(event_time=t.data + span)
        return dc.spool([p2, p])  # deliberately out of order

    @pytest.mark.parametrize("key", ["event_time", "event_time_min"])
    def test_sort_renamed_datetime_coord(self, renamed_spool, key):
        """A renamed datetime coord sorts through coord_defs."""
        srt = renamed_spool.sort(key)
        mins = [x.get_coord("event_time").min() for x in srt]
        assert mins == sorted(mins)
        # the realized relation agrees
        contents = srt.get_contents()
        assert contents["event_time_min"].is_monotonic_increasing

    def test_sort_non_hot_numeric_coord(self):
        """A numeric aux coord sorts through coord_defs."""
        p = dc.get_example_patch()
        n = p.shape[p.get_axis("distance")]
        lo = p.update_coords(sensor=("distance", np.arange(n, dtype=float)))
        hi = p.update_coords(sensor=("distance", np.arange(n, dtype=float) + 1000))
        srt = dc.spool([hi, lo]).sort("sensor")
        mins = [x.get_coord("sensor").min() for x in srt]
        assert mins == sorted(mins)

    def test_sort_string_coord(self):
        """A string coord sorts lexicographically through coord_defs."""
        p = dc.get_example_patch()
        n = p.shape[p.get_axis("distance")]
        pa = p.update_coords(station=("distance", np.array(["a"] * n)))
        pb = p.update_coords(station=("distance", np.array(["b"] * n)))
        srt = dc.spool([pb, pa]).sort("station")
        firsts = [x.get_coord("station").values[0] for x in srt]
        assert firsts == ["a", "b"]

    def test_hot_coords_still_sort(self):
        """time/distance keep the cached patches-column path."""
        p = dc.get_example_patch()
        t = p.get_coord("time")
        p2 = p.update_coords(time_min=t.max() + t.step)
        srt = dc.spool([p2, p]).sort("time")
        mins = [x.get_coord("time").min() for x in srt]
        assert mins == sorted(mins)


class TestInterleavedSourceOrder:
    """Directory time order across interleaved multi-patch files (F4)."""

    def test_multi_patch_file_straddles_another(self, tmp_path):
        """A patch between two patches of another file presents in order."""
        p0 = dc.get_example_patch()
        t = p0.get_coord("time")
        span = t.max() - t.min() + t.step
        p1 = p0.update_coords(time_min=t.min() + span)
        p2 = p0.update_coords(time_min=t.min() + 2 * span)
        dc.write(dc.spool([p0, p2]), tmp_path / "a.h5", "DASDAE")
        dc.write(p1, tmp_path / "b.h5", "DASDAE")
        spool = dc.spool(tmp_path).update(progress=None)
        contents = spool.get_contents()
        assert contents["time_min"].is_monotonic_increasing
        mins = [x.get_coord("time").min() for x in spool]
        assert mins == sorted(mins)
        # windows and sorting stay consistent with the presentation
        assert spool[1:2][0].get_coord("time").min() == mins[1]
        assert spool.sort("distance").get_contents().shape[0] == 3

    def test_default_order_is_not_view_state(self, tmp_path):
        """The presentation contract does not make a root a view."""
        dc.write(dc.get_example_patch(), tmp_path / "a.h5", "DASDAE")
        spool = dc.spool(tmp_path).update(progress=None)
        assert spool.update() is not None  # root update allowed


class TestInterruptedInitialUpdate:
    """The initial-update marker only sets after renumbering (F6)."""

    def test_interruption_before_renumber_recovers(self, tmp_path):
        """A crash after write_sources still renumbers on the retry."""
        p0 = dc.get_example_patch()
        t = p0.get_coord("time")
        span = t.max() - t.min() + t.step
        late = p0.update_coords(time_min=t.min() + span)
        # walk order (file names) disagrees with time order
        dc.write(late, tmp_path / "a_late.h5", "DASDAE")
        dc.write(p0, tmp_path / "b_early.h5", "DASDAE")

        class _InterruptedError(RuntimeError):
            pass

        indexer = DBDirectoryIndexer(tmp_path)
        original = type(indexer._backend).renumber_ordinals_by_time

        def _boom(self):
            raise _InterruptedError

        type(indexer._backend).renumber_ordinals_by_time = _boom
        try:
            with pytest.raises(_InterruptedError):
                indexer.ensure_updated()
        finally:
            type(indexer._backend).renumber_ordinals_by_time = original
        del indexer  # simulate the process dying after write_sources

        # a fresh open must not treat the interrupted update as done
        spool = dc.spool(tmp_path).update(progress=None)
        mins = list(spool.get_contents()["time_min"])
        assert mins == sorted(mins)
        sources = spool._catalog.backend.get_sources()
        by_ordinal = sources.sort_values("ordinal")["source_path"].tolist()
        assert [p.split("/")[-1] for p in by_ordinal] == [
            "b_early.h5",
            "a_late.h5",
        ]


class TestDefaultOrderThroughUnion:
    """Directory presentation order survives combining (round-5)."""

    @pytest.fixture()
    def interleaved_dir_spool(self, tmp_path):
        """A directory whose multi-patch file straddles another file."""
        p0 = dc.get_example_patch()
        t = p0.get_coord("time")
        span = t.max() - t.min() + t.step
        p1 = p0.update_coords(time_min=t.min() + span)
        p2 = p0.update_coords(time_min=t.min() + 2 * span)
        dc.write(dc.spool([p0, p2]), tmp_path / "a.h5", "DASDAE")
        dc.write(p1, tmp_path / "b.h5", "DASDAE")
        return dc.spool(tmp_path).update(progress=None)

    def test_empty_union_keeps_order_and_equality(self, interleaved_dir_spool):
        """Adding an empty spool preserves contents, order, and equality."""
        source = interleaved_dir_spool
        combined = source + dc.spool([])
        want = [x.get_coord("time").min() for x in source]
        got = [x.get_coord("time").min() for x in combined]
        assert got == want
        assert combined == source

    def test_live_append_keeps_directory_prefix(self, interleaved_dir_spool):
        """A live operand appends after the directory's presented rows."""
        source = interleaved_dir_spool
        later = dc.get_example_patch(time_min="2030-01-01")
        combined = source + dc.spool([later])
        mins = [x.get_coord("time").min() for x in combined]
        assert mins[:3] == [x.get_coord("time").min() for x in source]
        assert len(mins) == 4

    def test_non_interleaved_union_still_dedups(self, tmp_path):
        """Ordinary archives keep record-grain transfer and dedup."""
        p0 = dc.get_example_patch()
        t = p0.get_coord("time")
        span = t.max() - t.min() + t.step
        dc.write(p0, tmp_path / "a.h5", "DASDAE")
        dc.write(p0.update_coords(time_min=t.min() + span), tmp_path / "b.h5", "DASDAE")
        spool = dc.spool(tmp_path).update(progress=None)
        assert len(spool + spool) == len(spool)


class TestMissingTimeSortsLast:
    """Rows without a value sort last under any order (round-5)."""

    def test_directory_no_time_patch_presents_last(self, tmp_path):
        """A distance-only patch follows every time-bearing patch."""
        timed = dc.get_example_patch().update_attrs(tag="time")
        no_time = timed.mean("time").squeeze().update_attrs(tag="no_time")
        dc.write(timed, tmp_path / "a_time.h5", "DASDAE")
        dc.write(no_time, tmp_path / "b_no_time.h5", "DASDAE")
        spool = dc.spool(tmp_path).update(progress=None)
        assert list(spool.get_contents()["tag"]) == ["time", "no_time"]

    def test_sort_puts_missing_values_last(self):
        """Explicit sort also presents value-less rows last."""
        timed = dc.get_example_patch().update_attrs(tag="a_time")
        no_time = timed.mean("time").squeeze().update_attrs(tag="b_no_time")
        spool = dc.spool([no_time, timed]).sort("time")
        assert [x.attrs.tag for x in spool] == ["a_time", "b_no_time"]
