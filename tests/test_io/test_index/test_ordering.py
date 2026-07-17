"""
Tests for the catalog ordering contract (source ordinals).

Patch rows present in (ordinal, patch_id) order: live spools keep
construction order, unions concatenate (dedup keeps first-occurrence
position), and directory archives present in time order (the syncer
renumbers after each sync).
"""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc


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
        import sqlite3

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
        import copy

        dc.write(dc.get_example_patch(), tmp_path / "a.h5", "dasdae")
        spool = dc.spool(tmp_path).update(progress=None)
        assert copy.deepcopy(spool.indexer) is spool.indexer
        spool.indexer.close()
