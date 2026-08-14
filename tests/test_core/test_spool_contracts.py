"""
Tests for the unified Spool's equality and update contracts.

Equality is over rows, never backends: order-sensitive, metadata-level,
with pending residual selections included. update() is case-based:
directory spools sync, file spools rescan, in-memory spools are
trivially current, and combined spools with file rows raise.
"""

from __future__ import annotations

import pytest

import dascore as dc
import dascore.core.spool as spool_module
from dascore.exceptions import InvalidSpoolError
from dascore.io.index.planned import PlanResolver


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

    def test_removed_names_gone(self):
        """The old concrete class names are deleted outright."""
        for name in ("MemorySpool", "DirectorySpool", "FileSpool"):
            assert not hasattr(spool_module, name)
        with pytest.raises(ImportError):
            from dascore.clients.dirspool import DirectorySpool  # noqa

    def test_live_patch_predicate(self, patches, tmp_path):
        """has_live_patches distinguishes memory content, not class."""
        assert dc.spool(patches).has_live_patches
        file_path = tmp_path / "pred.h5"
        dc.write(patches[0], file_path, "dasdae")
        assert not dc.spool(file_path).has_live_patches
        mixed = dc.spool(file_path) + dc.spool(patches[:1])
        assert mixed.has_live_patches


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
