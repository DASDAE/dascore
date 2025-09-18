"""
Tests for coordinate manager utils.
"""

from __future__ import annotations

import numpy as np
import pytest

from dascore.core.coords import CoordArray, CoordMonotonicArray, CoordRange
from dascore.exceptions import CoordMergeError
from dascore.utils.coordmanager import merge_coord_managers


class TestMergeCoordManagers:
    """Tests for merging coord managers together."""

    def _get_offset_coord_manager(self, cm, from_max=True, **kwargs):
        """Get a new coord manager offset by some amount along a dim."""
        name, value = next(iter(kwargs.items()))
        coord = cm.coord_map[name]
        start = coord.max() if from_max else coord.min()
        attr_name = f"{name}_min"
        new, _ = cm.update_from_attrs({attr_name: start + value})
        return new

    @pytest.fixture(scope="class")
    def conflicting_non_dim_coords(self, cm_basic):
        """Get two coord managers with conflicting non-dimensional coordinates."""
        dist_ax = cm_basic.get_axis("distance")
        rand = np.random.RandomState(42)
        c1 = rand.random(cm_basic.shape[dist_ax])
        c2 = rand.random(c1.shape)

        time = cm_basic.get_coord("time")
        cm1 = cm_basic.update_coords(_bad_coord=("distance", c1))
        cm2 = cm1.update_coords(
            _bad_coord=("distance", c2), time=time + time.coord_range()
        )
        return cm1, cm2

    def test_merge_simple(self, cm_basic):
        """Ensure we can merge simple, contiguous, coordinates together."""
        cm1 = cm_basic
        time = cm1.coord_map["time"]
        cm2 = self._get_offset_coord_manager(cm1, time=time.step)
        out = merge_coord_managers([cm1, cm2], dim="time")
        new_time = out.coord_map["time"]
        assert isinstance(new_time, CoordRange)
        assert new_time.min() == time.min()
        assert new_time.max() == cm2.coord_map["time"].max()

    def test_merge_offset_close_no_snap(self, cm_basic):
        """When the coordinate don't line up, it should produce monotonic Coord."""
        cm1 = cm_basic
        dt = cm1.coord_map["time"].step
        # try a little more than dt
        cm2 = self._get_offset_coord_manager(cm1, time=dt * 1.1)
        out = merge_coord_managers([cm1, cm2], dim="time")
        assert isinstance(out.coord_map["time"], CoordMonotonicArray)
        # try a little less
        cm2 = self._get_offset_coord_manager(cm1, time=dt * 0.9)
        out = merge_coord_managers([cm1, cm2], dim="time")
        assert isinstance(out.coord_map["time"], CoordMonotonicArray)

    def test_merge_offset_overlap(self, cm_basic):
        """Ensure coordinates that have overlap produce Coord Array."""
        cm1 = cm_basic
        dt = cm1.coord_map["time"].step
        cm2 = self._get_offset_coord_manager(cm1, time=-dt * 1.1)
        out = merge_coord_managers([cm1, cm2], dim="time")
        assert isinstance(out.coord_map["time"], CoordArray)

    def test_merge_snap_but_not_needed(self, cm_basic):
        """Specifying a snap tolerance even if coords line up should work."""
        cm1 = cm_basic
        time = cm1.coord_map["time"]
        cm2 = self._get_offset_coord_manager(cm1, time=time.step)
        out = merge_coord_managers([cm1, cm2], dim="time", snap_tolerance=1.3)
        new_time = out.coord_map["time"]
        assert isinstance(new_time, CoordRange)
        assert new_time.min() == time.min()
        assert new_time.max() == cm2.coord_map["time"].max()

    @pytest.mark.parametrize("factor", (1.1, 0.9, 1.3, 1.01, 0))
    def test_merge_snap_when_needed(self, cm_basic, factor):
        """Snap should be applied because when other cm is close expected."""
        cm1 = cm_basic
        time = cm1.coord_map["time"]
        nt = time.step * factor
        cm2 = self._get_offset_coord_manager(cm1, time=nt)
        out = merge_coord_managers([cm1, cm2], dim="time", snap_tolerance=1.3)
        new_time = out.coord_map["time"]
        assert isinstance(new_time, CoordRange)
        assert new_time.min() == time.min()
        new_dim_len = out.shape[out.get_axis("time")]
        expected_end = time.min() + (new_dim_len - 1) * time.step
        assert new_time.max() == expected_end

    @pytest.mark.parametrize("factor", (10, -10, 6, -6))
    def test_merge_raise_snap_too_big(self, cm_basic, factor):
        """When snap is too big, an error should be raised."""
        cm1 = cm_basic
        time = cm1.coord_map["time"]
        nt = time.step * factor
        cm2 = self._get_offset_coord_manager(cm1, time=nt)
        with pytest.raises(CoordMergeError, match="Snap tolerance"):
            merge_coord_managers([cm1, cm2], dim="time", snap_tolerance=1.3)

    def test_different_dims_raises(self, cm_basic):
        """When dimensions differ merge should raise."""
        cm1 = cm_basic
        cm2 = cm1.rename_coord(distance="dist")
        with pytest.raises(CoordMergeError, match="same dimensions"):
            merge_coord_managers([cm1, cm2], "time")
        with pytest.raises(CoordMergeError, match="same dimensions"):
            merge_coord_managers([cm1, cm2], "distance")

    def test_different_units_raises(self, cm_basic):
        """When dimensions differ merge should raise."""
        cm1 = cm_basic
        cm2 = cm1.set_units(distance="furlong")
        with pytest.raises(CoordMergeError, match="share the same units"):
            merge_coord_managers([cm1, cm2], "distance")

    def test_unequal_non_merge_coords(self, cm_basic):
        """When coords that won't be merged arent equal merge should fail."""
        cm1 = cm_basic
        dist = cm1.coord_map["distance"]
        new_dist = dist.update_limits(min=dist.max())
        cm2 = cm1.update(distance=new_dist)
        with pytest.raises(CoordMergeError, match="Non merging coordinates"):
            merge_coord_managers([cm1, cm2], "time")

    def test_unshared_coord_dropped(self, cm_basic):
        """
        When one coord manager has coords and the other doesn't they should
        be dropped.
        """
        cm1 = cm_basic
        cm2 = cm1.update(time2=("time", cm1.get_array("time")))
        out_no_range = merge_coord_managers([cm1, cm2], "time")
        assert "time2" not in out_no_range.coord_map
        out_with_range = merge_coord_managers([cm1, cm2], "time")
        assert "time2" not in out_with_range.coord_map

    def test_slightly_different_dt(self, cm_dt_small_diff):
        """
        Ensure coord managers with slightly different dt can still merge
        but produce uneven sampled dimension.
        """
        cm = cm_dt_small_diff
        coord = cm.coord_map["time"]
        assert coord.sorted

    def test_conflicting_non_dimensional_coords(self, conflicting_non_dim_coords):
        """
        Ensure conflicting non-dimensional coords can be merged if drop_conflict=True,
        Otherwise raise.
        """
        c1, c2 = conflicting_non_dim_coords

        out = merge_coord_managers([c1, c2], dim="time", drop_conflicting=True)
        assert not any([x.startswith("_") for x in out.coord_map])

        with pytest.raises(CoordMergeError, match="cannot be merged"):
            merge_coord_managers([c1, c2], dim="time", drop_conflicting=False)
