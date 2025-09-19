"""Tests for performing aggregations."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

import dascore
import dascore as dc
from dascore.exceptions import ParameterError
from dascore.proc.aggregate import _AGG_FUNCS
from dascore.utils.misc import broadcast_for_index


class TestBasicAggregations:
    """Sanity checks for basic aggregations."""

    def test_first(self, random_patch):
        """Ensure aggregations can occur."""
        out = random_patch.aggregate(dim="distance", method="first")
        assert out.ndim == random_patch.ndim
        axis = random_patch.get_axis("distance")
        inds = broadcast_for_index(len(random_patch.data.shape), axis, 0)
        assert np.allclose(random_patch.data[inds].flatten(), out.data.flatten())

    def test_last(self, random_patch):
        """Ensure aggregations can occur."""
        out = random_patch.aggregate(dim="time", method="last")
        axis = random_patch.get_axis("time")
        inds = broadcast_for_index(len(random_patch.data.shape), axis, -1)
        assert np.allclose(random_patch.data[inds].flatten(), out.data.flatten())

    def test_no_dim(self, random_patch):
        """Ensure no dimension argument behaves like numpy."""
        out = random_patch.aggregate(method="mean")
        expected = np.mean(random_patch.data, keepdims=True)
        npt.assert_allclose(out.data, expected, rtol=1e-8, atol=0)

    def test_dtype_of_coord_unchanged(self, random_patch):
        """The dtype of the coord should not change."""
        out = random_patch.aggregate(dim="time", method="median")
        coord_new = out.get_coord("time")
        coord_old = random_patch.get_coord("time")
        assert coord_old.dtype == coord_new.dtype

    @pytest.mark.parametrize("method", list(_AGG_FUNCS))
    def test_named_aggregations(self, random_patch, method):
        """Simply run the named aggregations."""
        patch = getattr(random_patch, method)(dim="distance")
        assert isinstance(patch, dc.Patch)

    def test_dim_reduce_squeeze(self, random_patch):
        """Ensure the old dimension can be squeezed out."""
        out = random_patch.aggregate(dim="time", method="mean", dim_reduce="squeeze")
        assert "time" not in out.dims
        assert out.ndim == 1

    def test_dim_reduce_mean(self, random_patch):
        """Ensure the mean value can be left on the coord."""
        out = random_patch.aggregate(dim="time", method="mean", dim_reduce="mean")
        new_time = out.get_coord("time")
        assert len(new_time) == 1

    def test_dim_reduce_mean_time_delta(self, random_patch):
        """Ensure the mean value can be left on the coord."""
        time = random_patch.get_coord("time")
        dt = dc.to_timedelta64(time.values)
        patch = random_patch.update_coords(time=dt)
        out = patch.aggregate(dim="time", method="mean", dim_reduce="mean")
        new_time = out.get_coord("time")
        assert len(new_time) == 1

    def test_invalid_dim_reduce(self, random_patch):
        """Ensure an invalid dim_reduce argument raises."""
        msg = "dim_reduce must be"
        with pytest.raises(ParameterError, match=msg):
            random_patch.aggregate(dim="time", dim_reduce="invalid")

    def test_dim_reduce_first(self, random_patch):
        """Ensure first takes the first value"""
        out = random_patch.aggregate(dim="time", method="mean", dim_reduce="first")
        new_time = out.get_coord("time")
        assert len(new_time) == 1
        assert new_time[0] == out.get_array("time")[0]

    def test_dim_reduce_distance(self, random_patch):
        """Ensure non-time dims also work."""
        out = random_patch.aggregate(dim="distance", method="mean", dim_reduce=np.var)
        assert "distance" in out.dims
        expected = np.var(random_patch.get_array("distance"))
        assert out.get_coord("distance").values == expected

    def test_any(self, random_patch):
        """Ensure any works."""
        out = (random_patch > 0.5).any(dim="time")
        assert isinstance(out, dascore.Patch)
        assert np.issubdtype(out.dtype, np.bool_)


class TestApplyOperators:
    """Ensure aggregated patches can be used as operators for arithmetic."""

    def test_complete_reduction(self, random_patch):
        """Ensure a patch with complete reduction works."""
        agg = random_patch.min(None)
        assert np.all(agg.data == np.min(random_patch.data))
        # Ensure broadcasting works with reduced data
        out = random_patch - agg
        assert isinstance(out, dc.Patch)
        assert np.allclose(random_patch.data - agg.data, out.data)

    def test_single_reduction(self, random_patch):
        """Ensure a single patch reduced also works with broadcasting."""
        agg = random_patch.first("time")
        out1 = random_patch + agg
        assert isinstance(out1, dc.Patch)
