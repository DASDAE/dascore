"""
Tests for performing aggregations.
"""
import numpy as np
import pytest

from dascore.proc.aggregate import _AGG_FUNCS, aggregate


class TestBasicAggregations:
    """
    Sanity checks for basic aggregations.
    """

    @pytest.fixture(params=list(_AGG_FUNCS))
    def distance_aggregated_patch(self, request, random_patch):
        """Apply all supported aggregations along distance axis."""
        agg = request.param
        return aggregate(random_patch, dim="distance", method=agg)

    def test_dimension_collapsed(self, distance_aggregated_patch):
        """Ensure the aggregate dimension was collapsed to len 1."""
        axis = distance_aggregated_patch.dims.index("distance")
        axis_len = distance_aggregated_patch.data.shape[axis]
        assert axis_len == 1
        dist = distance_aggregated_patch.coords["distance"]
        assert len(dist) == axis_len

    def test_time_aggregate(self, random_patch):
        """Tests for aggregating on time axis."""
        out = random_patch.aggregate(dim="time")
        assert out.attrs["time_max"] == out.attrs["time_min"]
        assert 1 in out.data.shape

    def test_first(self, random_patch):
        """Ensure aggregations can occur"""
        out = random_patch.aggregate(dim="distance", method="first")
        axis = random_patch.dims.index("distance")
        assert out.data.shape[axis] == 1
        assert np.allclose(random_patch.data[0, :], out.data[0, :])

    def test_last(self, random_patch):
        """Ensure aggregations can occur"""
        out = random_patch.aggregate(dim="distance", method="last")
        axis = random_patch.dims.index("distance")
        assert out.data.shape[axis] == 1
        assert np.allclose(random_patch.data[-1, :], out.data[0, :])
