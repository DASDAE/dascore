"""Tests for performing aggregations."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.proc.aggregate import _AGG_FUNCS
from dascore.utils.misc import broadcast_for_index


class TestBasicAggregations:
    """Sanity checks for basic aggregations."""

    def test_first(self, random_patch):
        """Ensure aggregations can occur."""
        out = random_patch.aggregate(dim="distance", method="first")
        assert out.ndim == random_patch.ndim
        axis = random_patch.dims.index("distance")
        inds = broadcast_for_index(len(random_patch.data.shape), axis, 0)
        assert np.allclose(random_patch.data[inds].flatten(), out.data.flatten())

    def test_last(self, random_patch):
        """Ensure aggregations can occur."""
        out = random_patch.aggregate(dim="time", method="last")
        axis = random_patch.dims.index("time")
        inds = broadcast_for_index(len(random_patch.data.shape), axis, -1)
        assert np.allclose(random_patch.data[inds].flatten(), out.data.flatten())

    def test_no_dim(self, random_patch):
        """Ensure no dimension argument behaves like numpy."""
        out = random_patch.aggregate(method="mean")
        assert np.all(out.data == np.mean(random_patch.data, keepdims=True))

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


class TestApplyOperators:
    """Ensure aggregated patches can be used as operators for arithmetic."""

    def test_complete_reduction(self, random_patch):
        """Ensure a patch with complete reduction works."""
        agg = random_patch.min(None)
        assert np.all(agg.data == np.min(random_patch.data))
        # Ensure broadcasting works with reduced data
        out = random_patch - agg
        assert isinstance(out, dc.Patch)
        assert np.allclose(random_patch.data - agg, out.data)

    def test_single_reduction(self, random_patch):
        """Ensure a single patch reduced also works with broadcasting."""
        agg = random_patch.first("time")
        out1 = random_patch + agg
        assert isinstance(out1, dc.Patch)
