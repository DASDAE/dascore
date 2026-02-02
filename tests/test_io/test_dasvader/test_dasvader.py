"""
Tests specific to DASvader format.
"""

import numpy as np
import pytest
from h5py.h5r import Reference

import dascore as dc


class TestDASVader:
    """Test case for dasvader."""

    @pytest.fixture(scope="session")
    def das_vader_patch(self):
        """Get the dasvader patch."""
        return dc.get_example_patch("das_vader_1.jld2")

    def test_all_attrs_resolved(self, das_vader_patch):
        """Ensure all attributes are resolved (no h5py references)."""
        for _attr, value in das_vader_patch.attrs.items():
            assert not isinstance(value, Reference)

    def test_time_matches_julia_epoch(self, das_vader_patch):
        """Ensure Julia epoch offset is correct for htime decoding."""
        expected = np.datetime64("2023-08-24T14:06:17.550", "ms")
        got = das_vader_patch.coords.time.min()
        diff = np.abs(got - expected)
        assert diff <= np.timedelta64(1, "ms")
