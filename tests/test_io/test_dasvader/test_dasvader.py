"""
Tests specific to DASvader format.
"""

import pytest
from h5py.h5r import Reference

import dascore as dc


class TestDASVader:
    """Test case for dasvader."""

    @pytest.fixture(scope="session")
    def das_vader_patch(self):
        """Get the dasvader patch."""
        return dc.get_example_patch("das_vader_1.jld2")

    def test_all_attrs_resoved(self, das_vader_patch):
        """Ensure all attributes are resolted (no hf references)"""
        for attr, value in das_vader_patch.attrs.items():
            assert not isinstance(value, Reference)
