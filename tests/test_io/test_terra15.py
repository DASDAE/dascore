"""
Tests for reading terra15 file formats.
"""

import xarray as xr

from dfs.constants import REQUIRED_DAS_ATTRS


class TestReadTerra15:
    """Tests for reading the terra15 format."""

    def test_type(self, terra15_das):
        """Ensure the expected type is returned."""
        assert isinstance(terra15_das, xr.DataArray)

    def test_attributes(self, terra15_das):
        """Ensure the expected attrs exist in array."""
        attrs = terra15_das.attrs
        expected_attrs = {"dT", "dx", "nx", "nT", "recorder_id"}
        assert set(expected_attrs).issubset(set(attrs))

    def test_has_required_attrs(self, terra15_das):
        """ "Ensure the required das attrs are found"""
        assert set(REQUIRED_DAS_ATTRS).issubset(set(terra15_das.attrs))
