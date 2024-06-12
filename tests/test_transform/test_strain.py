"""Tests for converting velocity to strain-rate."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import PatchAttributeError
from dascore.units import get_quantity


class TestStrainRateConversion:
    """Tests for converting velocity to strain-rate."""

    @pytest.fixture(scope="class")
    def patch_strain_rate_default(self, terra15_das_patch):
        """Return the default terra15 converted to strain rate."""
        return terra15_das_patch.velocity_to_strain_rate()

    def test_attrs(self, patch_strain_rate_default):
        """Ensure the attributes were updated with strain_rate."""
        attrs = patch_strain_rate_default.attrs
        assert attrs["data_type"] == "strain_rate"

    def test_data_different(self, patch_strain_rate_default, terra15_das_patch):
        """The data should have changed."""
        data1 = patch_strain_rate_default.data
        data2 = terra15_das_patch.data
        assert not np.all(np.equal(data1, data2))

    def test_raises_on_strain_rate(self, patch_strain_rate_default):
        """It does not make sense to apply this twice."""
        with pytest.raises(PatchAttributeError, match="velocity"):
            _ = patch_strain_rate_default.velocity_to_strain_rate()

    def test_update_units(self, patch_strain_rate_default, terra15_das_patch):
        """Ensure units are updated. See issue #144."""
        new_units = get_quantity(patch_strain_rate_default.attrs.data_units)
        old_units = get_quantity(terra15_das_patch.attrs.data_units)
        assert new_units != old_units
        assert new_units == old_units / get_quantity("m")

    def test_gauge_length_multiple(self, terra15_das_patch):
        """Ensure strain rate multiples > 1 are supported."""
        # Note: this is a bit of a weak test. However, since we just call
        # Patch.differentiate under the hood, and that is better tested
        # for supporting different step sizes, its probably sufficient.
        out1 = terra15_das_patch.velocity_to_strain_rate(gauge_multiple=2)
        assert isinstance(out1, dc.Patch)
        out2 = terra15_das_patch.velocity_to_strain_rate(gauge_multiple=4)
        assert isinstance(out2, dc.Patch)


class TestFdStrainRateConversion:
    """Tests for converting velocity to strain-rate in the forward diff function."""

    @pytest.fixture(scope="class")
    def patch_strain_rate_default(self, terra15_das_patch):
        """Return the default terra15 converted to strain rate."""
        return terra15_das_patch.velocity_to_strain_rate_fd()

    def test_attrs(self, patch_strain_rate_default):
        """Ensure the attributes were updated with strain_rate."""
        attrs = patch_strain_rate_default.attrs
        assert attrs["data_type"] == "strain_rate"

    def test_raises_on_strain_rate(self, patch_strain_rate_default):
        """It does not make sense to apply this twice."""
        with pytest.raises(PatchAttributeError, match="velocity"):
            _ = patch_strain_rate_default.velocity_to_strain_rate_fd()

    def test_update_units(self, patch_strain_rate_default, terra15_das_patch):
        """Ensure units are updated. See issue #144."""
        new_units = get_quantity(patch_strain_rate_default.attrs.data_units)
        old_units = get_quantity(terra15_das_patch.attrs.data_units)
        assert new_units != old_units
        assert new_units == old_units / get_quantity("m")

    def test_even_gauge_multiple(self, terra15_das_patch):
        """Compare output shape with coord shape for even gauge multiple."""
        strain_rate_patch = terra15_das_patch.velocity_to_strain_rate_fd(4)
        assert strain_rate_patch.data.shape == strain_rate_patch.coords.shape

    def test_odd_gauge_multiple(self, terra15_das_patch):
        """Compare output shape with coord shape for odd gauge multiple."""
        strain_rate_patch = terra15_das_patch.velocity_to_strain_rate_fd(5)
        assert strain_rate_patch.data.shape == strain_rate_patch.coords.shape

    @pytest.mark.parametrize("sample", (1, 2, 3, 4, 5))
    def test_shape_different(self, terra15_das_patch, sample):
        """Ensure shape of the output is correctly changed."""
        strain_rate_patch = terra15_das_patch.velocity_to_strain_rate_fd(sample)
        shape_1 = len(strain_rate_patch.coords.get_array("distance"))
        shape_2 = len(terra15_das_patch.coords.get_array("distance"))
        assert shape_1 == shape_2 - sample
