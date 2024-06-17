"""Tests for converting velocity to strain-rate."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import PatchAttributeError
from dascore.units import get_quantity


@pytest.fixture(scope="class")
def linear_velocity_patch(random_patch):
    """A patch increasing linearly along distance dimension."""
    ind = random_patch.dims.index("distance")
    dist_len = random_patch.shape[ind]
    new = np.arange(dist_len, dtype=np.float64)
    new_data: np.ndarray = np.broadcast_to(new[:, None], random_patch.shape)
    new_attrs = {"data_type": "velocity", "data_units": "m/s"}
    # make distance and odd, non 1, value to test normalization
    dist = random_patch.get_array("distance")
    out = random_patch.new(data=new_data, attrs=new_attrs)
    return out.update_coords(distance=dist * 1.23)


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

    def test_step_multiple(self, terra15_das_patch):
        """Ensure strain rate multiples > 1 are supported."""
        # Note: this is a bit of a weak test. However, since we just call
        # Patch.differentiate under the hood, and that is better tested
        # for supporting different step sizes, its probably sufficient.
        out1 = terra15_das_patch.velocity_to_strain_rate(step_multiple=2)
        assert isinstance(out1, dc.Patch)
        out2 = terra15_das_patch.velocity_to_strain_rate(step_multiple=4)
        assert isinstance(out2, dc.Patch)

    def test_gauge_multiple_deprecated(self, terra15_das_patch):
        """Ensure using gauge_multiple issues deprecation warning."""
        with pytest.warns(DeprecationWarning):
            out1 = terra15_das_patch.velocity_to_strain_rate(gauge_multiple=1)
        out2 = terra15_das_patch.velocity_to_strain_rate(step_multiple=1)
        assert out1.equals(out2)

    def test_no_data_units(self, terra15_das_patch):
        """Ensure a patch with no data units still works."""
        patch = terra15_das_patch.update_attrs(data_units="").velocity_to_strain_rate()
        assert not patch.attrs.data_units

    def test_linear_patch(self, linear_velocity_patch):
        """Test conversion of analytical function."""
        step_mults = [1, 2, 3]
        order = [2, 4, 6]
        for step_mult, order in itertools.product(step_mults, order):
            if order > 2:  # Need findiff for orders > 2
                pytest.importorskip("findiff")
            step = linear_velocity_patch.get_coord("distance").step
            expected = 1.0 / step
            out1 = linear_velocity_patch.velocity_to_strain_rate(
                step_multiple=step_mult, order=order
            )
            assert np.allclose(out1.data, expected)


class TestStaggeredStrainRateConversion:
    """Tests for converting velocity to strain-rate with staggered function."""

    @pytest.fixture(scope="class")
    def patch_strain_rate_default(self, terra15_das_patch):
        """Return the default terra15 converted to strain rate."""
        return terra15_das_patch.staggered_velocity_to_strain_rate()

    def test_attrs(self, patch_strain_rate_default):
        """Ensure the attributes were updated with strain_rate."""
        attrs = patch_strain_rate_default.attrs
        assert attrs["data_type"] == "strain_rate"

    def test_raises_on_strain_rate(self, patch_strain_rate_default):
        """It does not make sense to apply this twice."""
        with pytest.raises(PatchAttributeError, match="velocity"):
            _ = patch_strain_rate_default.staggered_velocity_to_strain_rate()

    def test_update_units(self, patch_strain_rate_default, terra15_das_patch):
        """Ensure units are updated. See issue #144."""
        new_units = get_quantity(patch_strain_rate_default.attrs.data_units)
        old_units = get_quantity(terra15_das_patch.attrs.data_units)
        assert new_units != old_units
        assert new_units == old_units / get_quantity("m")

    def test_no_data_units(self, terra15_das_patch):
        """Ensure a patch with no data units still works."""
        patch = terra15_das_patch.update_attrs(
            data_units=""
        ).staggered_velocity_to_strain_rate()
        assert not patch.attrs.data_units

    def test_linear_patch(self, linear_velocity_patch):
        """Test conversion of analytical function."""
        step = linear_velocity_patch.get_coord("distance").step
        expected = 1.0 / step
        out1 = linear_velocity_patch.staggered_velocity_to_strain_rate()
        assert np.allclose(out1.data, expected)
