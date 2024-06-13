"""Tests for converting velocity to strain-rate."""

from __future__ import annotations

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


@pytest.fixture(scope="class")
def small_velocity_patch():
    """A small velocity patch with known values."""
    # This is useful for comparing output to easily validated results.
    # ar = np.arrange(100).reshape(10, 10)
    ar = np.array(
        [
            [8, 6, 7, 5],
            [3, 0, 9, 8],
            [5, 4, 4, 4],
            [9, 1, 1, 4],
        ]
    )

    dist = np.arange(4)
    time = np.arange(4)
    coords = {"time": time, "distance": dist}
    dims = ("time", "distance")
    attrs = {"data_type": "velocity", "data_units": "m/s"}

    patch = dc.Patch(data=ar, coords=coords, dims=dims, attrs=attrs)
    return patch


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


class TestFdStrainRateConversion:
    """Tests for converting velocity to strain-rate in the forward diff function."""

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

    def test_even_step_multiple(self, terra15_das_patch):
        """Compare output shape with coord shape for even gauge multiple."""
        strain_rate_patch = terra15_das_patch.staggered_velocity_to_strain_rate(4)
        assert strain_rate_patch.data.shape == strain_rate_patch.coords.shape

    def test_odd_step_multiple(self, terra15_das_patch):
        """Compare output shape with coord shape for odd gauge multiple."""
        strain_rate_patch = terra15_das_patch.staggered_velocity_to_strain_rate(5)
        assert strain_rate_patch.data.shape == strain_rate_patch.coords.shape

    @pytest.mark.parametrize("sample", (1, 2, 3, 4, 5))
    def test_shape_different(self, terra15_das_patch, sample):
        """Ensure shape of the output is correctly changed."""
        strain_rate_patch = terra15_das_patch.staggered_velocity_to_strain_rate(sample)
        shape_1 = len(strain_rate_patch.coords.get_array("distance"))
        shape_2 = len(terra15_das_patch.coords.get_array("distance"))
        assert shape_1 == shape_2 - sample


class TestCompareStrainRateConversion:
    """Tests for comparing the two strain-rate functions."""

    def test_compare_linear_patch(self, linear_velocity_patch):
        """For a simple analytical function, both functions have equal results."""
        step = linear_velocity_patch.get_coord("distance").step
        expected = 1.0 / step
        out1 = linear_velocity_patch.staggered_velocity_to_strain_rate()
        assert np.allclose(out1.data, expected)
        out2 = linear_velocity_patch.velocity_to_strain_rate()
        assert np.allclose(out2.data, expected)
