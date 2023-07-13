"""
Module for testing differentiation of patches.
"""
import numpy as np
import pytest

from dascore.transform.differentiate import differentiate
from dascore.units import get_quantity
from dascore.utils.time import to_float


class TestDifferentiateOrder2:
    """Simple tests for differentiation using order=2 (numpy.grad)"""

    def test_default_case(self, random_patch):
        """Ensure calling differentiate with all dims works."""
        patch = random_patch
        for dim in patch.dims:  # test all dimensions.
            out = random_patch.tran.differentiate(dim=dim)
            axis = patch.dims.index(dim)
            # ensure the differentiation was applied
            sampling = to_float(patch.get_coord(dim).step)
            expected = np.gradient(patch.data, sampling, axis=axis, edge_order=2)
            assert np.allclose(out.data, expected)
            assert not np.any(np.isnan(out.data))

    def test_units(self, random_patch):
        """Ensure data units are updated but coord units shouldn't change."""
        patch = random_patch.set_units("m/s")
        out = patch.tran.differentiate(dim="time")
        expected_units = get_quantity("m/s^2")
        # none of the coordinates/ units should have changed.
        assert out.coords == patch.coords
        # but the data units should now be divided by s
        assert get_quantity(out.attrs.data_units) == expected_units

    def test_all_axis(self, random_patch):
        """Ensure we can diff along all axis"""
        patch = random_patch.set_units("ohm")
        expected_units = get_quantity("ohm / (m * s)")
        out = differentiate(patch, dim=None)
        assert get_quantity(out.attrs.data_units) == expected_units

    def test_uneven_spacing(self, wacky_dim_patch):
        """Ensure we can diff over uneven dimensions."""
        patch = wacky_dim_patch.new(data=np.ones_like(wacky_dim_patch.data))
        out = differentiate(patch, "time")
        spacing = to_float(out.get_coord("time").data)
        ax = patch.dims.index("time")
        expected = np.gradient(patch.data, spacing, axis=ax, edge_order=2)
        assert np.allclose(expected, out.data, rtol=0.01)


class TestDifferentiateWFindiff:
    """Ensure differentiation with findiff works as well."""

    @pytest.fixture(scope="class")
    def linear_patch(self, random_patch):
        """A patch increasing linearly along distance dimension."""
        ind = random_patch.dims.index("distance")
        dist_len = random_patch.shape[ind]
        new = np.arange(dist_len, dtype=np.float64)
        new_data = np.broadcast_to(new[:, None], random_patch.shape)
        return random_patch.new(data=new_data)

    def test_default_case(self, random_patch):
        """Diff with order != 2 on each dimension."""
        pytest.importorskip("findiff")
        patch = random_patch
        for dim in patch.dims:  # test all dimensions.
            for order in [4, 6, 8, 10]:
                differentiate(random_patch, dim=dim, order=order)

    def test_different_orders(self, linear_patch):
        """Different order stencils should be approx equal with simple data."""
        pytest.importorskip("findiff")
        patch = linear_patch
        p1 = patch.tran.differentiate("distance", order=2)
        p2 = patch.tran.differentiate("distance", order=4)
        p3 = patch.tran.differentiate("distance", order=6)
        p4 = patch.tran.differentiate("distance", order=8)
        assert np.allclose(p1.data, p2.data)
        assert np.allclose(p2.data, p3.data)
        assert np.allclose(p3.data, p4.data)

    def test_diff_all_dims(self, linear_patch):
        """Ensure findiff can diff all dims."""
        pytest.importorskip("findiff")
        patch = linear_patch
        p1 = patch.tran.differentiate("distance", order=2)
        p2 = patch.tran.differentiate("distance", order=4)
        p3 = patch.tran.differentiate("distance", order=6)
        p4 = patch.tran.differentiate("distance", order=8)
        assert np.allclose(p1.data, p2.data)
        assert np.allclose(p2.data, p3.data)
        assert np.allclose(p3.data, p4.data)
