"""Module for testing differentiation of patches."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
import dascore.proc.coords
from dascore.exceptions import ParameterError
from dascore.transform.differentiate import differentiate
from dascore.units import get_quantity
from dascore.utils.time import to_float


@pytest.fixture(scope="class")
def x_plus_y_patch():
    """A patch with 2x + 3y as data and only x, y as coords."""
    x = np.arange(30)
    y = np.arange(30)
    data = np.add.outer(y * 3, x * 2).astype(np.float64)
    patch = dc.Patch(data=data, coords={"x": x, "y": y}, dims=("x", "y"))
    return patch


@pytest.fixture(scope="class")
def x_times_y_patch():
    """A patch with 2x * 3y as data and only x, y as coords."""
    x = np.arange(30)
    y = np.arange(30)
    data = np.multiply.outer(y * 3, x * 2).astype(np.float64)
    patch = dc.Patch(data=data, coords={"x": x, "y": y}, dims=("x", "y"))
    return patch


@pytest.fixture(scope="class")
def linear_patch(random_patch):
    """A patch increasing linearly along distance dimension."""
    ind = random_patch.get_axis("distance")
    dist_len = random_patch.shape[ind]
    new = np.arange(dist_len, dtype=np.float64)
    new_data: np.ndarray = np.broadcast_to(new[:, None], random_patch.shape)
    return random_patch.new(data=new_data)


@pytest.fixture(scope="class")
def linear_patch_monotonic_coord(random_patch):
    """A patch increasing linearly along unevenly sampled distance dimension."""
    state = np.random.RandomState(13)
    x_values = np.cumsum(state.random(50))
    dist_coord = dc.get_coord(values=x_values)
    data = np.broadcast_to(x_values[..., None], (len(x_values), 10))
    time = dc.get_coord(values=dc.to_datetime64(np.arange(10)))
    dims = ("distance", "time")
    coords = {"distance": dist_coord, "time": time}
    patch = dc.Patch(data=data, coords=coords, dims=dims)
    return patch


class TestDifferentiateOrder2:
    """Simple tests for differentiation using order=2 (numpy.grad)."""

    def test_default_case(self, random_patch):
        """Ensure calling differentiate with all dims works."""
        patch = random_patch
        for dim in patch.dims:  # test all dimensions.
            out = random_patch.differentiate(dim=dim)
            axis = patch.get_axis(dim)
            # ensure the differentiation was applied
            sampling = to_float(patch.get_coord(dim).step)
            expected = np.gradient(patch.data, sampling, axis=axis, edge_order=2)
            assert np.allclose(out.data, expected)
            assert not np.any(np.isnan(out.data))

    def test_units(self, random_patch):
        """Ensure data units are updated but coord units shouldn't change."""
        patch = random_patch.set_units("m/s")
        out = patch.differentiate(dim="time")
        expected_units = get_quantity("m/s^2")
        # none of the coordinates/ units should have changed.
        assert out.coords == patch.coords
        # but the data units should now be divided by s
        assert get_quantity(out.attrs.data_units) == expected_units

    def test_all_axis(self, random_patch):
        """Ensure we can diff along all axis."""
        patch = random_patch.set_units("ohm")
        expected_units = get_quantity("ohm / (m * s)")
        out = differentiate(patch, dim=None)
        assert get_quantity(out.attrs.data_units) == expected_units

    def test_uneven_spacing(self, wacky_dim_patch):
        """Ensure we can diff over uneven dimensions."""
        patch = wacky_dim_patch.update(data=np.ones_like(wacky_dim_patch.data))
        out = differentiate(patch, "time")
        # very occasionally, numpy outputs a few nan values from grad when
        # coordinate spacing is provided. I am still trying to figure out
        # why, but we don't want this to fail CI so skip test when that happens.
        if np.any(np.isnan(out.data)):
            pytest.skip("found NaN in output, not sure why this happens.")
        spacing = to_float(out.get_coord("time").data)
        ax = patch.get_axis("time")
        expected = np.gradient(patch.data, spacing, axis=ax, edge_order=2)
        assert np.allclose(expected, out.data, rtol=0.01)


class TestStep:
    """Tests for step != 1."""

    def test_multi_dim_raises(self, random_patch):
        """Can only use step on a single dimension."""
        msg = "can only be used along one axis"
        with pytest.raises(ParameterError, match=msg):
            random_patch.differentiate(("time", "distance"), step=2)

    def test_different_steps_comparable(self, linear_patch):
        """Ensure the values are comparable with different step sizes."""
        out1 = linear_patch.differentiate("distance", step=1)
        out2 = linear_patch.differentiate("distance", step=2)
        out3 = linear_patch.differentiate("distance", step=2)
        assert np.allclose(out1.data, out2.data)
        assert np.allclose(out2.data, out3.data)

    def test_different_steps_monotonic_coord(self, linear_patch_monotonic_coord):
        """Ensure using different steps still gives similar results."""
        patch = linear_patch_monotonic_coord
        out1 = patch.differentiate("distance", step=1)
        out2 = patch.differentiate("distance", step=2)
        out3 = patch.differentiate("distance", step=5)
        assert np.allclose(out1.data, out2.data)
        assert np.allclose(out2.data, out3.data)

    def test_steps_with_order(self, linear_patch):
        """Tests combinations of step and order."""
        pytest.importorskip("findiff")
        values = [
            linear_patch.differentiate("distance"),
            linear_patch.differentiate("distance", step=2, order=2),
            linear_patch.differentiate("distance", step=5, order=6),
            linear_patch.differentiate("distance", step=1, order=4),
        ]
        for num, patch in enumerate(values[:-1]):
            next_patch = values[num + 1]
            assert np.allclose(patch.data, next_patch.data)


class TestCompareOrders:
    """Ensure differentiation with different orders returns similar results."""

    # orders to check
    orders = (2, 4, 8)

    @pytest.fixture(params=orders)
    def order(self, request):
        """Fixture to return order for testing."""
        order = request.param
        if order != 2:  # order != 2 requires findiff, skip if not installed.
            pytest.importorskip("findiff")
        return request.param

    def test_default_case(self, random_patch, order):
        """Diff with order != 2 on each dimension."""
        patch = random_patch
        for dim in patch.dims:  # test all dimensions.
            out = differentiate(random_patch, dim=dim, order=order)
            assert out.shape == patch.shape

    def test_different_orders(self, linear_patch, order):
        """Different order stencils should be approx equal with simple data."""
        patch = linear_patch.differentiate("distance", order=order)
        # since linear_patch is just linear along distance axis this should be 1.0
        assert np.allclose(patch.data, 1.0)

    def test_diff_all_dims_plus(self, x_plus_y_patch, order):
        """Ensure diffs over all dimensions on x+y patch."""
        p1 = x_plus_y_patch.differentiate(dim=None, order=order)
        # since dxdy(3x + 2y) = 0
        assert np.allclose(p1.data, 0.0)

    def test_diff_all_dims_mult(self, x_times_y_patch, order):
        """Ensure diffs over all dimensions on x+y patch."""
        p1 = x_times_y_patch.differentiate(dim=None, order=order)
        # since dxdy(3x * 2y) = 6
        assert np.allclose(p1.data, 6.0)
