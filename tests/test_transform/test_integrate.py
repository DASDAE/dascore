"""Module for performing integrations."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
import dascore.proc.coords
from dascore.transform.integrate import integrate
from dascore.units import get_quantity
from dascore.utils.misc import broadcast_for_index
from dascore.utils.time import to_float


@pytest.fixture(scope="session")
def ones_patch(random_patch):
    """Return a patch of ones with normal axis."""
    array = np.ones_like(random_patch.data)
    return random_patch.new(data=array)


class TestIndefiniteIntegrals:
    """Tests for indefinite integrals."""

    @pytest.fixture(scope="class")
    def simple_func_patch(self, random_patch):
        """Create a simple function patch for testing. f(x) = x + 1."""
        time = np.arange(100) * 0.1
        data = time[:, None] + 1
        out = dc.Patch(
            data=data,
            coords={"time": time, "other": np.array([1])},
            dims=("time", "other"),
        )
        return out

    def test_indef_integration(self, ones_patch):
        """Happy path for default time/distance integrals."""
        for dim in ones_patch.dims:
            patch = integrate(ones_patch, dim=dim, definite=False)
            ax = patch.get_axis(dim)
            # We expect slice not on axis to be the identical
            non_dim_indexer = broadcast_for_index(
                len(patch.dims), ax, slice(None, None), fill=slice(0, 1)
            )
            first_along_axis = patch.data[non_dim_indexer]
            assert np.allclose(patch.data, first_along_axis)
            # the values along the axis of integration should strictly increase.
            # (because they are all 1s)
            flat = np.squeeze(first_along_axis)
            assert np.allclose(flat, np.sort(flat))

    def test_integrate_non_evenly_sampled(self, wacky_dim_patch):
        """Ensure we can integrate along non-evenly sampled dims."""
        out = wacky_dim_patch.integrate(dim="time", definite=False)
        assert isinstance(out, dc.Patch)

    def test_units(self, random_patch):
        """Ensure output units are as expected."""
        patch = random_patch.set_units("m/s")
        dims = ("time", "distance")
        expected_units = ("s", "m")
        for dim, eu in zip(dims, expected_units):
            out = patch.integrate(dim=dim)
            data_units1 = get_quantity(patch.attrs.data_units)
            data_units2 = get_quantity(out.attrs.data_units)
            assert data_units2 == (data_units1 * get_quantity(eu))
            for dim in patch.dims:
                coord1 = patch.get_coord(dim)
                coord2 = patch.get_coord(dim)
                assert coord2.units == coord1.units

    def test_simple_func(self, simple_func_patch):
        """Ensure the values are approximate correct for a simple function."""
        out = simple_func_patch.integrate(dim="time", definite=False)
        time = simple_func_patch.get_coord("time").values
        expected = (time**2) / 2 + time
        data_out = out.data.flatten()
        assert np.allclose(expected, data_out)

    def test_integrate_multiple_dims_matches_sequential(self):
        """Indefinite multi-dim integration should compose axis integrals."""
        data = np.arange(12, dtype=float).reshape(3, 4)
        coords = {"distance": np.arange(3), "time": np.arange(4)}
        patch = dc.Patch(data=data, coords=coords, dims=("distance", "time"))

        out = patch.integrate(dim=("distance", "time"), definite=False)
        expected = patch.integrate(dim="distance", definite=False).integrate(
            dim="time", definite=False
        )

        np.testing.assert_allclose(out.data, expected.data)


class TestDefiniteIntegration:
    """Test case for definite path integration."""

    def test_simple_integration(self, ones_patch):
        """Ensure simple integration works."""
        patch = ones_patch
        for dim in patch.dims:
            ax = patch.get_axis(dim)
            out = patch.integrate(dim=dim, definite=True)
            assert out.shape[ax] == 1
            step = to_float(patch.get_coord(dim).step)
            trap_name = "trapezoid" if hasattr(np, "trapezoid") else "trapz"
            trap = getattr(np, trap_name)
            expected_data = trap(patch.data, dx=step, axis=ax)
            ndims = len(patch.dims)
            indexer = broadcast_for_index(ndims, ax, None)
            assert np.allclose(out.data, expected_data[indexer])
            # Since the patch is just ones all values should equal the
            # dimensional length when dx == 1
            if dim == "distance" and np.isclose(patch.get_coord(dim).step, 1):
                assert np.allclose(out.data, patch.shape[ax] - 1)

    def test_units(self, random_patch):
        """Ensure data units are updated and coord units are unchanged."""
        patch = random_patch.set_units("m/s")
        out = patch.integrate(dim="time", definite=True)
        data_units1 = get_quantity(patch.attrs.data_units)
        data_units2 = get_quantity(out.attrs.data_units)
        assert data_units2 == (data_units1 * get_quantity("s"))
        for dim in patch.dims:
            coord1 = patch.get_coord(dim)
            coord2 = patch.get_coord(dim)
            assert coord2.units == coord1.units

    def test_integrate_all_dims(self, random_patch):
        """Ensure all dims can be integrated."""
        out = random_patch.integrate(dim=None, definite=True)
        assert out.shape == tuple([1] * len(random_patch.shape))

    def test_integrate_non_evenly_sampled_dim(self, wacky_dim_patch):
        """Simple test to integrate along non-evenly sampled dimension."""
        out = wacky_dim_patch.integrate(dim="time", definite=True)
        assert isinstance(out, dc.Patch)


class TestDataType:
    """An integral is a derivative read backwards, data_type included."""

    @pytest.mark.parametrize(
        ("dim", "start", "expected"),
        (
            ("time", "strain_rate", "strain"),
            ("time", "acceleration", "velocity"),
            ("time", "velocity", "displacement"),
            ("distance", "strain", "displacement"),
            ("distance", "strain_rate", "velocity"),
        ),
    )
    def test_known_pairs(self, random_patch, dim, start, expected):
        """An integral of a known type is the type it is known to give."""
        patch = random_patch.update_attrs(data_type=start)
        assert patch.integrate(dim).attrs.data_type == expected

    def test_unknown_type_is_cleared(self, random_patch):
        """An integral with no name here carries no label at all."""
        patch = random_patch.update_attrs(data_type="temperature")
        assert patch.integrate("time").attrs.data_type == ""

    def test_definite_integral_maps_too(self, random_patch):
        """Collapsing a dimension is still an integral along it."""
        patch = random_patch.update_attrs(data_type="strain_rate")
        out = patch.integrate("time", definite=True)
        assert out.attrs.data_type == "strain"

    def test_phase_rate_becomes_phase(self, random_patch):
        """The pair differentiate states forwards, read backwards."""
        patch = random_patch.update_attrs(data_type="phase_rate")
        assert patch.integrate("time").attrs.data_type == "phase"

    def test_round_trip(self, random_patch):
        """Differentiating then integrating gives the type back."""
        patch = random_patch.update_attrs(data_type="strain")
        out = patch.differentiate("time").integrate("time")
        assert out.attrs.data_type == "strain"


class TestNumericalIntegrals:
    """Analytic expectations for dtype and axis handling."""

    @pytest.mark.parametrize("definite", [False, True])
    @pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int64, np.bool_])
    def test_integer_data(self, definite, dtype):
        """Promote before adding samples, preserving fractional areas."""
        data = np.array([0, 1, 1, 0], dtype=dtype)
        patch = dc.Patch(
            data=data, coords={"distance": np.arange(4)}, dims=("distance",)
        )
        out = patch.integrate("distance", definite=definite)
        expected = np.array([0, 0.5, 1.5, 2])
        np.testing.assert_array_equal(out.data, expected[-1:] if definite else expected)
        assert out.data.dtype == np.float64
        np.testing.assert_array_equal(patch.data, data)

    @pytest.mark.parametrize("definite", [False, True])
    @pytest.mark.parametrize("dtype", [np.int8, np.uint8])
    def test_integer_overflow(self, definite, dtype):
        """Narrow integer sample sums must not overflow before promotion."""
        data = np.full(4, 100 if dtype == np.int8 else 200, dtype=dtype)
        patch = dc.Patch(
            data=data, coords={"distance": np.arange(4)}, dims=("distance",)
        )
        expected = np.arange(4, dtype=float) * float(data[0])
        out = patch.integrate("distance", definite=definite)
        np.testing.assert_array_equal(out.data, expected[-1:] if definite else expected)

    @pytest.mark.parametrize("definite", [False, True])
    @pytest.mark.parametrize(
        "dtype", [np.float32, np.float64, np.complex64, np.complex128]
    )
    def test_inexact_data(self, definite, dtype):
        """Floating precision and imaginary components survive integration."""
        value = 1 + 2j if np.issubdtype(dtype, np.complexfloating) else 1
        patch = dc.Patch(
            data=np.full(4, value, dtype=dtype),
            coords={"distance": np.arange(4)},
            dims=("distance",),
        )
        expected = np.arange(4) * value
        out = patch.integrate("distance", definite=definite)
        np.testing.assert_array_equal(out.data, expected[-1:] if definite else expected)
        if not definite:
            assert out.data.dtype == dtype

    @pytest.mark.parametrize("width", [2, 3])
    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_uneven_axis(self, width, axis):
        """Uneven intervals weight the named axis in any array position."""
        distance = np.array([0, 1, 3, 6])
        patch = dc.Patch(
            data=np.ones((4, 2, width)),
            coords={"distance": distance, "x": np.arange(2), "y": np.arange(width)},
            dims=("distance", "x", "y"),
        )
        dims = ["x", "y"]
        dims.insert(axis, "distance")
        out = patch.transpose(*dims).integrate("distance").transpose(*patch.dims)
        expected = np.broadcast_to(distance[:, None, None], patch.shape)
        np.testing.assert_array_equal(out.data, expected)
        np.testing.assert_array_equal(out.data, patch.integrate("distance").data)

    def test_multiple_uneven_axes(self):
        """Integrating a constant over two uneven axes gives their product."""
        x, y = np.array([0, 1, 3, 6]), np.array([0, 2, 5])
        patch = dc.Patch(data=np.ones((4, 3)), coords={"x": x, "y": y}, dims=("x", "y"))
        out = patch.integrate(None)
        np.testing.assert_array_equal(out.data, x[:, None] * y[None, :])
