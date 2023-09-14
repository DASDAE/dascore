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
    """Return a patch of ones with nornal axis."""
    array = np.ones_like(random_patch.data)
    return random_patch.new(data=array)


class TestIndefiniteIntegrals:
    """Tests for indefinite integrals."""

    def test_indef_integration(self, ones_patch):
        """Happy path for default time/distance integrals."""
        for dim in ones_patch.dims:
            patch = integrate(ones_patch, dim=dim, definite=False)
            ax = patch.dims.index(dim)
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


class TestDefiniteIntegration:
    """Test case for definite path integration."""

    def test_simple_integration(self, ones_patch):
        """Ensure simple integration works."""
        patch = ones_patch
        for dim in patch.dims:
            ax = patch.dims.index(dim)
            out = patch.integrate(dim=dim, definite=True)
            assert out.shape[ax] == 1
            step = to_float(patch.get_coord(dim).step)
            expected_data = np.trapz(patch.data, dx=step, axis=ax)
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
