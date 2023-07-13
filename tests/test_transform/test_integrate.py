"""
Module for performing integrations.
"""

import numpy as np

from dascore.transform.integrate import integrate
from dascore.units import get_quantity
from dascore.utils.misc import broadcast_for_index
from dascore.utils.time import to_float


class TestIntegrate:
    """Test case of patch integrations."""

    def test_simple_integration(self, random_patch):
        """Ensure simple integration works."""
        for dim in random_patch.dims:
            ax = random_patch.dims.index(dim)
            patch = random_patch.tran.integrate(dim=dim)
            assert patch.shape[ax] == 1
            step = to_float(random_patch.get_coord(dim).step)
            expected_data = np.trapz(random_patch.data, dx=step, axis=ax)
            ndims = len(patch.dims)
            indexer = broadcast_for_index(ndims, ax, None)
            assert np.allclose(patch.data, expected_data[indexer])

    def test_units(self, random_patch):
        """Ensure data units are updated and coord units are unchanged."""
        patch = random_patch.set_units("m/s")
        out = patch.tran.integrate(dim="time")
        data_units1 = get_quantity(patch.attrs.data_units)
        data_units2 = get_quantity(out.attrs.data_units)
        assert data_units2 == (data_units1 * get_quantity("s"))
        for dim in patch.dims:
            coord1 = patch.get_coord(dim)
            coord2 = patch.get_coord(dim)
            assert coord2.units == coord1.units

    def test_dont_keep_dims(self, random_patch):
        """Ensure dims are dropped when requested."""
        patch = random_patch
        out = integrate(patch, dim="time", keep_dims=False)
        assert "time" not in out.dims
        assert len(out.dims) == (len(patch.dims) - 1)

    def test_integrate_all_dims(self, random_patch):
        """Ensure all dims can be integrated."""
        out = random_patch.tran.integrate(dim=None)
        assert out.shape == tuple([1] * len(random_patch.shape))
