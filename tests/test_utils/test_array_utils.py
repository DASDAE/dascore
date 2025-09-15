"""
Tests for patch ufuncs.
"""

from __future__ import annotations

import numpy as np
import pytest

from dascore import get_quantity
from dascore.exceptions import UnitError
from dascore.units import furlongs, m, s
from dascore.utils.array import apply_ufunc


class TestApplyUfunc:
    """Tests for applying various ufunc-type operators."""

    def test_scalar(self, random_patch):
        """Test for a single scalar."""
        new = apply_ufunc(np.multiply, random_patch, 10)
        assert np.allclose(new.data, random_patch.data * 10)

    def test_array_like(self, random_patch):
        """Ensure array-like operations work."""
        ones = np.ones(random_patch.shape)
        new = apply_ufunc(np.add, random_patch, ones)
        assert np.allclose(new.data, ones + random_patch.data)

    def test_incompatible_coords(self, random_patch):
        """Ensure un-alignable coords returns degenerate patch."""
        time = random_patch.get_coord("time")
        new_time = time.max() + time.step
        new = random_patch.update_attrs(time_min=new_time)
        out = apply_ufunc(np.multiply, new, random_patch)
        assert 0 in set(out.shape)

    def test_quantity_scalar(self, random_patch):
        """Ensure operators work with quantities."""
        patch = random_patch.set_units("m/s")
        other = 10 * m / s
        # first try multiply
        new = apply_ufunc(np.multiply, patch, other)
        new_units = get_quantity("m/s") * get_quantity("m/s")
        assert get_quantity(new.attrs.data_units) == new_units
        assert isinstance(new.data, np.ndarray)
        # try add
        new = apply_ufunc(np.add, patch, other)
        new_units = get_quantity("m/s")
        assert get_quantity(new.attrs.data_units) == new_units
        assert isinstance(new.data, np.ndarray)
        # and divide
        new = apply_ufunc(np.divide, patch, other)
        new_units = get_quantity("m/s") / get_quantity("m/s")
        assert new.attrs.data_units is None or new.attrs.data_units == new_units
        assert isinstance(new.data, np.ndarray)

    def test_unit(self, random_patch):
        """Units should only affect the unit attr."""
        patch = random_patch.set_units("m/s")
        other = m / s
        # first try multiply
        new = apply_ufunc(np.multiply, patch, other)
        new_units = get_quantity("m/s") * get_quantity("m/s")
        assert get_quantity(new.attrs.data_units) == new_units
        assert np.allclose(new.data, random_patch.data)
        # try add
        new = apply_ufunc(np.add, patch, other)
        new_units = get_quantity("m/s")
        assert get_quantity(new.attrs.data_units) == new_units
        assert np.allclose(new.data, random_patch.data + 1)
        # and divide
        new = apply_ufunc(np.divide, patch, other)
        new_units = get_quantity("m/s") / get_quantity("m/s")
        assert new.attrs.data_units is None or new.attrs.data_units == new_units
        assert np.allclose(new.data, random_patch.data)

    def test_patch_with_units(self, random_patch):
        """Ensure when patch units are set they are applied as well."""
        # test add
        pa1 = random_patch.set_units("m/s")
        out1 = apply_ufunc(np.add, pa1, pa1)
        assert get_quantity(out1.attrs.data_units) == get_quantity("m/s")
        # test multiply
        out2 = apply_ufunc(
            np.multiply,
            pa1,
            pa1,
        )
        assert get_quantity(out2.attrs.data_units) == get_quantity("m**2/s**2")

    def test_array_with_units(self, random_patch):
        """Ensure an array with units works for multiplication."""
        patch1 = random_patch
        ones = np.ones(patch1.shape) * furlongs
        out1 = patch1 * ones
        assert get_quantity(out1.attrs.data_units) == get_quantity(furlongs)
        # test division with units
        patch2 = random_patch.set_units("m")
        out2 = patch2 / ones
        expected = get_quantity("m/furlongs")
        assert get_quantity(out2.attrs.data_units) == expected

    def test_incompatible_units(self, random_patch):
        """Ensure incompatible units raise."""
        pa1 = random_patch.set_units("m/s")
        other = 10 * get_quantity("m")
        with pytest.raises(UnitError):
            apply_ufunc(np.add, pa1, other)

    def test_patches_non_coords_len_1(self, random_patch):
        """Ensure patches with non-coords also work."""
        mean_patch = random_patch.mean("distance")
        out = mean_patch / mean_patch
        assert np.allclose(out.data, 1)

    def test_patches_non_coords_different_len(self, random_patch):
        """Ensure patches with non-coords of different lengths work."""
        patch_1 = random_patch.mean("distance")
        dist_ind = patch_1.dims.index("distance")
        old_shape = list(patch_1.shape)
        old_shape[dist_ind] = old_shape[dist_ind] + 2
        patch_2 = patch_1.make_broadcastable_to(tuple(old_shape))
        out = patch_1 / patch_2
        assert np.allclose(out.data, 1)
        assert out.shape == patch_2.shape

    def test_non_dim_coords(self, random_dft_patch):
        """Ensure ufuncs can still be applied to coords with non dim coords."""
        out = random_dft_patch * random_dft_patch
        assert set(out.coords.coord_map) == set(random_dft_patch.coords.coord_map)
