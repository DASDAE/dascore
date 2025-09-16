"""
Tests for patch ufuncs.
"""

from __future__ import annotations

import numpy as np
import pytest
from pint import DimensionalityError

import dascore as dc
from dascore import get_quantity
from dascore.exceptions import ParameterError, UnitError
from dascore.units import furlongs, m, s
from dascore.utils.array import apply_ufunc


class TestApplyUfunc:
    """Tests for applying various ufunc-type operators."""

    @pytest.mark.parametrize("func", (np.abs, np.tan, np.isfinite, np.exp))
    def test_uniary_ufuncs(self, func, random_patch):
        """Ensure ufuncs that take a single input also work."""
        out = func(random_patch)
        assert isinstance(out, dc.Patch)

    def test_scalar(self, random_patch):
        """Test for a single scalar."""
        new = apply_ufunc(np.multiply, random_patch, 10)
        assert np.allclose(new.data, random_patch.data * 10)

    def test_array_like(self, random_patch):
        """Ensure array-like operations work."""
        ones = np.ones(random_patch.shape)
        new = apply_ufunc(np.add, random_patch, ones)
        assert np.allclose(new.data, ones + random_patch.data)

    def test_reversed_scalar(self, random_patch):
        """Ensure reversed scalar works on patch."""
        out = 10 + random_patch  # np.add with Patch on RHS
        assert isinstance(out, dc.Patch)
        assert np.allclose(out.data, random_patch.data + 10)

    def test_reversed_array_like(self, random_patch):
        """Test reversed array works on patch."""
        ones = np.ones(random_patch.shape)
        out = ones * random_patch
        assert np.allclose(out.data, ones * random_patch.data)

    @pytest.mark.xfail(raises=(UnitError, DimensionalityError))
    def test_reversed_unit_and_quantity(self, random_patch):
        """
        Ensure reversed quantity works on patch.

        Currently, there is no way to make this pass without relying on internal
        Pint implementation details. This is because pint first handles the
        operation but doesn't know how to treat a Patch.
        """
        pa = random_patch.set_units("m/s")
        out1 = (m / s) + pa  # unit on LHS
        out2 = (10 * m / s) + pa  # quantity on LHS
        assert np.allclose(out1.data, random_patch.data + 1)
        assert np.allclose(out2.data, random_patch.data + 10)

    def test_incompatible_coords(self, random_patch):
        """Ensure un-alignable coords returns degenerate patch."""
        time = random_patch.get_coord("time")
        new_time = time.max() + time.step
        new = random_patch.update_attrs(time_min=new_time)
        out = apply_ufunc(np.multiply, new, random_patch)
        assert 0 in set(out.shape)
        assert out.data.size == 0
        assert out.dims == random_patch.dims

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
        # Dimensionless result: units may be omitted; accept None or "1".
        q = get_quantity(new.attrs.data_units)
        assert q is None or q == get_quantity("1")
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
        assert get_quantity(out1.attrs.data_units) == get_quantity("furlongs")
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
        out_coord_keys = set(out.coords.coord_map.keys())
        input_coord_keys = set(random_dft_patch.coords.coord_map.keys())
        assert out_coord_keys == input_coord_keys
        assert set(out.coords.coord_map) == set(random_dft_patch.coords.coord_map)

    @pytest.mark.parametrize(
        "op, other, expected_units, expected_data",
        [
            (np.multiply, 10 * m / s, "m**2/s**2", lambda d: d * 10),
            (np.add, 10 * m / s, "m/s", lambda d: d + 10),
            (np.divide, 10 * m / s, "1", lambda d: d / 10.0),
        ],
    )
    def test_quantity_ops_param(
        self, random_patch, op, other, expected_units, expected_data
    ):
        """Run several tests for quantities in various operations."""
        pa = random_patch.set_units("m/s")
        out = apply_ufunc(op, pa, other)
        quant = get_quantity(out.attrs.data_units)
        none_or_1 = quant is None and expected_units == "1"
        assert none_or_1 or quant == get_quantity(expected_units)
        assert np.allclose(out.data, expected_data(random_patch.data))

    def test_unsupported_raises(self, random_patch):
        """
        When ufuncs don't have the right number of input/ouput an error
        should be raised.
        """
        msg = "ufuncs with input/output"
        with pytest.raises(ParameterError, match=msg):
            apply_ufunc(np.frexp, random_patch)

    def test_apply_reduction(self, random_patch):
        """Ensure reductions also work."""
        out = np.multiply.reduce(random_patch, axis=1)
        out2 = np.multiply.reduce(random_patch, 1)
        assert isinstance(out, dc.Patch)
        assert out.shape[1] == 1

        assert out2.equals(out)
