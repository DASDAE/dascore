"""Tests for basic patch functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dascore.exceptions import IncompatiblePatchError, UnitError
from dascore.proc.basic import apply_operator
from dascore.units import furlongs, get_quantity, m, s


@pytest.fixture(scope="session")
def random_complex_patch(random_patch):
    """Swap out data for complex data."""
    shape = random_patch.shape
    rand = np.random.random
    new_data = rand(shape) + rand(shape) * 1j
    pa = random_patch.new(data=new_data)
    return pa


class TestAbs:
    """Test absolute values."""

    def test_no_negatives(self, random_patch):
        """Simply ensure the data has no negatives."""
        # add
        data = np.array(random_patch.data)
        data[:, 0] = -2
        new = random_patch.new(data=data)
        out = new.abs()
        assert np.all(out.data >= 0)

    def test_all_real(self, random_complex_patch):
        """Ensure all values are real after abs on complex data."""
        out = random_complex_patch.abs()
        assert np.issubdtype(out.data.dtype, np.float_)
        assert np.allclose(np.imag(out.data), 0)


class TestReal:
    """Test for getting real data values."""

    def test_real_from_complex(self, random_complex_patch):
        """Simply ensure data are real-valued."""
        out1 = random_complex_patch.real()
        assert np.all(out1.data == np.real(out1.data))

    def test_real_does_nothing_on_float(self, random_patch):
        """Read shouldn't do anything for float patches."""
        pa1 = random_patch.real()
        assert np.allclose(pa1.data, random_patch.data)


class TestImag:
    """Test for getting imaginary part of data values."""

    def test_imag_from_real(self, random_patch):
        """Real data should have close to 0 imaginary part."""
        out1 = random_patch.imag()
        assert np.allclose(out1.data, 0)

    def test_imag_from_complex(self, random_complex_patch):
        """Real data should have close to 0 imaginary part."""
        out1 = random_complex_patch.imag()
        assert np.allclose(out1.data, np.imag(random_complex_patch.data))


class TestsAngle:
    """Test for getting phase angle."""

    def test_angle_from_real(self, random_patch):
        """Ensure phase angles from real value are close to 0/360"""
        out = random_patch.angle()
        assert np.allclose(out.data, 0)

    def test_angle_from_complex(self, random_complex_patch):
        """Simple phase angle test."""
        out = random_complex_patch.angle()
        assert np.allclose(out.data, np.angle(random_complex_patch.data))


class TestTranspose:
    """Tests for transposing patch."""

    def test_transpose_no_args(self, random_patch):
        """Ensure transposing rotates dimensions."""
        pa = random_patch.transpose()
        assert pa.dims != random_patch.dims
        assert pa.dims == random_patch.dims[::-1]


class TestNormalize:
    """Tests for normalization."""

    def test_bad_norm_raises(self, random_patch):
        """Ensure an unsupported norm raises."""
        with pytest.raises(ValueError):
            random_patch.normalize("time", norm="bob_norm")

    def test_l2(self, random_patch):
        """Ensure after operation norms are 1."""
        dims = random_patch.dims
        # test along distance axis
        dist_norm = random_patch.normalize("distance", norm="l2")
        axis = dims.index("distance")
        norm = np.linalg.norm(dist_norm.data, axis=axis)
        assert np.allclose(norm, 1)
        # tests along time axis
        time_norm = random_patch.normalize("time", norm="l2")
        axis = dims.index("time")
        norm = np.linalg.norm(time_norm.data, axis=axis)
        assert np.allclose(norm, 1)

    def test_l1(self, random_patch):
        """Ensure after operation norms are 1."""
        dims = random_patch.dims
        # test along distance axis
        dist_norm = random_patch.normalize("distance", norm="l1")
        axis = dims.index("distance")
        norm = np.abs(np.sum(dist_norm.data, axis=axis))
        assert np.allclose(norm, 1)
        # tests along time axis
        time_norm = random_patch.normalize("time", norm="l1")
        axis = dims.index("time")
        norm = np.abs(np.sum(time_norm.data, axis=axis))
        assert np.allclose(norm, 1)

    def test_max(self, random_patch):
        """Ensure after operation norms are 1."""
        dims = random_patch.dims
        # test along distance axis
        dist_norm = random_patch.normalize("distance", norm="l1")
        axis = dims.index("distance")
        norm = np.abs(np.sum(dist_norm.data, axis=axis))
        assert np.allclose(norm, 1)
        # tests along time axis
        time_norm = random_patch.normalize("time", norm="l1")
        axis = dims.index("time")
        norm = np.abs(np.sum(time_norm.data, axis=axis))
        assert np.allclose(norm, 1)


class TestStandarize:
    """Tests for standardization."""

    def test_base_case(self, random_patch):
        """Ensure runs with default parameters."""
        # test along distance axis
        out = random_patch.standardize("distance")
        assert not np.any(pd.isnull(out.data))
        # test along time axis
        out = random_patch.standardize("time")
        assert not np.any(pd.isnull(out.data))

    def test_std(self, random_patch):
        """Ensure after operation standard deviations are 1."""
        dims = random_patch.dims
        # test along distance axis
        axis = dims.index("distance")
        out = random_patch.standardize("distance")
        assert np.allclose(
            np.round(np.std(out.data, axis=axis, keepdims=True), decimals=1), 1.0
        )
        # test along time axis
        out = random_patch.standardize("time")
        axis = dims.index("time")
        assert np.allclose(
            np.round(np.std(out.data, axis=axis, keepdims=True), decimals=1), 1.0
        )

    def test_mean(self, random_patch):
        """Ensure after operation means are 0."""
        dims = random_patch.dims
        # test along distance axis
        out = random_patch.standardize("distance")
        axis = dims.index("distance")
        assert np.allclose(
            np.round(np.mean(out.data, axis=axis, keepdims=True), decimals=1), 0.0
        )
        # test along time axis
        out = random_patch.standardize("time")
        axis = dims.index("time")
        assert np.allclose(
            np.round(np.mean(out.data, axis=axis, keepdims=True), decimals=1), 0.0
        )


class TestApplyOperator:
    """Tests for applying various ufunc-type operators."""

    def test_scalar(self, random_patch):
        """Test for a single scalar."""
        new = apply_operator(random_patch, 10, np.multiply)
        assert np.allclose(new.data, random_patch.data * 10)

    def test_array_like(self, random_patch):
        """Ensure array-like operations work."""
        ones = np.ones(random_patch.shape)
        new = apply_operator(random_patch, ones, np.add)
        assert np.allclose(new.data, ones + random_patch.data)

    def test_incompatible_coords(self, random_patch):
        """Ensure incompatible dimensions raises."""
        new = random_patch.update_attrs(time_min=random_patch.attrs.time_max)
        with pytest.raises(IncompatiblePatchError):
            apply_operator(new, random_patch, np.multiply)

    def test_quantity_scalar(self, random_patch):
        """Ensure operators work with quantities."""
        patch = random_patch.set_units("m/s")
        other = 10 * m / s
        # first try multiply
        new = apply_operator(patch, other, np.multiply)
        new_units = get_quantity("m/s") * get_quantity("m/s")
        assert get_quantity(new.attrs.data_units) == new_units
        assert isinstance(new.data, np.ndarray)
        # try add
        new = apply_operator(patch, other, np.add)
        new_units = get_quantity("m/s")
        assert get_quantity(new.attrs.data_units) == new_units
        assert isinstance(new.data, np.ndarray)
        # and divide
        new = apply_operator(patch, other, np.divide)
        new_units = get_quantity("m/s") / get_quantity("m/s")
        assert new.attrs.data_units is None or new.attrs.data_units == new_units
        assert isinstance(new.data, np.ndarray)

    def test_unit(self, random_patch):
        """Units should only affect the unit attr"""
        patch = random_patch.set_units("m/s")
        other = m / s
        # first try multiply
        new = apply_operator(patch, other, np.multiply)
        new_units = get_quantity("m/s") * get_quantity("m/s")
        assert get_quantity(new.attrs.data_units) == new_units
        assert np.allclose(new.data, random_patch.data)
        # try add
        new = apply_operator(patch, other, np.add)
        new_units = get_quantity("m/s")
        assert get_quantity(new.attrs.data_units) == new_units
        assert np.allclose(new.data, random_patch.data + 1)
        # and divide
        new = apply_operator(patch, other, np.divide)
        new_units = get_quantity("m/s") / get_quantity("m/s")
        assert new.attrs.data_units is None or new.attrs.data_units == new_units
        assert np.allclose(new.data, random_patch.data)

    def test_patch_with_units(self, random_patch):
        """Ensure when patch units are set they are applied as well."""
        # test add
        pa1 = random_patch.set_units("m/s")
        out1 = apply_operator(pa1, pa1, np.add)
        assert get_quantity(out1.attrs.data_units) == get_quantity("m/s")
        # test multiply
        out2 = apply_operator(pa1, pa1, np.multiply)
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
            apply_operator(pa1, other, np.add)
