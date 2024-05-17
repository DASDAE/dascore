"""Tests for basic patch functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore import get_example_patch
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
        """Ensure phase angles from real value are close to 0/360."""
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

    def test_bit(self):
        """Ensure after operation each sample is -1, 1, or 0."""
        patch = get_example_patch("dispersion_event")
        bit_norm = patch.normalize("time", norm="bit")
        assert np.all(np.unique(bit_norm.data) == np.array([-1.0, 0, 1.0]))

    def test_zero_channels(self, random_patch):
        """Ensure after operation each zero row or vector remains so."""
        zeroed_data = np.copy(random_patch.data)
        zeroed_data[0, :] = 0.0
        zeroed_data[:, 0] = 0.0
        zeroed_patch = random_patch.new(data=zeroed_data)
        for norm_type in ["l1", "l2", "max", "bit"]:
            norm = zeroed_patch.normalize("time", norm=norm_type)
            assert np.all(norm.data[0, :] == 0.0)
            assert np.all(norm.data[:, 0] == 0.0)
        for norm_type in ["l1", "l2", "max", "bit"]:
            norm = zeroed_patch.normalize("distance", norm=norm_type)
            assert np.all(norm.data[0, :] == 0.0)
            assert np.all(norm.data[:, 0] == 0.0)


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
        """Units should only affect the unit attr."""
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


class TestSqueeze:
    """Tests for squeeze."""

    def test_remove_dimension(self, random_patch):
        """Tests for removing random dimensions."""
        out = random_patch.aggregate("time").squeeze("time")
        assert "time" not in out.dims
        assert len(out.data.shape) == 1, "data should be 1d"

    def test_tutorial_example(self, random_patch):
        """Ensure the tutorial snippet works."""
        flat_patch = random_patch.select(distance=0, samples=True)
        squeezed = flat_patch.squeeze()
        assert len(squeezed.dims) < len(flat_patch.dims)


class TestDropNa:
    """Tests for dropping nullish values in a patch."""

    @pytest.fixture(scope="session")
    def patch_with_null(self):
        """Return a patch which has nullish values."""
        return dc.get_example_patch("patch_with_null")

    @pytest.fixture(scope="class")
    def patch_3d_with_null(self, range_patch_3d):
        """Return a patch which has nullish values."""
        data = np.array(range_patch_3d.data).astype(np.float_)
        nans = [(1, 1, 1), (0, 9, 9)]
        for nan_ind in nans:
            data[nan_ind] = np.NaN
        patch = range_patch_3d.update(data=data)
        return patch

    def test_drop_time_any(self, patch_with_null):
        """Ensure we can drop NaN along time axis."""
        patch = patch_with_null.dropna("time")
        # we know that each label has at least 1 nan so size should be 0
        coord = patch.get_coord("time")
        assert len(coord) == 0

    def test_drop_time_all(self, patch_with_null):
        """Ensure we can drop NaN along time axis."""
        patch = patch_with_null.dropna("time", how="all")
        # we should have dropped the first time label
        before_coord = patch_with_null.get_coord("time")
        after_coord = patch.get_coord("time")
        assert len(before_coord) == len(after_coord) + 1
        # also should have no columns with all NaN
        axis = patch_with_null.dims.index("time")
        assert axis == 1
        expected = np.all(pd.isnull(patch.data), axis=0)
        assert not np.any(expected)

    def test_drop_distance_all(self, patch_with_null):
        """Ensure we can drop NaN along distance axis."""
        patch = patch_with_null.dropna("distance", how="all")
        # we should have dropped the first time label
        before_coord = patch_with_null.get_coord("distance")
        after_coord = patch.get_coord("distance")
        assert len(before_coord) == len(after_coord) + 1
        # also should have no columns with all NaN
        axis = patch_with_null.dims.index("distance")
        assert axis == 0
        expected = np.all(pd.isnull(patch.data), axis=1)
        assert not np.any(expected)

    def test_3d(self, patch_3d_with_null):
        """Ensure 3D patches can have NaNs dropped."""
        patch = patch_3d_with_null
        # first test all; it shouldn't change anything
        out = patch.dropna("time", how="all")
        assert out.shape == patch.shape
        # then test any; it should drop 2 labels
        axis = out.dims.index("time")
        out = patch.dropna("time", how="any")
        assert out.shape[axis] == patch.shape[axis] - 2


class TestFillNa:
    """Tests for dropping nullish values in a patch."""

    @pytest.fixture(scope="session")
    def patch_with_null(self):
        """Return a patch which has nullish values."""
        return dc.get_example_patch("patch_with_null")

    @pytest.fixture(scope="class")
    def patch_3d_with_null(self, range_patch_3d):
        """Return a patch which has nullish values."""
        data = np.array(range_patch_3d.data).astype(np.float_)
        nans = [(1, 1, 1), (0, 9, 9)]
        for nan_ind in nans:
            data[nan_ind] = np.NaN
        patch = range_patch_3d.update(data=data)
        return patch

    def test_fillna(self, patch_with_null):
        """Ensure we can fillna and keep the other values the same."""
        patch = patch_with_null.fillna(0)

        assert all(patch.data[pd.isnull(patch_with_null.data)] == 0)
        assert all(
            patch.data[~pd.isnull(patch_with_null.data)]
            == patch_with_null.data[~pd.isnull(patch_with_null.data)]
        )

    def test_3d(self, patch_3d_with_null):
        """Ensure 3D patches can fillna and keep the other values the same."""
        patch = patch_3d_with_null.fillna(0)

        assert all(patch.data[pd.isnull(patch_3d_with_null.data)] == 0)
        assert all(
            patch.data[~pd.isnull(patch_3d_with_null.data)]
            == patch_3d_with_null.data[~pd.isnull(patch_3d_with_null.data)]
        )


class TestPad:
    """Tests for the padding functionality in a patch."""

    def test_pad_time_dimension_samples_true(self, random_patch, samples=True):
        """Test padding the time dimension with zeros before and after."""
        padded_patch = random_patch.pad(time=(2, 3), samples=samples)
        # Check if the padding is applied correctly
        original_shape = random_patch.shape
        new_shape = padded_patch.shape
        time_axis = random_patch.dims.index("time")
        assert new_shape[time_axis] == original_shape[time_axis] + 5
        # Ensure that padded values are zeros
        assert np.all(padded_patch.select(time=(None, 2), samples=samples).data == 0)
        assert np.all(padded_patch.select(time=(-3, None), samples=samples).data == 0)

    def test_pad_distance_dimension(self, random_patch):
        """Test padding the distance with same number of zeros on both sides."""
        padded_patch = random_patch.pad(distance=7)
        original_shape = random_patch.shape
        new_shape = padded_patch.shape
        distance_axis = random_patch.dims.index("distance")
        ch_spacing = random_patch.attrs["distance_step"]
        assert (
            new_shape[distance_axis] == original_shape[distance_axis] + 14 * ch_spacing
        )
        # Ensure that padded values are zeros
        assert np.all(padded_patch.select(distance=(None, 7), samples=True).data == 0)
        assert np.all(padded_patch.select(distance=(-7, None), samples=True).data == 0)

    def test_pad_distance_dimension_expand_coords(
        self, random_patch, expand_coords=True
    ):
        """Test padding the distance with same number of zeros on both sides."""
        padded_patch = random_patch.pad(distance=4, expand_coords=expand_coords)
        original_shape = random_patch.shape
        new_shape = padded_patch.shape
        distance_axis = random_patch.dims.index("distance")
        ch_spacing = random_patch.attrs["distance_step"]
        dist_max = random_patch.attrs["distance_max"]
        assert (
            new_shape[distance_axis] == original_shape[distance_axis] + 8 * ch_spacing
        )
        # Ensure that padded values are zeros
        assert np.all(padded_patch.select(distance=(None, -1)).data == 0)
        assert np.all(padded_patch.select(distance=(dist_max + 1, None)).data == 0)

    def test_pad_multiple_dimensions_samples_true(self, random_patch, samples=True):
        """Test padding multiple dimensions with different pad values."""
        padded_patch = random_patch.pad(
            time=(6, 7), distance=(1, 4), constant_values=np.pi, samples=samples
        )
        # Check dimensions individually
        time_axis = random_patch.dims.index("time")
        distance_axis = random_patch.dims.index("distance")
        assert padded_patch.shape[time_axis] == random_patch.shape[time_axis] + 13
        assert (
            padded_patch.shape[distance_axis] == random_patch.shape[distance_axis] + 5
        )
        # Check padding values
        assert np.allclose(
            padded_patch.select(time=(None, 6), samples=samples).data, np.pi, atol=1e-6
        )
        assert np.allclose(
            padded_patch.select(time=(-7, None), samples=samples).data, np.pi, atol=1e-6
        )
        assert np.allclose(
            padded_patch.select(distance=(None, 1), samples=samples).data,
            np.pi,
            atol=1e-6,
        )
        assert np.allclose(
            padded_patch.select(distance=(-4, None), samples=samples).data,
            np.pi,
            atol=1e-6,
        )

    def test_error_on_sequence_constant_values(self, random_patch):
        """Test that providing a sequence for constant_values raises a TypeError."""
        with pytest.raises(TypeError):
            random_patch.pad(time=(0, 5), constant_values=(0, 0))
