"""Tests for basic patch functions."""

from __future__ import annotations

import operator

import numpy as np
import pandas as pd
import pytest
from scipy.fft import next_fast_len

import dascore as dc
from dascore import get_example_patch
from dascore.exceptions import ParameterError, PatchBroadcastError, UnitError
from dascore.proc.basic import apply_operator
from dascore.units import furlongs, get_quantity, m, s
from dascore.utils.misc import _merge_tuples

OP_NAMES = ("add", "sub", "pow", "truediv", "floordiv", "mul", "mod")
TEST_OPS = tuple(getattr(operator, x) for x in OP_NAMES)


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
        assert np.issubdtype(out.data.dtype, np.float64)
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


class TestApplyUfunc:
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
        """Ensure un-alignable coords returns degenerate patch."""
        time = random_patch.get_coord("time")
        new_time = time.max() + time.step
        new = random_patch.update_attrs(time_min=new_time)
        out = apply_operator(new, random_patch, np.multiply)
        assert 0 in set(out.shape)

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


class TestPatchBroadcasting:
    """Tests for patches broadcasting to allow operations on each other."""

    def test_broadcast_sub_patch(self, random_patch):
        """Ensure a patch with a subset of dimensions broadcasts."""
        sub_patch = random_patch.min("time")
        out = random_patch - sub_patch
        assert isinstance(out, dc.Patch)
        axis = random_patch.dims.index("time")
        to_sub = np.min(random_patch.data, axis=axis, keepdims=True)
        # Ensure the time aggregation worked.
        assert np.allclose(sub_patch.data, to_sub)
        # Then test that the resulting data is as expected.
        expected = random_patch.data - to_sub
        assert np.allclose(expected, out.data)
        # The reverse should also be true (after accounting for negative sign)
        out2 = sub_patch - random_patch
        assert out == -out2

    def test_broadcast_single_shared_dim(self, random_patch):
        """Ensure two 2d patches can get broad-casted when one dim is the same."""
        # get two smallish patches to test multidimensional broadcasting.
        patch1 = random_patch.select(time=(1, 10), distance=(1, 10), samples=True)
        patch2 = patch1.transpose("distance", "time").rename_coords(distance="length")
        expected_dims = _merge_tuples(patch1.dims, patch2.dims)
        out = patch1 * patch2
        assert out.dims == expected_dims
        assert out.shape == (9, 9, 9)

    def test_patches_different_coords_same_shape(self, random_patch):
        """Ensure the intersection of coordinates in output when coords differ."""
        distance = random_patch.get_array("distance")
        mid = distance[len(distance) // 2]
        shifted_patch = random_patch.update_coords(distance_min=mid)
        out = random_patch + shifted_patch
        # Ensure distance is equal to intersecting values
        dist_out = out.get_array("distance")
        overlap_dist = np.intersect1d(
            random_patch.get_array("distance"), out.get_array("distance")
        )
        assert np.all(dist_out == overlap_dist)

    def test_patch_broadcast_array(self, random_patch):
        """Ensure a patch is broadcastable with an array up to the same ndims."""
        should_work = (np.array(1), np.ones(1), np.ones((1, 1)))
        for ar in should_work:
            out = random_patch * ar
            assert isinstance(out, dc.Patch)

    def test_patch_broadcast_array_more_dims_raises(self, random_patch):
        """Ensure a patch cannot broadcast to an array which has more dims."""
        ar = np.ones((1, 1, 1))
        with pytest.raises(PatchBroadcastError, match="Cannot broadcast"):
            _ = random_patch * ar

    def test_broadcast_up_array(self, random_patch):
        """Ensure a patch with empty coords can broadcast up."""
        # This patch still has empty dims of len 1
        patch = random_patch.mean()
        ar = np.ones(random_patch.shape)
        for out in [patch * ar, ar * patch]:
            assert isinstance(out, dc.Patch)
            assert np.allclose(out.data, patch.data)

    def test_broadcast_with_array(self, random_patch):
        """Ensure a patch can broadcast up and results are correct."""
        agg = random_patch.min(None)
        out1 = random_patch - agg
        out2 = random_patch.data - agg
        assert np.allclose(out1.data, out2.data)

    @pytest.mark.parametrize("test_op", TEST_OPS)
    def test_broadcast_collapsed_patch(self, random_patch, test_op):
        """Ensure a collapsed patch can still broadcast."""
        collapsed_patch = random_patch.min(None)
        scalar = 10
        # Arrays should raise since we don't know the name of the
        # dims that would be expanded.
        mat1 = np.array([1, 2, 3])
        mat2 = np.arange(4).reshape(2, 2) + 1  # + 1 to avoid divide by 0
        # A collapsed patch should broadcast to all these things.
        for val in (scalar, mat1, mat2, random_patch):
            p1 = test_op(collapsed_patch, val)
            p2 = test_op(val, collapsed_patch)
            assert isinstance(p1, dc.Patch)
            assert isinstance(p2, dc.Patch)


class TestDropNa:
    """Tests for dropping nullish values in a patch."""

    @pytest.fixture(scope="session")
    def patch_with_null(self):
        """Return a patch which has nullish values."""
        return dc.get_example_patch("patch_with_null")

    @pytest.fixture(scope="class")
    def patch_3d_with_null(self, range_patch_3d):
        """Return a patch which has nullish values."""
        data = np.array(range_patch_3d.data).astype(np.float64)
        nans = [(1, 1, 1), (0, 9, 9)]
        for nan_ind in nans:
            data[nan_ind] = np.nan
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
    """Tests for replacing nullish values in a patch."""

    @pytest.fixture(scope="session")
    def patch_with_null(self):
        """Return a patch which has nullish values."""
        return dc.get_example_patch("patch_with_null")

    @pytest.fixture(scope="class")
    def patch_3d_with_null(self, range_patch_3d):
        """Return a patch which has nullish values."""
        data = np.array(range_patch_3d.data).astype(np.float64)
        nans = [(1, 1, 1), (0, 9, 9)]
        for nan_ind in nans:
            data[nan_ind] = np.nan
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

    @pytest.mark.parametrize("length", [10, 13, 200, 293])
    def test_fft_pad(self, random_patch, length):
        """Tests for padding to next fast length."""
        # First trim patch to match length
        patch_in = random_patch.select(time=(0, length), samples=True)
        axis = patch_in.dims.index("time")
        length_old = patch_in.shape[axis]
        next_fast = next_fast_len(length_old)
        out = patch_in.pad(time="fft")
        assert out.dims == patch_in.dims, "dims should not have changed."
        assert out.shape[axis] == next_fast

    def test_correlate_pad(self, random_patch):
        """Ensure the fft correlate pad returns sensible values."""
        axis = random_patch.dims.index("time")
        length_old = random_patch.shape[axis]
        next_fast = next_fast_len(length_old * 2 - 1)
        out = random_patch.pad(time="correlate")
        assert out.shape[axis] == next_fast

    def test_pad_no_expand(self, random_patch):
        """Ensure we can pad without expanding dimensions."""
        out = random_patch.pad(time=10, samples=True, expand_coords=False)
        coord_array = out.get_array("time")
        # Check the first and last nans
        assert np.all(pd.isnull(coord_array[:10]))
        assert np.all(pd.isnull(coord_array[-10:]))
        # Check the middle values are equal to old values.
        old_array = random_patch.get_array("time")
        # The datatype should not have changed
        assert coord_array.dtype == old_array.dtype
        assert np.all(old_array == coord_array[10:-10])

    def test_pad_no_expand_int_coord(self, random_patch):
        """Ensure we can pad an integer coordinate."""
        # Ensure distances in an in coordinate.
        coord = random_patch.get_coord("distance")
        new_vals = np.arange(len(coord), dtype=np.int64)
        patch = random_patch.update_coords(distance=new_vals)
        # Apply padding, ensure NaN values appear.
        padded = patch.pad(distance=4, expand_coords=False)
        coord_array = padded.get_array("distance")
        assert np.all(pd.isnull(coord_array[:4]))
        assert np.all(pd.isnull(coord_array[-4:]))

    def test_error_on_sequence_constant_values(self, random_patch):
        """Test that providing a sequence for constant_values raises a TypeError."""
        with pytest.raises(ParameterError):
            random_patch.pad(time=(0, 5), constant_values=(0, 0))


class TestConj:
    """Tests for complex conjugate."""

    def test_imaginary_part_reversed(self, random_dft_patch):
        """Ensure the imaginary part of the array is reversed."""
        imag1 = np.imag(random_dft_patch.data)
        conj = random_dft_patch.conj()
        imag2 = np.imag(conj.data)
        assert np.allclose(imag1, -imag2)


class TestRoll:
    """Test cases for patch roll method."""

    def test_time_roll(self, random_patch):
        """Test basic sample roll."""
        rand_patcht = random_patch.transpose("time", ...)
        rolled_patch = rand_patcht.roll(time=5, samples=True)
        assert rand_patcht.shape == rolled_patch.shape
        assert np.all(rand_patcht.data[0] == rolled_patch.data[5])

    def test_dist_roll(self, random_patch):
        """Test roll when samples=False."""
        rolled_patch = random_patch.roll(distance=30, samples=False)
        coord = random_patch.get_coord("distance")
        value = coord.get_sample_count(30)
        assert random_patch.shape == rolled_patch.shape
        assert np.all(random_patch.data[0] == rolled_patch.data[value])

    def test_coord_update_roll(self, random_patch):
        """Test roll when update_coord=True."""
        patcht = random_patch.transpose("time", ...)
        rolled_patch = patcht.roll(time=5, samples=True, update_coord=True)
        assert (
            patcht.coords.get_array("time")[0]
            == rolled_patch.coords.get_array("time")[5]
        )

    def test_dist_coord_roll(self, random_patch):
        """Test roll when samples=False and coords will be updated."""
        rolled_patch = random_patch.roll(distance=30, samples=False, update_coord=True)
        coord = random_patch.get_coord("distance")
        value = coord.get_sample_count(30)
        assert (
            random_patch.coords.get_array("distance")[0]
            == rolled_patch.coords.get_array("distance")[value]
        )
