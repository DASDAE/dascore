"""Tests for basic patch functions."""

from __future__ import annotations

import operator

import numpy as np
import pandas as pd
import pytest
from scipy.fft import next_fast_len

import dascore as dc
from dascore import get_example_patch
from dascore.exceptions import ParameterError, PatchBroadcastError
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


@pytest.fixture(scope="session")
def patch_with_null():
    """Return a patch which has nullish values."""
    return dc.get_example_patch("patch_with_null")


@pytest.fixture(scope="session")
def patch_with_inf(random_patch):
    """Return a patch which has nullish values."""
    pa = random_patch
    ar = np.array(pa.data)
    ar[:, 2] = np.inf
    return pa.new(data=ar)


@pytest.fixture(scope="class")
def patch_3d_with_null(range_patch_3d):
    """Return a patch which has nullish values."""
    data = np.array(range_patch_3d.data).astype(np.float64)
    nans = [(1, 1, 1), (0, 9, 9)]
    for nan_ind in nans:
        data[nan_ind] = np.nan
    patch = range_patch_3d.update(data=data)
    return patch


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
        # test along the distance axis
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
        # test along the distance axis
        dist_norm = random_patch.normalize("distance", norm="l1")
        axis = dims.index("distance")
        norm = np.abs(np.sum(dist_norm.data, axis=axis))
        assert np.allclose(norm, 1)
        # tests along time axis
        time_norm = random_patch.normalize("time", norm="l1")
        axis = dims.index("time")
        norm = np.abs(np.sum(time_norm.data, axis=axis))
        assert np.allclose(norm, 1)

    def test_l1_norm(self, random_patch):
        """Ensure after l1 normalization operation norms are 1."""
        dims = random_patch.dims
        # test along the distance axis
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


class TestStandardize:
    """Tests for standardization."""

    def test_base_case(self, random_patch):
        """Ensure runs with default parameters."""
        # test along the distance axis
        out = random_patch.standardize("distance")
        assert not np.any(pd.isnull(out.data))
        # test along the time axis
        out = random_patch.standardize("time")
        assert not np.any(pd.isnull(out.data))

    def test_std(self, random_patch):
        """Ensure after operation standard deviations are 1."""
        dims = random_patch.dims
        # test along the distance axis
        axis = dims.index("distance")
        out = random_patch.standardize("distance")
        assert np.allclose(
            np.round(np.std(out.data, axis=axis, keepdims=True), decimals=1), 1.0
        )
        # test along the time axis
        out = random_patch.standardize("time")
        axis = dims.index("time")
        assert np.allclose(
            np.round(np.std(out.data, axis=axis, keepdims=True), decimals=1), 1.0
        )

    def test_mean(self, random_patch):
        """Ensure after operation means are 0."""
        dims = random_patch.dims
        # test along the distance axis
        out = random_patch.standardize("distance")
        axis = dims.index("distance")
        assert np.allclose(
            np.round(np.mean(out.data, axis=axis, keepdims=True), decimals=1), 0.0
        )
        # test along the time axis
        out = random_patch.standardize("time")
        axis = dims.index("time")
        assert np.allclose(
            np.round(np.mean(out.data, axis=axis, keepdims=True), decimals=1), 0.0
        )


class TestPatchBroadcasting:
    """Tests for patches broadcasting to allow operations on each other."""

    def test_broadcast_sub_patch(self, random_patch):
        """Ensure a patch with a subset of dimensions broadcasts."""
        sub_patch = random_patch.min("time")
        out = random_patch - sub_patch
        assert isinstance(out, dc.Patch)
        axis = random_patch.get_axis("time")
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
        axis = patch_with_null.get_axis("time")
        assert patch_with_null.dims[axis] == "time"
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
        axis = patch_with_null.get_axis("distance")
        assert patch_with_null.dims[axis] == "distance"
        expected = np.all(pd.isnull(patch.data), axis=1)
        assert not np.any(expected)

    def test_3d(self, patch_3d_with_null):
        """Ensure 3D patches can have NaNs dropped."""
        patch = patch_3d_with_null
        # first test all; it shouldn't change anything
        out = patch.dropna("time", how="all")
        assert out.shape == patch.shape
        # then test any; it should drop 2 labels
        axis = patch.get_axis("time")
        out = patch.dropna("time", how="any")
        assert out.shape[axis] == patch.shape[axis] - 2

    def test_inf_dropped(self, patch_with_inf):
        """Ensure inf are also dropped when include_inf is True."""
        patch = patch_with_inf.dropna("time", include_inf=True)
        assert not np.any(np.isinf(patch.data))

    def test_inf_not_dropped(self, patch_with_inf):
        """Ensure inf are not dropped when include_inf is False."""
        patch = patch_with_inf.dropna("time", include_inf=False)
        assert np.any(np.isinf(patch.data))


class TestFillNa:
    """Tests for replacing nullish values in a patch."""

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

    def test_fill_inf(self, patch_with_inf):
        """Ensure inf get filled."""
        zero_count = np.sum(patch_with_inf.data == 0)
        pa = patch_with_inf.fillna(0, include_inf=True)
        assert zero_count < (np.sum(pa.data == 0))

    def test_fill_no_inf(self, patch_with_inf):
        """Ensure when include_inf=False inf don't get filled."""
        zero_count = np.sum(patch_with_inf.data == 0)
        pa = patch_with_inf.fillna(0, include_inf=False)
        assert zero_count == (np.sum(pa.data == 0))


class TestPad:
    """Tests for the padding functionality in a patch."""

    def test_pad_time_dimension_samples_true(self, random_patch, samples=True):
        """Test padding the time dimension with zeros before and after."""
        padded_patch = random_patch.pad(time=(2, 3), samples=samples)
        # Check if the padding is applied correctly
        original_shape = random_patch.shape
        new_shape = padded_patch.shape
        time_axis = random_patch.get_axis("time")
        assert new_shape[time_axis] == original_shape[time_axis] + 5
        # Ensure that padded values are zeros
        assert np.all(padded_patch.select(time=(None, 2), samples=samples).data == 0)
        assert np.all(padded_patch.select(time=(-3, None), samples=samples).data == 0)

    def test_pad_distance_dimension(self, random_patch):
        """Test padding the distance with same number of zeros on both sides."""
        padded_patch = random_patch.pad(distance=7)
        original_shape = random_patch.shape
        new_shape = padded_patch.shape
        distance_axis = random_patch.get_axis("distance")
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
        distance_axis = random_patch.get_axis("distance")
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
        time_axis = random_patch.get_axis("time")
        distance_axis = random_patch.get_axis("distance")
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
        axis = patch_in.get_axis("time")
        length_old = patch_in.shape[axis]
        next_fast = next_fast_len(length_old)
        out = patch_in.pad(time="fft")
        assert out.dims == patch_in.dims, "dims should not have changed."
        assert out.shape[axis] == next_fast

    def test_correlate_pad(self, random_patch):
        """Ensure the fft correlate pad returns sensible values."""
        axis = random_patch.get_axis("time")
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


class TestWhere:
    """Tests for the where method of Patch."""

    def test_where_with_boolean_array(self, random_patch):
        """Test where with a boolean array condition."""
        condition = random_patch.data > random_patch.data.mean()
        result = random_patch.where(condition)

        # Check that the result has the same shape
        assert result.shape == random_patch.shape

        # Check that values where condition is True are preserved
        assert np.allclose(result.data[condition], random_patch.data[condition])

        # Check that values where condition is False are NaN
        assert np.all(np.isnan(result.data[~condition]))

    def test_where_with_other_value(self, random_patch):
        """Test where with a replacement value."""
        condition = random_patch.data > 0
        other_value = -999
        result = random_patch.where(condition, other=other_value)

        # Check that values where condition is True are preserved
        assert np.allclose(result.data[condition], random_patch.data[condition])

        # Check that values where condition is False are replaced
        assert np.all(result.data[~condition] == other_value)

    def test_where_with_patch_condition(self, random_patch):
        """Test where with another patch as condition."""
        boolean_data = (random_patch.data > random_patch.data.mean()).astype(bool)
        boolean_patch = random_patch.new(data=boolean_data)
        result = random_patch.where(boolean_patch, other=0)

        # Check that the result has the same shape
        assert result.shape == random_patch.shape

        # Check that values where condition is True are preserved
        true_mask = boolean_data
        assert np.allclose(result.data[true_mask], random_patch.data[true_mask])

        # Check that values where condition is False are 0
        false_mask = ~boolean_data
        assert np.all(result.data[false_mask] == 0)

    def test_where_preserves_metadata(self, random_patch):
        """Test that where preserves patch metadata."""
        condition = random_patch.data > 0
        result = random_patch.where(condition)

        # Check that coordinates are preserved
        assert result.coords == random_patch.coords

        # Check that dimensions are preserved
        assert result.dims == random_patch.dims

        # Check that attributes are preserved (except history)
        assert result.attrs.model_dump(
            exclude={"history"}
        ) == random_patch.attrs.model_dump(exclude={"history"})

    def test_where_non_boolean_condition_raises(self, random_patch):
        """Test that non-boolean condition raises ValueError."""
        non_boolean_condition = random_patch.data  # Not boolean

        with pytest.raises(ValueError, match="Condition must be a boolean array"):
            random_patch.where(non_boolean_condition)

    def test_where_broadcasts_condition(self, random_patch):
        """Test that condition can be broadcast to patch shape."""
        # Create a condition that can be broadcast to the full shape
        # Create a boolean array that matches the first dimension
        condition = np.ones(random_patch.shape[0], dtype=bool)
        condition[0] = False  # Make first element False

        # This should broadcast across the second dimension
        result = random_patch.where(condition[:, np.newaxis], other=-1)
        assert result.shape == random_patch.shape

        # Check that first row is all -1 and others are preserved
        assert np.all(result.data[0, :] == -1)
        assert np.allclose(result.data[1:, :], random_patch.data[1:, :])

    def test_where_with_broadcastable_patch_other(self, random_patch):
        """Test where with a broadcastable patch as other parameter."""
        # Get the actual dimensions of the patch to create the right broadcasting
        broadcastable_patch1 = random_patch.mean("distance").squeeze()
        broadcastable_patch2 = random_patch.mean("time").squeeze()

        # Create condition
        condition = random_patch.data > random_patch.data.mean()

        for castable in [broadcastable_patch1, broadcastable_patch2]:
            result = random_patch.where(condition, other=castable)
            assert result.shape == random_patch.shape
            # Check that values where condition is True are preserved
            assert np.allclose(result.data[condition], random_patch.data[condition])
            # Check that values where condition is False come from the broadcasted other
            false_mask = ~condition
            # The exact values depend on how the broadcasting worked
            assert np.all(
                ~np.isnan(result.data[false_mask])
            )  # Should have valid values

    def test_where_with_misaligned_coords(self, random_patch):
        """Test where with condition patch having misaligned coordinates."""
        # Create a subset of the original patch with partial overlap
        time_coord = random_patch.coords.get_array("time")
        # Take only part of the time coordinates to create a partial overlap
        partial_time = time_coord[10:20]  # Use a subset

        # Create a boolean condition patch with partial time coordinates
        shifted_patch = random_patch.new(
            coords={
                "time": partial_time,
                "distance": random_patch.coords.get_array("distance"),
            },
            data=(random_patch.data[:, 10:20] > random_patch.data[:, 10:20].mean()),
        )

        # This should work with coordinate alignment (union)
        result = random_patch.where(shifted_patch, other=0)

        # The result should have the union of coordinates and correct shape
        assert result is not None
        assert isinstance(result.data, np.ndarray)
        # After alignment, coords should have the overlapping range
        result_time = result.coords.get_array("time")
        partial_time_len = len(partial_time)
        assert len(result_time) == partial_time_len

    def test_where_both_cond_and_other_misaligned(self, random_patch):
        """Test where with both condition and other patches having misaligned coords."""
        # Create condition patch with partial time overlap (first part)
        time_coord = random_patch.coords.get_array("time")
        cond_time = time_coord[5:15]  # indices 5-14

        condition_patch = random_patch.new(
            coords={
                "time": cond_time,
                "distance": random_patch.coords.get_array("distance"),
            },
            data=(random_patch.data[:, 5:15] > random_patch.data[:, 5:15].mean()),
        )

        # Create other patch with different partial time overlap (shifted range)
        other_time = time_coord[8:18]  # indices 8-17, overlaps with condition
        other_patch = random_patch.new(
            coords={
                "time": other_time,
                "distance": random_patch.coords.get_array("distance"),
            },
            data=random_patch.data[:, 8:18] * 0.5,  # Use different values
        )

        # This should work with coordinate alignment handling both patches
        result = random_patch.where(condition_patch, other=other_patch)

        # The result should have coordinates that are the intersection of all three
        assert result is not None
        assert isinstance(result.data, np.ndarray)

        # After alignment, the time coordinate should be the intersection
        result_time = result.coords.get_array("time")
        # The intersection of [5:15], [8:18], and full range should be [8:15]
        expected_overlap_len = 7  # indices 8, 9, 10, 11, 12, 13, 14
        assert len(result_time) == expected_overlap_len

        # Verify the actual time values match the expected overlap
        expected_time_values = time_coord[8:15]
        assert np.array_equal(result_time, expected_time_values)


class TestFlip:
    """Test for flipping patch."""

    def test_flipped_time(self, random_patch):
        """Test basic properties of flipped patch."""
        flipped_patch = random_patch.flip("time")

        # Dimensions and shape should be the same.
        assert flipped_patch.data.shape == random_patch.shape
        assert flipped_patch.dims == random_patch.dims

        # Time coord should have reversed.
        flipped_time = flipped_patch.coords.get_array("time")
        time = random_patch.get_array("time")
        assert np.all(flipped_time == time[::-1])

    def test_flipped_time_and_distance(self, random_patch):
        """Test basic properties of flipped patch."""
        flipped_patch = random_patch.flip("time", "distance")

        # Dimensions and shape should be the same.
        assert flipped_patch.data.shape == random_patch.shape
        assert flipped_patch.dims == random_patch.dims

        # Time and distance should have reversed.
        flipped_time = flipped_patch.coords.get_array("time")
        time = random_patch.get_array("time")
        assert np.all(flipped_time == time[::-1])
        flipped_distance = flipped_patch.coords.get_array("distance")
        distance = random_patch.get_array("distance")
        assert np.all(flipped_distance == distance[::-1])

    def test_no_coords(self, random_patch):
        """Ensure coord doesn't change with flip_coords=False"""
        flipped_patch = random_patch.flip("time", flip_coords=False)
        assert flipped_patch.get_coord("time") == random_patch.get_coord("time")

    def test_flip_reverses_data_time(self, random_patch):
        """Test that data is actually reversed along time axis."""
        axis = random_patch.get_axis("time")
        out = random_patch.flip("time")
        expected = np.flip(random_patch.data, axis=axis)
        assert np.array_equal(out.data, expected)

    def test_double_flip_restores(self, random_patch):
        """Test that double flipping restores original patch."""
        out = random_patch.flip("time").flip("time")
        assert random_patch.equals(out)

    def test_flip_coords_false_reverses_data_only(self, random_patch):
        """Test that flip_coords=False only reverses data, not coordinates."""
        axis = random_patch.get_axis("time")
        out = random_patch.flip("time", flip_coords=False)
        assert np.array_equal(out.data, np.flip(random_patch.data, axis=axis))
        assert out.get_coord("time") == random_patch.get_coord("time")

    def test_no_op(self, random_patch):
        """Ensure passing no dims does nothing."""
        out = random_patch.flip()
        assert out is random_patch
