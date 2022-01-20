"""
Tests for decimation
"""
import numpy as np
import pytest

from dascore.exceptions import FilterValueError
from dascore.utils.patch import get_start_stop_step


class TestInterpolate:
    """Tests for interpolating data along an axis in patch."""

    def test_interp_upsample_distance(self, random_patch):
        """Ensure interpolation between distance works."""
        start, stop, step = get_start_stop_step(random_patch, "distance")
        axis = random_patch.dims.index("distance")
        new_sampling = step / 2.2
        new_coord = np.arange(start, stop, new_sampling)
        out = random_patch.interpolate(distance=new_coord)
        assert out.data.shape[axis] == len(new_coord)
        assert np.all(out.coords["distance"] == new_coord)

    def test_interp_down_sample_distance(self, random_patch):
        """Ensure interp can be used to downsample data."""
        start, stop, step = get_start_stop_step(random_patch, "distance")
        axis = random_patch.dims.index("distance")
        new_sampling = step / 0.1
        new_coord = np.arange(start, stop, new_sampling)
        out = random_patch.interpolate(distance=new_coord)
        assert out.data.shape[axis] == len(new_coord)
        assert np.all(out.coords["distance"] == new_coord)

    def test_uneven_sampling_rates(self, random_patch):
        """Ensure uneven sampling raises an exception."""
        start, stop, step = get_start_stop_step(random_patch, "distance")
        new = [start, start + step, start + 4 * step, stop]
        with pytest.raises(FilterValueError):
            random_patch.interpolate(distance=new)

    def test_upsample_time(self, random_patch):
        """Ensure time can be upsampled."""
        start, stop, step = get_start_stop_step(random_patch, "time")
        new = np.arange(start, stop, step / 2)
        # out = random_patch.interpolate(time=new)
        breakpoint()


#
# class TestDecimate:
#     """Ensure Patch can be decimated."""
#
#     def test_decimate_no_lowpass(self, random_patch):
#         """Simple decimation"""
#         p1 = random_patch
#         old_time = p1.coords["time"]
#         old_dt = old_time[1:] - old_time[:-1]
#         # apply decimation,
#         pa2 = random_patch.decimate(2)
#         new_time = pa2.coords["time"]
#         new_dt = new_time[1:] - new_time[:-1]
#         # ensure distance between time samples and shapes have changed
#         len_ratio = np.round(len(old_dt) / len(new_dt))
#         assert np.isclose(len_ratio, 2.0)
#         dt_ratio = np.round(new_dt[0] / old_dt[0])
#         assert np.isclose(dt_ratio, 2.0)
#
#     def test_update_time_max(self, random_patch):
#         """Ensure the time_max is updated after decimation."""
#         out = random_patch.decimate(10)
#         assert out.attrs["time_max"] == out.coords["time"].max()
#
#     def test_update_delta_dim(self, random_patch):
#         """
#         Since decimate changes the spacing of dimension this should be updated.
#         """
#         dt1 = random_patch.attrs["d_time"]
#         out = random_patch.decimate(10)
#         assert out.attrs["d_time"] == dt1 * 10
#
#
# class TestResample:
#     """
#     Tests for resampling along a given dimension.
#     """
#
#     def test_resample_time(self, random_patch):
#         """Resample the patch to a specified dt."""
#         patch = random_patch
#         axis = patch.dims.index('time')
#         new_dt = np.timedelta64(10, "ms")
#         new = patch.resample(time=new_dt)
#         assert new_dt == new.attrs['d_time']
#         assert np.all(np.diff(new.coords['time']) == new_dt)
#         # ensure only the time dimension has changed.
#         shape1, shape2 = random_patch.data.shape, new.data.shape
#         for ax, (len1, len2) in enumerate(zip(shape1, shape2)):
#             if ax == axis:  # Only resampled axis should have changed len
#                 assert len1 > len2
#             else:
#                 assert len1 == len2
#
#     def test_upsample_time(self, random_patch):
#         """Ensure time dimension can be upsampled."""
#         current_dt = random_patch.attrs["d_time"]
#         axis = random_patch.dims.index('time')
#         new_dt = current_dt / 2
#         new = random_patch.resample(time=new_dt)
#         assert new_dt == new.attrs['d_time']
#         assert np.all(np.diff(new.coords['time']) == new_dt)
#         shape1, shape2 = random_patch.data.shape, new.data.shape
#         for ax, (len1, len2) in enumerate(zip(shape1, shape2)):
#             if ax == axis:  # Only resampled axis should have changed len
#                 assert len1 < len2
#             else:
#                 assert len1 == len2
#
#     def test_resample_distance(self, random_patch):
#         """Ensure distance dimension is also resample-able."""
#         current_dx = random_patch.attrs['d_distance']
#         new_dx = current_dx / 2
#         new = random_patch.resample(distance=new_dx)
#         axis = random_patch.dims.index('distance')
#         assert new_dx == new.attrs['d_distance']
#         assert np.allclose(np.diff(new.coords['distance']), new_dx)
#         shape1, shape2 = random_patch.data.shape, new.data.shape
#         for ax, (len1, len2) in enumerate(zip(shape1, shape2)):
#             if ax == axis:  # Only resampled axis should have changed len
#                 assert len1 < len2
#             else:
#                 assert len1 == len2
#
#
# class TestIResample:
#     """Tests for resampling based on numbers of sample per dimension."""
#
#     def test_iresample_time(self, random_patch):
#         """Tests iresample in time dim."""
#         time_samples = 40
#         out = random_patch.iresample(time=time_samples)
#         assert len(out.coords['time']) == time_samples
