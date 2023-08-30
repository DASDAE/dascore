"""Tests for applying rolling functions on patch's data."""

import numpy as np
import pytest
import pandas as pd

import dascore as dc
from dascore.exceptions import ParameterError
from dascore.units import m


class TestRolling:
    """Tests for applying rolling functions on patch's data."""

    window = 16
    step = 8

    @pytest.fixture(scope="class")
    def range_patch(self, random_patch):
        """Return a patch with sequential values for its 0th dim."""
        new_data = np.ones_like(random_patch.data)
        range_array = np.arange(0, new_data.shape[0])[:, None]
        return random_patch.new(data=new_data * range_array)

    @pytest.fixture(scope="class")
    def dist_roll_range_patch(self, range_patch):
        """Return a patch with sequential values for its 0th dim."""
        dist_step = range_patch.get_coord("distance").step
        out = range_patch.rolling(distance=dist_step * self.window).max()
        return out

    def test_apply_correct_no_step(self, dist_roll_range_patch):
        """Ensure the apply is correct without using a step size."""
        data_no_nan = dist_roll_range_patch.dropna("distance").data
        # determine what the values should be:
        ax_0_len = dist_roll_range_patch.shape[0]
        samps = self.window - 1
        expected_values = np.arange(samps, ax_0_len)[:, None]
        # and assert they are indeed that.
        assert np.allclose(data_no_nan, expected_values)

    def test_along_time(self, range_patch):
        """Ensure time axis also works."""
        time_step = range_patch.get_coord("time").step
        out = range_patch.rolling(time=time_step * self.window).sum()
        shape = range_patch.shape
        # Note: we start at 0 here because range is along 0th axis and
        # time is first axis so all windows are the same values.
        expected = (np.arange(0, shape[0]) * self.window)[:, None]
        assert np.allclose(out.dropna("time").data, expected)
        # this should also work if patch is transposed.
        trans = range_patch.transpose()
        out = trans.rolling(time=time_step * self.window).sum()
        assert np.allclose(out.dropna("time").data, expected.transpose())

    def test_apply_with_step(self, range_patch):
        """Ensure apply works with various step sizes."""
        # first calculate rolling max on time axis.
        step = range_patch.get_coord("distance").step
        out = (
            range_patch.rolling(distance=self.window * step, step=self.step * step)
            .max()
            .dropna("distance")
        )
        # Determine what output should be.
        vals = np.arange(self.window - 1, range_patch.shape[0])
        start = (self.step - ((self.window - 2) % self.step)) % self.step
        expected = vals[start :: self.step, None]
        assert np.allclose(out.data, expected)

    def test_1D_patch(self):
        """Ensure rolling works with 1D patch."""
        patch = dc.Patch(
            data=np.arange(10),
            coords={"time": dc.to_datetime64(np.arange(10))},
            dims=("time",),
        )
        expected = patch.data[:-2]
        out = patch.rolling(time=3).min().dropna("time")
        assert np.allclose(expected, out.data)

    def test_3D_patch(self, range_patch_3d):
        """Ensure rolling works with 3D patch."""
        patch = range_patch_3d
        # first try along time axis.
        out = patch.rolling(time=3).min().dropna("time")
        expected = np.arange(range_patch_3d.shape[0] - 2)
        assert np.allclose(out.data, expected[:, None, None])
        # then distance
        out = patch.rolling(distance=3).min().dropna("distance")
        expected = np.arange(range_patch_3d.shape[0])
        assert np.allclose(out.data, expected[:, None, None])
        # then smell
        out = patch.rolling(smell=3).min().dropna("smell")
        expected = np.arange(range_patch_3d.shape[0])
        assert np.allclose(out.data, expected[:, None, None])

    def test_window_too_big_raises(self, random_patch):
        """When the window is too large it should raise."""
        coord = random_patch.get_coord("time")
        duration = coord.max() - coord.min()
        msg = "Window or step size is larger than"
        with pytest.raises(ParameterError, match=msg):
            random_patch.rolling(time=duration * 2)

    def test_misc(self, random_patch):
        """Test miscellaneous functionality."""
        time_step = random_patch.get_coord("time").step
        rolling = random_patch.rolling(time=6 * time_step, step=2 * time_step)
        assert isinstance(rolling.mean(), dc.Patch)
        assert isinstance(rolling.median(), dc.Patch)
        assert isinstance(rolling.min(), dc.Patch)
        assert isinstance(rolling.max(), dc.Patch)
        assert isinstance(rolling.std(), dc.Patch)

    @pytest.mark.parametrize("_", list(range(10)))
    def test_apply_axis_0_dist_time(self, range_patch, _):
        """Test the apply method of PatchRoller when distance coordinate is entered
        and the first axis is distance.
        """
        patch = range_patch
        axis = patch.dims.index("distance")
        # random window and step sizes for each trial run.
        window = np.random.randint(1, patch.shape[axis])
        step = np.random.randint(1, patch.shape[axis])
        # setup patch and apply mean.
        channel_spacing = patch.attrs["distance_step"]
        roller = patch.rolling(
            distance=(window * channel_spacing) * m,
            step=(step * channel_spacing) * m,
        )
        applied_result = roller.apply(np.mean).data
        # do the same with pandas and compare result
        df = pd.DataFrame(patch.data)
        rolling_mean_pandas = df.rolling(window, step=step, axis=axis).mean()
        valid_data = ~np.isnan(np.array(rolling_mean_pandas)).any(axis=1)
        filtered_data_pandas = np.array(rolling_mean_pandas)[valid_data]
        assert applied_result.shape == np.array(rolling_mean_pandas).shape
        assert np.allclose(applied_result, filtered_data_pandas)

    # @pytest.mark.parametrize("_", list(range(10)))
    # def test_apply_axis_1_dist_time(self, _):
    #     """Test the apply method of PatchRoller when time coordinate is entered
    #     and the first axis is distance.
    #     """
    #     random_das_dist_time = dc.get_example_patch("random_das")
    #     axis = 1
    #     sampling_interval = random_das_dist_time.attrs["time_step"] / np.timedelta64(
    #         1, "s"
    #     )
    #     window = np.random.randint(1, 100)
    #     step = np.random.randint(1, 100)
    #     roller = PatchRoller(
    #         random_das_dist_time,
    #         time=(window * sampling_interval) * s,
    #         step=(step * sampling_interval) * s,
    #     )
    #     applied_result = roller.apply(np.mean).data

    #     valid_data = ~np.isnan(applied_result).any(axis=0)
    #     filtered_data_rolling = applied_result[:, valid_data]

    #     df = pd.DataFrame(random_das_dist_time.data)
    #     rolling_mean_pandas = df.rolling(window, step=step, axis=axis).mean()

    #     valid_data = ~np.isnan(np.array(rolling_mean_pandas)).any(axis=0)
    #     filtered_data_pandas = np.array(rolling_mean_pandas)[:, valid_data]

    #     assert applied_result.shape == np.array(rolling_mean_pandas).shape
    #     assert np.allclose(filtered_data_rolling, filtered_data_pandas)

    # @pytest.mark.parametrize("_", list(range(10)))
    # def test_apply_axis_0_time_dist(self, _):
    #     """Test the apply method of PatchRoller when time coordinate is entered
    #     and the first axis is time.
    #     """
    #     random_das_time_dist = dc.get_example_patch("random_das_time_dist")
    #     axis = 0
    #     sampling_interval = random_das_time_dist.attrs["time_step"] / np.timedelta64(
    #         1, "s"
    #     )
    #     window = np.random.randint(1, 100)
    #     step = np.random.randint(1, 100)
    #     roller = PatchRoller(
    #         random_das_time_dist,
    #         time=(window * sampling_interval) * s,
    #         step=(step * sampling_interval) * s,
    #     )
    #     applied_result = roller.apply(np.mean).data

    #     valid_data = ~np.isnan(applied_result).any(axis=1)
    #     filtered_data_rolling = applied_result[valid_data]

    #     df = pd.DataFrame(random_das_time_dist.data)
    #     rolling_mean_pandas = df.rolling(window, step=step, axis=axis).mean()

    #     valid_data = ~np.isnan(np.array(rolling_mean_pandas)).any(axis=1)
    #     filtered_data_pandas = np.array(rolling_mean_pandas)[valid_data]

    #     assert applied_result.shape == np.array(rolling_mean_pandas).shape
    #     assert np.allclose(filtered_data_rolling, filtered_data_pandas)

    # @pytest.mark.parametrize("_", list(range(10)))
    # def test_apply_axis_1_time_dist(self, _):
    #     """Test the apply method of PatchRoller when distance coordinate is entered
    #     and the first axis is time."""
    #     random_das_time_dist = dc.get_example_patch("random_das_time_dist")
    #     axis = 1
    #     channel_spacing = random_das_time_dist.attrs["distance_step"]
    #     window = np.random.randint(1, 100)
    #     step = np.random.randint(1, 100)
    #     roller = PatchRoller(
    #         random_das_time_dist,
    #         distance=(window * channel_spacing) * m,
    #         step=(step * channel_spacing) * m,
    #     )
    #     applied_result = roller.apply(np.mean).data

    #     valid_data = ~np.isnan(applied_result).any(axis=0)
    #     filtered_data_rolling = applied_result[:, valid_data]

    #     df = pd.DataFrame(random_das_time_dist.data)
    #     rolling_mean_pandas = df.rolling(window, step=step, axis=axis).mean()

    #     valid_data = ~np.isnan(np.array(rolling_mean_pandas)).any(axis=0)
    #     filtered_data_pandas = np.array(rolling_mean_pandas)[:, valid_data]

    #     assert applied_result.shape == np.array(rolling_mean_pandas).shape
    #     assert np.allclose(filtered_data_rolling, filtered_data_pandas)

    # def test_center(self):
    #     """Test the center option in PatchRoller."""
    #     random_das_dist_time = dc.get_example_patch("random_das")
    #     axis = 0
    #     channel_spacing = random_das_dist_time.attrs["distance_step"]
    #     window = np.random.randint(1, 100)
    #     step = np.random.randint(1, 100)
    #     roller = PatchRoller(
    #         random_das_dist_time,
    #         distance=(window * channel_spacing) * m,
    #         step=(step * channel_spacing) * m,
    #     )
    #     applied_result = roller.apply(np.mean).data

    #     valid_data = ~np.isnan(applied_result).any(axis=1)
    #     filtered_data_rolling = applied_result[valid_data]

    #     df = pd.DataFrame(random_das_dist_time.data)
    #     rolling_mean_pandas = df.rolling(window, step=step, axis=axis).mean()

    #     valid_data = ~np.isnan(np.array(rolling_mean_pandas)).any(axis=1)
    #     filtered_data_pandas = np.array(rolling_mean_pandas)[valid_data]

    #     assert applied_result.shape == np.array(rolling_mean_pandas).shape
    #     assert np.allclose(filtered_data_rolling, filtered_data_pandas)
