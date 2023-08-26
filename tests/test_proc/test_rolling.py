"""Tests for applying rolling functions on patch's data."""

import pandas as pd
import numpy as np

import dascore as dc
from dascore.proc.rolling import PatchRoller
from dascore.units import s


class TestRolling:
    """Tests for applying rolling functions on patch's data."""

    # @pytest.fixture(scope="class")
    # def my_patch_dis_time(self, random_patch):
    #     """Create a Patch containing known data with (distance, time)."""
    #     num_rows, num_cols = random_patch.data.shape
    #     my_data = np.random.random((num_rows,num_cols))
    #     new_patch = random_patch.new(data=my_data)
    #     return new_patch

    # @pytest.fixture(scope="class")
    # def my_patch_time_dis(self):
    #     """Create a Patch containing known data with (time, distance)."""

    #     array = np.random.random(size=(2_000, 300))
    #     t1 = np.datetime64("2017-09-18")
    #     attrs = dict(
    #         d_distance=1,
    #         d_time=to_timedelta64(1 / 250),
    #         category="DAS",
    #         id="test_data1",
    #         time_min=t1,
    #     )
    #     coords = dict(
    #         time=np.arange(array.shape[0]) * attrs["d_time"],
    #         distance=np.arange(array.shape[1]) * attrs["d_distance"] ,)
    #     dims = ('time', 'distance')

    # pa = dc.Patch(data=array, coords=coords, attrs=attrs, dims=dims)

    #     return pa

    def test_apply_axis_0_dist_time(self):
        """Test the apply method of PatchRoller when time coordinate is entered
        and the fist axis is distance.
        """
        random_das_dist_time = dc.get_example_patch("random_das")
        axis = 0
        if random_das_dist_time.dims[1] == "time":
            axis = 1
        sampling_interval = random_das_dist_time.attrs["time_step"] / np.timedelta64(
            1, "s"
        )
        window = 6
        step = 2
        roller = PatchRoller(
            random_das_dist_time,
            time=(window * sampling_interval) * s,
            step=(step * sampling_interval) * s,
        )
        applied_result = roller.apply(np.mean)

        valid_data = ~np.isnan(applied_result).any(axis=axis)
        filtered_data_rolling = applied_result[valid_data]

        df = pd.DataFrame(random_das_dist_time.data)
        rolling_mean_pandas = df.rolling(window, step=step, axis=axis).mean()

        valid_data = ~np.isnan(np.array(rolling_mean_pandas)).any(axis=axis)
        filtered_data_pandas = np.array(rolling_mean_pandas)[valid_data]

        assert applied_result.shape == np.array(rolling_mean_pandas).shape
        assert np.allclose(filtered_data_rolling, filtered_data_pandas)

    def test_apply_axis_1_dis_time(self):
        """Test the apply method of PatchRoller when distance coordinate is entered
        and the fist axis is distance.
        """
        random_das_dist_time = dc.get_example_patch("random_das")
        axis = 1
        if random_das_dist_time.dims[0] == "distance":
            axis = 0
        # channel_spacing = my_patch_dis_time.attrs['distance_step']
        window = 6
        step = 2
        roller = PatchRoller(random_das_dist_time, distance=window, step=step)
        applied_result = roller.apply(np.mean)

        valid_data = ~np.isnan(applied_result).any(axis=axis)
        filtered_data_rolling = applied_result[:, valid_data]

        df = pd.DataFrame(random_das_dist_time.data)
        rolling_mean_pandas = df.rolling(window, step=step, axis=axis).mean()

        valid_data = ~np.isnan(np.array(rolling_mean_pandas)).any(axis=axis)
        filtered_data_pandas = np.array(rolling_mean_pandas)[:, valid_data]

        assert applied_result.shape == np.array(rolling_mean_pandas).shape
        assert np.allclose(filtered_data_rolling, filtered_data_pandas)

    def test_apply_axis_0_time_dist(self):
        """Test the apply method of PatchRoller when time coordinate is entered
        and the fist axis is time.
        """
        random_das_time_dist = dc.get_example_patch("random_das_time_dist")
        axis = 0
        if random_das_time_dist.dims[1] == "time":
            axis = 1
        sampling_interval = random_das_time_dist.attrs["time_step"] / np.timedelta64(
            1, "s"
        )
        window = 6
        step = 2
        roller = PatchRoller(
            random_das_time_dist,
            time=(window * sampling_interval) * s,
            step=(step * sampling_interval) * s,
        )
        applied_result = roller.apply(np.mean)

        valid_data = ~np.isnan(applied_result).any(axis=axis)
        filtered_data_rolling = applied_result[valid_data]

        df = pd.DataFrame(random_das_time_dist.data)
        rolling_mean_pandas = df.rolling(window, step=step, axis=axis).mean()

        valid_data = ~np.isnan(np.array(rolling_mean_pandas)).any(axis=axis)
        filtered_data_pandas = np.array(rolling_mean_pandas)[valid_data]

        assert applied_result.shape == np.array(rolling_mean_pandas).shape
        assert np.allclose(filtered_data_rolling, filtered_data_pandas)

    def test_apply_axis_1_time_dis(self):
        """Test the apply method of PatchRoller when distance coordinate is entered
        and the fist axis is time."""
        random_das_time_dist = dc.get_example_patch("random_das_time_dist")
        axis = 1
        if random_das_time_dist.dims[0] == "distance":
            axis = 0
        # channel_spacing = my_patch.attrs['distance_step']
        window = 6
        step = 2
        roller = PatchRoller(random_das_time_dist, distance=window, step=step)
        applied_result = roller.apply(np.mean)

        valid_data = ~np.isnan(applied_result).any(axis=axis)
        filtered_data_rolling = applied_result[:, valid_data]

        df = pd.DataFrame(random_das_time_dist.data)
        rolling_mean_pandas = df.rolling(window, step=step, axis=axis).mean()

        valid_data = ~np.isnan(np.array(rolling_mean_pandas)).any(axis=axis)
        filtered_data_pandas = np.array(rolling_mean_pandas)[:, valid_data]

        assert applied_result.shape == np.array(rolling_mean_pandas).shape
        assert np.allclose(filtered_data_rolling, filtered_data_pandas)

    # def test_center(self, my_patch):
    #     """Test the center option in PatchRoller."""
    #     sampling_interval = my_patch.attrs['time_step']/np.timedelta64(1, 's')
    #     window = 6
    #     step = 2
    #     roll = my_patch.rolling(time=(window*sampling_interval)*s,
    # step=(step*sampling_interval)*s, center=True)
    #     centered_result = roll.mean()

    #     df = pd.DataFrame(my_patch.data)
    #     rolling_mean_pandas = df.rolling(3, step=2, axis=0, center=True).mean()

    #     assert centered_result.shape == np.array(rolling_mean_pandas).shape
    #     assert np.allclose(centered_result, rolling_mean_pandas)

    def test_misc(self, random_patch):
        """Test miscellaneous functionality."""
        sampling_interval = random_patch.attrs["time_step"] / np.timedelta64(1, "s")
        window = 6
        step = 2
        rolling = random_patch.rolling(
            time=(window * sampling_interval) * s, step=(step * sampling_interval) * s
        )
        assert isinstance(rolling.mean())
        assert isinstance(rolling.median())
        assert isinstance(rolling.min())
        assert isinstance(rolling.max())
