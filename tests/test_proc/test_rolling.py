"""Tests for applying rolling functions on patch's data."""

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.proc.rolling import PatchRoller
from dascore.units import s, m


class TestRolling:
    """Tests for applying rolling functions on patch's data."""

    @pytest.mark.parametrize("_", list(range(10)))
    def test_apply_axis_0_dist_time(self, _):
        """Test the apply method of PatchRoller when distance coordinate is entered
        and the first axis is distance.
        """
        random_das_dist_time = dc.get_example_patch("random_das")
        axis = 0
        channel_spacing = random_das_dist_time.attrs["distance_step"]
        window = np.random.randint(1, 100)
        step = np.random.randint(1, 100)
        roller = PatchRoller(
            random_das_dist_time,
            distance=(window * channel_spacing) * m,
            step=(step * channel_spacing) * m,
        )
        applied_result = roller.apply(np.mean).data

        valid_data = ~np.isnan(applied_result).any(axis=1)
        filtered_data_rolling = applied_result[valid_data]

        df = pd.DataFrame(random_das_dist_time.data)
        rolling_mean_pandas = df.rolling(window, step=step, axis=axis).mean()

        valid_data = ~np.isnan(np.array(rolling_mean_pandas)).any(axis=1)
        filtered_data_pandas = np.array(rolling_mean_pandas)[valid_data]

        assert applied_result.shape == np.array(rolling_mean_pandas).shape
        assert np.allclose(filtered_data_rolling, filtered_data_pandas)

    @pytest.mark.parametrize("_", list(range(10)))
    def test_apply_axis_1_dist_time(self, _):
        """Test the apply method of PatchRoller when time coordinate is entered
        and the first axis is distance.
        """
        random_das_dist_time = dc.get_example_patch("random_das")
        axis = 1
        sampling_interval = random_das_dist_time.attrs["time_step"] / np.timedelta64(
            1, "s"
        )
        window = np.random.randint(1, 100)
        step = np.random.randint(1, 100)
        roller = PatchRoller(
            random_das_dist_time,
            time=(window * sampling_interval) * s,
            step=(step * sampling_interval) * s,
        )
        applied_result = roller.apply(np.mean).data

        valid_data = ~np.isnan(applied_result).any(axis=0)
        filtered_data_rolling = applied_result[:, valid_data]

        df = pd.DataFrame(random_das_dist_time.data)
        rolling_mean_pandas = df.rolling(window, step=step, axis=axis).mean()

        valid_data = ~np.isnan(np.array(rolling_mean_pandas)).any(axis=0)
        filtered_data_pandas = np.array(rolling_mean_pandas)[:, valid_data]

        assert applied_result.shape == np.array(rolling_mean_pandas).shape
        assert np.allclose(filtered_data_rolling, filtered_data_pandas)

    @pytest.mark.parametrize("_", list(range(10)))
    def test_apply_axis_0_time_dist(self, _):
        """Test the apply method of PatchRoller when time coordinate is entered
        and the first axis is time.
        """
        random_das_time_dist = dc.get_example_patch("random_das_time_dist")
        axis = 0
        sampling_interval = random_das_time_dist.attrs["time_step"] / np.timedelta64(
            1, "s"
        )
        window = np.random.randint(1, 100)
        step = np.random.randint(1, 100)
        roller = PatchRoller(
            random_das_time_dist,
            time=(window * sampling_interval) * s,
            step=(step * sampling_interval) * s,
        )
        applied_result = roller.apply(np.mean).data

        valid_data = ~np.isnan(applied_result).any(axis=1)
        filtered_data_rolling = applied_result[valid_data]

        df = pd.DataFrame(random_das_time_dist.data)
        rolling_mean_pandas = df.rolling(window, step=step, axis=axis).mean()

        valid_data = ~np.isnan(np.array(rolling_mean_pandas)).any(axis=1)
        filtered_data_pandas = np.array(rolling_mean_pandas)[valid_data]

        assert applied_result.shape == np.array(rolling_mean_pandas).shape
        assert np.allclose(filtered_data_rolling, filtered_data_pandas)

    @pytest.mark.parametrize("_", list(range(10)))
    def test_apply_axis_1_time_dist(self, _):
        """Test the apply method of PatchRoller when distance coordinate is entered
        and the first axis is time."""
        random_das_time_dist = dc.get_example_patch("random_das_time_dist")
        axis = 1
        channel_spacing = random_das_time_dist.attrs["distance_step"]
        window = np.random.randint(1, 100)
        step = np.random.randint(1, 100)
        roller = PatchRoller(
            random_das_time_dist,
            distance=(window * channel_spacing) * m,
            step=(step * channel_spacing) * m,
        )
        applied_result = roller.apply(np.mean).data

        valid_data = ~np.isnan(applied_result).any(axis=0)
        filtered_data_rolling = applied_result[:, valid_data]

        df = pd.DataFrame(random_das_time_dist.data)
        rolling_mean_pandas = df.rolling(window, step=step, axis=axis).mean()

        valid_data = ~np.isnan(np.array(rolling_mean_pandas)).any(axis=0)
        filtered_data_pandas = np.array(rolling_mean_pandas)[:, valid_data]

        assert applied_result.shape == np.array(rolling_mean_pandas).shape
        assert np.allclose(filtered_data_rolling, filtered_data_pandas)

    def test_center(self):
        """Test the center option in PatchRoller."""
        random_das_dist_time = dc.get_example_patch("random_das")
        axis = 0
        channel_spacing = random_das_dist_time.attrs["distance_step"]
        window = np.random.randint(1, 100)
        step = np.random.randint(1, 100)
        roller = PatchRoller(
            random_das_dist_time,
            distance=(window * channel_spacing) * m,
            step=(step * channel_spacing) * m,
        )
        applied_result = roller.apply(np.mean).data

        valid_data = ~np.isnan(applied_result).any(axis=1)
        filtered_data_rolling = applied_result[valid_data]

        df = pd.DataFrame(random_das_dist_time.data)
        rolling_mean_pandas = df.rolling(window, step=step, axis=axis).mean()

        valid_data = ~np.isnan(np.array(rolling_mean_pandas)).any(axis=1)
        filtered_data_pandas = np.array(rolling_mean_pandas)[valid_data]

        assert applied_result.shape == np.array(rolling_mean_pandas).shape
        assert np.allclose(filtered_data_rolling, filtered_data_pandas)

    def test_misc(self, random_patch):
        """Test miscellaneous functionality."""
        sampling_interval = random_patch.attrs["time_step"] / np.timedelta64(1, "s")
        window = np.random.randint(1, 100)
        step = np.random.randint(1, 100)
        rolling = random_patch.rolling(
            time=(window * sampling_interval) * s, step=(step * sampling_interval) * s
        )
        assert isinstance(rolling.mean(), dc.Patch)
        assert isinstance(rolling.median(), dc.Patch)
        assert isinstance(rolling.min(), dc.Patch)
        assert isinstance(rolling.max(), dc.Patch)
        assert isinstance(rolling.std(), dc.Patch)
        assert isinstance(rolling.sum(), dc.Patch)
