"""Tests for applying rolling functions on patch's data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dascore as dc
import dascore.proc.coords
from dascore.exceptions import ParameterError
from dascore.units import m
from dascore.utils.misc import all_close
from dascore.utils.pd import rolling_df


@pytest.fixture(scope="class")
def range_patch(random_patch):
    """Return a patch with sequential values for its 0th dim."""
    new_data = np.ones_like(random_patch.data)
    range_array = np.arange(0, new_data.shape[0])[:, None]
    return random_patch.new(data=new_data * range_array)


class TestRolling:
    """Tests for applying rolling functions on patch's data."""

    window = 16
    step = 8

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

    def test_rolling_timdelta(self, random_patch):
        """Ensure rolling works with timedeltas."""
        time_step = random_patch.get_coord("time").step
        time = time_step * self.window
        out1 = random_patch.rolling(time=dc.to_timedelta64(time)).sum()
        out2 = random_patch.rolling(time=time * dc.get_quantity("s")).sum()
        assert all_close(out1, out2)

    def test_apply_with_step(self, range_patch):
        """Ensure apply works with various step sizes."""
        # first calculate rolling max on time axis.
        step = range_patch.get_coord("distance").step
        roll = range_patch.rolling(distance=self.window * step, step=self.step * step)

        out = roll.max().dropna("distance")
        start = roll.get_start_index()
        # Determine what output should be.
        vals = np.arange(self.window - 1, range_patch.shape[0])
        expected = vals[start :: self.step, None]
        assert np.allclose(out.data, expected)

    def test_window_size_one(self, random_patch):
        """Ensure we can get a window size of one."""
        time_step = random_patch.get_coord("time").step
        out = random_patch.rolling(time=time_step, step=None).mean()
        # window size of 1 with no step results in the same as input.
        assert out.equals(random_patch)

    def test_1d_patch(self):
        """Ensure rolling works with 1D patch."""
        patch = dc.Patch(
            data=np.arange(10),
            coords={"time": dc.to_datetime64(np.arange(10))},
            dims=("time",),
        )
        expected = patch.data[:-2]
        roll = patch.rolling(time=3)
        out = roll.min().dropna("time")
        assert np.allclose(expected, out.data)

    def test_3d_patch(self, range_patch_3d):
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

    def test_window_zero_raises(self, random_patch):
        """When the window or step is entered 0 it should raise."""
        random_patch.get_coord("time")
        msg = "Window or step size can't be zero"
        with pytest.raises(ParameterError, match=msg):
            random_patch.rolling(time=0)

    def test_window_too_big_raises(self, random_patch):
        """When the window or step is too large it should raise."""
        coord = random_patch.get_coord("time")
        duration = coord.max() - coord.min()
        msg = "results in a window larger than coordinate"
        with pytest.raises(ParameterError, match=msg):
            random_patch.rolling(time=duration * 2)

    def test_center(self, random_patch):
        """Ensure the center option places NaN at start and end."""
        out = random_patch.rolling(time=1, center=True).mean()
        time_ax = random_patch.get_axis("time")
        first_label = np.take(out.data, -1, axis=time_ax)
        assert np.all(np.isnan(first_label))
        last_label = np.take(out.data, 0, axis=time_ax)
        assert np.all(np.isnan(last_label))

    @pytest.mark.parametrize("_", list(range(5)))
    def test_compare_to_pandas(self, range_patch, _):
        """Test the apply method of PatchRoller when distance coordinate is entered
        and the first axis is distance.
        """
        random = np.random.RandomState(42)
        patch = range_patch
        axis = patch.get_axis("distance")
        # random window and step sizes for each trial run.
        window = random.randint(1, patch.shape[axis])
        step = random.randint(1, patch.shape[axis])
        # setup patch and apply mean.
        channel_spacing = patch.attrs["distance_step"]
        roller = patch.rolling(
            distance=(window * channel_spacing) * m,
            step=(step * channel_spacing) * m,
        )
        applied_result = roller.apply(np.mean).dropna("distance").data
        # do the same with pandas and compare result
        df = pd.DataFrame(patch.data)
        rolling_mean_pandas = rolling_df(df, window, step=step, axis=axis).mean()
        filtered_data_pandas = rolling_mean_pandas.dropna(axis=axis).values
        assert applied_result.shape == filtered_data_pandas.shape
        assert np.allclose(applied_result, filtered_data_pandas)

    def test_pandas_engine_raises_3d_patch(self, range_patch_3d):
        """Pandas engine should raise when required with a 4d patch."""
        msg = "Cannot use Pandas engine on patches with more than"
        with pytest.raises(ParameterError, match=msg):
            range_patch_3d.rolling(time=1, engine="pandas").mean()

    def test_misc(self, random_patch):
        """Test miscellaneous functionality."""
        time_step = random_patch.get_coord("time").step
        window = 10 * time_step
        step = 10 * time_step
        rollers = (
            random_patch.rolling(time=window, step=step, engine="numpy"),
            random_patch.rolling(time=window, step=step, engine="pandas"),
        )
        for roller in rollers:
            assert isinstance(roller.mean(), dc.Patch)
            assert isinstance(roller.median(), dc.Patch)
            assert isinstance(roller.min(), dc.Patch)
            assert isinstance(roller.max(), dc.Patch)
            assert isinstance(roller.std(), dc.Patch)
            assert isinstance(roller.sum(), dc.Patch)

    def test_pandas_apply(self, random_patch):
        """Test pandas apply works."""
        # This can be very slow so we use a large window and step size.
        dt = random_patch.get_coord("time").step
        time_len = random_patch.shape[random_patch.get_axis("time")]
        window = (time_len - 1) * dt
        step = (time_len - 1) * dt
        roll1 = random_patch.rolling(time=window, step=step, engine="pandas")
        roll2 = random_patch.rolling(time=window, step=step, engine="numpy")
        patch_1 = roll1.apply(np.sum)
        patch_2 = roll2.apply(np.sum)
        assert patch_2.shape == patch_1.shape


class TestNumpyVsPandasRolling:
    """Ensure numpy rolling return the same results as pandas rolling."""

    # window size/step in terms of steps
    combinations = (
        (1, 1),
        (120, 60),
        (100, 100),
        (13, 7),
        (40, 80),
    )

    @pytest.mark.parametrize("data", combinations)
    def test_time_dim(self, data: tuple[int, int], range_patch):
        """Ensure pandas and numpy engine return same results along time."""
        coord = range_patch.get_coord("time")
        dt = coord.step
        win, step = dt * data[0], dt * data[1]
        numpy_roll = range_patch.rolling(time=win, step=step, engine="numpy")
        pandas_roll = range_patch.rolling(time=win, step=step, engine="pandas")

        nump_out = numpy_roll.mean()
        pand_out = pandas_roll.mean()

        is_close = np.isclose(nump_out.data, pand_out.data)
        is_nan = np.isnan(nump_out.data) & np.isnan(pand_out.data)
        assert np.all(is_close | is_nan)

    @pytest.mark.parametrize("data", combinations)
    def test_dist_dim(self, data: tuple[int, int], range_patch):
        """Ensure pandas and numpy engine return same results along distance."""
        coord = range_patch.get_coord("distance")
        dt = coord.step
        win, step = dt * data[0], dt * data[1]
        numpy_roll = range_patch.rolling(distance=win, step=step, engine="numpy")
        pandas_roll = range_patch.rolling(distance=win, step=step, engine="pandas")

        nump_out = numpy_roll.mean()
        pand_out = pandas_roll.mean()

        is_close = np.isclose(nump_out.data, pand_out.data)
        is_nan = np.isnan(nump_out.data) & np.isnan(pand_out.data)
        assert np.all(is_close | is_nan)

    def test_center_same(self, range_patch):
        """Ensure center values are handled the same."""
        dt = range_patch.get_coord("time").step
        numpy_out = range_patch.rolling(time=13 * dt, center=True, engine="numpy").sum()
        pandas_out = range_patch.rolling(
            time=13 * dt, center=True, engine="pandas"
        ).sum()
        numpy_isnan = np.isnan(numpy_out.data)
        pandas_isnan = np.isnan(pandas_out.data)
        assert np.all(
            np.equal(numpy_isnan, pandas_isnan)
        ), "The NaN indices do not match"

    def test_center_same_stepped(self, range_patch):
        """Ensure center values are handled the same."""
        dt = range_patch.get_coord("time").step
        numpy_out = range_patch.rolling(
            time=13 * dt, step=3 * dt, center=True, engine="numpy"
        ).sum()
        pandas_out = range_patch.rolling(
            time=13 * dt, step=3 * dt, center=True, engine="pandas"
        ).sum()
        numpy_isnan = np.isnan(numpy_out.data)
        pandas_isnan = np.isnan(pandas_out.data)
        assert np.all(
            np.equal(numpy_isnan, pandas_isnan)
        ), "The NaN indices do not match"

    def test_dimension_order(self, range_patch):
        """Ensure the dimension order doesn't matter."""
        for patch in [range_patch, range_patch.transpose()]:
            for dim in patch.dims:
                coord = patch.get_coord(dim)
                step = coord.step
                total_len = len(coord) - 2
                kwargs_pandas = {
                    dim: step * total_len,
                    "step": total_len * step,
                    "engine": "pandas",
                }
                kwargs_numpy = {
                    dim: step * total_len,
                    "step": total_len * step,
                    "engine": "numpy",
                }
                pandas_out = patch.rolling(**kwargs_pandas).mean().dropna(dim)
                numpy_out = patch.rolling(**kwargs_numpy).mean().dropna(dim)
                assert pandas_out == numpy_out
