"""Tests for applying rolling functions on patch's data."""

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import ParameterError


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

    @pytest.fixture(scope="class")
    def range_patch_3d(self):
        """Return a 3D patch for testing."""
        data = np.broadcast_to(np.arange(10)[:, None, None], (10, 10, 10))
        coords = {
            "time": np.arange(10),
            "distance": np.arange(10),
            "smell": np.arange(10),
        }
        patch = dc.Patch(data=data, coords=coords, dims=tuple(coords))
        return patch

    def test_apply_correct_no_step(self, dist_roll_range_patch):
        """Ensure the apply is correct without using a step size."""
        # determine what the values should be:
        ax_0_len = dist_roll_range_patch.shape[0]
        samps = self.window - 1
        expected_values = np.arange(samps, ax_0_len + samps)[:, None]
        # and assert they are indeed that.
        assert np.allclose(dist_roll_range_patch.data, expected_values)

    def test_along_time(self, range_patch):
        """Ensure time axis also works."""
        time_step = range_patch.get_coord("time").step
        out = range_patch.rolling(time=time_step * self.window).sum()
        shape = range_patch.shape
        # Note: we start at 0 here because range is along 0th axis and
        # time is first axis so all windows are the same values.
        expected = (np.arange(0, shape[0]) * self.window)[:, None]
        assert np.allclose(out.data, expected)
        # this should also work if patch is transposed.
        trans = range_patch.transpose()
        out = trans.rolling(time=time_step * self.window).sum()
        assert np.allclose(out.data, expected.transpose())

    def test_apply_with_step(self, range_patch):
        """Ensure apply works with various step sizes."""
        # first calculate rolling max on time axis.
        step = range_patch.get_coord("distance").step
        out = range_patch.rolling(
            distance=self.window * step, step=self.step * step
        ).max()
        # Determine what output should be.
        vals = np.arange(self.window - 1, range_patch.shape[0])
        expected = vals[:: self.step, None]
        assert np.allclose(out.data, expected)

    def test_1D_patch(self):
        """Ensure rolling works with 1D patch."""
        patch = dc.Patch(
            data=np.arange(10),
            coords={"time": dc.to_datetime64(np.arange(10))},
            dims=("time",),
        )
        expected = patch.data[:-2]
        out = patch.rolling(time=3).min()
        assert np.allclose(expected, out.data)

    def test_3D_patch(self, range_patch_3d):
        """Ensure rolling works with 3D patch."""
        patch = range_patch_3d
        out = patch.rolling(time=3).min()

        expected = patch.data[:-2]
        out = patch.rolling(time=3).min()
        assert np.allclose(expected, out.data)

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
