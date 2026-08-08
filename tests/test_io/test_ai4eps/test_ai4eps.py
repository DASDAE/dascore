"""Tests for the AI4EPS event format."""

import numpy as np
import pytest

import dascore as dc
from dascore.io.ai4eps import AI4EPSV1
from dascore.utils.downloader import fetch


@pytest.fixture(scope="module")
def ai4eps_path():
    """Return the path to the AI4EPS test file."""
    return fetch("ai4eps_1.h5")


@pytest.fixture(scope="module")
def ai4eps_patch(ai4eps_path):
    """Read the AI4EPS test file into a patch."""
    return AI4EPSV1().read(ai4eps_path)[0]


class TestAI4EPS:
    """Format-specific tests not covered by the common IO tests."""

    def test_coords(self, ai4eps_patch):
        """Coordinates come from the dt_s/dx_m/begin_time attrs."""
        time = ai4eps_patch.get_coord("time")
        distance = ai4eps_patch.get_coord("distance")
        assert time.step == dc.to_timedelta64(0.01)
        assert distance.step == 8.0
        assert distance.min() == 0.0
        # begin_time in the file is tz-aware (UTC); coord must be naive UTC.
        assert time.min() == dc.to_datetime64("2020-06-24T17:56:48.910000")

    def test_event_attrs(self, ai4eps_patch):
        """The event metadata lands on the patch attrs."""
        attrs = ai4eps_patch.attrs
        assert attrs.event_id == "ci37280444"
        assert attrs.magnitude == 2.67
        assert attrs.event_time == dc.to_datetime64("2020-06-24T17:56:58.910000")
        assert np.isclose(attrs.event_latitude, 36.4673)
        assert np.isclose(attrs.event_longitude, -117.9907)
        assert np.isclose(attrs.event_depth_km, 16.36)

    def test_data_units(self, ai4eps_patch):
        """Units come from the unit attr."""
        assert dc.get_quantity(ai4eps_patch.attrs.data_units) == dc.get_quantity(
            "microstrain/s"
        )

    def test_event_time_within_patch(self, ai4eps_patch):
        """The event occurs inside the recorded window."""
        time = ai4eps_patch.get_coord("time")
        assert time.min() < ai4eps_patch.attrs.event_time < time.max()

    def test_select_trims(self, ai4eps_patch, ai4eps_path):
        """Reading with a time filter trims the patch."""
        time = ai4eps_patch.get_coord("time")
        mid = time.min() + (time.max() - time.min()) / 2
        trimmed = AI4EPSV1().read(ai4eps_path, time=(mid, None))[0]
        assert trimmed.shape[1] < ai4eps_patch.shape[1]
        assert trimmed.get_coord("time").min() >= mid

    def test_magnitude_type(self, ai4eps_patch):
        """The magnitude scale comes through so magnitudes stay comparable."""
        assert ai4eps_patch.attrs.magnitude_type == "ml"

    def test_inconsistent_end_time_warns(self, ai4eps_path, tmp_path):
        """A file whose end_time disagrees with begin_time + n*dt warns."""
        import shutil

        import h5py

        path = tmp_path / "bad_dt.h5"
        shutil.copy(ai4eps_path, path)
        with h5py.File(path, "a") as h5:
            h5["data"].attrs["dt_s"] = 0.02
        with pytest.warns(UserWarning, match="end_time"):
            AI4EPSV1().read(path)

    def test_near_miss_not_claimed(self, tmp_path):
        """A file with generic timing attrs but no event metadata isn't claimed."""
        import h5py

        path = tmp_path / "near_miss.h5"
        with h5py.File(path, "w") as h5:
            dataset = h5.create_dataset("data", data=np.zeros((3, 4)))
            dataset.attrs["begin_time"] = "2020-01-01T00:00:00"
            dataset.attrs["dt_s"] = 0.01
            dataset.attrs["dx_m"] = 8
            dataset.attrs["unit"] = "m/s"
        assert not AI4EPSV1().get_format(path)
