"""
Febus specific tests.
"""

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.io.febus import Febus2
from dascore.io.febus.a1utils import _flatten_febus_info
from dascore.utils.downloader import fetch
from dascore.utils.time import to_float


class TestFebus:
    """Special test cases for febus."""

    @pytest.fixture(scope="class")
    def febus_path(self):
        """Return the path to a test febus file."""
        return fetch("febus_1.h5")

    def test_time_coords_consistent_with_metadata(self, febus_path):
        """
        Ensure the time coords returned have the same length as
        metadata indicates.
        """
        patch = Febus2().read(febus_path)[0]
        coords = patch.coords
        time = coords.get_coord("time")
        time_span = to_float((time.max() - time.min()) + time.step)

        with h5py.File(febus_path, "r") as f:
            feb = _flatten_febus_info(f)[0]
            # First check total time extent
            n_blocks = feb.zone[feb.data_name].shape[0]
            block_time = 1 / (feb.zone.attrs["BlockRate"] / 1_000)
            expected_time_span = block_time * n_blocks
            assert np.isclose(expected_time_span, time_span)
            # Then check absolute time
            time_offset = feb.zone.attrs["Origin"][1] / 1_000
            time_start = feb.source["time"][0] + time_offset
            assert np.isclose(to_float(time.min()), time_start)

    def test_valencia_data_sampling_rate(self):
        """
        The valencia file has two indicated sampling rates. One from
        the spacing array (500 Hz) and one from the sampling_rate attributes (250).
        Looking at the entire archive on pubDAS, it is clear the correct value
        is 250, otherwise there 50% gaps every file. This test just ensure that
        The sampling rate doesnt change.
        """
        patch = dc.get_example_patch("valencia_febus_example.h5")
        time = patch.get_coord("time")
        sampling_rate = np.timedelta64(1, "s") / time.step
        assert np.isclose(sampling_rate, 250)
