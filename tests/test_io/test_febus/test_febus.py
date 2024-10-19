"""
Febus specific tests.
"""

import h5py
import numpy as np
import pytest

from dascore.io.febus import Febus2
from dascore.io.febus.utils import _flatten_febus_info
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
