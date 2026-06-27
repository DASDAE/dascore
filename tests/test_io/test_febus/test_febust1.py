"""
FEBUS T1 DTS specific tests.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dascore.core.coords import CoordMonotonicArray
from dascore.io.febus import FebusT1V1
from dascore.utils.downloader import fetch


class TestFebusT1:
    """Tests for the FEBUS T1 DTS reader."""

    parser = FebusT1V1()

    @pytest.fixture(scope="class")
    def t1_path(self):
        """Path to a 12-reading T1 test file."""
        return fetch("febus_dts.h5")

    @pytest.fixture(scope="class")
    def t1_single_reading_path(self):
        """Path to a single-reading T1 test file."""
        return fetch("febus_dts_single_reading.h5")

    @pytest.fixture(scope="class")
    def t1_patch(self, t1_path):
        """Get the febus t1 patch"""
        return self.parser.read(t1_path)[0]

    @pytest.fixture(scope="class")
    def t1_single_reading_patch(self, t1_single_reading_path):
        """Get the febus t1 patch with a single time"""
        return self.parser.read(t1_single_reading_path)[0]

    def test_time_spacing(self, t1_patch):
        """Time steps should be approximately 5 minutes apart."""
        time = t1_patch.get_coord("time")
        step_minutes = time.step / np.timedelta64(1, "m")
        assert_allclose(step_minutes, 5.35, rtol=1e-3)

    def test_distance_range(self, t1_patch):
        """Distance should span roughly 0-90 m."""
        dist = t1_patch.get_coord("distance")
        assert dist.min() >= 0
        assert_allclose(dist.max(), 89.9, rtol=1e-3)
        assert_allclose(dist.step, 0.0816, rtol=1e-3)

    def test_temperature_reasonable(self, t1_patch):
        """Temperature values should be plausible (5 to 50 °C)."""
        assert t1_patch.data.min() > 5
        assert t1_patch.data.max() < 50

    def test_single_reading_does_not_raise(self, t1_single_reading_patch):
        """Single-reading files should parse without error."""
        assert t1_single_reading_patch.data.shape[0] == 1
        time = t1_single_reading_patch.get_coord("time")
        assert isinstance(time, CoordMonotonicArray)
        assert time.min() == time.max()
