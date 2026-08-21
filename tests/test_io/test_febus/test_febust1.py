"""
FEBUS T1 DTS specific tests.
"""

import shutil

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

import dascore as dc
from dascore.core.coords import CoordMonotonicArray
from dascore.io.febus import FebusT1V1
from dascore.utils.downloader import fetch
from dascore.utils.misc import unbyte


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


class TestFebusT1Interrogator:
    """T1 root attrs name the unit and the kind of instrument it ran as."""

    parser = FebusT1V1()

    @pytest.fixture(
        scope="class", params=["febus_dts.h5", "febus_dts_single_reading.h5"]
    )
    def t1_file(self, request):
        """Paths to both T1 test files."""
        return fetch(request.param)

    def test_interrogator_from_root_attrs(self, t1_file):
        """Name and instrument_type come from the file, not a constant."""
        with h5py.File(t1_file, "r") as f:
            device_name = unbyte(f.attrs["device_name"])
            device = unbyte(f.attrs["device"])
        attrs = dict(dc.scan(t1_file)[0].attrs)
        assert attrs["interrogator.name"] == device_name
        assert attrs["interrogator.instrument_type"] == device

    def test_format_constants_still_set(self, t1_file):
        """
        Manufacturer and model come from the format, not the header.

        T1 files state no maker or model, so these are what claiming the
        format asserts rather than facts read out of the file.
        """
        attrs = dict(dc.scan(t1_file)[0].attrs)
        assert attrs["interrogator.manufacturer"] == "FEBUS"
        assert attrs["interrogator.model"] == "T1"

    def test_scan_and_read_agree(self, t1_file):
        """A read states the same interrogator a scan does."""
        scanned = dict(dc.scan(t1_file)[0].attrs)
        read = dict(self.parser.read(t1_file)[0].attrs)
        keys = [x for x in scanned if x.startswith("interrogator.")]
        assert keys
        assert all(scanned[x] == read[x] for x in keys)

    def test_blank_root_attrs_dropped(self, t1_file, tmp_path):
        """An empty device_name is not passed off as a name."""
        path = tmp_path / "blank_device.h5"
        shutil.copy2(t1_file, path)
        with h5py.File(path, "r+") as f:
            f.attrs["device_name"] = b"   "
        attrs = dict(dc.scan(path)[0].attrs)
        assert "interrogator.name" not in attrs
        assert attrs["interrogator.model"] == "T1"
