"""Tests for the Silixa H5 iDAS v2 (raw_das_data) variant."""

import shutil

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.io.silixah5 import SilixaH5V1, SilixaH5V2, SilixaH5V3
from dascore.utils.downloader import fetch


@pytest.fixture(scope="module")
def idas2_path():
    """Return the path to the iDAS v2 test file (Rhonegletscher)."""
    return fetch("silixa_h5_idas2_1.h5")


@pytest.fixture(scope="module")
def idas2_patch(idas2_path):
    """Read the iDAS v2 test file into a patch."""
    return SilixaH5V3().read(idas2_path)[0]


class TestDetection:
    """Each Silixa variant claims only its own files."""

    def test_get_format(self, idas2_path):
        """The file resolves to Silixa_H5 v3 through the full dispatch."""
        assert dc.get_format(idas2_path) == ("Silixa_H5", "3")

    def test_siblings_do_not_claim(self, idas2_path):
        """The Acoustic and Carina readers do not claim an iDAS v2 file."""
        assert not SilixaH5V1().get_format(idas2_path)
        assert not SilixaH5V2().get_format(idas2_path)

    @pytest.mark.parametrize(
        ("name", "version"),
        [("silixa_h5_1.hdf5", "1"), ("silixa_h5_ingv_1.h5", "2")],
    )
    def test_sibling_files_keep_their_reader(self, name, version):
        """The sibling files still resolve to their own versions, not v3."""
        path = fetch(name)
        assert not SilixaH5V3().get_format(path)
        assert dc.get_format(path) == ("Silixa_H5", version)

    def test_missing_attr_not_claimed(self, idas2_path, tmp_path):
        """A raw_das_data file missing a required attr is not claimed."""
        path = tmp_path / "no_rate.h5"
        shutil.copy(idas2_path, path)
        with h5py.File(path, "a") as h5:
            del h5["raw_das_data"].attrs["sampling_frequency_Hz"]
        assert not SilixaH5V3().get_format(path)


class TestCoords:
    """Coordinates follow the dataset's string attrs exactly."""

    def test_time(self, idas2_patch):
        """GPS start, 1 kHz, 400 samples ends at the trimmed endtime attr."""
        time = idas2_patch.get_coord("time")
        assert time.evenly_sampled
        assert time.min() == np.datetime64("2020-07-15T14:26:50.719")
        assert time.step == np.timedelta64(1, "ms")
        assert time.max() == np.datetime64("2020-07-15T14:26:51.118")

    def test_distance(self, idas2_patch):
        """Start Distance plus i * spatial_resolution * fibre multiplier."""
        distance = idas2_patch.get_coord("distance")
        step = 4 * 1.020952
        assert len(distance) == 40
        assert np.isclose(distance.step, step)
        assert np.isclose(distance.min(), -124.020952)
        assert np.isclose(distance.max(), 35.247560)  # trimmed stop_distance_m

    def test_scan_matches_read(self, idas2_path):
        """Scan reports the coords and attrs the read gives."""
        payload = dc.scan_payloads(idas2_path)[0]
        patch = dc.read(idas2_path)[0]
        assert payload.coords == patch.coords
        assert payload.attrs == patch.attrs


class TestAttrs:
    """Attrs share the sibling variants' vocabulary."""

    def test_attrs(self, idas2_patch):
        """Gauge length, pulse width and host name; data stay raw counts."""
        attrs = idas2_patch.attrs
        assert attrs.gauge_length == 10
        assert np.isclose(attrs.pulse_width, 50e-9)
        assert dict(attrs)["interrogator.name"] == "iDAS18065"
        assert not attrs.data_units
        assert idas2_patch.data.dtype == np.int16

    def test_cpu_time_fallback(self, idas2_path, tmp_path):
        """With no GPS time the CPU time stamp starts the time axis."""
        path = tmp_path / "no_gps.h5"
        shutil.copy(idas2_path, path)
        with h5py.File(path, "a") as h5:
            h5["raw_das_data"].attrs["gps_time_stamp"] = ""
        time = SilixaH5V3().read(path)[0].get_coord("time")
        assert time.min() == np.datetime64("2020-07-15T14:26:50.706")
