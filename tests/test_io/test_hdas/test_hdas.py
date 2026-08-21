"""Tests for the Aragon Photonics HDAS format."""

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import InvalidFiberFileError
from dascore.io.core import FiberIO
from dascore.io.hdas import HDASV1, HDASV2
from dascore.utils.downloader import fetch


@pytest.fixture(scope="module")
def hdas_v1_path():
    """Return the path to the V1 (IGN-ADIF) test file."""
    return fetch("hdas_1.h5")


@pytest.fixture(scope="module")
def hdas_v2_path():
    """Return the path to the V2 (ICM-CSIC Canalink) test file."""
    return fetch("hdas_2.h5")


@pytest.fixture(scope="module")
def hdas_v1_patch(hdas_v1_path):
    """Read the V1 file into a patch."""
    return HDASV1().read(hdas_v1_path)[0]


@pytest.fixture(scope="module")
def hdas_v2_patch(hdas_v2_path):
    """Read the V2 file into a patch."""
    return HDASV2().read(hdas_v2_path)[0]


class TestHDASV1:
    """Tests for the vendor (File_Header) layout."""

    def test_header_decode(self, hdas_v1_patch):
        """Coordinates decode from the fixed header positions."""
        time = hdas_v1_patch.get_coord("time")
        distance = hdas_v1_patch.get_coord("distance")
        # trigger 2000 Hz / 20 / 1 -> 100 Hz
        assert time.step == dc.to_timedelta64(0.01)
        # epoch at header position 101 (1-based); the source file was
        # recorded starting 2023-02-14T00:00:28 (also in its filename).
        # The full float64 epoch precision (~0.24 us) must survive decode.
        assert time.min() == np.datetime64("2023-02-14T00:00:28.084310528")
        assert distance.step == 10.0
        assert distance.min() == 60.0

    def test_orientation_and_units(self, hdas_v1_patch):
        """V1 data is (time, distance) strain rate."""
        assert hdas_v1_patch.dims == ("time", "distance")
        assert dc.get_quantity(hdas_v1_patch.attrs.data_units) == dc.get_quantity(
            "strain/s"
        )


class TestHDASV2:
    """Tests for the attr-timed (hdas_header) variant."""

    def test_header_decode(self, hdas_v2_patch):
        """Coordinates decode from the header and start_time attr."""
        time = hdas_v2_patch.get_coord("time")
        distance = hdas_v2_patch.get_coord("distance")
        # trigger 1000 Hz / 20 / 1 -> 50 Hz
        assert time.step == dc.to_timedelta64(0.02)
        # start_time attr, not the imprecise float32 header epoch.
        assert time.min() == dc.to_datetime64("2023-02-09T21:12:28")
        assert distance.step == 10.0
        assert distance.min() == 60.0

    def test_orientation(self, hdas_v2_patch):
        """V2 data is (distance, time) with units unset."""
        assert hdas_v2_patch.dims == ("distance", "time")
        assert not hdas_v2_patch.attrs.data_units


class TestHDASDetection:
    """Detection edge cases."""

    def test_no_blank_version_registered(self):
        """Sharing logic between variants must not register a blank version."""
        hdas_ios = [
            io
            for io in FiberIO.manager.yield_fiberio(format="HDAS")
            if io.name == "HDAS"
        ]
        assert hdas_ios
        assert all(io.version for io in hdas_ios)

    def test_variant_header_shape_enforced(self, tmp_path):
        """A V1-named header with the V2 shape isn't claimed."""
        header = np.zeros(200)
        header[0] = 200
        path = tmp_path / "flat_v1_header.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("File_Header", data=header)
            h5.create_dataset("HDAS_DATA", data=np.zeros((10, 10)))
        assert not HDASV1().get_format(path)

    def test_wrong_header_length_not_claimed(self, tmp_path):
        """A file whose header is not 200 values isn't claimed."""
        path = tmp_path / "bad_header.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("File_Header", data=np.zeros((1, 100)))
            h5.create_dataset("HDAS_DATA", data=np.zeros((10, 10)))
        assert not HDASV1().get_format(path)

    def test_wrong_header_marker_not_claimed(self, tmp_path):
        """A correctly-shaped header whose length marker isn't 200 is rejected."""
        path = tmp_path / "bad_marker.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("File_Header", data=np.zeros((1, 200)))
            h5.create_dataset("HDAS_DATA", data=np.zeros((10, 10)))
        assert not HDASV1().get_format(path)

    def test_missing_data_not_claimed(self, tmp_path):
        """A valid header without a 2d data array is rejected."""
        header = np.zeros((1, 200))
        header[0, 0] = 200
        path = tmp_path / "no_data.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("File_Header", data=header)
        assert not HDASV1().get_format(path)

    def test_corrupt_header_raises_on_read(self, tmp_path):
        """A zero-filled header refuses to decode instead of yielding nan."""
        header = np.zeros((1, 200))
        header[0, 0] = 200
        path = tmp_path / "zero_header.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("File_Header", data=header)
            h5.create_dataset("HDAS_DATA", data=np.zeros((10, 10)))
        with pytest.raises(InvalidFiberFileError, match="acquisition fields"):
            HDASV1().read(path)

    def test_bad_header_fields_raise(self, tmp_path):
        """Infinite spatial steps and negative divisors are rejected."""

        def _make(path, spatial, div_a, div_b):
            header = np.zeros((1, 200))
            header[0, 0] = 200
            header[0, 1] = spatial
            header[0, 6] = 2000
            header[0, 15] = div_a
            header[0, 98] = div_b
            header[0, 100] = 1.6e9
            with h5py.File(path, "w") as h5:
                h5.create_dataset("File_Header", data=header)
                h5.create_dataset("HDAS_DATA", data=np.zeros((10, 10)))
            return path

        inf_step = _make(tmp_path / "inf_step.h5", np.inf, 20, 1)
        with pytest.raises(InvalidFiberFileError, match="non-finite"):
            HDASV1().read(inf_step)
        neg_divs = _make(tmp_path / "neg_divs.h5", 10, -20, -1)
        with pytest.raises(InvalidFiberFileError, match="non-positive"):
            HDASV1().read(neg_divs)

    def test_wrong_size_header_raises_on_direct_read(self, tmp_path):
        """Reading a file with a wrong-size header raises a format error."""
        path = tmp_path / "small_header.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("File_Header", data=np.zeros((1, 100)))
            h5.create_dataset("HDAS_DATA", data=np.zeros((10, 10)))
        with pytest.raises(InvalidFiberFileError, match="header"):
            HDASV1().read(path)

    def test_missing_start_time_not_claimed_by_v2(self, tmp_path):
        """A V2-shaped file without start_time isn't claimed."""
        header = np.zeros(200, dtype=np.float32)
        header[0] = 200
        path = tmp_path / "no_start_time.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("hdas_header", data=header)
            h5.create_dataset("data", data=np.zeros((10, 10)))
        assert not HDASV2().get_format(path)
