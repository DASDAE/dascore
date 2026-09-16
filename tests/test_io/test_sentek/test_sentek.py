"""
Tests specific to the Sentek format.
"""

from io import BytesIO

import numpy as np
import pytest

import dascore as dc
from dascore.compat import random_state
from dascore.io.sentek import SentekV5
from dascore.io.sentek.utils import _get_patch_attrs
from dascore.utils.downloader import fetch


class TestSentekV5:
    """Tests for Sentek format that aren;t covered by common tests."""

    def test_das_extension_not_sentek(self, tmp_path_factory):
        """Ensure a non-sentek file with a das extension isn't id as sentek."""
        path = tmp_path_factory.mktemp("sentek_test") / "not_sentek.das"
        ar = random_state.random(10)
        with path.open("wb") as fi:
            np.save(fi, ar)
        sentek = SentekV5()
        assert not sentek.get_format(path)


class TestStoredLayout:
    """Known binary buffers establish the independent sample and coordinate oracle."""

    @pytest.fixture
    def stored_file(self, tmp_path):
        """Write the six-value header, both coordinate arrays, then time-major data."""
        path = tmp_path / "DASDMSShot00_20230328155653619.das"
        distance = np.array([10, 15, 22, 26], dtype="float32")
        offsets = np.array([0.5, 0.75, 1.125, 1.25, 1.5, 1.75], dtype="float32")
        data = (10_000 + 100 * np.arange(4)[:, None] + 3 * np.arange(6)).astype(
            "float32"
        )
        header = np.array([4, 6, 250_000_000, 1, 0, 1], dtype="float32")
        with path.open("wb") as stream:
            for values in (header, distance, offsets, data.T):
                stream.write(values.tobytes())
        time = np.datetime64("2023-03-28T15:56:53.619", "ns") + np.array(
            [
                500_000_000,
                750_000_000,
                1_125_000_000,
                1_250_000_000,
                1_500_000_000,
                1_750_000_000,
            ],
            dtype="timedelta64[ns]",
        )
        return path, data, distance, time

    @pytest.mark.parametrize("snap", [True, False, "time", "distance"])
    def test_full_and_bounded_samples(self, stored_file, snap):
        """Stored headers never leak into the signal or displace its final values."""
        path, data, distance, time = stored_file
        full = dc.read(path, snap=snap)[0]
        np.testing.assert_array_equal(full.data, data)
        selected = dc.read(path, snap=snap, samples=True, distance=(1, 3), time=(2, 5))[
            0
        ]
        np.testing.assert_array_equal(selected.data, data[1:3, 2:5])
        if snap is False or snap == "time":
            np.testing.assert_array_equal(full.get_coord("distance").values, distance)
        if snap is False or snap == "distance":
            np.testing.assert_array_equal(full.get_coord("time").values, time)
        metadata = dc.scan_payloads(path, snap=snap)[0]
        assert metadata.coords == full.coords
        assert metadata.attrs.patch_id == full.attrs.patch_id
        assert metadata.dtype == data.dtype

    def test_default_scan_reads_only_endpoints(self, stored_file):
        """Default scans read coordinate endpoints without reading signal samples."""
        path, data, _, _ = stored_file

        class HeaderReader(BytesIO):
            def __init__(self):
                super().__init__(path.read_bytes())
                self.name = str(path)
                self.read_bytes = 0

            def read(self, size=-1):
                self.read_bytes += size
                assert self.tell() + size <= (6 + sum(data.shape)) * 4
                return super().read(size)

        stream = HeaderReader()
        _, coords, offsets = _get_patch_attrs(stream)
        assert coords.shape == data.shape
        assert offsets[0] == (6 + sum(data.shape)) * 4
        assert stream.read_bytes == 10 * 4

    def test_real_file_signal_boundary(self):
        """The real recording has exactly the declared payload after both arrays."""
        path = fetch("DASDMSShot00_20230328155653619.das")
        raw = np.fromfile(path, dtype="float32")
        channels, samples = map(int, raw[:2])
        start = 6 + channels + samples
        expected = raw[start:].reshape(samples, channels).T
        patch = dc.read(path)[0]
        np.testing.assert_array_equal(patch.data, expected)
