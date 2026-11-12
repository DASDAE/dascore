"""Tests for the PUNIU HiFi-DAS HDF5 reader."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import InvalidFiberFileError
from dascore.io.puniu import PuniuV1

_START = np.datetime64("2024-08-01T00:03:36.552952169")
_ATTRS = {
    "version": 1,
    "format": "differential",
    "start_channel": "306",
    "step": "5",
    "epoch": 1722470616,
    "ns": 552952169,
    "sampling_rate": 1000,
    "spatial_resolution": 1.6,
    "spatial_sampling_rate": 0.2,
}


def _samples(shape, dtype="float32"):
    """Return the deterministic samples `write_puniu` stores."""
    return np.arange(np.prod(shape)).reshape(shape).astype(dtype)


def write_puniu(path, shape=(12, 4), order="time, channel", dtype="float32", **attrs):
    """Write a PUNIU version 1 file with the observed schema; return its path."""
    names = [x.strip() for x in order.split(",")]
    count = shape[names.index("channel")] if "channel" in names else shape[1]
    attrs = {
        **_ATTRS,
        "row_major_order": order,
        "column_major_order": ", ".join(names[::-1]),
        "channel_count": count,
        **attrs,
    }
    start, step = int(attrs["start_channel"]), int(attrs["step"])
    attrs.setdefault("end_channel", str(start + count * step - 1))
    with h5py.File(path, "w") as h5:
        h5.create_dataset("default", data=_samples(shape, dtype)).attrs.update(attrs)
    return path


@pytest.fixture
def puniu_path(tmp_path):
    """A small (time, channel) PUNIU file."""
    return write_puniu(tmp_path / "puniu.h5")


@pytest.fixture
def transposed_path(tmp_path):
    """A small PUNIU file stored as (channel, time)."""
    return write_puniu(tmp_path / "transposed.h5", shape=(4, 12), order="channel, time")


class TestGetFormat:
    """Tests for detecting PUNIU files."""

    @pytest.mark.parametrize("version", [1, "1", b"1", 1.0, "1.0", [1]])
    def test_version_spellings(self, tmp_path, version):
        """Version 1 is recognized however the attribute encodes it."""
        path = write_puniu(tmp_path / "file.h5", version=version)
        assert dc.get_format(path) == ("PUNIU", "1")

    @pytest.mark.parametrize(
        "attrs", [{"version": 2}, {"version": "one"}, {"format": "other"}]
    )
    def test_other_schema(self, tmp_path, attrs):
        """Other versions and signal formats are not claimed or read."""
        path = write_puniu(tmp_path / "file.h5", **attrs)
        assert PuniuV1().get_format(path) is False
        with pytest.raises(InvalidFiberFileError, match="version 1"):
            PuniuV1().scan(path)

    def test_group_named_default(self, tmp_path):
        """A group at /default is not a PUNIU dataset."""
        path = tmp_path / "other.h5"
        with h5py.File(path, "w") as h5:
            h5.create_group("default")
        assert PuniuV1().get_format(path) is False
        with pytest.raises(InvalidFiberFileError, match="dataset"):
            PuniuV1().scan(path)


class TestCoords:
    """Tests for the coordinates built from the header."""

    def test_absolute_distance(self, puniu_path):
        """Distance starts at the first channel's position on the fiber."""
        patch = dc.read(puniu_path)[0]
        assert patch.dims == ("time", "distance")
        distance = patch.get_coord("distance")
        assert distance.units == dc.get_quantity("m")
        np.testing.assert_allclose(patch.get_array("distance"), 61.2 + np.arange(4))
        np.testing.assert_array_equal(patch.get_array("channel"), [306, 311, 316, 321])

    @pytest.mark.parametrize(
        "spacing, step", [(0.2, 1.0), (0.20000000298, 1.0), (0.25, 1.5)]
    )
    def test_spacing_rounded(self, tmp_path, spacing, step):
        """Base spacing is rounded half-up to 0.1 m before striding."""
        path = write_puniu(tmp_path / "file.h5", spatial_sampling_rate=spacing)
        coord = dc.read(path)[0].get_coord("distance")
        assert coord.step == step
        assert coord.min() == pytest.approx(306 * step / 5)

    def test_nanosecond_start(self, puniu_path):
        """Time starts at epoch seconds plus the ns attribute."""
        time = dc.read(puniu_path)[0].get_coord("time")
        assert time.min() == _START
        assert time.step == np.timedelta64(1, "ms")

    @pytest.mark.parametrize("rate", [1500, 3000, 6000])
    def test_rate_not_whole_ns(self, tmp_path, rate):
        """Rates whose period is not a whole number of ns are still read."""
        path = write_puniu(tmp_path / "file.h5", shape=(rate, 2), sampling_rate=rate)
        time = dc.read(path)[0].get_coord("time")
        expected = _START + np.timedelta64(round(1e9 * (rate - 1) / rate), "ns")
        assert abs(time.max() - expected) <= np.timedelta64(1, "ns")

    def test_gauge_length(self, puniu_path):
        """spatial_resolution is reported as the gauge length."""
        assert dc.read(puniu_path)[0].attrs.gauge_length == 1.6

    def test_scan_matches_read(self, puniu_path, monkeypatch):
        """A scan reads no samples and agrees with the read."""
        patch = dc.read(puniu_path)[0]

        def reject(*args, **kwargs):
            raise AssertionError("samples read")

        monkeypatch.setattr(h5py.Dataset, "__getitem__", reject)
        summary = dc.scan(puniu_path)[0]
        assert summary.attrs.gauge_length == patch.attrs.gauge_length
        assert summary.coords["distance"].min == patch.get_coord("distance").min()


class TestRead:
    """Tests for reading samples."""

    def test_transposed(self, transposed_path):
        """A (channel, time) file is returned as (time, distance)."""
        patch = dc.read(transposed_path)[0]
        assert patch.dims == ("time", "distance")
        np.testing.assert_array_equal(patch.data, _samples((4, 12)).T)

    def test_select_time_and_channel(self, puniu_path):
        """Time and channel selections pick the matching samples."""
        time = (_START + np.timedelta64(2, "ms"), _START + np.timedelta64(5, "ms"))
        patch = dc.read(puniu_path, time=time, channel=(311, 316))[0]
        np.testing.assert_array_equal(patch.data, _samples((12, 4))[2:6, 1:3])

    def test_distance_matches_channel(self, puniu_path):
        """A distance window selects the same samples as its channels."""
        by_distance = dc.read(puniu_path, distance=(62.2, 63.2))[0]
        by_channel = dc.read(puniu_path, channel=(311, 316))[0]
        assert by_distance.equals(by_channel)

    @pytest.mark.parametrize("fixture", ["puniu_path", "transposed_path"])
    def test_bounded_read(self, request, monkeypatch, fixture):
        """A selection reads only its own hyperslab from the file."""
        path = request.getfixturevalue(fixture)
        original = h5py.Dataset.__getitem__
        keys = []

        def capture(dataset, key):
            keys.append(key)
            return original(dataset, key)

        monkeypatch.setattr(h5py.Dataset, "__getitem__", capture)
        time = (_START + np.timedelta64(2, "ms"), _START + np.timedelta64(5, "ms"))
        dc.read(path, time=time, distance=(62.2, 63.2))
        time_slice, distance_slice = slice(2, 6), slice(1, 3)
        if fixture == "transposed_path":
            assert keys == [(distance_slice, time_slice)]
        else:
            assert keys == [(time_slice, distance_slice)]

    @pytest.mark.parametrize(
        "selection",
        [{"channel": (999, 1000)}, {"distance": (0, 10)}],
    )
    def test_empty_selection(self, puniu_path, selection):
        """A selection outside the file returns an empty spool."""
        assert len(dc.read(puniu_path, **selection)) == 0

    def test_unsigned(self, tmp_path):
        """Unsigned integer samples are read."""
        path = write_puniu(tmp_path / "file.h5", dtype="uint16")
        patch = dc.read(path)[0]
        np.testing.assert_array_equal(patch.data, _samples((12, 4), "uint16"))


class TestSpool:
    """Tests for files from different parts of the same fiber."""

    @pytest.fixture
    def two_starts(self, tmp_path):
        """Two time-adjacent files whose first channels differ."""
        write_puniu(tmp_path / "a.h5")
        write_puniu(tmp_path / "b.h5", start_channel="1306", ns=552952169 + 12_000_000)
        return dc.spool(tmp_path).update()

    def test_select_distance(self, two_starts):
        """A distance selection keeps only the file covering it."""
        selected = two_starts.select(distance=(260, 270))
        assert len(selected) == 1
        assert selected[0].get_array("channel")[0] == 1306

    def test_chunk_keeps_files_apart(self, two_starts):
        """Files on different fiber segments are not merged into one."""
        assert len(two_starts.chunk(time=None)) == 2

    def test_chunk_merges_same_segment(self, tmp_path):
        """Time-adjacent files on the same segment merge into one patch."""
        write_puniu(tmp_path / "a.h5")
        write_puniu(tmp_path / "b.h5", ns=552952169 + 12_000_000)
        merged = dc.spool(tmp_path).update().chunk(time=None, conflict="drop")
        assert len(merged) == 1
        assert merged[0].shape == (24, 4)


class TestBadMetadata:
    """Tests for refusing inconsistent headers."""

    @pytest.mark.parametrize(
        "attrs",
        [
            {"sampling_rate": 0},
            {"sampling_rate": np.nan},
            {"ns": 1_000_000_000},
            {"channel_count": 99},
            {"end_channel": "100"},
            {"epoch": 1.5},
            {"epoch": 10_000_000_000},
            {"row_major_order": "unknown"},
            {"column_major_order": "time, channel"},
            {"spatial_resolution": 0},
            {"spatial_sampling_rate": 0.04},
            {"spatial_sampling_rate": np.inf},
            {"channel_count": [4, 4]},
        ],
    )
    def test_refused(self, tmp_path, attrs):
        """Contradictory or invalid metadata raises InvalidFiberFileError."""
        path = write_puniu(tmp_path / "file.h5", **attrs)
        with pytest.raises(InvalidFiberFileError):
            dc.read(path, file_format="PUNIU")

    def test_text_samples(self, tmp_path):
        """A text dataset is refused even with a valid header."""
        path = write_puniu(tmp_path / "file.h5", dtype="S8")
        with pytest.raises(InvalidFiberFileError, match="numeric"):
            dc.read(path, file_format="PUNIU")

    def test_missing_clock(self, tmp_path):
        """A file without epoch is not claimed, and refused when forced."""
        path = write_puniu(tmp_path / "file.h5")
        with h5py.File(path, "r+") as h5:
            del h5["default"].attrs["epoch"]
        assert PuniuV1().get_format(path) is False
        with pytest.raises(InvalidFiberFileError):
            PuniuV1().scan(path)
