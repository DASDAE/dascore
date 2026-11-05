"""Tests for PUNIU HiFi-DAS HDF5 support."""

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import InvalidFiberFileError
from dascore.io.puniu import PuniuV1

from .conftest import make_file


class TestPuniu:
    """Validate metadata, coordinates, and partial reads."""

    def test_declared_short_time_axis(self, tmp_path):
        """Declared short time axis."""
        p, data = make_file(tmp_path, shape=(3, 7))
        patch = PuniuV1().read(p)[0]
        assert patch.shape == (3, 7)
        np.testing.assert_array_equal(patch.data, data)
        np.testing.assert_array_equal(
            patch.get_array("channel"), 306 + np.arange(7) * 5
        )
        assert patch.attrs.data_units is None
        assert patch.attrs.acquisition_clock_basis == "unspecified"

    def test_transposed_declaration(self, tmp_path):
        """Transposed declaration."""
        p, data = make_file(tmp_path, shape=(7, 3), order="channel, time")
        patch = PuniuV1().read(p)[0]
        np.testing.assert_array_equal(patch.data, data.T)

    def test_selection_and_nanosecond_time(self, tmp_path):
        """Selection and nanosecond time."""
        p, data = make_file(tmp_path)
        start = np.datetime64("2024-08-01T00:03:36.552952169")
        patch = PuniuV1().read(
            p,
            time=(start + np.timedelta64(2, "ms"), start + np.timedelta64(5, "ms")),
            channel=(311, 316),
        )[0]
        np.testing.assert_array_equal(patch.data, data[2:6, 1:3])
        assert patch.attrs.time_min == start + np.timedelta64(2, "ms")

    def test_scan_matches_read_without_data_read(self, tmp_path, monkeypatch):
        """Scan matches read without data read."""
        p, _ = make_file(tmp_path)
        reader = PuniuV1()
        attrs = reader.read(p)[0].attrs

        def reject(*args, **kwargs):
            raise AssertionError("payload accessed")

        monkeypatch.setattr(h5py.Dataset, "__getitem__", reject)
        scan = reader.scan(p)[0]
        assert scan.coords == attrs.coords
        assert reader.get_format(p) == ("PUNIU", "1")

    @pytest.mark.parametrize(
        "changes",
        [
            {"sampling_rate": 0},
            {"sampling_rate": np.nan},
            {"sampling_rate": 3},
            {"ns": 1_000_000_000},
            {"channel_count": 99},
            {"end_channel": "100"},
            {"epoch": 1.5},
            {"row_major_order": "unknown"},
            {"column_major_order": "time, channel"},
            {"version": 2},
        ],
    )
    def test_bad_metadata_refused(self, tmp_path, changes):
        """Bad metadata refused."""
        p, _ = make_file(tmp_path, **changes)
        reader = PuniuV1()
        with pytest.raises(InvalidFiberFileError):
            reader.read(p)

    def test_unrelated_hdf5_refused(self, tmp_path):
        """Unrelated hdf5 refused."""
        p = tmp_path / "other.h5"
        with h5py.File(p, "w") as f:
            f.create_group("default")
        assert not PuniuV1().get_format(p)

    def test_missing_clock_refused(self, tmp_path):
        """Missing clock refused."""
        p, _ = make_file(tmp_path)
        with h5py.File(p, "r+") as f:
            del f["default"].attrs["epoch"]
        with pytest.raises(InvalidFiberFileError):
            PuniuV1().scan(p)

    def test_empty_selection(self, tmp_path):
        """Empty selection."""
        p, _ = make_file(tmp_path)
        assert len(PuniuV1().read(p, channel=(999, 1000))) == 0

    def test_public_api_explicit_format(self, tmp_path):
        """Public api explicit format."""
        p, data = make_file(tmp_path)
        patch = dc.read(p, file_format="PUNIU")[0]
        np.testing.assert_array_equal(patch.data, data)

    @pytest.mark.parametrize(
        "spacing, expected_dx", [(0.2, 1.0), (0.20000000298, 1.0), (0.25, 1.5)]
    )
    def test_matlab_spacing_rule_and_relative_origin(
        self, tmp_path, spacing, expected_dx
    ):
        """Matlab spacing rule and relative origin."""
        p, _ = make_file(tmp_path, spatial_sampling_rate=spacing)
        patch = PuniuV1().read(p)[0]
        assert patch.dims == ("time", "distance")
        np.testing.assert_allclose(
            patch.get_array("distance"), np.arange(4) * expected_dx
        )
        assert patch.coords.get_coord("distance").units == dc.get_quantity("m")
        assert patch.attrs.spatial_reference == "file_start"
        assert patch.get_coord("distance").step == expected_dx
        np.testing.assert_array_equal(patch.get_array("channel"), [306, 311, 316, 321])

    def test_distance_selection_keeps_file_origin_and_matches_channel(self, tmp_path):
        """Distance selection keeps file origin and matches channel."""
        p, data = make_file(tmp_path)
        by_distance = dc.read(p, file_format="PUNIU", distance=(1, 2))[0]
        by_channel = dc.read(p, file_format="PUNIU", channel=(311, 316))[0]
        np.testing.assert_array_equal(by_distance.data, data[:, 1:3])
        np.testing.assert_array_equal(by_distance.data, by_channel.data)
        np.testing.assert_array_equal(by_distance.get_array("distance"), [1, 2])
        np.testing.assert_array_equal(by_distance.get_array("channel"), [311, 316])
        selected = PuniuV1().read(p, distance=(1, 3), channel=(316, 321))[0]
        np.testing.assert_array_equal(selected.data, data[:, 2:4])

    def test_distance_selection_is_a_bounded_hdf5_read(self, tmp_path, monkeypatch):
        """Distance selection is a bounded hdf5 read."""
        p, data = make_file(tmp_path, shape=(4, 12), order="channel, time")
        original = h5py.Dataset.__getitem__
        keys = []

        def capture(ds, key):
            keys.append(key)
            return original(ds, key)

        monkeypatch.setattr(h5py.Dataset, "__getitem__", capture)
        start = np.datetime64("2024-08-01T00:03:36.552952169")
        patch = PuniuV1().read(
            p,
            time=(start + np.timedelta64(2, "ms"), start + np.timedelta64(5, "ms")),
            distance=(1, 2),
        )[0]
        np.testing.assert_array_equal(patch.data, data[1:3, 2:6].T)
        assert keys == [(slice(1, 3), slice(2, 6))]

    @pytest.mark.parametrize("value", [0, -0.2, np.nan, np.inf, 0.04])
    def test_invalid_spatial_sampling_refused(self, tmp_path, value):
        """Invalid spatial sampling refused."""
        p, _ = make_file(tmp_path, spatial_sampling_rate=value)
        with pytest.raises(InvalidFiberFileError):
            PuniuV1().read(p)

    def test_empty_distance_window(self, tmp_path):
        """Empty distance window."""
        p, _ = make_file(tmp_path)
        assert len(PuniuV1().read(p, distance=(100, 101))) == 0

    def test_detect(self, puniu_path):
        """The installed entry point detects PUNIU without an explicit format."""
        assert dc.get_format(puniu_path) == ("PUNIU", "1")
        assert len(dc.read(puniu_path)) == 1

    @pytest.mark.parametrize("attrs", [{"version": 2}, {"format": "other"}])
    def test_other_schema(self, tmp_path, attrs):
        """Other schema versions and signal formats are not claimed."""
        path, _ = make_file(tmp_path, **attrs)
        assert not PuniuV1().get_format(path)

    def test_scan_attributes(self, puniu_path):
        """Scan and read agree on acquisition and coordinate metadata."""
        scan = PuniuV1().scan(puniu_path)[0]
        patch = PuniuV1().read(puniu_path)[0]
        assert scan.gauge_length == patch.attrs.gauge_length == 1.6
        assert (
            scan.recording_start_channel == patch.attrs.recording_start_channel == 306
        )
        assert scan.spatial_reference == patch.attrs.spatial_reference == "file_start"
        assert scan.coords == patch.attrs.coords
