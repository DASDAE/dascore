"""Tests for the Uptech HDF5 reader."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.io.uptech import UptechH5V1
from dascore.utils.downloader import fetch
from tests.test_io.test_common_io import skip_timeout

# An epoch (in seconds) recent enough that float64 timestamps only resolve
# to a few hundred ns, which is what the reader has to tolerate.
_T0 = 1.7815e9

_ATTRS = {
    "acquisition_frequency": 2.0,
    "fiber_length": 12800.0,
    "gauge_length": 10.0,
    "sampling_interval": 2.5,
    "spatial_resolution": 2.0,
}


def _write_uptech(path, time=None, shape=(4, 3), **attrs):
    """Write a minimal Uptech export, returning its path."""
    attrs = {**_ATTRS, **attrs}
    if time is None:
        time = _T0 + np.arange(shape[0]) / attrs["acquisition_frequency"]
    with h5py.File(path, "w") as h5:
        group = h5.create_group("Acquisition")
        data = group.create_dataset("StrainRate", data=np.ones(shape))
        data.attrs.update(**attrs)
        group.create_dataset("Time", data=np.asarray(time, dtype=float))
    return path


@pytest.fixture
def uptech_path(tmp_path):
    """A minimal, valid Uptech file."""
    return _write_uptech(tmp_path / "sample.hdf5")


@pytest.fixture(scope="class")
def real_patch():
    """The patch read from the registered Uptech test file."""
    with skip_timeout():
        path = fetch("uptech_as1000_1.hdf5")
    return UptechH5V1().read(path)[0]


class TestGetFormat:
    """Tests for Uptech format detection."""

    def test_detected(self, uptech_path):
        """A well formed file reports the format and version."""
        assert UptechH5V1().get_format(uptech_path) == ("Uptech_H5", "1")

    def test_not_detected_without_metadata(self, tmp_path):
        """Reject a file without the identifying metadata."""
        path = tmp_path / "other.hdf5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("Acquisition/StrainRate", shape=(2, 2))
            h5.create_dataset("Acquisition/Time", shape=(2,))
        assert not UptechH5V1().get_format(path)

    def test_not_detected_without_datasets(self, tmp_path):
        """Reject a file which lacks the expected datasets entirely."""
        path = tmp_path / "empty.hdf5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("something_else", shape=(2, 2))
        assert not UptechH5V1().get_format(path)


class TestRead:
    """Tests for reading Uptech files."""

    def test_read(self, uptech_path):
        """Read a minimal export and check attrs come through."""
        patch = UptechH5V1().read(uptech_path)[0]
        assert patch.shape == (4, 3)
        assert patch.attrs.data_type == "strain_rate"
        assert patch.attrs.data_category == "DAS"
        assert patch.attrs.fiber_length == 12800.0
        assert patch.attrs.gauge_length == 10.0

    def test_select_distance(self, uptech_path):
        """A distance range trims the distance axis."""
        patch = UptechH5V1().read(uptech_path, distance=(2.5, None))[0]
        assert patch.shape == (4, 2)
        assert patch.get_coord("distance").min() == 2.5

    def test_select_outside_data_returns_empty(self, uptech_path):
        """A selection with no overlap returns an empty spool."""
        spool = UptechH5V1().read(uptech_path, distance=(1e6, 2e6))
        assert len(spool) == 0

    def test_real_file_uses_channel_spacing(self, real_patch):
        """Use sampling interval for spacing, not spatial resolution."""
        assert real_patch.get_coord("distance").step == 2.5
        assert real_patch.attrs.spatial_resolution == 2.0

    def test_real_file_time_is_evenly_sampled(self, real_patch):
        """Jittered float64 timestamps still yield an evenly sampled coord."""
        assert real_patch.get_coord("time").evenly_sampled


class TestScan:
    """Tests for scanning Uptech files."""

    def test_scan(self, uptech_path):
        """Scan reports the format and the file path."""
        summary = dc.scan(uptech_path)[0]
        assert summary.source_format == "Uptech_H5"
        assert summary.source_version == "1"
        assert str(summary.source_path) == str(uptech_path)

    def test_scan_matches_read(self, uptech_path):
        """Scanned coords must match the coords produced by read."""
        io = UptechH5V1()
        scanned = io.scan(uptech_path)[0]
        patch = io.read(uptech_path)[0]
        assert scanned["coords"] == patch.coords
        assert scanned["dims"] == patch.dims


class TestTimeValidation:
    """Tests for the Acquisition/Time sanity checks."""

    @pytest.mark.parametrize("frequency", [193.5, 500.0, 1000.0, 10_000.0])
    def test_high_sample_rates_accepted(self, tmp_path, frequency):
        """Float64 timestamp jitter must not reject fast acquisitions."""
        path = _write_uptech(
            tmp_path / f"rate_{frequency}.hdf5",
            shape=(100, 3),
            acquisition_frequency=frequency,
        )
        patch = UptechH5V1().read(path)[0]
        assert patch.shape == (100, 3)

    def test_irregular_timing_preserved(self, tmp_path):
        """Genuinely uneven timing is kept, not silently regularized."""
        offsets = np.array([0.0, 0.3, 1.0, 1.4, 2.0])
        path = _write_uptech(
            tmp_path / "irregular.hdf5",
            shape=(5, 2),
            time=_T0 + offsets,
            acquisition_frequency=2.0,
        )
        coord = UptechH5V1().read(path)[0].get_coord("time")
        assert not coord.evenly_sampled
        assert np.array_equal(coord.values, dc.to_datetime64(_T0 + offsets))

    @pytest.mark.parametrize(
        ("frequency", "expected"), [(2.06, True), (1.94, True), (2.2, False)]
    )
    def test_frequency_tolerance(self, tmp_path, frequency, expected):
        """A small nominal/actual mismatch is tolerated, a large one is not."""
        # The true step is 0.5 s; frequency is the (possibly rounded) nominal.
        path = _write_uptech(
            tmp_path / f"tolerance_{frequency}.hdf5",
            shape=(5, 2),
            time=_T0 + 0.5 * np.arange(5),
            acquisition_frequency=frequency,
        )
        if expected:
            assert UptechH5V1().read(path)[0].shape == (5, 2)
        else:
            with pytest.raises(ValueError, match="acquisition_frequency"):
                UptechH5V1().read(path)

    def test_non_monotonic_rejected(self, tmp_path):
        """Reject a time array which does not increase."""
        path = _write_uptech(
            tmp_path / "backwards.hdf5",
            shape=(3, 2),
            time=[_T0, _T0 + 0.5, _T0 + 0.25],
        )
        with pytest.raises(ValueError, match="must be increasing"):
            UptechH5V1().read(path)

    def test_non_finite_rejected(self, tmp_path):
        """Reject a time array containing non-finite values."""
        path = _write_uptech(
            tmp_path / "nan_time.hdf5",
            shape=(3, 2),
            time=[_T0, np.nan, _T0 + 1.0],
        )
        with pytest.raises(ValueError, match="non-finite"):
            UptechH5V1().read(path)

    def test_time_must_match_frequency(self, tmp_path):
        """Reject a time coordinate inconsistent with acquisition frequency."""
        path = _write_uptech(
            tmp_path / "invalid_time.hdf5",
            shape=(3, 2),
            time=[_T0, _T0 + 1.0, _T0 + 2.0],
            acquisition_frequency=2.0,
        )
        with pytest.raises(ValueError, match="acquisition_frequency"):
            UptechH5V1().read(path)

    @pytest.mark.parametrize("frequency", [0.0, -1.0, np.nan, np.inf])
    def test_frequency_must_be_finite_and_positive(self, tmp_path, frequency):
        """Reject invalid acquisition frequency for a one-sample file."""
        path = _write_uptech(
            tmp_path / "invalid_frequency.hdf5",
            shape=(1, 2),
            time=[_T0],
            acquisition_frequency=frequency,
        )
        with pytest.raises(ValueError, match="finite and positive"):
            UptechH5V1().read(path)
