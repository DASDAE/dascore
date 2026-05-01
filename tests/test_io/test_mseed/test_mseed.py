"""Tests for MiniSEED IO."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import MissingOptionalDependencyError
from dascore.io.mseed import utils as mseed_utils
from dascore.io.mseed.core import MSeedV2

pytest.importorskip("pymseed")


def _write_mseed(
    path,
    format_version=3,
    starts=None,
    sample_rates=None,
    source_ids=None,
):
    """Write a small MiniSEED file."""
    import pymseed

    source_ids = source_ids or [f"FDSN:XX_{ind:05d}__H_S_F" for ind in range(3)]
    starts = starts or [pymseed.timestr2nstime("2024-01-01T00:00:00Z")] * len(
        source_ids
    )
    sample_rates = sample_rates or [10.0] * len(source_ids)
    traces = pymseed.MS3TraceList()
    for ind, (source_id, start, sample_rate) in enumerate(
        zip(source_ids, starts, sample_rates, strict=True)
    ):
        data = np.arange(10, dtype=np.int32) + ind * 100
        traces.add_data(
            sourceid=source_id,
            data_samples=data,
            sample_type="i",
            sample_rate=sample_rate,
            start_time=start,
        )
    traces.to_file(
        str(path),
        overwrite=True,
        format_version=format_version,
        encoding=pymseed.DataEncoding.INT32,
    )
    return path


@pytest.fixture
def mseed_v2_path(tmp_path):
    """Return a small MiniSEED v2 path."""
    return _write_mseed(tmp_path / "test_v2.mseed", format_version=2)


@pytest.fixture
def mseed_v3_path(tmp_path):
    """Return a small MiniSEED v3 path."""
    return _write_mseed(tmp_path / "test_v3.mseed", format_version=3)


class TestMiniSeedGetFormat:
    """Tests for detecting MiniSEED files."""

    def test_get_format_v2(self, mseed_v2_path):
        """MiniSEED v2 files can be detected."""
        assert MSeedV2().get_format(mseed_v2_path) == ("MSEED", "2")

    def test_get_format_v3(self, mseed_v3_path):
        """MiniSEED v3 files can be detected."""
        assert MSeedV2().get_format(mseed_v3_path) == ("MSEED", "3")

    def test_get_format_from_dascore(self, mseed_v3_path):
        """DASCore can detect MiniSEED files through plugin discovery."""
        assert dc.get_format(mseed_v3_path) == ("MSEED", "3")

    def test_get_format_too_small(self, tmp_path):
        """Small invalid files return False."""
        path = tmp_path / "bad.mseed"
        path.write_bytes(b"abc")
        assert MSeedV2().get_format(path) is False


class TestMiniSeedRead:
    """Tests for reading MiniSEED files."""

    def test_read_multichannel(self, mseed_v3_path):
        """Compatible traces are merged into one channel-time patch."""
        patch = dc.read(mseed_v3_path, file_format="MSEED", file_version="3")[0]
        assert patch.dims == ("channel", "time")
        assert patch.shape == (3, 10)
        assert np.all(patch.data[1] == np.arange(10) + 100)
        assert patch.get_coord("time").step == np.timedelta64(100_000_000, "ns")
        assert tuple(patch.coords.get_array("source_id")) == (
            "FDSN:XX_00000__H_S_F",
            "FDSN:XX_00001__H_S_F",
            "FDSN:XX_00002__H_S_F",
        )
        assert tuple(patch.coords.get_array("network")) == ("XX", "XX", "XX")
        assert tuple(patch.coords.get_array("station")) == ("00000", "00001", "00002")
        assert tuple(patch.coords.get_array("location")) == ("", "", "")
        assert tuple(patch.coords.get_array("seed_channel")) == ("HSF", "HSF", "HSF")

    def test_same_channel_different_stations_merge(self, tmp_path):
        """Generic MiniSEED sources with compatible timing merge by station."""
        path = _write_mseed(
            tmp_path / "multi_station.mseed",
            source_ids=[
                "FDSN:XX_00066__H_S_F",
                "FDSN:XX_00067__H_S_F",
                "FDSN:XX_00068__H_S_F",
            ],
        )
        patch = dc.read(path, file_format="MSEED", file_version="3")[0]
        assert patch.shape == (3, 10)
        assert tuple(patch.coords.get_array("station")) == ("00066", "00067", "00068")
        assert tuple(patch.coords.get_array("seed_channel")) == ("HSF", "HSF", "HSF")

    def test_different_seed_channels_are_separate_patches(self, tmp_path):
        """Different physical components should not be merged into one patch."""
        path = _write_mseed(
            tmp_path / "components.mseed",
            source_ids=[
                "FDSN:XX_TEST__B_H_N",
                "FDSN:XX_TEST__B_H_E",
                "FDSN:XX_TEST__B_H_Z",
            ],
        )
        spool = dc.read(path, file_format="MSEED", file_version="3")
        assert len(spool) == 3
        assert {x.coords.get_array("seed_channel")[0] for x in spool} == {
            "BHE",
            "BHN",
            "BHZ",
        }

    def test_read_select_time_and_channel(self, mseed_v3_path):
        """Time and channel selections are applied to the output patch."""
        time_min = np.datetime64("2024-01-01T00:00:00.200000000")
        time_max = np.datetime64("2024-01-01T00:00:00.500000000")
        patch = dc.read(
            mseed_v3_path,
            file_format="MSEED",
            file_version="3",
            time=(time_min, time_max),
            channel=(1, 3),
        )[0]
        assert patch.shape == (2, 4)
        assert np.all(patch.get_coord("channel").values == np.array([1, 2]))
        assert patch.get_coord("time").min() == time_min

    def test_incompatible_sample_rates_are_separate_patches(self, tmp_path):
        """Incompatible traces are returned as separate patches."""
        path = _write_mseed(
            tmp_path / "mixed.mseed",
            format_version=3,
            sample_rates=[10.0, 20.0, 10.0],
        )
        spool = dc.read(path, file_format="MSEED", file_version="3")
        assert len(spool) == 2
        assert sorted(x.shape[0] for x in spool) == [1, 2]

    def test_read_source_patch_id(self, tmp_path):
        """source_patch_id filters logical MiniSEED groups."""
        path = _write_mseed(
            tmp_path / "mixed.mseed",
            format_version=3,
            sample_rates=[10.0, 20.0, 10.0],
        )
        summaries = dc.scan(path, file_format="MSEED", file_version="3")
        target = summaries[0].source_patch_id
        spool = dc.read(
            path, file_format="MSEED", file_version="3", source_patch_id=target
        )
        assert len(spool) == 1
        assert spool[0].attrs["_source_patch_id"] == target

    def test_read_source_patch_id_with_time(self, mseed_v3_path):
        """source_patch_id from scan works with partial time reads."""
        target = dc.scan(mseed_v3_path, file_format="MSEED", file_version="3")[
            0
        ].source_patch_id
        time_min = np.datetime64("2024-01-01T00:00:00.200000000")
        time_max = np.datetime64("2024-01-01T00:00:00.500000000")
        spool = dc.read(
            mseed_v3_path,
            file_format="MSEED",
            file_version="3",
            source_patch_id=target,
            time=(time_min, time_max),
        )
        assert len(spool) == 1
        assert spool[0].attrs["_source_patch_id"] == target
        assert spool[0].shape == (3, 4)
        assert spool[0].get_coord("time").min() == time_min

    def test_read_source_patch_id_with_non_overlapping_time(self, mseed_v3_path):
        """source_patch_id reads return empty spools for non-overlapping times."""
        target = dc.scan(mseed_v3_path, file_format="MSEED", file_version="3")[
            0
        ].source_patch_id
        spool = dc.read(
            mseed_v3_path,
            file_format="MSEED",
            file_version="3",
            source_patch_id=target,
            time=(
                np.datetime64("2024-01-01T01:00:00"),
                np.datetime64("2024-01-01T01:00:01"),
            ),
        )
        assert len(spool) == 0

    def test_discontinuous_source_records_are_separate_patches(self, tmp_path):
        """Discontinuous records for one source ID are not coalesced."""
        import pymseed

        start = pymseed.timestr2nstime("2024-01-01T00:00:00Z")
        path = _write_mseed(
            tmp_path / "discontinuous.mseed",
            source_ids=["FDSN:XX_TEST__H_S_F", "FDSN:XX_TEST__H_S_F"],
            starts=[start, start + 2_000_000_000],
        )
        spool = dc.read(path, file_format="MSEED", file_version="3")
        assert len(spool) == 2


class TestMiniSeedScan:
    """Tests for scanning MiniSEED files."""

    def test_scan(self, mseed_v3_path):
        """Scan returns metadata for the logical MiniSEED patch."""
        summary = dc.scan(mseed_v3_path, file_format="MSEED", file_version="3")[0]
        assert summary.dims == ("channel", "time")
        assert summary.shape == (3, 10)
        assert summary.dtype == "int32"
        assert summary.source_format == "MSEED"
        assert summary.source_version == "3"
        assert summary.source_patch_id

    def test_missing_pymseed_raises(self, mseed_v3_path, monkeypatch):
        """Explicit MiniSEED reads require PyMseed."""
        import dascore.io.mseed.core as mseed_core

        def _optional_import(*args, **kwargs):
            raise MissingOptionalDependencyError("missing")

        monkeypatch.setattr(mseed_core, "optional_import", _optional_import)
        with pytest.raises(MissingOptionalDependencyError):
            dc.read(mseed_v3_path, file_format="MSEED", file_version="3")


class TestMiniSeedUtils:
    """Focused tests for MiniSEED helper edge cases."""

    def test_zero_sample_rate_raises(self):
        """A zero sample rate cannot be converted to a sample step."""
        with pytest.raises(ValueError, match="sample rate"):
            mseed_utils._sample_step_ns(0)

    def test_negative_sample_rate_is_sample_period(self):
        """Negative MiniSEED sample rates are interpreted as sample periods."""
        assert mseed_utils._sample_step_ns(-0.1) == 100_000_000

    def test_open_time_limits(self):
        """None and ellipsis are open time bounds."""
        assert mseed_utils._get_time_limits(...) == (None, None)
        assert mseed_utils._get_time_limits((None, ...)) == (None, None)

    def test_bad_source_id_falls_back_to_station(self):
        """Unparseable source IDs are still preserved."""

        class BadPyMseed:
            @staticmethod
            def sourceid2nslc(source_id):
                raise ValueError("bad source id")

        assert mseed_utils._source_id_to_nslc(BadPyMseed, "not-fdsn") == (
            "",
            "not-fdsn",
            "",
            "",
        )

    def test_record_to_segment_skips_non_overlapping_records(self):
        """Record metadata can be rejected before sample unpacking."""

        class Record:
            starttime = 0
            endtime = 10

            def unpack_data(self):
                raise AssertionError("should not unpack")

        assert mseed_utils._record_to_segment(Record(), None, (20, 30)) is None

    def test_detect_format_no_records(self, tmp_path):
        """Files with no records are not MiniSEED."""

        class FakeMS3Record:
            @staticmethod
            def from_file(path, unpack_data=False):
                return iter(())

        class FakePyMseed:
            MS3Record = FakeMS3Record

        out = mseed_utils._detect_format(tmp_path / "empty.mseed", FakePyMseed)
        assert out is False

    def test_detect_format_unsupported_version(self, tmp_path):
        """Unsupported MiniSEED versions are rejected."""

        class Record:
            formatversion = 4

        class FakeMS3Record:
            @staticmethod
            def from_file(path, unpack_data=False):
                yield Record()

        class FakePyMseed:
            MS3Record = FakeMS3Record

        assert mseed_utils._detect_format(tmp_path / "v4.mseed", FakePyMseed) is False


class TestRealMiniSeed:
    """Tests using real DAS MiniSEED data from the test-data registry."""

    def test_read_merged_das_station_channels(self):
        """Etna DAS station-coded channels read as one channel-time patch."""
        from dascore.utils.downloader import fetch

        path = fetch("etna_9n_3chan_10s.mseed")
        patch = dc.read(path)[0]
        assert patch.dims == ("channel", "time")
        assert patch.shape[0] == 3
        assert tuple(patch.coords.get_array("network")) == ("9N", "9N", "9N")
        assert tuple(patch.coords.get_array("station")) == ("00066", "00067", "00068")
        assert tuple(patch.coords.get_array("seed_channel")) == ("HSF", "HSF", "HSF")
