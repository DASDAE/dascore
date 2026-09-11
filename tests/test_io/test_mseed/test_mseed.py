"""Tests for MiniSEED IO."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import InvalidFiberFileError, MissingOptionalDependencyError
from dascore.io.mseed import core as mseed_core
from dascore.io.mseed import utils as mseed_utils
from dascore.io.mseed.core import MSeedV2, MSeedV3
from dascore.utils.downloader import fetch
from tests.test_io._common_io_test_utils import skip_timeout


def _mseed_v2_header():
    """Return enough fixed header bytes to identify a MiniSEED 2 file."""
    header = bytearray(48)
    header[0:8] = b"000001D "
    header[8:13] = b"STAT "
    header[13:15] = b"  "
    header[15:18] = b"HSF"
    header[18:20] = b"XX"
    header[20:32] = bytes.fromhex("07e800010000000000000001")
    header[44:48] = bytes.fromhex("00300000")
    return header


def _write_mseed(
    path,
    format_version=3,
    starts=None,
    sample_rates=None,
    source_ids=None,
    data_samples=None,
    encoding=None,
):
    """Write a small MiniSEED file."""
    pymseed = pytest.importorskip("pymseed")

    source_ids = source_ids or [f"FDSN:XX_{ind:05d}__H_S_F" for ind in range(3)]
    starts = starts or [pymseed.timestr2nstime("2024-01-01T00:00:00Z")] * len(
        source_ids
    )
    sample_rates = sample_rates or [10.0] * len(source_ids)
    data_samples = data_samples or [
        np.arange(10, dtype=np.int32) + ind * 100 for ind in range(len(source_ids))
    ]
    traces = pymseed.MS3TraceList()
    for source_id, start, sample_rate, data in zip(
        source_ids, starts, sample_rates, data_samples, strict=True
    ):
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
        encoding=pymseed.DataEncoding.INT32 if encoding is None else encoding,
    )
    return path


def _write_mseed_v2_header(path):
    """Write enough fixed header bytes to identify a MiniSEED 2 file."""
    path.write_bytes(_mseed_v2_header())
    return path


def _trace_segment(**kwargs):
    """Return a small decoded MiniSEED trace segment for helper tests."""
    data = kwargs.pop("data", np.arange(3, dtype=np.int32))
    # Annotated because the values are heterogeneous; without it the
    # inferred value type is object and every field of the splat below
    # is rejected.
    defaults: dict[str, Any] = dict(
        source_id="FDSN:XX_00000__H_S_F",
        network="XX",
        station="00000",
        location="",
        seed_channel="HSF",
        format_version="3",
        start_ns=0,
        sample_rate=10.0,
        sample_count=len(data),
        sample_type="i",
        encoding="11",
        publication_version=0,
        record_length=256,
    )
    defaults.update(kwargs)
    return mseed_utils._TraceSegment(**defaults, data=data)


def _trace_summary(**kwargs):
    """Return a small MiniSEED trace summary for helper tests."""
    segment = _trace_segment(**kwargs)
    summary_kwargs = {
        key: getattr(segment, key) for key in mseed_utils._TraceInfo.__annotations__
    }
    return mseed_utils._TraceSummary(
        **summary_kwargs,
        dtype=str(segment.data.dtype),
    )


class _MiniSeedRecord:
    """Small record stub for exercising read-planning paths."""

    def __init__(
        self,
        sourceid="FDSN:XX_00000__H_S_F",
        data=None,
        samprate=10.0,
        starttime=0,
        fail_unpack=False,
    ):
        self.sourceid = sourceid
        self.formatversion = 3
        self.starttime = starttime
        self.samprate = samprate
        self.np_datasamples = np.arange(3, dtype=np.int32) if data is None else data
        self.samplecnt = len(self.np_datasamples)
        self.endtime = starttime + int((self.samplecnt - 1) * 1_000_000_000 / samprate)
        self.sampletype = "i"
        self.encoding = 3
        self.pubversion = 0
        self.reclen = 256
        self.fail_unpack = fail_unpack

    def unpack_data(self):
        """Raise when a test expects this record to be skipped."""
        if self.fail_unpack:
            raise AssertionError(f"{self.sourceid} should not unpack")


def _pymseed_for_records(records):
    """Return a small PyMseed-like object for record iteration tests."""

    class PyMseed:
        class MS3Record:
            @staticmethod
            def from_file(path, *, unpack_data):
                assert path == "unused"
                assert unpack_data is False
                yield from records

        @staticmethod
        def sourceid2nslc(source_id):
            _, rest = source_id.split(":", maxsplit=1)
            network, station, location, band, source, subsource = rest.split("_")
            return network, station, location, f"{band}{source}{subsource}"

    return PyMseed


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

    def test_get_format_from_dascore(self, mseed_v3_path):
        """DASCore can detect MiniSEED files through plugin discovery."""
        assert dc.get_format(mseed_v3_path) == ("MSEED", "3")

    def test_get_format_without_pymseed(self, tmp_path, hide_module):
        """MiniSEED headers can be detected without PyMseed installed."""
        hide_module("pymseed")
        v2_path = _write_mseed_v2_header(tmp_path / "test_v2.mseed")
        v3_path = tmp_path / "test_v3.mseed"
        v3_path.write_bytes(b"MS\x03" + bytes(45))
        assert MSeedV2().get_format(v2_path) == ("MSEED", "2")
        assert MSeedV2().get_format(v3_path) == ("MSEED", "3")

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

    def test_fractional_rate_is_exact(self, tmp_path):
        """A 40 Hz trace gets an exact 1/40 s step, not a rounded one."""
        path = _write_mseed(tmp_path / "forty.mseed", sample_rates=[40.0] * 3)
        patch = dc.read(path, file_format="MSEED", file_version="3")[0]
        time = patch.get_coord("time")
        assert time.step_exact == Fraction(1, 40)
        assert time.step == np.timedelta64(25_000_000, "ns")
        path = _write_mseed(tmp_path / "third.mseed", sample_rates=[3000.0] * 3)
        time = dc.read(path, file_format="MSEED", file_version="3")[0].get_coord("time")
        assert time.step_exact == Fraction(1, 3000)
        assert time.stop == time.min() + np.timedelta64(10 * 10**9 // 3000, "ns")

    def test_trim_keeps_grid_and_boundary_sample(self, tmp_path):
        """Trimming at a whole second keeps that sample and the grid's phase."""
        pymseed = pytest.importorskip("pymseed")
        start = pymseed.timestr2nstime("2024-01-01T00:00:00Z")
        data = [np.arange(3000, dtype=np.int32)] * 3
        path = _write_mseed(
            tmp_path / "trim.mseed", sample_rates=[1024.0] * 3, data_samples=data
        )
        full = dc.read(path, file_format="MSEED", file_version="3")[0]
        t0 = np.datetime64(start, "ns")
        lo, hi = t0 + np.timedelta64(1, "s"), t0 + np.timedelta64(2, "s")
        trimmed = dc.read(path, file_format="MSEED", file_version="3", time=(lo, hi))[0]
        time = trimmed.get_coord("time")
        assert time.min() == lo and time.max() == hi
        assert time.step_exact == Fraction(1, 1024)
        # The trim discards the first record, so the grid is anchored on the
        # writer's rounded start of the second; labels agree to the ns.
        expected = full.get_coord("time").values[1024:2049]
        assert np.all(np.abs(time.values - expected) <= np.timedelta64(1, "ns"))
        odd = dc.read(
            path,
            file_format="MSEED",
            file_version="3",
            time=(full.get_coord("time")[3], None),
        )[0].get_coord("time")
        assert np.array_equal(odd.values, full.get_coord("time").values[3:])

    def test_nanosecond_gap_on_whole_grid_is_not_merged(self):
        """A one-nanosecond offset on a whole-nanosecond grid is a real gap."""
        assert mseed_utils._continues(1_000_000, 1_000_000, 1000.0)
        assert not mseed_utils._continues(1_000_000, 1_000_001, 1000.0)
        assert mseed_utils._continues(985_351_562, 985_351_563, 1024.0)
        assert not mseed_utils._continues(985_351_562, 985_351_565, 1024.0)

    def test_group_key_includes_phase(self, tmp_path):
        """Segments trimmed onto different phases are not stacked together."""
        path = _write_mseed(tmp_path / "phase.mseed", sample_rates=[1024.0] * 3)
        segments = mseed_utils._read_segments(path, pytest.importorskip("pymseed"))
        shifted = replace(segments[1], origin_offset=1)
        keys = {mseed_utils._get_group_key(s) for s in (segments[0], shifted)}
        assert len(keys) == 2

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

    def test_unequal_sample_counts_are_separate_patches(self, tmp_path):
        """Unequal trace lengths are preserved in separate patches."""
        data_samples = [
            np.arange(10, dtype=np.int32),
            np.arange(7, dtype=np.int32) + 100,
            np.arange(10, dtype=np.int32) + 200,
        ]
        path = _write_mseed(tmp_path / "unequal.mseed", data_samples=data_samples)
        spool = dc.read(path, file_format="MSEED", file_version="3")
        assert len(spool) == 2
        assert sorted(x.shape for x in spool) == [(1, 7), (2, 10)]
        assert sorted(len(x.get_coord("time")) for x in spool) == [7, 10]
        assert sorted(tuple(x.get_coord("channel").values) for x in spool) == [
            (0, 2),
            (1,),
        ]

    def test_read_source_patch_key(self, tmp_path):
        """source_patch_key filters logical MiniSEED groups."""
        path = _write_mseed(
            tmp_path / "mixed.mseed",
            format_version=3,
            sample_rates=[10.0, 20.0, 10.0],
        )
        summaries = dc.scan(path, file_format="MSEED", file_version="3")
        target = summaries[0].source_patch_key
        spool = dc.read(
            path, file_format="MSEED", file_version="3", source_patch_key=target
        )
        assert len(spool) == 1
        assert spool[0]._source.key == target

    def test_read_source_patch_key_with_time(self, mseed_v3_path):
        """source_patch_key from scan works with partial time reads."""
        target = dc.scan(mseed_v3_path, file_format="MSEED", file_version="3")[
            0
        ].source_patch_key
        time_min = np.datetime64("2024-01-01T00:00:00.200000000")
        time_max = np.datetime64("2024-01-01T00:00:00.500000000")
        spool = dc.read(
            mseed_v3_path,
            file_format="MSEED",
            file_version="3",
            source_patch_key=target,
            time=(time_min, time_max),
        )
        assert len(spool) == 1
        assert spool[0]._source.key == target
        assert spool[0].shape == (3, 4)
        assert spool[0].get_coord("time").min() == time_min

    def test_read_source_patch_key_with_non_overlapping_time(self, mseed_v3_path):
        """source_patch_key reads return empty spools for non-overlapping times."""
        target = dc.scan(mseed_v3_path, file_format="MSEED", file_version="3")[
            0
        ].source_patch_key
        spool = dc.read(
            mseed_v3_path,
            file_format="MSEED",
            file_version="3",
            source_patch_key=target,
            time=(
                np.datetime64("2024-01-01T01:00:00"),
                np.datetime64("2024-01-01T01:00:01"),
            ),
        )
        assert len(spool) == 0

    def test_source_patch_key_skips_unselected_record_unpack(self, monkeypatch):
        """source_patch_key reads do not decode records outside selected groups."""
        records = [
            _MiniSeedRecord("FDSN:XX_00000__H_S_F", samprate=10.0),
            _MiniSeedRecord("FDSN:XX_00001__H_S_F", samprate=20.0, fail_unpack=True),
        ]
        target = "v3:XX..HSF:0:10:3"
        monkeypatch.setattr(
            "dascore.io.mseed.core.optional_import",
            lambda name: _pymseed_for_records(records),
        )
        patches = list(
            MSeedV3().read(
                "unused",
                source_patch_key=target,
            )
        )
        assert len(patches) == 1
        assert patches[0]._source.key == target
        assert tuple(patches[0].get_coord("channel").values) == (0,)

    def test_channel_filter_skips_unselected_record_unpack(self, monkeypatch):
        """Channel reads do not decode records outside selected channels."""
        records = [
            _MiniSeedRecord("FDSN:XX_00000__H_S_F"),
            _MiniSeedRecord("FDSN:XX_00001__H_S_F", fail_unpack=True),
        ]
        monkeypatch.setattr(
            "dascore.io.mseed.core.optional_import",
            lambda name: _pymseed_for_records(records),
        )
        patches = list(
            MSeedV3().read(
                "unused",
                channel=(0, 0),
            )
        )
        assert len(patches) == 1
        assert tuple(patches[0].get_coord("channel").values) == (0,)

    def test_unmatched_source_patch_key_returns_no_patches(self, monkeypatch):
        """Unknown source patch IDs produce no decoded patches."""
        records = [_MiniSeedRecord("FDSN:XX_00000__H_S_F", fail_unpack=True)]
        monkeypatch.setattr(
            "dascore.io.mseed.core.optional_import",
            lambda name: _pymseed_for_records(records),
        )
        patches = list(
            MSeedV3().read(
                "unused",
                source_patch_key="v3:XX..HSF:1:10:3",
            )
        )
        assert patches == []

    def test_unmatched_channel_returns_no_patches(self, monkeypatch):
        """Channel selections with no sources produce no decoded patches."""
        records = [_MiniSeedRecord("FDSN:XX_00000__H_S_F", fail_unpack=True)]
        monkeypatch.setattr(
            "dascore.io.mseed.core.optional_import",
            lambda name: _pymseed_for_records(records),
        )
        patches = list(
            MSeedV3().read(
                "unused",
                channel=(10, 10),
            )
        )
        assert patches == []

    def test_discontinuous_source_records_are_separate_patches(self, tmp_path):
        """Discontinuous records for one source ID are not coalesced."""
        pymseed = pytest.importorskip("pymseed")

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
        assert summary.source_patch_key

    @pytest.mark.parametrize(
        ("fixture_name", "file_version"),
        (("mseed_v2_path", "2"), ("mseed_v3_path", "3")),
    )
    def test_scan_matches_read_summary(self, request, fixture_name, file_version):
        """Light scan metadata matches the corresponding read summary."""
        path = request.getfixturevalue(fixture_name)
        summary = dc.scan(path, file_format="MSEED", file_version=file_version)[0]
        patch_summary = dc.read(path, file_format="MSEED", file_version=file_version)[
            0
        ].summary
        assert summary.dims == patch_summary.dims
        assert summary.shape == patch_summary.shape
        assert summary.dtype == patch_summary.dtype
        assert summary.source_patch_key == patch_summary.source_patch_key
        for dim in summary.dims:
            scan_coord = summary.get_coord_summary(dim)
            read_coord = patch_summary.get_coord_summary(dim)
            assert scan_coord.min == read_coord.min
            assert scan_coord.max == read_coord.max
            assert scan_coord.step == read_coord.step

    def test_scan_v2_metadata(self, mseed_v2_path):
        """MiniSEED v2 scan reports expected metadata from headers."""
        summary = dc.scan(mseed_v2_path, file_format="MSEED", file_version="2")[0]
        assert summary.shape == (3, 10)
        assert summary.dtype == "int32"
        assert summary.source_format == "MSEED"
        assert summary.source_version == "2"
        assert summary.source_patch_key.startswith("v2:")

    def test_scan_does_not_unpack_records(self, monkeypatch):
        """Scan uses record headers without decoding sample payloads."""

        class Record:
            sourceid = "FDSN:XX_00000__H_S_F"
            formatversion = 3
            starttime = 0
            samprate = 10.0
            samplecnt = 10
            sampletype = None
            encoding = 3
            pubversion = 0
            reclen = 256

            def unpack_data(self):
                raise AssertionError("scan should not unpack samples")

        class PyMseed:
            class MS3Record:
                @staticmethod
                def from_file(path, *, unpack_data):
                    assert path == "unused"
                    assert unpack_data is False
                    yield Record()

            @staticmethod
            def sourceid2nslc(source_id):
                assert source_id == Record.sourceid
                return "XX", "00000", "", "HSF"

        payload = mseed_utils._scan_patches("unused", PyMseed)[0]
        assert payload.shape == (1, 10)
        assert payload.dtype == "int32"

    def test_missing_pymseed_raises(self, mseed_v3_path, hide_module):
        """Explicit MiniSEED reads require PyMseed."""
        hide_module("pymseed")
        with pytest.raises(MissingOptionalDependencyError):
            dc.read(mseed_v3_path, file_format="MSEED", file_version="3")

        with pytest.raises(MissingOptionalDependencyError):
            dc.scan(mseed_v3_path, file_format="MSEED", file_version="3")


class TestMiniSeedUtils:
    """Focused tests for MiniSEED helper edge cases."""

    def test_zero_sample_rate_raises(self):
        """A zero sample rate cannot be converted to a sample step."""
        with pytest.raises(ValueError, match="sample rate"):
            mseed_utils._duration_ns(0)

    def test_negative_sample_rate_is_sample_period(self):
        """Negative MiniSEED sample rates are interpreted as sample periods."""
        assert mseed_utils._duration_ns(-0.1) == 100_000_000

    def test_next_start_ns_avoids_sample_step_drift(self):
        """Record adjacency uses total duration instead of rounded sample steps."""
        segment = _trace_segment(sample_rate=3.0, sample_count=3)
        assert segment.sample_step_ns == 333_333_333
        assert segment.next_start_ns == 1_000_000_000

    def test_open_time_limits(self):
        """None and ellipsis are open time bounds."""
        assert mseed_utils._get_time_limits(...) == (None, None)
        assert mseed_utils._get_time_limits((None, ...)) == (None, None)

    def test_exact_time_limits_and_trim(self):
        """Trimming at a fractional grid keeps the selected samples and origin."""
        limits = mseed_utils._get_time_limits(
            (np.datetime64(333333334, "ns"), np.datetime64(1400000000, "ns"))
        )
        assert limits == (333333334, 1400000000)
        segment = _trace_segment(sample_rate=3.0, data=np.arange(6, dtype=np.int32))
        selected = mseed_utils._trim_segment_time(segment, limits)
        np.testing.assert_array_equal(selected.data, [2, 3, 4])
        assert selected.start_ns == 666666666
        assert selected.origin_offset == 2
        assert selected.sample_count == 3

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

    def test_source_id_error_falls_back_to_station(self):
        """PyMseed source ID parser errors are treated as unparseable IDs."""

        class BadPyMseed:
            class SourceIDError(Exception):
                pass

            @staticmethod
            def sourceid2nslc(source_id):
                raise BadPyMseed.SourceIDError("bad source id")

        assert mseed_utils._source_id_to_nslc(BadPyMseed, "not-fdsn") == (
            "",
            "not-fdsn",
            "",
            "",
        )

    def test_unexpected_source_id_errors_propagate(self):
        """Only expected source ID parser errors are swallowed."""

        class BadPyMseed:
            @staticmethod
            def sourceid2nslc(source_id):
                raise RuntimeError("unexpected bug")

        with pytest.raises(RuntimeError, match="unexpected bug"):
            mseed_utils._source_id_to_nslc(BadPyMseed, "not-fdsn")

    def test_record_to_segment_skips_non_overlapping_records(self):
        """Record metadata can be rejected before sample unpacking."""

        class Record:
            starttime = 0
            endtime = 10

            def unpack_data(self):
                raise AssertionError("should not unpack")

        assert mseed_utils._record_to_segment(Record(), None, (20, 30)) is None

    def test_trim_segment_time_returns_none_for_empty_selection(self):
        """Time trimming can remove all samples from a decoded segment."""
        segment = _trace_segment()
        assert mseed_utils._trim_segment_time(segment, (25_000_000, 75_000_000)) is None

    def test_record_dtype_from_sample_type(self):
        """Populated sample types are preferred when inferring scan dtype."""

        class Record:
            sampletype = "d"
            encoding = 3

        assert mseed_utils._record_dtype(Record()) == "float64"

    # One encoding per distinct dtype the table maps to, plus one it does
    # not know: repeating int32 for 3, 10 and 11 says the same thing thrice.
    @pytest.mark.parametrize(
        ("encoding", "dtype"),
        (
            (0, "S1"),
            (1, "int32"),
            (3, "int32"),
            (4, "float32"),
            (5, "float64"),
            (999, ""),
        ),
    )
    def test_record_dtype_from_encoding(self, encoding, dtype):
        """MiniSEED scan dtype can be inferred from known encodings."""

        class Record:
            sampletype = None

        Record.encoding = encoding
        assert mseed_utils._record_dtype(Record()) == dtype

    def test_contiguous_source_summaries_are_coalesced(self):
        """Contiguous scan summaries from one source are merged."""
        first = _trace_summary()
        second = _trace_summary(start_ns=first.next_start_ns, record_length=512)
        out = mseed_utils._coalesce_source_summaries([second, first])
        assert len(out) == 1
        assert out[0].sample_count == first.sample_count + second.sample_count
        assert out[0].record_length == 512

    def test_discontinuous_source_summaries_are_not_coalesced(self):
        """Discontinuous scan summaries from one source stay separate."""
        first = _trace_summary()
        second = _trace_summary(start_ns=first.next_start_ns + first.sample_step_ns)
        out = mseed_utils._coalesce_source_summaries([first, second])
        assert len(out) == 2

    def test_source_segments_with_different_encodings_are_not_coalesced(self):
        """Read and scan coalescing both preserve MiniSEED encoding changes."""
        first = _trace_segment(encoding="3")
        second = _trace_segment(start_ns=first.next_start_ns, encoding="11")
        out = mseed_utils._coalesce_source_segments([first, second])
        assert len(out) == 2

    def test_patch_from_segments_defaults_to_local_channel_indices(self):
        """Patches can still be built without a global channel map."""
        segment = _trace_segment()
        coords = mseed_utils._get_coords([segment])
        assert tuple(coords["channel"].values) == (0,)

    def test_source_patch_key_uses_group_key_fields(self):
        """Source patch IDs are built from the typed MiniSEED group key."""
        group_key = mseed_utils._TraceGroupKey(
            format_version="3",
            network="XX",
            location="",
            seed_channel="HSF",
            start_ns=10,
            sample_rate=10.0,
            sample_count=12,
        )
        assert mseed_utils._source_patch_key(group_key) == "v3:XX..HSF:10:10:12"

    def test_detect_format_no_records(self, tmp_path):
        """Files with no records are not MiniSEED."""
        path = tmp_path / "empty.mseed"
        path.write_bytes(b"")
        out = mseed_utils._detect_format(path)
        assert out is False

    def test_detect_format_missing_file(self, tmp_path):
        """Missing paths are not MiniSEED."""
        assert mseed_utils._detect_format(tmp_path / "missing.mseed") is False

    def test_detect_format_unsupported_version(self, tmp_path):
        """Unsupported MiniSEED versions are rejected."""
        path = tmp_path / "v4.mseed"
        path.write_bytes(b"MS\x04" + bytes(45))
        assert mseed_utils._detect_format(path) is False

    @pytest.mark.parametrize(
        "field,value",
        (
            ("sequence", b"A00001"),
            ("record_type", b"X"),
            ("reserved", b"X"),
            ("station", b"     "),
            ("channel", b"H F"),
            ("network", b"  "),
            ("year", bytes.fromhex("0700")),
            ("data_offset", bytes.fromhex("002f")),
            ("sample_count", bytes.fromhex("0000")),
        ),
    )
    def test_detect_mseed_v2_header_rejects_invalid_fields(self, field, value):
        """Each fixed-header guard can reject invalid MiniSEED 2 headers."""
        header = _mseed_v2_header()
        slices = {
            "sequence": slice(0, 6),
            "record_type": slice(6, 7),
            "reserved": slice(7, 8),
            "station": slice(8, 13),
            "channel": slice(15, 18),
            "network": slice(18, 20),
            "year": slice(20, 22),
            "data_offset": slice(44, 46),
            "sample_count": slice(30, 32),
        }
        header[slices[field]] = value
        assert not mseed_utils._detect_mseed_v2_header(header)

    def test_detect_mseed_v2_header_rejects_short_header(self):
        """Truncated fixed headers are not MiniSEED 2."""
        assert not mseed_utils._detect_mseed_v2_header(bytes(47))

    def test_detect_mseed_v2_header_handles_unpack_error(self, monkeypatch):
        """Header unpack failures are treated as non-MiniSEED."""

        def _raise_unpack(*args, **kwargs):
            raise mseed_utils.struct.error("bad unpack")

        monkeypatch.setattr(mseed_utils, "unpack", _raise_unpack)
        assert not mseed_utils._detect_mseed_v2_header(_mseed_v2_header())

    def test_close_sample_rates_have_distinct_source_patch_keys(self):
        """Full precision sample rates avoid source patch ID collisions."""
        group_a = mseed_utils._TraceGroupKey("3", "XX", "", "HSF", 0, 1.00000001, 10)
        group_b = mseed_utils._TraceGroupKey("3", "XX", "", "HSF", 0, 1.00000002, 10)
        assert mseed_utils._source_patch_key(group_a) != mseed_utils._source_patch_key(
            group_b
        )


class TestRealMiniSeed:
    """Tests using real DAS MiniSEED data from the test-data registry."""

    def test_read_das_station_channels(self):
        """Etna DAS station-coded channels preserve their full sample counts."""
        pytest.importorskip("pymseed")

        with skip_timeout():
            path = fetch("etna_9n_3chan_10s.mseed")
        spool = dc.read(path)
        assert len(spool) == 3
        assert sorted(x.shape for x in spool) == [(1, 13556), (1, 13729), (1, 13735)]
        assert {x.coords.get_array("network")[0] for x in spool} == {"9N"}
        assert {x.coords.get_array("station")[0] for x in spool} == {
            "00066",
            "00067",
            "00068",
        }
        assert {x.coords.get_array("seed_channel")[0] for x in spool} == {"HSF"}


class TestDecodedArrayContract:
    """Stored encodings must describe the array returned by the decoder."""

    @pytest.mark.parametrize(
        "rate, count, offset", [(1024.0, 8, 7812500), (3.0, 1, 333333334)]
    )
    def test_fractional_final_record(self, tmp_path, rate, count, offset):
        """A final one-sample packet survives fractional endpoint rounding."""
        start = 1704067200000000000
        first = _write_mseed(
            tmp_path / "first.mseed",
            starts=[start],
            sample_rates=[rate],
            source_ids=["FDSN:XX_00000__H_S_F"],
            data_samples=[np.arange(count, dtype=np.int32)],
        )
        last = _write_mseed(
            tmp_path / "last.mseed",
            starts=[start + offset],
            sample_rates=[rate],
            source_ids=["FDSN:XX_00000__H_S_F"],
            data_samples=[np.array([101], dtype=np.int32)],
        )
        path = tmp_path / "joined.mseed"
        path.write_bytes(first.read_bytes() + last.read_bytes())
        full = dc.read(path)[0]
        np.testing.assert_array_equal(full.data[0], np.r_[np.arange(count), 101])
        tail = dc.read(path, time=(-1, None), samples=True)[0]
        np.testing.assert_array_equal(tail.data, [[101]])
        assert full.attrs.patch_id == tail.attrs.patch_id

    def test_overlapping_groups_same_source(self, tmp_path):
        """Overlapping records with different rates load only their logical group."""
        parts = []
        for index, rate in enumerate([10.0, 20.0]):
            part = _write_mseed(
                tmp_path / f"rate{index}.mseed",
                sample_rates=[rate],
                source_ids=["FDSN:XX_00000__H_S_F"],
                data_samples=[np.arange(10, dtype=np.int32) + index * 100],
            )
            parts.append(part.read_bytes())
        path = tmp_path / "overlap.mseed"
        path.write_bytes(b"".join(parts))
        summaries = dc.scan(path)
        assert len(summaries) == 2
        for index, summary in enumerate(summaries):
            patch = dc.read(path, source_patch_key=summary.source_patch_key)[0]
            np.testing.assert_array_equal(patch.data, [np.arange(10) + index * 100])
            assert patch.attrs.patch_id == summary.attrs.patch_id

    def test_mixed_channel_dtypes(self, tmp_path):
        """A channel subset retains the dtype promoted across its source group."""
        pymseed = pytest.importorskip("pymseed")
        rows = [np.arange(10, dtype=np.int32), np.arange(10, dtype=np.float32) + 0.25]
        parts = []
        for index, (data, sample_type, encoding) in enumerate(
            zip(
                rows,
                ["i", "f"],
                [pymseed.DataEncoding.INT32, pymseed.DataEncoding.FLOAT32],
                strict=True,
            )
        ):
            traces = pymseed.MS3TraceList()
            traces.add_data(
                sourceid=f"FDSN:XX_{index:05d}__H_S_F",
                data_samples=data,
                sample_type=sample_type,
                sample_rate=10.0,
                starttime=pymseed.timestr2nstime("2024-01-01T00:00:00Z"),
            )
            part = tmp_path / f"{index}.mseed"
            traces.to_file(
                str(part), overwrite=True, format_version=3, encoding=encoding
            )
            parts.append(part.read_bytes())
        path = tmp_path / "mixed.mseed"
        path.write_bytes(b"".join(parts))
        summary = dc.scan(path)[0]
        full = dc.read(path)[0]
        selected = dc.read(path, channel=(1, 1))[0]
        assert summary.dtype == str(full.dtype) == str(selected.dtype) == "float64"
        np.testing.assert_array_equal(full.data, np.stack(rows))
        np.testing.assert_array_equal(selected.data, rows[1][None, :])
        assert full.attrs.patch_id == selected.attrs.patch_id == summary.attrs.patch_id
        assert full._source.key == selected._source.key == summary.source_patch_key

    @pytest.mark.parametrize("version", [2, 3])
    def test_int16_decodes_as_int32(self, tmp_path, version):
        """INT16 storage expands to int32 on full and bounded reads."""
        pymseed = pytest.importorskip("pymseed")
        path = _write_mseed(
            tmp_path / "int16.mseed",
            format_version=version,
            encoding=pymseed.DataEncoding.INT16,
        )
        summary = dc.scan(path)[0]
        full = dc.read(path)[0]
        selected = dc.read(path, time=(2, 7), samples=True)[0]
        expected = np.arange(10, dtype=np.int32) + np.arange(3)[:, None] * 100
        assert summary.dtype == str(full.dtype) == "int32"
        assert summary.attrs.sample_type == full.attrs.sample_type == "i"
        np.testing.assert_array_equal(full.data, expected)
        np.testing.assert_array_equal(selected.data, expected[:, 2:7])
        assert full.attrs.patch_id == selected.attrs.patch_id

    @pytest.mark.parametrize(
        "windows, shape",
        [
            ({"channel": (0, 0)}, (0, 10)),
            ({"time": (3, 3)}, (3, 0)),
        ],
    )
    def test_empty_windows(self, mseed_v3_path, windows, shape):
        """Empty source windows retain the declared axes and decoded dtype."""
        out = MSeedV3().read_array(mseed_v3_path, windows)
        assert out.shape == shape
        assert out.dtype == np.dtype("int32")


class TestDecodedSourceMatching:
    """Malformed decoded groups fail as file errors before patch assembly."""

    @pytest.mark.parametrize("kind", ["missing", "duplicate", "wrong_count"])
    def test_invalid_decoded_group(self, mseed_v3_path, monkeypatch, kind):
        """Require one decoded segment per declared source and sample grid."""
        original = mseed_core._read_segments

        def invalid_segments(*args, **kwargs):
            segments = original(*args, **kwargs)
            if kind == "missing":
                return segments[1:]
            if kind == "duplicate":
                return [*segments, segments[0]]
            return [
                replace(segments[0], sample_count=1, data=segments[0].data[:1]),
                *segments[1:],
            ]

        monkeypatch.setattr(mseed_core, "_read_segments", invalid_segments)
        with pytest.raises(InvalidFiberFileError, match="decoded segments"):
            dc.read(mseed_v3_path)

    def test_decoded_sources_follow_metadata_order(self, mseed_v3_path, monkeypatch):
        """The decoder's iteration order cannot swap channel data."""
        expected = dc.read(mseed_v3_path)[0]
        original = mseed_core._read_segments
        monkeypatch.setattr(
            mseed_core, "_read_segments", lambda *a, **kw: original(*a, **kw)[::-1]
        )
        actual = dc.read(mseed_v3_path)[0]
        np.testing.assert_array_equal(actual.data, expected.data)
        assert actual.coords == expected.coords
