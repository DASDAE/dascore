"""Tests for TDMS utilities to improve coverage."""

from __future__ import annotations

import datetime
import io
import struct
from pathlib import Path

import numpy as np
import pytest

import dascore as dc
from dascore.constants import INVENTORY_ATTRS
from dascore.exceptions import InvalidFiberFileError, RemoteCacheError
from dascore.io.silixah5.utils import _ATTR_MAP as _SILIXA_ATTR_MAP
from dascore.io.tdms import utils as tdms_utils
from dascore.io.tdms.core import TDMSFormatterV4713
from dascore.io.tdms.utils import parse_time_stamp, type_not_supported
from dascore.utils.downloader import fetch
from dascore.utils.io import IOResourceManager, LocalBinaryReader

IDAS = "iDAS005_tdms_example.626.tdms"


def _edited(path, edits=(), size=None):
    """Write the iDAS005 example to path, bytes replaced at offsets, cut to size."""
    raw = bytearray(Path(fetch(IDAS)).read_bytes())
    for offset, value in edits:
        raw[offset : offset + len(value)] = value
    path.write_bytes(raw[:size])
    return path


# struct formats of the fixed-size property types the example files hold
_PROPERTY_FORMATS = {0x03: "i", 0x05: "B", 0x06: "H", 0x07: "I", 0x0A: "d", 0x21: "?"}


def _big_endian(raw):
    """The one-segment int16 TDMS file `raw`, written as a big-endian file."""
    src, out = io.BytesIO(raw), bytearray()

    def copy(fmt):
        """Copy one number, swapping its bytes, and return it."""
        (value,) = struct.unpack("<" + fmt, src.read(struct.calcsize("<" + fmt)))
        out.extend(struct.pack(">" + fmt, value))
        return value

    def text():
        out.extend(src.read(copy("I")))

    tag, toc = struct.unpack("<4si", src.read(8))
    # The ToC itself is always little-endian.
    out.extend(struct.pack("<4si", tag, toc | tdms_utils.BIG_ENDIAN_MASK))
    copy("i"), copy("Q")
    raw_data = 28 + copy("Q")
    for _ in range(copy("I")):  # objects
        text()
        if copy("I") != 0xFFFFFFFF:  # a raw data index
            assert copy("I") == 0x02, "an int16 file"
            copy("I"), copy("Q")
        for _ in range(copy("I")):  # properties
            text()
            kind = copy("I")
            if kind == 0x20:
                text()
            elif kind == 0x44:  # a timestamp's halves swap places
                fractions, seconds = struct.unpack("<Qq", src.read(16))
                out.extend(struct.pack(">qQ", seconds, fractions))
            else:
                copy(_PROPERTY_FORMATS[kind])
    out.extend(src.read(raw_data - src.tell()))
    out.extend(np.frombuffer(src.read(), "<i2").astype(">i2").tobytes())
    return bytes(out)


class _FakeTDMSFile(io.BytesIO):
    """A BytesIO object with the minimum file API TDMS utils expect."""

    name = "fake.tdms"

    def fileno(self):
        """Return a dummy file descriptor for monkeypatched mmap."""
        return 0


class TestReadWork:
    """A read does not fetch the TDMS header twice."""

    @pytest.mark.parametrize(
        "selection",
        [
            {"source_patch_key": "missing"},
            {"samples": True, "time": (10**9, 10**9 + 1)},
        ],
    )
    def test_rejected_selection_does_not_materialize(self, selection):
        """Metadata rejection works even when local materialization is disabled."""

        class NoLocalCopy(IOResourceManager):
            def get_resource(self, required_type):
                if required_type is LocalBinaryReader:
                    raise RemoteCacheError("Local materialization is disabled")
                return super().get_resource(required_type)

        path = fetch("sample_tdms_file_v4713.tdms")
        with path.open("rb") as resource:
            manager = NoLocalCopy(resource)
            assert len(TDMSFormatterV4713().read(manager, **selection)) == 0
            assert not resource.closed

    def test_header_read_volume(self):
        """Reading samples needs only the scan header plus the chunk-size fields."""

        class Counter(io.BufferedReader):
            bytes_read = 0

            def read(self, size=-1):
                out = super().read(size)
                self.bytes_read += len(out)
                return out

        reader = TDMSFormatterV4713()
        path = fetch("sample_tdms_file_v4713.tdms")
        with Counter(path.open("rb")) as resource:
            metadata = reader.get_metadata(resource)[0]
            scan_bytes = resource.bytes_read
            resource.seek(0)
            resource.bytes_read = 0
            patch = reader.read(resource)[0]
            assert resource.bytes_read <= scan_bytes + 8
            assert not resource.closed
        assert patch.shape == metadata.shape
        assert patch.dtype == metadata.dtype


class TestStatedVersion:
    """Files stating another TDMS version are read where the layout allows."""

    version = ((8, struct.pack("<i", 4712)),)

    def test_read(self, tmp_path):
        """The parts the reader uses are the same, so the file reads."""
        path = _edited(tmp_path / "v4712.tdms", self.version)
        assert dc.get_format(path) == ("TDMS", "4713")
        patch, expected = dc.read(path)[0], dc.read(fetch(IDAS))[0]
        assert patch.coords == expected.coords
        np.testing.assert_array_equal(patch.data, expected.data)

    def test_header_failure_names_version(self, tmp_path):
        """A header the reader cannot parse names the version and the part."""
        path = _edited(tmp_path / "cut.tdms", self.version, size=200)
        match = "header of a TDMS file stating version 4712.*follows version 4713"
        with pytest.raises(InvalidFiberFileError, match=match):
            dc.read(path)

    def test_data_failure_names_version(self, tmp_path):
        """Data the reader cannot lay out names the version and the part."""
        with open(fetch(IDAS), "rb") as fi:
            tdms_utils._get_all_attrs(fi)
            # the chunk size follows the array dimension
            chunk_size = fi.tell() + 4
        edits = [*self.version, (chunk_size, struct.pack("<i", 0))]
        path = _edited(tmp_path / "chunk.tdms", edits)
        match = "data of a TDMS file stating version 4712.*ZeroDivisionError"
        with pytest.raises(InvalidFiberFileError, match=match):
            dc.read(path)


class TestByteOrder:
    """Every field is read in the byte order the ToC states."""

    def test_big_endian(self, tmp_path):
        """A big-endian file reads as the little-endian one it was made from."""
        path = tmp_path / "big_endian.tdms"
        path.write_bytes(_big_endian(Path(fetch(IDAS)).read_bytes()))
        assert dc.get_format(path) == ("TDMS", "4713")
        patch, expected = dc.read(path)[0], dc.read(fetch(IDAS))[0]
        assert patch.coords == expected.coords == dc.scan_payloads(path)[0].coords
        assert patch.dtype == expected.dtype == np.dtype("int16")
        np.testing.assert_array_equal(patch.data, expected.data)
        ids = {"origin_id": "", "data_id": ""}
        assert patch.attrs.update(**ids) == expected.attrs.update(**ids)


class TestDistanceGrid:
    """The channel count sets the distance coordinate's length."""

    @staticmethod
    def _value(name):
        """The offset of a property's value in the example file."""
        name = name.encode()
        # the name is followed by its four byte type
        return Path(fetch(IDAS)).read_bytes().index(name) + len(name) + 4

    @staticmethod
    def _renamed(name):
        """An edit hiding a property, its name changed in place."""
        return name.encode(), name.replace("[m]", "[x]").encode()

    @pytest.mark.parametrize("measure_length", [1000, 2000])
    def test_channel_count_wins(self, tmp_path, measure_length):
        """A measure length stating another channel count warns."""
        value = struct.pack("<I", measure_length)
        edits = [(self._value("MeasureLength[m]"), value)]
        path = _edited(tmp_path / "grid.tdms", edits)
        with pytest.warns(UserWarning, match=f"disagree.*gives {measure_length}"):
            patch = dc.read(path)[0]
        expected = dc.read(fetch(IDAS))[0]
        assert patch.get_coord("distance") == expected.get_coord("distance")
        np.testing.assert_array_equal(patch.data, expected.data)

    def test_unstated_start_and_length(self, tmp_path):
        """Without them the channels start at 0, and nothing disagrees."""
        raw = Path(fetch(IDAS)).read_bytes()
        for name in ("StartPosition[m]", "MeasureLength[m]"):
            raw = raw.replace(*self._renamed(name))
        path = tmp_path / "unstated.tdms"
        path.write_bytes(raw)
        distance = dc.read(path)[0].get_coord("distance")
        expected = dc.read(fetch(IDAS))[0].get_coord("distance")
        assert distance.min() == 0 and len(distance) == len(expected)
        assert distance.step == expected.step

    def test_unstated_spacing(self, tmp_path):
        """Without the spacing there is no coordinate to build."""
        raw = Path(fetch(IDAS)).read_bytes()
        path = tmp_path / "no_spacing.tdms"
        path.write_bytes(raw.replace(*self._renamed("SpatialResolution[m]")))
        match = "header of a TDMS file stating version 4713.*no SpatialResolution"
        with pytest.raises(InvalidFiberFileError, match=match):
            dc.read(path)


class TestTDMSUtils:
    """Tests for TDMS utility functions."""

    def test_type_not_supported(self):
        """Test that type_not_supported raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="Reading of this tdsDataType is not implemented"
        ):
            type_not_supported("any_input")

    def test_parse_time_stamp_none(self):
        """Test that parse_time_stamp returns None for invalid input."""
        # Test with invalid timestamp values that should return None
        result = parse_time_stamp(0, 0)  # epoch start should return None
        assert result is None

        # Test with None values
        result = parse_time_stamp(None, 100)
        assert result is None

        result = parse_time_stamp(100, None)
        assert result is None

        result = parse_time_stamp(None, None)
        assert result is None

    def test_parse_time_stamp_valid(self):
        """Test that parse_time_stamp works with valid input."""
        # Test with valid timestamp - using a reasonable epoch timestamp
        # LabVIEW epoch starts at 1904-01-01, so we need a positive value

        # Use a timestamp that represents a valid date after 1904
        seconds = 365 * 24 * 3600 * 100  # 100 years after 1904
        fractions = 0.5  # Some fractional seconds

        result = parse_time_stamp(fractions, seconds)
        assert isinstance(result, datetime.datetime)
        assert result.year >= 1904

    def test_get_all_attrs_unsupported_data_type(self, monkeypatch):
        """Unsupported TDMS channel types should raise clearly."""
        lead_in = struct.pack("<4siiQQ", b"TDSm", 0, 4713, 0, 0)
        payload = b"".join(
            [
                struct.pack("<i", 3),  # object count -> 1 channel after adjustment
                struct.pack("<i", 0),  # object path len
                struct.pack("<i", 0),  # raw data index len
                struct.pack("<i", 0),  # property count
                struct.pack("<i", 0),  # group info len
                b"\x00" * 8,  # skipped group info bytes
                struct.pack("<i", 0),  # first channel path len
                struct.pack("<i", 0),  # index len
                struct.pack("<i", 0x21),  # bool -> unsupported
            ]
        )
        fake = _FakeTDMSFile(lead_in + payload)
        monkeypatch.setattr(
            tdms_utils,
            "get_buffer_size",
            lambda _: len(lead_in + payload),
        )
        monkeypatch.setattr(
            tdms_utils,
            "_get_distance_coord",
            lambda *_: tdms_utils.get_coord(start=0, stop=1, step=1, units="m"),
        )
        with pytest.raises(Exception, match="Unsupported TDMS data type"):
            tdms_utils._get_all_attrs(fake)

    def test_get_data_decimated_multi_segment(self, monkeypatch):
        """Decimated multi-segment data should use the append/update path."""
        fileinfo = {
            "decimated": True,
            "chunk_size": 2,
            "data_type": "float32",
            "file_size": 52,
            "raw_data_offset": 0,
            "n_channels": 1,
            "next_segment_offset": 12,
            "order": "<",
        }
        attrs = {"tag": "example"}
        data = bytearray(52)
        data[0:12] = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
        data[24:40] = struct.pack("<qq", 12, 0)
        data[40:52] = np.array([4.0, 5.0, 6.0], dtype=np.float32).tobytes()
        fake = _FakeTDMSFile(bytes(data))
        monkeypatch.setattr(tdms_utils, "_get_fileinfo", lambda _: (fileinfo, attrs))
        monkeypatch.setattr(tdms_utils.mmap, "mmap", lambda *args, **kwargs: data)
        out_data, channel_length, out_attrs = tdms_utils._get_data(fake)
        # Three samples on one channel from each of the two segments.
        assert out_data.shape == (6, 1)
        assert channel_length == 6
        assert out_attrs == attrs

    def test_get_data_decimated_whole_chunks(self, monkeypatch):
        """A decimated segment whose chunks are all full still has its data."""
        fileinfo = {
            "decimated": True,
            "chunk_size": 2,
            "data_type": "float32",
            "file_size": 8,
            "raw_data_offset": 0,
            "n_channels": 1,
            "next_segment_offset": 8,
            "order": "<",
        }
        attrs = {"tag": "example"}
        data = np.array([1.0, 2.0], dtype=np.float32).tobytes()
        fake = _FakeTDMSFile(data)
        monkeypatch.setattr(tdms_utils, "_get_fileinfo", lambda _: (fileinfo, attrs))
        monkeypatch.setattr(tdms_utils.mmap, "mmap", lambda *args, **kwargs: data)
        out_data, channel_length, _ = tdms_utils._get_data(fake)
        assert out_data.shape == (2, 1)
        assert channel_length == 2


class TestSampleOrder:
    """Raw chunks must become chronological samples, including windowed reads."""

    @pytest.mark.parametrize("shape", [(6, 1), (6, 2), (7, 2)])
    @pytest.mark.parametrize("decimated", [True, False])
    @pytest.mark.parametrize("bounds", [(0, None), (1, 5)])
    def test_chunk_order(self, tmp_path, shape, decimated, bounds):
        """Full chunks and a partial tail retain time and channel identities."""
        expected = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        chunk_size = 2
        if decimated:
            # Storage writes each channel's samples in one chunk before
            # moving to the next channel, then starts the next time chunk.
            payload = b"".join(
                expected[start : start + chunk_size, channel].tobytes()
                for start in range(0, shape[0], chunk_size)
                for channel in range(shape[1])
            )
        else:
            payload = expected.tobytes()
        path = tmp_path / "samples.raw"
        path.write_bytes(payload)
        # A finite declared end describes a closed segment. Its partial
        # chunk, if any, has the same remaining sample count per channel.
        fileinfo = {
            "decimated": decimated,
            "chunk_size": chunk_size,
            "data_type": "float32",
            "n_channels": shape[1],
            "raw_data_offset": 0,
            "next_segment_offset": len(payload),
            "file_size": len(payload),
            "order": "<",
        }
        start, stop = bounds
        with path.open("rb") as resource:
            actual = tdms_utils._read_sample_range(resource, fileinfo, start, stop)
        np.testing.assert_array_equal(actual, expected[start:stop])
        assert actual.dtype == expected.dtype


class TestMultiSegment:
    """A TDMS file can hold several segments, each a stretch of time."""

    @pytest.fixture(scope="class")
    def two_segment_path(self, tmp_path_factory):
        """The example file's segment, written twice.

        Appending segments is how a TDMS file grows, and this example
        declares its segment length exactly, so the second copy reads as a
        second segment. sample_tdms_file_v4713.tdms cannot stand in for it:
        its lead-in claims 146 MB for a 1 MB file, so anything after it is
        read as part of the first segment.
        """
        raw = Path(fetch("iDAS005_tdms_example.626.tdms")).read_bytes()
        path = tmp_path_factory.mktemp("tdms_multi_segment") / "two_segment.tdms"
        path.write_bytes(raw + raw)
        return path

    def test_segments_stack_along_time(self, two_segment_path):
        """Both segments are read, and they stack along time, not distance."""
        with open(two_segment_path, "rb") as fi:
            data, channel_length, _ = tdms_utils._get_data(fi)
        assert data.shape == (2000, 1152)
        assert channel_length == 2000
        # One segment written twice, so the halves are the same samples.
        assert np.array_equal(data[:1000], data[1000:])

    def test_read(self, two_segment_path):
        """A multi-segment file reads as one patch holding every segment."""
        patch = dc.spool(two_segment_path)[0]
        single = dc.spool(fetch("iDAS005_tdms_example.626.tdms"))[0]
        assert patch.shape == (2000, 1152)
        assert np.array_equal(patch.data[:1000], single.data)
        assert len(patch.get_coord("time")) == 2000

    def test_unclosed_last_segment_runs_to_the_end_of_the_file(self):
        """All ones for a segment's length means the rest of the file.

        TDMS writes that when a file was never closed -- the recording was
        cut short -- and the offsets are unsigned, so reading them as signed
        would make the last segment a backwards one.
        """
        fileinfo = {
            "raw_data_offset": 28,
            "next_segment_offset": 40,
            "file_size": 200,
            "n_channels": 1,
            "data_type": "float32",
            "order": "<",
        }
        buffer = bytearray(200)
        buffer[52:68] = struct.pack("<QQ", 0xFFFFFFFFFFFFFFFF, 0)
        fake = _FakeTDMSFile(bytes(buffer))
        bounds = list(tdms_utils._iter_segment_bounds(fake, fileinfo))
        assert bounds == [(28, 40), (68, 200)]
        assert tdms_utils._get_sample_count(fake, fileinfo) == 3 + 33

    def test_scan_agrees_with_read(self, two_segment_path):
        """Counting bytes to the end of the file would over-count the time.

        A scan never reads the data, so nothing else would catch it saying
        the file covers more time than it holds.
        """
        summary = dc.scan(two_segment_path)[0]
        time = dc.spool(two_segment_path)[0].get_coord("time")
        assert summary.coords["time"].len == 2000
        assert summary.coords["time"].min == time.min()
        assert summary.coords["time"].max == time.max()


class TestReadArray:
    """Tests for reading only the segments a window touches."""

    @pytest.fixture(scope="class")
    def two_segment_path(self, tmp_path_factory):
        """The example file's segment written twice (see TestMultiSegment)."""
        raw = Path(fetch("iDAS005_tdms_example.626.tdms")).read_bytes()
        path = tmp_path_factory.mktemp("tdms_read_array") / "two_segment.tdms"
        path.write_bytes(raw + raw)
        return path

    def test_matches_default_across_segments(self, two_segment_path):
        """A window spanning the segment boundary matches the default."""
        io = TDMSFormatterV4713()
        out = io.read_array(two_segment_path, ((990, 1010), (5, 9)))
        expected = (
            io.read(two_segment_path, source_patch_key="")[0]
            .select(samples=True, time=(990, 1010), distance=(5, 9))
            .data
        )
        assert out.dtype == expected.dtype
        assert np.array_equal(out, expected)
        assert out.shape == (20, 4)

    def test_only_touched_segments_are_decoded(self, two_segment_path, monkeypatch):
        """A window inside the second segment decodes that segment alone."""
        from dascore.io.tdms import utils as tdms_utils  # noqa: PLC0415

        decoded = []
        original = tdms_utils._get_segment_data

        def spy(fileinfo, nch, dmap, nso, rdo):
            decoded.append(rdo)
            return original(fileinfo, nch, dmap, nso, rdo)

        single = dc.spool(fetch("iDAS005_tdms_example.626.tdms"))[0]
        with open(two_segment_path, "rb") as fi:
            fileinfo, _ = tdms_utils._get_fileinfo(fi)
            first, second = (
                rdo for rdo, _ in tdms_utils._iter_segment_bounds(fi, fileinfo)
            )
        monkeypatch.setattr(tdms_utils, "_get_segment_data", spy)
        io = TDMSFormatterV4713()
        # windows and the segment each should decode; the boundary is 1000
        cases = {
            (1200, 1300): [second],
            (100, 200): [first],
            (0, 1000): [first],
            (1000, 1100): [second],
            (990, 1010): [first, second],
        }
        for (start, stop), expected in cases.items():
            decoded.clear()
            out = io.read_array(two_segment_path, ((start, stop),))
            assert decoded == expected, (start, stop)
            # both segments hold the example's samples
            rows = np.concatenate([single.data, single.data])[start:stop]
            assert np.array_equal(out, rows)

    def test_empty_window(self, two_segment_path):
        """A window past the end is empty with the right width and dtype."""
        io = TDMSFormatterV4713()
        out = io.read_array(two_segment_path, ((5000, 6000),))
        assert out.shape == (0, 1152)
        assert out.dtype == io.read(two_segment_path, source_patch_key="")[0].dtype


class TestTDMSInterrogator:
    """A TDMS file names the interrogator by its host name."""

    @pytest.fixture(
        scope="class",
        params=["sample_tdms_file_v4713.tdms", "iDAS005_tdms_example.626.tdms"],
    )
    def tdms_path(self, request):
        """Paths to each TDMS test file."""
        return fetch(request.param)

    @pytest.fixture(scope="class")
    def tdms_attrs(self, tdms_path):
        """Attrs read back from a TDMS file."""
        return dict(dc.spool(tdms_path)[0].attrs)

    @pytest.fixture(scope="class")
    def raw_host_name(self, tdms_path):
        """The HostName property as the TDMS header states it."""
        with open(tdms_path, "rb") as fi:
            header, _ = tdms_utils._get_all_attrs(fi)
        return header["SystemInfomation.OS.HostName"]

    def test_flat_inventory_attr(self, tdms_path, monkeypatch):
        """A canonical property already named in the header survives filtering."""
        read_attr = tdms_utils._read_attr

        def canonical_attr(*args):
            name, value = read_attr(*args)
            if name == "GaugeLength":
                name = "gauge_length"
            return name, value

        monkeypatch.setattr(tdms_utils, "_read_attr", canonical_attr)
        patch = dc.read(tdms_path)[0]
        metadata = dc.scan_payloads(tdms_path)[0]
        summary = dc.scan(tdms_path)[0]
        assert patch.attrs.gauge_length == 10.0
        assert metadata.attrs.gauge_length == summary.attrs.gauge_length == 10.0

    def test_canonical_scan_and_read(self, tdms_path, raw_host_name):
        """Scan and read omit raw vendor fields while retaining canonical facts."""
        patch = dc.read(tdms_path)[0]
        metadata = dc.scan_payloads(tdms_path)[0]
        summary = dc.scan(tdms_path)[0]
        attrs = dict(patch.attrs)
        allowed = set(dc.PatchAttrs.model_fields) | set(INVENTORY_ATTRS)
        assert set(attrs) <= allowed
        assert attrs == dict(metadata.attrs) == dict(summary.attrs)
        assert attrs["interrogator.name"] == raw_host_name
        assert attrs["data_type"] == "strain_rate"
        assert metadata.coords == patch.coords
        assert metadata.dtype == patch.dtype
        selected = dc.read(tdms_path, samples=True, time=(1, 4), distance=(2, 5))[0]
        np.testing.assert_array_equal(selected.data, patch.data[1:4, 2:5])
        # A trim names a window of the file, so its data_id alone moves.
        assert selected.attrs.data_id != patch.attrs.data_id
        assert selected.attrs.update(data_id=patch.attrs.data_id) == patch.attrs
        assert patch._source == metadata._source
        # The selection keeps the same provenance and narrows the windows.
        assert selected._source.detach() == patch._source.detach()
        assert selected._source.windows == ((1, 4), (2, 5))

    def test_name_is_host_name(self, tdms_attrs, raw_host_name):
        """The name is exactly the HostName property, eg "iDAS005"."""
        assert raw_host_name
        assert tdms_attrs["interrogator.name"] == raw_host_name

    def test_no_component_serial_as_interrogator(self, tdms_attrs, tdms_path):
        """No card or crate serial is passed off as the interrogator's."""
        with open(tdms_path, "rb") as fi:
            header, _ = tdms_utils._get_all_attrs(fi)
        serials = {
            v
            for k, v in header.items()
            if k.startswith("SystemInfomation.") and k.endswith(".SerialNum") and v
        }
        stated = {v for k, v in tdms_attrs.items() if k.startswith("interrogator.")}
        assert serials
        assert not (stated & serials)

    def test_no_serial_claimed(self, tdms_attrs):
        """The Devices and Chassis serials name parts, not the interrogator."""
        assert "interrogator.serial_number" not in tdms_attrs

    def test_agrees_with_silixa_h5(self, tdms_attrs, raw_host_name):
        """Both Silixa readers key the interrogator off the same attr."""
        key = "SystemInfomation.OS.HostName"
        assert _SILIXA_ATTR_MAP[key] == "interrogator.name"
        assert tdms_attrs["interrogator.name"] == raw_host_name
