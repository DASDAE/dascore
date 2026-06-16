"""
SR-4731 OTDR SOR specific tests.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from numpy.testing import assert_allclose

import dascore as dc
from dascore.io.sr4731 import SR4731V200
from dascore.io.sr4731.utils import (
    _get_distance_coord,
    _parse_blocks,
    _parse_data_points,
    _parse_fixed_params,
    _parse_sor,
    _parse_text_fields,
    _unpack_from,
    get_format,
)
from dascore.utils.downloader import fetch

OTDR_NAME = "ofl100_2.sor"


def _make_map_data(*entries, map_size=32, version=200, extra_size=256):
    """Create bytes with a SOR map block for parser tests."""
    block_count = len(entries) + 1
    data = bytearray(b"Map\0")
    data.extend(np.uint16(version).tobytes())
    data.extend(np.uint32(map_size).tobytes())
    data.extend(np.uint16(block_count).tobytes())
    for name, entry_version, size in entries:
        data.extend(name.encode())
        data.extend(b"\0")
        data.extend(np.uint16(entry_version).tobytes())
        data.extend(np.uint32(size).tobytes())
    target_size = sum(size for _, _, size in entries) + map_size + extra_size
    data.extend(b"\0" * max(0, target_size - len(data)))
    return bytes(data)


class TestSR4731:
    """Tests for SR-4731 SOR support."""

    parser = SR4731V200()

    @pytest.fixture(scope="class")
    @staticmethod
    def sor_path():
        """Return the SR-4731 SOR test path."""
        return fetch(OTDR_NAME)

    @pytest.fixture(scope="class")
    @staticmethod
    def sor_patch(sor_path):
        """Return the parsed SR-4731 patch."""
        return TestSR4731.parser.read(sor_path)[0]

    def test_get_format(self, sor_path):
        """Ensure the SOR file is identified."""
        assert self.parser.get_format(sor_path) == (
            self.parser.name,
            self.parser.version,
        )

    def test_scan(self, sor_path):
        """Scan returns expected SR-4731 metadata."""
        attrs = self.parser.scan(sor_path)
        assert len(attrs) == 1
        attr = attrs[0]
        assert isinstance(attr, dc.PatchAttrs)
        assert str(attr.path) == str(sor_path)
        assert attr.file_format == self.parser.name
        assert attr.file_version == self.parser.version
        assert attr.dim_tuple == ("time", "distance")
        assert attr.data_type == "otdr"
        assert attr.data_units == dc.get_quantity("dB")
        assert attr.instrument_id == "FIBERCLOUD-FC4000-0901001"
        assert attr.wavelength_nm == 1550.0
        assert attr.acquisition_range_m == 204805
        assert attr.trace_count == 1
        assert attr.sample_scale == 1000

    def test_read(self, sor_patch):
        """Read returns one singleton-time OTDR patch."""
        assert isinstance(sor_patch, dc.Patch)
        assert sor_patch.shape == (1, 16384)
        assert sor_patch.dims == ("time", "distance")
        assert sor_patch.attrs.data_type == "otdr"
        assert sor_patch.attrs.data_units == dc.get_quantity("dB")

    def test_time_coord(self, sor_patch):
        """The SOR acquisition timestamp is used as singleton time coordinate."""
        time = sor_patch.get_coord("time")
        assert time.min() == np.datetime64("2026-06-12T10:56:08")
        assert time.max() == np.datetime64("2026-06-12T10:56:08")
        assert time.step is None

    def test_distance_coord(self, sor_patch):
        """Distance coordinate is based on acquisition range and sample count."""
        distance = sor_patch.get_coord("distance")
        assert_allclose(distance.min(), 0.0)
        assert_allclose(distance.step, 204805 / 16384)
        assert len(distance) == 16384

    def test_sample_values(self, sor_patch):
        """Sample values match the reference parser."""
        assert_allclose(
            sor_patch.data[0, :10],
            [
                56.609,
                55.368,
                54.409,
                53.804,
                53.086,
                52.742,
                52.333,
                52.085,
                51.795,
                51.674,
            ],
        )
        assert_allclose(
            sor_patch.data[0, -10:],
            [
                61.310,
                65.535,
                60.764,
                65.535,
                62.525,
                65.535,
                65.535,
                65.535,
                65.535,
                65.535,
            ],
        )

    def test_select(self, sor_path, sor_patch):
        """Partial distance reads reduce coords and data consistently."""
        distance = sor_patch.get_coord("distance")
        out = self.parser.read(
            sor_path,
            distance=(
                distance.min() + 5 * distance.step,
                distance.min() + 10 * distance.step,
            ),
        )[0]
        assert out.shape == (1, 6)
        assert_allclose(out.get_coord("distance").min(), distance.values[5])
        assert_allclose(out.get_coord("distance").max(), distance.values[10])

    def test_out_of_range_selects_empty_spool(self, sor_path, sor_patch):
        """Out-of-range selectors return an empty spool."""
        time = sor_patch.get_coord("time")
        distance = sor_patch.get_coord("distance")
        assert not len(
            self.parser.read(sor_path, time=(time.max() + np.timedelta64(1, "s"), ...))
        )
        assert not len(self.parser.read(sor_path, distance=(distance.max() + 1, ...)))

    def test_read_stream(self, sor_path, sor_patch):
        """BytesIO streams can be read."""
        bio = BytesIO(sor_path.read_bytes())
        out = self.parser.read(bio)[0]
        assert out.update_attrs(path=sor_patch.attrs.path).equals(sor_patch)

    def test_get_format_false_for_version_mismatch(self, sor_path):
        """A valid SOR with the wrong map version should not be claimed."""
        data = bytearray(sor_path.read_bytes())
        map_version_offset = len(b"Map\0")
        data[map_version_offset : map_version_offset + 2] = np.uint16(201).tobytes()
        assert not get_format(BytesIO(data), self.parser.name, self.parser.version)


class TestSR4731Utils:
    """Tests for SR-4731 parser details."""

    def test_text_fields_keep_empty_positions(self):
        """Empty text fields should not shift positional metadata."""
        payload = b"FIBERCLOUD\x00\x000901001\x00\x00"
        assert _parse_text_fields(payload) == ["FIBERCLOUD", "", "0901001", ""]

    def test_unpack_truncated_field_raises(self):
        """Truncated binary fields should raise DASCore IO errors."""
        with pytest.raises(dc.exceptions.InvalidFiberFileError):
            _unpack_from("<I", b"\x00", 0)

    def test_missing_null_terminator_raises(self):
        """SOR block names must be null terminated."""
        with pytest.raises(dc.exceptions.InvalidFiberFileError):
            _parse_blocks(b"Map")

    def test_first_block_must_be_map(self):
        """SOR files must start with a Map block."""
        data = b"Bad\0" + np.uint16(200).tobytes() + np.uint32(12).tobytes()
        data += np.uint16(1).tobytes()
        with pytest.raises(dc.exceptions.InvalidFiberFileError):
            _parse_blocks(data)

    def test_block_count_must_include_data_blocks(self):
        """A SOR map with only the Map entry is invalid."""
        data = _make_map_data(map_size=32)
        with pytest.raises(dc.exceptions.InvalidFiberFileError):
            _parse_blocks(data)

    def test_block_size_must_be_positive(self):
        """SOR block sizes must be positive."""
        data = _make_map_data(("GenParams", 200, 1), map_size=0)
        with pytest.raises(dc.exceptions.InvalidFiberFileError):
            _parse_blocks(data)

    def test_block_must_fit_in_file(self):
        """SOR blocks cannot extend past the end of the file."""
        data = b"Map\0" + np.uint16(200).tobytes() + np.uint32(999).tobytes()
        data += np.uint16(2).tobytes()
        data += b"GenParams\0" + np.uint16(200).tobytes() + np.uint32(10).tobytes()
        with pytest.raises(dc.exceptions.InvalidFiberFileError):
            _parse_blocks(data)

    def test_missing_required_block_raises(self):
        """SOR files must include all required blocks."""
        data = _make_map_data(("GenParams", 200, 16), map_size=32)
        with pytest.raises(dc.exceptions.InvalidFiberFileError):
            _parse_sor(BytesIO(data), load_samples=False)

    def test_get_format_false_for_invalid_sor(self):
        """Invalid bytes should not be claimed as SR-4731."""
        assert not get_format(BytesIO(b"Map"), "SR4731", "200")

    def test_fixed_params_has_no_datetime_utc_field(self):
        """The fixed params parser should not depend on Python 3.11 datetime.UTC."""
        payload = bytearray(44)
        payload[4:6] = b"km"
        payload[6:8] = np.uint16(15500).tobytes()
        payload[40:44] = np.uint32(204805).tobytes()
        out = _parse_fixed_params(bytes(payload))
        assert out == {
            "timestamp": 0,
            "distance_unit": "km",
            "wavelength_nm": 1550.0,
            "acquisition_range_m": 204805,
        }

    @pytest.mark.parametrize("trace_points, scale", [(0, 1000), (10, 0)])
    def test_invalid_data_points_raise(self, trace_points, scale):
        """Invalid point counts and scales should fail with DASCore IO errors."""
        payload = bytearray(12 + trace_points * 2)
        payload[0:4] = np.uint32(trace_points).tobytes()
        payload[4:6] = np.uint16(1).tobytes()
        payload[6:10] = np.uint32(trace_points).tobytes()
        payload[10:12] = np.uint16(scale).tobytes()
        with pytest.raises(dc.exceptions.InvalidFiberFileError):
            _parse_data_points(bytes(payload))

    def test_segmented_data_points_raise(self):
        """Only single-trace unsegmented DataPts blocks are supported."""
        payload = bytearray(14)
        payload[0:4] = np.uint32(2).tobytes()
        payload[4:6] = np.uint16(2).tobytes()
        payload[6:10] = np.uint32(1).tobytes()
        payload[10:12] = np.uint16(1000).tobytes()
        with pytest.raises(dc.exceptions.InvalidFiberFileError):
            _parse_data_points(bytes(payload))

    def test_truncated_data_points_raise(self):
        """A DataPts block must include all declared samples."""
        payload = bytearray(12)
        payload[0:4] = np.uint32(1).tobytes()
        payload[4:6] = np.uint16(1).tobytes()
        payload[6:10] = np.uint32(1).tobytes()
        payload[10:12] = np.uint16(1000).tobytes()
        with pytest.raises(dc.exceptions.InvalidFiberFileError):
            _parse_data_points(bytes(payload))

    def test_distance_coord_rejects_zero_points(self):
        """Distance coord creation guards against invalid point counts."""
        parsed = {
            "fixed": {"acquisition_range_m": 10},
            "data_points": {"trace_points": 0},
        }
        with pytest.raises(dc.exceptions.InvalidFiberFileError):
            _get_distance_coord(parsed)
