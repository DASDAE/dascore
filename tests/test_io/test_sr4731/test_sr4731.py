"""
SR-4731 OTDR SOR specific tests.
"""

from __future__ import annotations

import struct
from io import BytesIO

import numpy as np
import pytest
from numpy.testing import assert_allclose

import dascore as dc
from dascore.io.sr4731 import SR4731V200
from dascore.io.sr4731.utils import (
    _get_format,
    _parse_blocks,
    _parse_data_points,
    _parse_fixed_params,
    _parse_sor,
    _parse_text_fields,
    _unpack_from,
)
from dascore.utils.downloader import fetch

OTDR_NAMES = ("ofl100_1.sor", "ofl100_2.sor", "ofl100_3.sor")
SPEED_OF_LIGHT_KM_PER_USEC = 0.299792458


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


def _make_block(name, payload):
    """Create an SOR block with a null-terminated block name."""
    return name.encode() + b"\0" + payload


def _make_sor_data(fixed_n_samples=3, data_n_samples=3):
    """Create a minimal supported SOR byte stream."""
    fixed = bytearray(44)
    fixed[4:6] = b"km"
    fixed[6:8] = np.uint16(15500).tobytes()
    fixed[20:24] = np.uint32(125003).tobytes()
    fixed[24:28] = np.uint32(fixed_n_samples).tobytes()
    fixed[28:32] = np.uint32(146832).tobytes()
    fixed[40:44] = np.uint32(204805).tobytes()

    data_points = bytearray(12 + data_n_samples * 2)
    data_points[0:4] = np.uint32(data_n_samples).tobytes()
    data_points[4:6] = np.uint16(1).tobytes()
    data_points[6:10] = np.uint32(data_n_samples).tobytes()
    data_points[10:12] = np.uint16(1000).tobytes()
    data_points[12:] = np.arange(data_n_samples, dtype="<u2").tobytes()

    blocks = [
        _make_block("GenParams", b"gen\0"),
        _make_block("SupParams", b"FIBERCLOUD\0" + b"OFL100\0" + b"0901001\0"),
        _make_block("FxdParams", bytes(fixed)),
        _make_block("DataPts", bytes(data_points)),
        _make_block("Cksum", b"\0\0"),
    ]
    entries = [(block.split(b"\0", 1)[0].decode(), 200, len(block)) for block in blocks]
    entry_bytes = b"".join(
        name.encode() + b"\0" + np.uint16(version).tobytes() + np.uint32(size).tobytes()
        for name, version, size in entries
    )
    map_size = len(b"Map\0") + 2 + 4 + 2 + len(entry_bytes)
    map_block = (
        b"Map\0"
        + np.uint16(200).tobytes()
        + np.uint32(map_size).tobytes()
        + np.uint16(len(blocks) + 1).tobytes()
        + entry_bytes
    )
    return map_block + b"".join(blocks)


def _read_c_string(data: bytes, offset: int = 0) -> tuple[str, int]:
    """Read a null-terminated ASCII string for independent fixture checks."""
    end = data.index(0, offset)
    return data[offset:end].decode("ascii", "replace"), end + 1


def _get_block_payload(path, block_name):
    """Return a named block payload from a SOR fixture."""
    data = path.read_bytes()
    block = _parse_blocks(data)[block_name]
    raw = data[block.offset : block.offset + block.size]
    _, payload_start = _read_c_string(raw)
    return raw[payload_start:]


def _expected_fixed_values(payload):
    """Decode fixed params independently of the production helper."""
    sample_spacing_usec = struct.unpack_from("<I", payload, 20)[0] * 1e-8
    n_samples = struct.unpack_from("<I", payload, 24)[0]
    refractive_index = struct.unpack_from("<I", payload, 28)[0] * 1e-5
    resolution_m = (
        sample_spacing_usec * SPEED_OF_LIGHT_KM_PER_USEC / refractive_index * 1000
    )
    return {
        "timestamp": struct.unpack_from("<I", payload, 0)[0],
        "wavelength_nm": struct.unpack_from("<H", payload, 6)[0] / 10,
        "sample_spacing_usec": sample_spacing_usec,
        "n_samples": n_samples,
        "refractive_index": refractive_index,
        "acquisition_range_m": resolution_m * n_samples,
        "distance_step": resolution_m,
    }


def _expected_data_values(payload):
    """Decode DataPts samples independently of the production helper."""
    n_samples = struct.unpack_from("<I", payload, 6)[0]
    scale = struct.unpack_from("<H", payload, 10)[0]
    raw = np.frombuffer(payload[12 : 12 + n_samples * 2], dtype="<u2")
    data = (raw.max() - raw.astype(np.float64)) * scale / 1_000_000
    return {"n_samples": n_samples, "scale": scale, "data": data}


def _expected_supplier_values(payload):
    """Decode supplier fields independently of the production helper."""
    parts = payload.split(b"\0")
    if parts and parts[-1] == b"":
        parts = parts[:-1]
    return [part.decode("ascii", "replace") for part in parts]


class TestSR4731:
    """Tests for SR-4731 SOR support."""

    parser = SR4731V200()

    @pytest.fixture(params=OTDR_NAMES)
    def sor_path(self, request):
        """Return an SR-4731 SOR test path."""
        return fetch(request.param)

    @pytest.fixture()
    def sor_patch(self, sor_path):
        """Return the parsed SR-4731 patch."""
        return self.parser.read(sor_path)[0]

    def test_get_format(self, sor_path):
        """Ensure the SOR file is identified."""
        assert self.parser.get_format(sor_path) == (
            self.parser.name,
            self.parser.version,
        )

    def test_scan(self, sor_path):
        """Scan returns expected SR-4731 metadata."""
        fixed = _expected_fixed_values(_get_block_payload(sor_path, "FxdParams"))
        data_points = _expected_data_values(_get_block_payload(sor_path, "DataPts"))
        supplier = _expected_supplier_values(_get_block_payload(sor_path, "SupParams"))
        manufacturer, model, serial_number = [*supplier, "", "", ""][:3]
        instrument_id = "-".join(x for x in (manufacturer, model, serial_number) if x)
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
        assert attr.instrument_id == instrument_id
        assert attr.wavelength_nm == fixed["wavelength_nm"]
        assert attr.acquisition_range_m == pytest.approx(fixed["acquisition_range_m"])
        assert attr.sample_spacing_usec == pytest.approx(fixed["sample_spacing_usec"])
        assert attr.refractive_index == pytest.approx(fixed["refractive_index"])
        assert attr.trace_count == 1
        assert attr.sample_scale == data_points["scale"]

    def test_read(self, sor_path, sor_patch):
        """Read returns one singleton-time OTDR patch."""
        data_points = _expected_data_values(_get_block_payload(sor_path, "DataPts"))
        assert isinstance(sor_patch, dc.Patch)
        assert sor_patch.shape == (1, data_points["n_samples"])
        assert sor_patch.dims == ("time", "distance")
        assert sor_patch.attrs.data_type == "otdr"
        assert sor_patch.attrs.data_units == dc.get_quantity("dB")

    def test_time_coord(self, sor_path, sor_patch):
        """The SOR acquisition timestamp is used as singleton time coordinate."""
        fixed = _expected_fixed_values(_get_block_payload(sor_path, "FxdParams"))
        time = sor_patch.get_coord("time")
        expected_time = np.datetime64(fixed["timestamp"], "s").astype("datetime64[ns]")
        assert time.min() == expected_time
        assert time.max() == time.min()
        assert time.step is None

    def test_distance_coord(self, sor_path, sor_patch):
        """Distance coordinate is based on sample spacing and refractive index."""
        fixed = _expected_fixed_values(_get_block_payload(sor_path, "FxdParams"))
        data_points = _expected_data_values(_get_block_payload(sor_path, "DataPts"))
        distance = sor_patch.get_coord("distance")
        assert_allclose(distance.min(), 0.0)
        assert_allclose(distance.step, fixed["distance_step"])
        assert distance.units == dc.get_quantity("m")
        assert len(distance) == data_points["n_samples"]

    def test_sample_values_match_pyotdr_display_convention(self, sor_path):
        """Sample values match pyotdr's display dB convention."""
        sor_patch = self.parser.read(sor_path)[0]
        data_points = _expected_data_values(_get_block_payload(sor_path, "DataPts"))
        assert_allclose(sor_patch.data[0], data_points["data"])

    def test_golden_values_ofl100_1(self):
        """Pin known values for ofl100_1.sor to catch wrong scaling constants.

        Unlike the other tests, these expected values are hard-coded rather than
        re-derived from the file, so a wrong offset or scale factor in both the
        parser and the test helpers cannot hide.
        """
        patch = self.parser.read(fetch("ofl100_1.sor"))[0]
        attrs = patch.attrs
        distance = patch.get_coord("distance")
        assert patch.shape == (1, 16384)
        assert attrs.wavelength_nm == 1550.0
        assert attrs.refractive_index == pytest.approx(1.46832)
        assert attrs.sample_spacing_usec == pytest.approx(0.00125003)
        assert attrs.acquisition_range_m == pytest.approx(4181.579556111035)
        assert attrs.trace_count == 1
        assert attrs.sample_scale == 1000
        assert attrs.instrument_id == "FIBERCLOUD-FC4000-0901001"
        assert distance.step == pytest.approx(0.2552233615790427)
        assert patch.get_coord("time").min() == np.datetime64("2026-06-12T10:58:14")
        assert_allclose(patch.data[0, :5], [9.064, 10.146, 11.439, 11.98, 12.539])
        assert patch.data.min() == 0.0
        assert patch.data.max() == pytest.approx(20.304)

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
        assert not _get_format(BytesIO(data), self.parser.name, self.parser.version)


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
        assert not _get_format(BytesIO(b"Map"), "SR4731", "200")

    def test_fixed_and_data_sample_counts_must_match(self):
        """FxdParams and DataPts must agree on trace length."""
        data = _make_sor_data(fixed_n_samples=4, data_n_samples=3)
        with pytest.raises(dc.exceptions.InvalidFiberFileError, match="does not match"):
            _parse_sor(BytesIO(data), load_samples=False)

    def test_parse_fixed_params(self):
        """FxdParams fields decode to the documented physical values."""
        payload = bytearray(44)
        payload[4:6] = b"km"
        payload[6:8] = np.uint16(15500).tobytes()
        payload[20:24] = np.uint32(125003).tobytes()
        payload[24:28] = np.uint32(16384).tobytes()
        payload[28:32] = np.uint32(146832).tobytes()
        payload[40:44] = np.uint32(204805).tobytes()
        out = _parse_fixed_params(bytes(payload))
        assert out == {
            "timestamp": 0,
            "distance_unit": "km",
            "wavelength_nm": 1550.0,
            "sample_spacing_usec": pytest.approx(0.00125003),
            "n_samples": 16384,
            "refractive_index": pytest.approx(1.46832),
            "display_range_km": pytest.approx(4.0961),
            "distance_step_m": pytest.approx(0.2552233615795309),
            "acquisition_range_m": pytest.approx(4181.579556111035),
        }

    def test_invalid_fixed_distance_scale_raises(self):
        """Fixed params require sample spacing and refractive index."""
        payload = bytearray(44)
        payload[4:6] = b"km"
        with pytest.raises(dc.exceptions.InvalidFiberFileError):
            _parse_fixed_params(bytes(payload))

    def test_data_points_use_pyotdr_display_convention(self):
        """Raw samples are zero-referenced against the maximum sample."""
        payload = bytearray(18)
        payload[0:4] = np.uint32(3).tobytes()
        payload[4:6] = np.uint16(1).tobytes()
        payload[6:10] = np.uint32(3).tobytes()
        payload[10:12] = np.uint16(1000).tobytes()
        payload[12:18] = np.array([100, 50, 0], dtype="<u2").tobytes()
        out = _parse_data_points(bytes(payload))
        assert_allclose(out["samples"], [0.0, 0.05, 0.1])

    @pytest.mark.parametrize("n_samples, scale", [(0, 1000), (10, 0)])
    def test_invalid_data_points_raise(self, n_samples, scale):
        """Invalid sample counts and scales should fail with DASCore IO errors."""
        payload = bytearray(12 + n_samples * 2)
        payload[0:4] = np.uint32(n_samples).tobytes()
        payload[4:6] = np.uint16(1).tobytes()
        payload[6:10] = np.uint32(n_samples).tobytes()
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
