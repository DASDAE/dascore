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
    _get_attr_dict,
    _get_coords,
    _get_format,
    _get_time_coord,
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


def _make_sor_data(
    fixed_n_samples=3,
    data_n_samples=3,
    averaging_raw=50,
    timestamp=0,
    manufacturer=b"FIBERCLOUD",
):
    """Create a minimal supported SOR byte stream."""
    fixed = bytearray(44)
    fixed[0:4] = np.uint32(timestamp).tobytes()
    fixed[4:6] = b"km"
    fixed[6:8] = np.uint16(15500).tobytes()
    fixed[16:18] = np.uint16(1).tobytes()
    fixed[18:20] = np.uint16(50).tobytes()
    fixed[20:24] = np.uint32(125003).tobytes()
    fixed[24:28] = np.uint32(fixed_n_samples).tobytes()
    fixed[28:32] = np.uint32(146832).tobytes()
    fixed[34:38] = np.uint32(5).tobytes()
    fixed[38:40] = np.uint16(averaging_raw).tobytes()
    fixed[40:44] = np.uint32(204805).tobytes()

    data_points = bytearray(12 + data_n_samples * 2)
    data_points[0:4] = np.uint32(data_n_samples).tobytes()
    data_points[4:6] = np.uint16(1).tobytes()
    data_points[6:10] = np.uint32(data_n_samples).tobytes()
    data_points[10:12] = np.uint16(1000).tobytes()
    data_points[12:] = np.arange(data_n_samples, dtype="<u2").tobytes()

    blocks = [
        _make_block("GenParams", b"gen\0"),
        _make_block("SupParams", manufacturer + b"\0OFL100\0" + b"0901001\0"),
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
        "pulse_width": struct.unpack_from("<H", payload, 18)[0] * 1e-9,
        "n_averages": struct.unpack_from("<I", payload, 34)[0],
        "averaging_time_raw": struct.unpack_from("<H", payload, 38)[0],
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

    def test_scan(self, sor_path):
        """Scan returns expected SR-4731 metadata."""
        fixed = _expected_fixed_values(_get_block_payload(sor_path, "FxdParams"))
        data_points = _expected_data_values(_get_block_payload(sor_path, "DataPts"))
        supplier = _expected_supplier_values(_get_block_payload(sor_path, "SupParams"))
        manufacturer, model, serial_number = [*supplier, "", "", ""][:3]
        payloads = self.parser.scan(sor_path)
        assert len(payloads) == 1
        payload = payloads[0]
        attr = payload["attrs"]
        assert isinstance(attr, dc.PatchAttrs)
        assert "path" not in attr.model_dump()
        assert "file_format" not in attr.model_dump()
        assert "file_version" not in attr.model_dump()
        assert payload["dims"] == ("time", "distance")
        assert attr.data_type == "otdr"
        assert attr.data_units == dc.get_quantity("dB")
        assert attr.get("interrogator.manufacturer") == manufacturer
        assert attr.get("interrogator.model") == model
        assert attr.get("interrogator.serial_number") == serial_number
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
        """The timestamp is the sample and the averaging time its extent."""
        fixed = _expected_fixed_values(_get_block_payload(sor_path, "FxdParams"))
        time = sor_patch.get_coord("time")
        expected_time = np.datetime64(fixed["timestamp"], "s").astype("datetime64[ns]")
        assert time.min() == expected_time
        assert time.max() == time.min()
        assert len(time) == 1
        expected_step = np.timedelta64(fixed["averaging_time_raw"] * 10**8, "ns")
        assert time.step == expected_step

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
        assert attrs.pulse_width == pytest.approx(5e-8)
        assert attrs.n_averages == 5
        assert attrs.averaging_time == pytest.approx(5.0)
        assert attrs.refractive_index == pytest.approx(1.46832)
        assert attrs.sample_spacing_usec == pytest.approx(0.00125003)
        assert attrs.acquisition_range_m == pytest.approx(4181.579556111035)
        assert attrs.trace_count == 1
        assert attrs.sample_scale == 1000
        assert attrs.get("interrogator.manufacturer") == "FIBERCLOUD"
        assert attrs.get("interrogator.model") == "FC4000"
        assert attrs.get("interrogator.serial_number") == "0901001"
        assert distance.step == pytest.approx(0.2552233615790427)
        assert patch.get_coord("time").min() == np.datetime64("2026-06-12T10:58:14")
        assert patch.get_coord("time").step == np.timedelta64(5, "s")
        assert_allclose(patch.data[0, :5], [9.064, 10.146, 11.439, 11.98, 12.539])
        assert patch.data.min() == 0.0
        assert patch.data.max() == pytest.approx(20.304)

    def test_no_averaging_time_keeps_a_bare_instant(self):
        """A file stating no averaging time reads as it always did.

        The step says how much fiber time the sample stands for, and a
        file which does not say must not have one invented for it.
        """
        data = _make_sor_data(averaging_raw=0)
        parsed = _parse_sor(BytesIO(data), load_samples=False)
        time = _get_time_coord(parsed)
        assert time.step is None
        assert len(time) == 1

    def test_unlisted_supplier_states_no_averaging_time(self):
        """A vendor scale this reader cannot read is never made a duration.

        The same half minute is written 300 by the standard, 3000 by
        Noyes and 30 by EXFO, so an unrecognised supplier keeps its bare
        instant and reports no averaging time, rather than an axis and
        an attr which may each be wrong by a factor of ten.
        """
        data = _make_sor_data(manufacturer=b"NOYES")
        parsed = _parse_sor(BytesIO(data), load_samples=False)
        assert _get_time_coord(parsed).step is None
        assert _get_attr_dict(parsed)["averaging_time"] is None
        # the number the file states is read, it just names no duration
        assert parsed["fixed"]["averaging_time_raw"] == 50

    def test_separated_traces_report_a_gap(self):
        """Two traces an hour apart are two measurements, not one run.

        This is what the averaging time buys: without a step the
        continuity tolerance has no sample to scale, so the pair reads
        as one unbroken recording and `get_coverage` calls the hour
        between them covered.
        """
        first = _parse_sor(BytesIO(_make_sor_data(timestamp=1_000_000)))
        later = _parse_sor(BytesIO(_make_sor_data(timestamp=1_003_600)))
        patches = [
            dc.Patch(
                data=np.zeros((1, 3)),
                coords=_get_coords(parsed),
                dims=("time", "distance"),
            )
            for parsed in (first, later)
        ]
        spool = dc.spool(patches)
        assert len(spool.get_gaps("time")) == 1
        coverage = spool.get_coverage("time")["coverage"].iloc[0]
        assert coverage < 0.01

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
        payload[16:18] = np.uint16(1).tobytes()
        payload[18:20] = np.uint16(50).tobytes()
        payload[20:24] = np.uint32(125003).tobytes()
        payload[24:28] = np.uint32(16384).tobytes()
        payload[28:32] = np.uint32(146832).tobytes()
        payload[34:38] = np.uint32(5).tobytes()
        payload[38:40] = np.uint16(50).tobytes()
        payload[40:44] = np.uint32(204805).tobytes()
        out = _parse_fixed_params(bytes(payload))
        assert out == {
            "timestamp": 0,
            "distance_unit": "km",
            "wavelength_nm": 1550.0,
            "pulse_width": pytest.approx(5e-8),
            "n_averages": 5,
            "averaging_time_raw": 50,
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
        payload[16:18] = np.uint16(1).tobytes()
        with pytest.raises(dc.exceptions.InvalidFiberFileError):
            _parse_fixed_params(bytes(payload))

    def test_several_pulse_width_entries_raise(self):
        """A second pulse-width entry shifts every field after it.

        The reader trusts fixed offsets past the pulse width, so a file
        which declares another entry must be refused rather than read
        into a silently wrong distance axis.
        """
        payload = bytearray(44)
        payload[4:6] = b"km"
        payload[16:18] = np.uint16(2).tobytes()
        payload[20:24] = np.uint32(125003).tobytes()
        payload[28:32] = np.uint32(146832).tobytes()
        with pytest.raises(dc.exceptions.InvalidFiberFileError, match="pulse-width"):
            _parse_fixed_params(bytes(payload))

    @pytest.mark.parametrize("entries", [0, 3])
    def test_pulse_width_count_must_be_one(self, entries):
        """Any count but one shifts the fields the reader takes on faith."""
        payload = bytearray(44)
        payload[4:6] = b"km"
        payload[16:18] = np.uint16(entries).tobytes()
        payload[20:24] = np.uint32(125003).tobytes()
        payload[28:32] = np.uint32(146832).tobytes()
        with pytest.raises(dc.exceptions.InvalidFiberFileError, match="pulse-width"):
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
