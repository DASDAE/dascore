"""
Utilities for reading SR-4731 OTDR SOR files.

SR-4731 is the Telcordia OTDR Data Format. This implementation currently
supports the OFL100/FIBERCLOUD subset seen in DASCore's ``ofl100_*.sor`` test
files and makes these specific assumptions:

- The file starts with a ``Map`` block and contains ``GenParams``,
  ``SupParams``, ``FxdParams``, ``DataPts``, and ``Cksum`` blocks.
- The map block version is ``200``.
- ``SupParams`` stores manufacturer, model, and serial number in the first
  three null-separated fields.
- ``FxdParams`` exposes the timestamp, distance unit, wavelength, sample
  spacing, index of refraction, and display range fields documented in
  ``_parse_fixed_params``.
- ``DataPts`` is a single unsegmented trace with unsigned little-endian
  16-bit samples and the display scale documented in ``_parse_data_points``.
- Trace samples follow pyotdr's SR-4731 display convention:
  ``(max_raw - raw) * scale / 1_000_000``.
- Distance spacing is derived from sample spacing and index of refraction.
- Key events, proprietary blocks, and checksums are not parsed or validated.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

import dascore as dc
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import CoordManager, get_coord_manager
from dascore.core.coords import BaseCoord, get_coord
from dascore.exceptions import InvalidFiberFileError
from dascore.io.core import _make_scan_payload

DIMS = ("time", "distance")
REQUIRED_BLOCKS = frozenset(
    ("Map", "GenParams", "SupParams", "FxdParams", "DataPts", "Cksum")
)
SPEED_OF_LIGHT_KM_PER_USEC = 0.299792458


@dataclass(frozen=True)
class Block:
    """A SOR block map entry."""

    name: str
    version: int
    size: int
    offset: int


class SR4731PatchAttrs(PatchAttrs):
    """Patch attributes for supported SR-4731 SOR files."""

    wavelength_nm: float = np.nan
    acquisition_range_m: float = np.nan
    sample_spacing_usec: float = np.nan
    refractive_index: float = np.nan
    trace_count: int = 0
    sample_scale: int = 0
    manufacturer: str = ""
    model: str = ""
    serial_number: str = ""


def _read_all(resource) -> bytes:
    """Read all bytes from a binary resource without depending on ``Path``."""
    resource.seek(0)
    data = resource.read()
    resource.seek(0)
    return data


def _c_string(data: bytes, offset: int = 0) -> tuple[str, int]:
    """Read a null-terminated ASCII string from bytes."""
    try:
        end = data.index(0, offset)
    except ValueError:
        msg = "SOR block string is not null terminated."
        raise InvalidFiberFileError(msg)
    text = data[offset:end].decode("ascii", "replace")
    return text, end + 1


def _unpack_from(fmt: str, data: bytes, offset: int) -> tuple[Any, ...]:
    """Unpack binary data and convert struct errors to DASCore IO errors."""
    try:
        return struct.unpack_from(fmt, data, offset)
    except struct.error as exc:
        msg = "SOR file ended before expected fields could be read."
        raise InvalidFiberFileError(msg) from exc


def _parse_blocks(data: bytes) -> dict[str, Block]:
    """Parse the SOR block map."""
    name, pos = _c_string(data, 0)
    if name != "Map":
        msg = f"not a SOR map block: first block is {name!r}"
        raise InvalidFiberFileError(msg)
    map_version = _unpack_from("<H", data, pos)[0]
    map_size = _unpack_from("<I", data, pos + 2)[0]
    block_count = _unpack_from("<H", data, pos + 6)[0]
    if block_count < 2:
        msg = "SOR map block does not contain data blocks."
        raise InvalidFiberFileError(msg)
    pos += 8

    entries: list[tuple[str, int, int]] = [("Map", map_version, map_size)]
    for _ in range(block_count - 1):
        block_name, pos = _c_string(data, pos)
        version = _unpack_from("<H", data, pos)[0]
        size = _unpack_from("<I", data, pos + 2)[0]
        pos += 6
        entries.append((block_name, version, size))

    offset = 0
    blocks: dict[str, Block] = {}
    for block_name, version, size in entries:
        if size <= 0:
            msg = f"SOR block {block_name!r} has invalid size {size}."
            raise InvalidFiberFileError(msg)
        next_offset = offset + size
        if next_offset > len(data):
            msg = f"SOR block {block_name!r} extends past end of file."
            raise InvalidFiberFileError(msg)
        blocks[block_name] = Block(block_name, version, size, offset)
        offset = next_offset
    return blocks


def _block_payload(data: bytes, block: Block) -> bytes:
    """Return payload bytes for a SOR block, excluding the block name."""
    raw = data[block.offset : block.offset + block.size]
    _, payload_start = _c_string(raw, 0)
    return raw[payload_start:]


def _parse_text_fields(payload: bytes) -> list[str]:
    """Parse null-separated ASCII fields while preserving field positions."""
    parts = payload.split(b"\0")
    if parts and parts[-1] == b"":
        parts = parts[:-1]
    return [part.decode("ascii", "replace") for part in parts]


def _parse_fixed_params(payload: bytes) -> dict[str, Any]:
    """Parse the fixed-parameter SOR block fields needed by DASCore."""
    # OFL100/FIBERCLOUD FxdParams fields used here:
    #   0: uint32 unix timestamp, 4: 2-byte distance unit,
    #   6: uint16 wavelength in tenths of nm,
    #   20: uint32 sample spacing in 1e-8 microseconds,
    #   24: uint32 sample count,
    #   28: uint32 group index of refraction in 1e-5,
    #   40: uint32 display range in 2e-5 km.
    timestamp = _unpack_from("<I", payload, 0)[0]
    unit = payload[4:6].decode("ascii", "replace")
    wavelength_nm = _unpack_from("<H", payload, 6)[0] / 10
    sample_spacing_usec = _unpack_from("<I", payload, 20)[0] * 1e-8
    n_samples = _unpack_from("<I", payload, 24)[0]
    refractive_index = _unpack_from("<I", payload, 28)[0] * 1e-5
    display_range_km = _unpack_from("<I", payload, 40)[0] * 2e-5
    if sample_spacing_usec <= 0 or refractive_index <= 0:
        msg = "SR-4731 SOR FxdParams block has invalid distance scaling fields."
        raise InvalidFiberFileError(msg)
    resolution_km = sample_spacing_usec * SPEED_OF_LIGHT_KM_PER_USEC / refractive_index
    distance_step_m = resolution_km * 1000
    acquisition_range_m = distance_step_m * n_samples
    return {
        "timestamp": timestamp,
        "distance_unit": unit,
        "wavelength_nm": wavelength_nm,
        "sample_spacing_usec": sample_spacing_usec,
        "n_samples": n_samples,
        "refractive_index": refractive_index,
        "display_range_km": display_range_km,
        "distance_step_m": distance_step_m,
        "acquisition_range_m": acquisition_range_m,
    }


def _parse_data_points(payload: bytes, load_samples: bool = True) -> dict[str, Any]:
    """Parse the DataPts block."""
    # OFL100/FIBERCLOUD DataPts fields used here:
    #   0: uint32 total points, 4: uint16 trace count,
    #   6: uint32 samples per trace, 10: uint16 sample scale.
    #   12: uint16 sample array, converted to pyotdr-compatible display dB.
    total_points = _unpack_from("<I", payload, 0)[0]
    trace_count = _unpack_from("<H", payload, 4)[0]
    n_samples = _unpack_from("<I", payload, 6)[0]
    scale = _unpack_from("<H", payload, 10)[0]
    if n_samples <= 0:
        msg = "SR-4731 SOR DataPts block contains no trace samples."
        raise InvalidFiberFileError(msg)
    if scale <= 0:
        msg = "SR-4731 SOR DataPts block has an invalid sample scale."
        raise InvalidFiberFileError(msg)
    if trace_count != 1 or total_points != n_samples:
        msg = (
            "Only single-trace, unsegmented SR-4731 SOR DataPts blocks are "
            "currently supported."
        )
        raise InvalidFiberFileError(msg)
    sample_start = 12
    sample_end = sample_start + n_samples * 2
    if sample_end > len(payload):
        msg = "SOR DataPts block ended before all trace samples were read."
        raise InvalidFiberFileError(msg)
    out: dict[str, Any] = {
        "trace_count": trace_count,
        "n_samples": n_samples,
        "scale": scale,
    }
    if load_samples:
        raw = np.frombuffer(payload[sample_start:sample_end], dtype="<u2")
        out["samples"] = (raw.max() - raw.astype(np.float64)) * scale / 1_000_000
    return out


def _validate_sample_counts(fixed: dict[str, Any], data_points: dict[str, Any]) -> None:
    """Ensure FxdParams and DataPts agree on trace sample count."""
    fixed_samples = fixed["n_samples"]
    data_samples = data_points["n_samples"]
    if fixed_samples != data_samples:
        msg = (
            "SR-4731 SOR FxdParams sample count "
            f"({fixed_samples}) does not match DataPts samples ({data_samples})."
        )
        raise InvalidFiberFileError(msg)


def _parse_sor(resource, load_samples: bool = True) -> dict[str, Any]:
    """Parse the supported SR-4731 SOR file."""
    data = _read_all(resource)
    blocks = _parse_blocks(data)
    if missing := REQUIRED_BLOCKS - set(blocks):
        msg = f"SOR file is missing required block(s): {sorted(missing)}"
        raise InvalidFiberFileError(msg)
    fixed = _parse_fixed_params(_block_payload(data, blocks["FxdParams"]))
    data_points = _parse_data_points(
        _block_payload(data, blocks["DataPts"]), load_samples=load_samples
    )
    _validate_sample_counts(fixed, data_points)
    supplier = _parse_text_fields(_block_payload(data, blocks["SupParams"]))
    general = _parse_text_fields(_block_payload(data, blocks["GenParams"]))
    return {
        "blocks": blocks,
        "fixed": fixed,
        "data_points": data_points,
        "supplier": supplier,
        "general": general,
    }


def _get_time_coord(parsed: dict[str, Any]) -> BaseCoord:
    """Create a singleton time coordinate from the fixed-params timestamp."""
    timestamp = parsed["fixed"]["timestamp"]
    time = np.asarray([timestamp], dtype="datetime64[s]").astype("datetime64[ns]")
    return get_coord(data=time)


def _get_distance_coord(parsed: dict[str, Any]) -> BaseCoord:
    """Create the distance coordinate."""
    fixed = parsed["fixed"]
    step = fixed["distance_step_m"]
    stop = fixed["acquisition_range_m"]
    # DASCore keeps coordinate units on coords; the SOR unit token is parser metadata.
    return get_coord(start=0, stop=stop, step=step, units="m")


def _get_coords(parsed: dict[str, Any]) -> CoordManager:
    """Create the coordinate manager for an SR-4731 trace."""
    coords = {
        "time": _get_time_coord(parsed),
        "distance": _get_distance_coord(parsed),
    }
    return get_coord_manager(coords=coords, dims=DIMS)


def _get_attr_dict(parsed: dict[str, Any], extras: dict | None = None) -> dict:
    """Create DASCore patch attrs from parsed SR-4731 metadata."""
    supplier = parsed["supplier"]
    manufacturer, model, serial_number = [*supplier, "", "", ""][:3]
    instrument_id = "-".join(x for x in (manufacturer, model, serial_number) if x)
    data_points = parsed["data_points"]
    out = {
        "data_type": "otdr",
        "data_units": "dB",
        "instrument_id": instrument_id,
        "wavelength_nm": parsed["fixed"]["wavelength_nm"],
        "acquisition_range_m": parsed["fixed"]["acquisition_range_m"],
        "sample_spacing_usec": parsed["fixed"]["sample_spacing_usec"],
        "refractive_index": parsed["fixed"]["refractive_index"],
        "trace_count": data_points["trace_count"],
        "sample_scale": data_points["scale"],
        "manufacturer": manufacturer,
        "model": model,
        "serial_number": serial_number,
    }
    out.update(extras or {})
    return out


def _get_patch_attrs(
    resource,
    attr_class: type[PatchAttrs] = SR4731PatchAttrs,
    extras: dict | None = None,
) -> dict:
    """Return patch attrs for an SR-4731 SOR file."""
    parsed = _parse_sor(resource, load_samples=False)
    coords = _get_coords(parsed)
    attrs = _get_attr_dict(parsed, extras)
    return _make_scan_payload(
        attrs=attr_class(**attrs),
        coords=coords,
        dims=coords.dims,
        shape=coords.shape,
        dtype="float64",
    )


def _get_patches(
    resource,
    attr_class: type[PatchAttrs] = SR4731PatchAttrs,
    extras: dict | None = None,
    time=None,
    distance=None,
):
    """Read an SR-4731 SOR file and return patches."""
    parsed = _parse_sor(resource, load_samples=True)
    cm = _get_coords(parsed)
    attrs = _get_attr_dict(parsed, extras)
    data = parsed["data_points"]["samples"][np.newaxis, :]
    cm, data = cm.select(data, time=time, distance=distance)
    if not data.size:
        return []
    patch = dc.Patch(data=data, coords=cm, attrs=attr_class(**attrs))
    return [patch]


def _get_format(resource, name: str, version: str) -> tuple[str, str] | bool:
    """Return SR-4731 format/version if the resource is supported."""
    try:
        parsed = _parse_sor(resource, load_samples=False)
    except (InvalidFiberFileError, OSError, ValueError, TypeError):
        return False
    blocks = parsed["blocks"]
    if blocks["Map"].version == int(version):
        return name, version
    return False
