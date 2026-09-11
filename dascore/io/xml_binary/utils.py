"""Utilities for Binary."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_pascal

import dascore as dc
from dascore.core import get_coord, get_coord_manager
from dascore.core.source import PatchSource
from dascore.io.utils import step_from_rate
from dascore.models import DateTime64
from dascore.utils.misc import iterate
from dascore.utils.remote_io import ensure_local_file
from dascore.utils.xml import xml_to_dict

# -- Create a pydantic model for the metadata info to help keep thins organized.

DATE_TIME_PATTERN = r"\b\d{8}T\d{6}_d{6}Z\b"
STANDARD_DIMS = ("time", "distance")


class _XMLModel(BaseModel):
    """Base model which converts camel case to snake."""

    model_config = ConfigDict(
        alias_generator=to_pascal,
    )


class XMLLaserZones(_XMLModel):
    """Zones in the xml header."""

    start_channel: int
    end_channel: int
    stride: int
    number_of_channels: int


class XMLBinaryInfo(_XMLModel):
    """Base level of information about XML index file."""

    file_format: str
    date_time: DateTime64
    das_interrogator_serial: dict[str, str]
    gauge_length_m: float
    pulse_width_ns: float
    data_type: str
    number_of_lasers: int
    i_t_u_channels: dict[str, int]
    original_temporal_sampling_rate: float
    output_temporal_sampling_rate: float
    original_spatial_sampling_interval: float
    units: str
    zones: dict[str, XMLLaserZones]
    number_of_channels: int
    number_of_frames: int
    use_relative_strain: bool
    transposed_data: bool


@lru_cache
def _read_xml_metadata(path):
    """A function to read metadata from the xml file."""
    contents = xml_to_dict(ensure_local_file(path).read_bytes())
    return XMLBinaryInfo.model_validate(contents)


def _make_distance_coord(metadata: XMLBinaryInfo):
    """
    Make the base coordinates from the metadata.
    """
    zones = metadata.zones
    zone = next(iter(zones.values()))
    dx = metadata.original_spatial_sampling_interval
    distance = get_coord(
        start=(zone.start_channel - 1) * dx,
        stop=zone.end_channel * dx,
        step=dx,
        units=metadata.units,
    )
    return distance


def _make_time_coord(file_start_times, metadata: XMLBinaryInfo):
    """Create time coord for each file."""
    dt = step_from_rate(metadata.output_temporal_sampling_rate)
    nt = metadata.number_of_frames
    for start in file_start_times:
        yield get_coord(start=start, step=dt, shape=(nt,), units="s")


def _make_base_attrs_dict(metadata: XMLBinaryInfo):
    """
    Make the base attributes and coordinates from metadata.
    """
    zones = metadata.zones
    assert len(zones) == 1, "expecting single zone per metadata file."
    zone_name = next(iter(zones))

    ius = metadata.das_interrogator_serial
    assert len(ius) == 1, "expecting one interrogator."
    iu_name = next(iter(ius.values()))
    attrs = {
        # The metadata states nanoseconds and meters; attrs use seconds.
        "pulse_width": metadata.pulse_width_ns * 1e-9,
        "gauge_length": metadata.gauge_length_m,
        "interrogator.serial_number": iu_name,
        "zone_name": zone_name,
    }
    return attrs


def _get_path_datetime_64(paths):
    """Get a series of datetime64 and path as index."""
    ser = pd.Series(x.name for x in paths)
    split = ser.str.split("_", expand=True)
    year = split[1].str[:4]
    month = split[1].str[4:6]
    day = split[1].str[6:8]
    hour = split[1].str[9:11]
    minute = split[1].str[11:13]
    second = split[1].str[13:15]
    frac_sec = split[2].str.split("Z", expand=True)[0]
    iso8601 = (
        year
        + "-"
        + month
        + "-"
        + day
        + "T"
        + hour
        + ":"
        + minute
        + ":"
        + second
        + "."
        + frac_sec
    )
    dt = dc.to_datetime64(iso8601.values)
    return pd.Series(dt, index=ser.values)


def _paths_to_scan_patches(
    paths: Sequence[Path],
    metadata,
    attr_cls=dc.PatchAttrs,
    extra_attrs=None,
) -> list[dc.Patch]:
    """Convert paths to patch summaries for scan/index workflows."""
    extra_attrs = {} if not extra_attrs else extra_attrs
    paths = list(iterate(paths))
    if not paths:
        return []
    base_attrs = _make_base_attrs_dict(metadata)
    distance_coord = _make_distance_coord(metadata)
    dt_ser = _get_path_datetime_64(paths)
    dims = STANDARD_DIMS[::-1] if metadata.transposed_data else STANDARD_DIMS
    out = []
    for path, time_coord in zip(
        paths, _make_time_coord(dt_ser.values, metadata), strict=True
    ):
        coords = get_coord_manager(
            {"time": time_coord, "distance": distance_coord},
            dims=dims,
        )
        attrs = attr_cls(**base_attrs, **extra_attrs)
        out.append(
            dc.Patch(
                attrs=attrs,
                coords=coords,
                dtype=metadata.data_type,
                source=PatchSource(path=str(path)),
            )
        )
    return out
