"""Utilities for Binary."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ConfigDict
from pydantic.alias_generators import to_pascal

import dascore as dc
from dascore.core import get_coord, get_coord_manager
from dascore.utils.misc import iterate
from dascore.utils.models import BaseModel, DateTime64
from dascore.utils.pd import _model_list_to_df, adjust_segments, filter_df
from dascore.utils.time import to_float
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
    contents = xml_to_dict(path.read_bytes())
    return XMLBinaryInfo.model_validate(contents)


def _make_distance_coord(metadata: XMLLaserZones):
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


def _make_time_coord(file_start_times, metadata: XMLLaserZones):
    """Create time coord for each file."""
    dt = dc.to_timedelta64(1.0 / metadata.output_temporal_sampling_rate)
    nt = metadata.number_of_frames
    for start in file_start_times:
        yield get_coord(start=start, stop=start + dt * nt, step=dt, units="s")


def _make_base_attrs_dict(metadata: XMLLaserZones):
    """
    Make the base attributes and coordinates from metadata.
    """
    zones = metadata.zones
    assert len(zones) == 1, "expecting single zone per metadata file."
    zone_name = next(iter(zones))

    ius = metadata.das_interrogator_serial
    assert len(ius) == 1, "expecting one interrogator."
    iu_name = next(iter(ius.values()))
    attrs = dict(
        pulse_width_ns=metadata.pulse_width_ns,
        gauge_length=metadata.gauge_length_m,
        instrument_id=iu_name,
        zone_name=zone_name,
    )
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


def _paths_to_df(paths, metadata, attr_cls):
    """Convert paths to dataframe of info."""
    attrs = _paths_to_attrs(paths=paths, metadata=metadata, attr_cls=attr_cls)
    return _model_list_to_df(attrs)


def _read_single_file(path, metadata, time, distance):
    """Read a single file into a patch."""
    assert not metadata.transposed_data, "Cant handle data transposition yet."
    attr = _paths_to_attrs(path, metadata, summarize=False)[0]
    time_coord = attr.coords["time"].to_coord()
    distance_coord = attr.coords["distance"].to_coord()
    cm = get_coord_manager(
        {"time": time_coord, "distance": distance_coord},
        dims=attr.dim_tuple,
    )
    memmap = np.memmap(path, dtype=metadata.data_type)
    size = np.prod(cm.shape)
    assert memmap.size == size, f"wrong data shape for {path}"
    data = memmap.reshape(cm.shape)
    patch = dc.Patch(
        data=data,
        dims=attr.dim_tuple,
        coords=cm,
        attrs=attr.update(coord=None),
    )
    return patch.select(time=time, distance=distance)


def _load_patches(paths, metadata, time, distance, attr_cls):
    """Load the data file or file into a patch."""
    # Fast case for single file.
    if isinstance(paths, Path) and paths.is_file():
        return _read_single_file(paths, metadata, time, distance)
    # Since there could be **MANY** files, we have to create a mini-index
    # here to determine which files to read. Under normal circumstances
    # this isn't required as a spool can manage it.
    df = _paths_to_df(paths=paths, metadata=metadata, attr_cls=attr_cls)
    df["path"] = paths
    filtered_df = df[filter_df(df, time=time, distance=distance)].pipe(
        adjust_segments, time=time, distance=distance
    )
    # Then we can just recurse into this function.
    out = [
        _load_patches(
            Path(ser["path"]),
            metadata=metadata,
            time=time,
            distance=distance,
            attr_cls=attr_cls,
        )
        for _, ser in filtered_df.iterrows()
    ]
    return out


def _paths_to_attrs(
    paths: Path | Sequence[Path],
    metadata,
    summarize=True,
    attr_cls=dc.PatchAttrs,
    extra_attrs=None,
    timestamp=None,
):
    """Convert a path to a Patch attribute."""
    extra_attrs = {} if not extra_attrs else extra_attrs
    paths = iterate(paths)
    # filter based on timestamp
    if timestamp is not None:
        ts = to_float(timestamp)
        paths = list(x for x in paths if Path(x).stat().st_mtime >= ts)
    # No data to index.
    if not len(paths):
        return []
    base_attrs = _make_base_attrs_dict(metadata)
    distance_coord = _make_distance_coord(metadata)
    dt_ser = _get_path_datetime_64(paths)
    dims = STANDARD_DIMS[::-1] if metadata.transposed_data else STANDARD_DIMS
    out = []
    for path, time_coord in zip(paths, _make_time_coord(dt_ser.values, metadata)):
        cm = get_coord_manager(
            {"time": time_coord, "distance": distance_coord},
            dims=dims,
        )
        if summarize:
            cm = cm.to_summary_dict()
        attr_dict = {**base_attrs, **extra_attrs}
        attrs = attr_cls(coords=cm, path=path, **attr_dict)
        out.append(attrs)
    return out
