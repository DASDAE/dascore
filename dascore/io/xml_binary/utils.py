"""Utilities for Binary."""
from __future__ import annotations

from functools import lru_cache

from pydantic import ConfigDict
from pydantic.alias_generators import to_pascal

from dascore.utils.models import BaseModel, DateTime64
from dascore.utils.xml import xml_to_dict

# -- Create a pydantic model for the metadata info to help keep thins organized.


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
def read_xml_metadata(path):
    """A function to read metadata from the xml file."""
    contents = xml_to_dict(path.read_bytes())
    return XMLBinaryInfo.model_validate(contents)
