"""Utility functions for the Aragon Photonics HDAS format."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.core import get_coord, get_coord_manager
from dascore.exceptions import InvalidFiberFileError
from dascore.utils.misc import unbyte

# 0-based positions in the 200-value HDAS header, per the vendor reader
# script distributed with the data (H5_Reader.m, 1-based there).
_HEADER_LENGTH_POS = 0  # holds the header length itself (200)
_SPATIAL_SAMPLING_POS = 1  # meters between measured points
_TRIGGER_FREQ_POS = 6  # raw trigger frequency, Hz
_DOWNSAMPLE_A_POS = 15  # first downsampling divisor
_DOWNSAMPLE_B_POS = 98  # second downsampling divisor
_POSITION_OFFSET_POS = 11  # fiber position where processing starts, m
_EPOCH_START_POS = 100  # start time, seconds since 1970-01-01 UTC

# Dataset names for the two container variants (also used by core.py).
V1_HEADER_NAME = "File_Header"
V1_DATA_NAME = "HDAS_DATA"
V2_HEADER_NAME = "hdas_header"
V2_DATA_NAME = "data"


def _get_header(resource, name):
    """Return the flattened 200-value header array or None."""
    header = resource.get(name)
    if getattr(header, "size", None) != 200:
        return None
    return np.asarray(header[()], dtype=np.float64).ravel()


def _is_hdas(resource, header_name, header_shape, data_name, required_attr) -> bool:
    """Return True if resource has an HDAS header and 2d data."""
    if getattr(resource.get(header_name), "shape", None) != header_shape:
        return False
    header = _get_header(resource, header_name)
    if header is None or header[_HEADER_LENGTH_POS] != 200:
        return False
    dataset = resource.get(data_name)
    if len(getattr(dataset, "shape", None) or ()) != 2:
        return False
    return required_attr is None or required_attr in dataset.attrs


def _decode_header(header):
    """Decode the acquisition metadata from the header array."""
    if header is None:
        msg = "HDAS header is missing or does not hold 200 values."
        raise InvalidFiberFileError(msg)
    trigger = header[_TRIGGER_FREQ_POS]
    divisor_a = header[_DOWNSAMPLE_A_POS]
    divisor_b = header[_DOWNSAMPLE_B_POS]
    spatial_step = float(header[_SPATIAL_SAMPLING_POS])
    # Each factor must be finite and positive on its own; e.g. two negative
    # divisors would otherwise cancel into a bogus positive rate.
    fields = (trigger, divisor_a, divisor_b, spatial_step)
    if not all(np.isfinite(x) and x > 0 for x in fields):
        msg = (
            "HDAS header decodes to non-positive or non-finite acquisition "
            "fields; the file is likely truncated or corrupt."
        )
        raise InvalidFiberFileError(msg)
    rate = trigger / divisor_a / divisor_b
    return {
        "spatial_step": spatial_step,
        "sampling_rate": float(rate),
        "position_offset": float(header[_POSITION_OFFSET_POS]),
        "epoch_start": float(header[_EPOCH_START_POS]),
    }


def _get_coords(header_dict, time_start, n_time, n_distance, time_first):
    """Build the coordinate manager for an HDAS file."""
    time = get_coord(
        start=time_start,
        step=dc.to_timedelta64(1 / header_dict["sampling_rate"]),
        shape=(n_time,),
    )
    distance = get_coord(
        start=header_dict["position_offset"],
        step=header_dict["spatial_step"],
        shape=(n_distance,),
        units="m",
    )
    coords = {"time": time, "distance": distance}
    dims = ("time", "distance") if time_first else ("distance", "time")
    return get_coord_manager(coords, dims=dims)


def _get_v1_coords_and_attrs(resource):
    """Get coords and attrs for a V1 (File_Header/HDAS_DATA) file."""
    header = _decode_header(_get_header(resource, V1_HEADER_NAME))
    n_time, n_distance = resource[V1_DATA_NAME].shape
    time_start = dc.to_datetime64(header["epoch_start"])
    coords = _get_coords(header, time_start, n_time, n_distance, time_first=True)
    # HDAS_DATA holds strain rate per the vendor reader script.
    attrs = {
        "data_category": "DAS",
        "data_type": "strain_rate",
        "data_units": "strain/s",
    }
    return coords, attrs


def _get_v2_coords_and_attrs(resource):
    """Get coords and attrs for a V2 (hdas_header/data) file."""
    header = _decode_header(_get_header(resource, V2_HEADER_NAME))
    dataset = resource[V2_DATA_NAME]
    n_distance, n_time = dataset.shape
    # The float32 header cannot hold an exact epoch; the start_time attr
    # (naive UTC ISO string) is authoritative for this variant. to_datetime64
    # also handles a numeric epoch-seconds attr correctly.
    time_start = dc.to_datetime64(unbyte(dataset.attrs["start_time"]))
    coords = _get_coords(header, time_start, n_time, n_distance, time_first=False)
    # This variant does not record the measurand, so units stay unset.
    attrs = {"data_category": "DAS"}
    return coords, attrs


def _get_patches(resource, data_name, coords, attrs_dict, time=None, distance=None):
    """Read patches from an HDAS file, optionally trimming coords."""
    data = resource[data_name]
    attrs = dc.PatchAttrs.model_validate(attrs_dict)
    if time is not None or distance is not None:
        coords, data = coords.select(array=data, time=time, distance=distance)
        if not data.size:
            return []
    return [dc.Patch(data=data[:], coords=coords, attrs=attrs)]
