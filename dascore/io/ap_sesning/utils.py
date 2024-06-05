"""
Utility functions for AP sensing module.
"""

import dascore as dc
from dascore.core import get_coord, get_coord_manager
from dascore.utils.misc import _maybe_unpack, unbyte


def _get_version_string(resource):
    """Determine if the file is AP sensing and what version."""
    attrs = {"AppVersion", "FileVersion", "OpticalChannelNumber"}
    groups = {"DAQ", "Interrogator", "Metadata", "ProcessingServer"}
    attrs_names = set(resource.attrs)
    group_names = set(resource)
    # This file doesn't have expected attrs and groups.
    if not attrs.issubset(attrs_names) and not groups.issubset(group_names):
        return False
    file_version = resource.attrs["FileVersion"]
    return str(_maybe_unpack(file_version))


def _get_time_coord(resource, shape):
    """Create the time coordinate."""
    # Load needed data from file and perform sanity checks.
    sr = _maybe_unpack(resource["ProcessingServer"]["DataRate"])
    start_time_str = unbyte(_maybe_unpack(resource["Metadata"]["Timestamp"]))
    time_zone = unbyte(_maybe_unpack(resource["Metadata"]["Timezone"]))
    assert time_zone in {"UTC", "UTC+0"}, "can't handle non-UTC timezones."
    trace_count = _maybe_unpack(resource["Metadata"]["TraceCount"])
    assert trace_count == shape[0], "trace count doesn't match dim shape"
    # Create coord
    start = dc.to_datetime64(start_time_str)
    step = dc.to_timedelta64(1 / sr)
    stop = start + trace_count * step
    coord = get_coord(start=start, stop=stop, step=step, units="s")
    return coord


def _get_distance_coord(resource, data_shape):
    """Get distance coordinate from AP sensing data."""
    start = resource["DAQ"]["PositionStart"]
    stop = resource["DAQ"]["PositionEnd"]
    start_unit = unbyte(_maybe_unpack(start.attrs.get("Unit", "m")))
    stop_unit = unbyte(_maybe_unpack(stop.attrs.get("Unit", "m")))
    assert start_unit == stop_unit

    # Oddly, SpatialSampling in ProcessServer group doesn't quite work to get
    # the right shape, so we either have to recalculate the spatial step, or
    # slightly change the end of the cable. Here we recalculate the sampling
    # rate, which is off from the stated rate by ~0.08% in ap_sensing_1 file.
    # step = _maybe_unpack(resource['ProcessingServer']['SpatialSampling'])
    dist_length = data_shape[1]
    x_start = _maybe_unpack(start)
    # x_end_calc = x_start + step * dist_length
    x_end = _maybe_unpack(stop)
    step_calc = (x_end - x_start) / dist_length
    coord = get_coord(start=x_start, stop=x_end, step=step_calc, units=start_unit)
    coord = coord.change_length(dist_length)
    return coord


def _get_coords(resource):
    """Get coordinates of AP_sensing file."""
    data_shape = resource["DAS"].shape
    # first get time
    time_coord = _get_time_coord(resource, data_shape)
    distance_coord = _get_distance_coord(resource, data_shape)
    cm = get_coord_manager(
        {"time": time_coord, "distance": distance_coord},
        dims=("time", "distance"),
    )
    return cm


def _get_attrs_dict(resource):
    """Get attributes."""
    daq = resource["DAQ"]
    pserver = resource["ProcessingServer"]
    out = dict(
        coords=_get_coords(resource),
        data_category="DAS",
        instrumet_id=unbyte(_maybe_unpack(daq["SerialNumber"])),
        gauge_length=_maybe_unpack(pserver["GaugeLength"]),
        radians_to_nano_strain=_maybe_unpack(pserver["RadiansToNanoStrain"]),
    )
    return out


def _get_patch(resource, time=None, distance=None, attr_cls=dc.PatchAttrs):
    """Get a patch from ap_sensing file."""
    attrs = _get_attrs_dict(resource)
    coords = attrs["coords"]
    data = resource["DAS"]
    if time is not None or distance is not None:
        coords, data = coords.select(array=data, time=time, distance=distance)
        attrs["coords"] = coords
    attrs = attr_cls.model_validate(attrs)
    return dc.Patch(data=data[:], coords=coords, attrs=attrs)
