"""
Utility functions for AP sensing module.
"""

import pandas as pd

import dascore as dc
from dascore.core import get_coord, get_coord_manager
from dascore.utils.misc import maybe_get_items

_ATTR_MAP = {
    "GaugeLength": "gauge_length",
    "SamplingFrequency[Hz]": "sampling_frequency",
    "Fibre Length Multiplier": "fiber_length_multiplier",
    "Start Distance (m)": "start_distance",
    "Stop Distance (m)": "stop_distance",
    "Fibre Length per Metre": "fiber_length_per_metre",
    "GPSTimeStamp": "gps_timestamp",
    "CPUTimeStamp": "cpu_timestamp",
    "Tags": "tag",
    "PulseWidth[ns]": "pulse_width",
    "MeasureLength[m]": "measured_length",
    "StartPosition[m]": "start_position",
    "SpatialResolution[m]": "spatial_resolution",
    # oops, they spelled information "infomation"
    "SystemInfomation.Devices1.SerialNum": "instrument_id",
}

_EXPECTED_ATTRS = set(_ATTR_MAP)


def _get_version_string(resource, version):
    """Return version string if silixa h5 format else False."""
    dataset = resource.get("Acoustic", {})
    attrs_names = set(getattr(dataset, "attrs", dataset))
    has_attrs = _EXPECTED_ATTRS.issubset(attrs_names)
    if dataset is None or not has_attrs:
        return False
    return version


def _read_time_string(time_str):
    """Read the timestring like dd/mm/yyyy."""
    out = pd.to_datetime(time_str.replace(" (UTC)", ""), dayfirst=True)
    return dc.to_datetime64(out)


def _get_time_coord(attr_dict, shape):
    """Create the time coordinate."""
    gps_time = _read_time_string(attr_dict["gps_timestamp"])
    cpu_time = _read_time_string(attr_dict["cpu_timestamp"])
    time_min = cpu_time if pd.isnull(gps_time) else gps_time
    sampling_rate = 1.0 / float(attr_dict["sampling_frequency"])
    step = dc.to_timedelta64(sampling_rate)
    length = shape[0]
    coord = get_coord(start=time_min, step=step, shape=(length,))
    return coord


def _get_distance_coord(attr_dict, data_shape):
    """Get distance coordinate from AP sensing data."""
    # Note: To be consistent with TDMS reader (see tdms.utils.get_distance_coord)
    # We use this method for calculating distance, although start distance
    # and stop distance are included.
    multiplier = float(attr_dict["fiber_length_multiplier"])
    total_length = float(attr_dict["measured_length"]) * multiplier
    start = float(attr_dict["start_position"]) + total_length
    step = float(attr_dict["spatial_resolution"]) * multiplier
    stop = start + data_shape[1] * step
    coord = get_coord(start=start, stop=stop, step=step, units="m")
    return coord.change_length(data_shape[1])


def _get_coords(attrs_dict, shape):
    """Get coordinates of AP_sensing file."""
    # first get time
    time_coord = _get_time_coord(attrs_dict, shape)
    distance_coord = _get_distance_coord(attrs_dict, shape)
    cm = get_coord_manager(
        {"time": time_coord, "distance": distance_coord},
        dims=("time", "distance"),
    )
    return cm


def _get_attr_dict(resource):
    """Get the attribute map."""
    ds = resource["Acoustic"]
    attrs_dict = maybe_get_items(ds.attrs, _ATTR_MAP)
    attrs_dict["coords"] = _get_coords(attrs_dict, ds.shape)
    return attrs_dict


def _get_attr(resource, attr_cls, extras=None):
    """Get the attribute class"""
    attrs = _get_attr_dict(resource)
    expected_fields = set(attr_cls.model_fields)
    attrs_sub = {i: v for i, v in attrs.items() if i in expected_fields}
    attrs_sub.update(extras if extras else {})
    attrs = attr_cls.model_validate(attrs_sub)
    return attrs


def _get_patch(resource, time=None, distance=None, attr_cls=dc.PatchAttrs):
    """Get a patch from ap_sensing file."""
    attrs = _get_attr_dict(resource)
    coords = attrs["coords"]
    data = resource["Acoustic"]
    if time is not None or distance is not None:
        coords, data = coords.select(array=data, time=time, distance=distance)
        attrs["coords"] = coords
    expected_fields = set(attr_cls.model_fields)
    attrs_sub = {i: v for i, v in attrs.items() if i in expected_fields}
    attrs = attr_cls.model_validate(attrs_sub)
    return dc.Patch(data=data[:], coords=coords, attrs=attrs)
