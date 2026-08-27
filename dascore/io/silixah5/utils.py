"""Utility functions for Silixa HDF5 I/O."""

import numpy as np
import pandas as pd

import dascore as dc
from dascore.core import get_coord, get_coord_manager
from dascore.exceptions import InvalidFiberFileError
from dascore.io.utils import (
    build_patches,
    convert_attr_units,
    drop_blank_attrs,
    get_attr_names,
)
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
    # HostName names the unit ("iDAS20110", "Carina-P52"). The Chassis and
    # Devices<N> entries name COTS parts inside it, so their serials are not
    # the interrogator's; these files state no interrogator serial.
    "SystemInfomation.OS.HostName": "interrogator.name",
}

# A file may carry the host name but leave it empty.
_BLANKABLE_ATTRS = ("interrogator.name",)

# Read for metadata but not required to claim the format: a file which omits
# HostName is still a Silixa file. Kept out of the fingerprints below so that
# changing what is extracted cannot change which files the reader accepts.
_OPTIONAL_ATTRS = frozenset({"SystemInfomation.OS.HostName"})

# The detection fingerprints. Devices1.SerialNum is named here rather than in
# _ATTR_MAP: it is no longer read (it names a card, not the interrogator) but
# is still required to claim a file, so detection is unchanged.
_FINGERPRINT_EXTRAS = frozenset({"SystemInfomation.Devices1.SerialNum"})

# The header states these units in the key; patch attrs use seconds.
_PULSE_WIDTH_UNITS = "ns"

_EXPECTED_ATTRS = (frozenset(_ATTR_MAP) - _OPTIONAL_ATTRS) | _FINGERPRINT_EXTRAS


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
    """
    Get the distance coordinate.

    Channel i sits at ``start_distance + i * spatial_resolution *
    fiber_length_multiplier``, the same convention as the Carina variant;
    it lands on the file's Stop Distance to within the multiplier's rounding.
    """
    multiplier = float(attr_dict["fiber_length_multiplier"])
    start = float(attr_dict["start_distance"])
    step = float(attr_dict["spatial_resolution"]) * multiplier
    return get_coord(start=start, step=step, shape=(data_shape[1],), units="m")


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
    drop_blank_attrs(attrs_dict, _BLANKABLE_ATTRS)
    convert_attr_units(attrs_dict, "pulse_width", "s", from_units=_PULSE_WIDTH_UNITS)
    coords = _get_coords(attrs_dict, ds.shape)
    return attrs_dict, coords


def _validate_attrs(attrs_dict, attr_cls, extras=None):
    """Validate the subset of attrs the attr class knows about."""
    expected_fields = get_attr_names(attr_cls)
    attrs_sub = {i: v for i, v in attrs_dict.items() if i in expected_fields}
    attrs_sub.update(extras if extras else {})
    return attr_cls.model_validate(attrs_sub)


def _get_attr(resource, attr_cls, extras=None):
    """Get the attribute class and coordinates."""
    attrs, coords = _get_attr_dict(resource)
    return _validate_attrs(attrs, attr_cls, extras), coords


def _build_patches(attrs_dict, coords, data, time, distance, attr_cls):
    """Assemble patches from attrs, coords, and a data node."""
    attrs = _validate_attrs(attrs_dict, attr_cls)
    return build_patches(
        coords, data, attrs, selection={"time": time, "distance": distance}
    )


def _get_patches(resource, time=None, distance=None, attr_cls=dc.PatchAttrs):
    """Get a patch from a Silixa V1 (Acoustic) file."""
    attrs, coords = _get_attr_dict(resource)
    return _build_patches(attrs, coords, resource["Acoustic"], time, distance, attr_cls)


# --- Carina (netCDF-shell) variant helpers.
# These files (e.g. the INGV Mt Etna deployment) are written through a
# netCDF library: the Silixa attrs sit on the file root rather than on an
# "Acoustic" dataset, samples live in a "Fiber" int16 dataset of shape
# (time, channel), and a "ChannelMap" dataset places each stored column on
# the physical fiber. The netCDF coordinate variables (t, x, cm) are empty
# or zeroed, so everything derives from the root attrs.

_CARINA_DATA_NAME = "Fiber"
_CARINA_CHANNEL_MAP = "ChannelMap"

# Carina files carry the Silixa attr family minus "Tags", plus the
# decimated output rate and epoch start the time coordinate needs.
_CARINA_ATTR_MAP = {k: v for k, v in _ATTR_MAP.items() if k != "Tags"} | {
    "StartTime": "start_time_us",
    "Samplerate": "sample_rate",
}
_CARINA_EXPECTED_ATTRS = (
    frozenset(_CARINA_ATTR_MAP) - _OPTIONAL_ATTRS
) | _FINGERPRINT_EXTRAS


def _get_carina_version_string(resource, version):
    """Return version string if Carina-variant Silixa h5, else False."""
    # Check datasets first: single link lookups reject most files without
    # iterating all root attr names.
    fiber = resource.get(_CARINA_DATA_NAME)
    if len(getattr(fiber, "shape", None) or ()) != 2:
        return False
    channel_map = resource.get(_CARINA_CHANNEL_MAP)
    if getattr(channel_map, "shape", None) is None:
        return False
    if not _CARINA_EXPECTED_ATTRS.issubset(set(resource.attrs)):
        return False
    return version


def _get_carina_time_coord(attrs_dict, n_time):
    """Time from the root StartTime (µs epoch) and Samplerate attrs."""
    start_us = int(attrs_dict["start_time_us"])
    rate = float(attrs_dict["sample_rate"])
    if not (np.isfinite(rate) and rate > 0):
        msg = f"Silixa Carina file has an unusable Samplerate attr ({rate})."
        raise InvalidFiberFileError(msg)
    start = dc.to_datetime64(np.datetime64(start_us, "us"))
    return get_coord(start=start, step=dc.to_timedelta64(1 / rate), shape=(n_time,))


def _get_carina_distance_coord(attrs_dict, resource, n_columns):
    """
    Distance from Start Distance/SpatialResolution through the ChannelMap.

    ChannelMap holds, for each physical channel index, the column of
    ``Fiber`` storing it (or -1 when unmapped). Physical channel i sits at
    ``start_distance + i * spatial_resolution * fiber_length_multiplier``
    (the same multiplier convention as the V1/TDMS readers; it reproduces
    the file's Stop Distance exactly).
    """
    step = float(attrs_dict["spatial_resolution"]) * float(
        attrs_dict["fiber_length_multiplier"]
    )
    start = float(attrs_dict["start_distance"])
    map_node = resource[_CARINA_CHANNEL_MAP]
    if len(getattr(map_node, "shape", None) or ()) != 1:
        msg = "Silixa Carina ChannelMap must be a one-dimensional dataset."
        raise InvalidFiberFileError(msg)
    channel_map = map_node[()]
    physical = np.flatnonzero(channel_map >= 0)
    if not len(physical):
        msg = "Silixa Carina ChannelMap maps no channels at all."
        raise InvalidFiberFileError(msg)
    columns = channel_map[physical]
    order = np.argsort(columns)
    if not np.array_equal(columns[order], np.arange(n_columns)):
        msg = (
            "Silixa Carina ChannelMap does not bijectively map the data "
            "columns onto physical channels; refusing to guess distances."
        )
        raise InvalidFiberFileError(msg)
    physical = physical[order]
    if np.all(np.diff(physical) == 1):
        coord = get_coord(
            start=start + physical[0] * step, step=step, shape=(n_columns,), units="m"
        )
    else:
        coord = get_coord(data=start + physical * step, units="m")
    return coord


def _get_carina_attrs_and_coords(resource):
    """Get the attr dict and coordinates for a Carina-variant file."""
    attrs_dict = maybe_get_items(
        resource.attrs, _CARINA_ATTR_MAP, unpack_names=set(_CARINA_ATTR_MAP)
    )
    drop_blank_attrs(attrs_dict, _BLANKABLE_ATTRS)
    convert_attr_units(attrs_dict, "pulse_width", "s", from_units=_PULSE_WIDTH_UNITS)
    n_time, n_columns = resource[_CARINA_DATA_NAME].shape
    time_coord = _get_carina_time_coord(attrs_dict, n_time)
    distance_coord = _get_carina_distance_coord(attrs_dict, resource, n_columns)
    coords = get_coord_manager(
        {"time": time_coord, "distance": distance_coord},
        dims=("time", "distance"),
    )
    return attrs_dict, coords


def _get_carina_attr(resource, attr_cls, extras=None):
    """Get the attribute class and coordinates for a Carina-variant file."""
    attrs, coords = _get_carina_attrs_and_coords(resource)
    return _validate_attrs(attrs, attr_cls, extras), coords


def _get_carina_patches(resource, time=None, distance=None, attr_cls=dc.PatchAttrs):
    """Get a patch from a Carina-variant file."""
    attrs, coords = _get_carina_attrs_and_coords(resource)
    data = resource[_CARINA_DATA_NAME]
    return _build_patches(attrs, coords, data, time, distance, attr_cls)
