"""Utilities for terra15."""

from tables.exceptions import NoSuchNodeError

from dascore.constants import timeable_types
from dascore.core import Patch
from dascore.core.coords import get_coord
from dascore.core.schema import PatchFileSummary
from dascore.utils.time import to_datetime64, to_timedelta64

# --- Getting format/version


def _get_terra15_version_str(hdf_fi) -> str:
    """
    Return the version string for terra15 file.
    """

    # define a few root attrs that act as a "fingerprint" for terra15 files
    expected_attrs = [
        "acoustic_bandwidth_end",
        "amplifier_incoming_current",
        "file_start_computer_time",
        "file_version",
    ]
    root_attrs = hdf_fi.root._v_attrs
    is_terra15 = all([hasattr(root_attrs, x) for x in expected_attrs])
    if not is_terra15:
        return ""
    return str(root_attrs.file_version)


# --- Getting File summaries


def _get_time_node(data_node):
    """
    Get the time node from data.

    This will prefer GPS time but gets posix time if it is missing.
    """
    try:
        time = data_node["gps_time"]
    except (NoSuchNodeError, IndexError):
        time = data_node["posix_time"]
    return time


def _get_scanned_time_info(data_node):
    """Get the min, max, len, and dt from time array."""
    time = _get_time_node(data_node)
    t_len = len(time)
    # first try fast path by tacking first/last of time
    tmin, tmax = time[0], time[-1]
    # This doesn't work if an incomplete datablock exists at the end of
    # the file. In this case we need to read/filter time array (slower).
    if tmin > tmax:
        time = time[:]
        time_filtered = time[time > 0]
        t_len = len(time_filtered)
        tmin, tmax = time_filtered[0], time_filtered[-1]
    # surprisingly, using gps time column, dt is much different than dt
    # reported in data attrs so we calculate it this way.
    dt = (tmax - tmin) / (t_len - 1)
    starttime = to_datetime64(tmin)
    d_time = to_timedelta64(dt)
    endtime = starttime + d_time * (t_len - 1)
    return starttime, endtime, t_len, d_time


def _get_extra_scan_attrs(self, file_version, path, data_node):
    """Get the extra attributes that go into summary information."""
    tmin, tmax, _, dt = _get_scanned_time_info(data_node)
    out = {
        "time_min": to_datetime64(tmin),
        "time_max": to_datetime64(tmax),
        "d_time": to_timedelta64(dt),
        "path": path,
        "file_format": self.name,
        "file_version": str(file_version),
    }
    return out


def _get_version_data_node(root):
    """Get the version, time, and data node from terra15 file."""
    version = str(root._v_attrs.file_version)
    if version == "4":
        data_type = root._v_attrs.data_product
        data_node = root[data_type]
    elif version in {"5", "6"}:
        data_node = root["data_product"]
    else:
        raise NotImplementedError("Unknown Terra15 version")
    return version, data_node


def _scan_terra15(self, fi):
    """Scan a terra15 file, return metadata."""
    root = fi.root
    root_attrs = fi.root._v_attrs
    version, data_node = _get_version_data_node(root)
    out = _get_default_attrs(root_attrs)
    out.update(_get_extra_scan_attrs(self, version, fi.filename, data_node))
    return [PatchFileSummary(**out)]


# --- Reading patch


def _get_dar_attrs(data_node, root, tar, dar):
    """Get the attributes for the terra15 data array (loaded)"""
    attrs = _get_default_attrs(root._v_attrs)
    attrs["time_min"] = tar.min()
    attrs["time_max"] = tar.max()
    attrs["distance_min"] = dar.min()
    attrs["distance_max"] = dar.max()
    return attrs


def _get_raw_time_coord(data_node):
    """Read the time from the data node and return it."""
    time = _get_time_node(data_node)[:]
    return get_coord(values=to_datetime64(time))


def _read_terra15(
    root,
    time: tuple[timeable_types, timeable_types] | None = None,
    distance: tuple[float, float] | None = None,
    snap_dims: bool = True,
) -> Patch:
    """
    Read a terra15 file.

    Notes
    -----
    The time array is complicated. There is GPS time and Posix time included
    in the file. In version 0.0.6 and less of dascore we just used gps time.
    However, sometimes this results in subsequent samples having a time before
    the previous sample (time did not increase monotonically).

    So now, we use the first GPS sample, the last sample, and length
    to determine the dt (new in dascore>0.0.11).
    """
    _, data_node = _get_version_data_node(root)
    t_min, t_max, time_len, d_time = _get_scanned_time_info(data_node)
    if snap_dims:
        kwargs = dict(start=t_min, stop=t_max + d_time, step=d_time, units="s")
        time_coord = get_coord(**kwargs)
    else:
        time_coord = _get_raw_time_coord(data_node)
    time_coord, time_slice = time_coord.select(time)
    # get data and sliced distance coord
    dist_coord = _get_distance_coord(root)
    dist_coord, dslice = dist_coord.select(distance)
    _data = data_node.data
    # checks for incomplete data blocks
    if _data.shape[0] > time_len:
        new_stop = time_slice.stop or time_len
        time_slice = slice(time_slice.start, new_stop)
    data = data_node.data[time_slice, dslice]
    coords = {"time": time_coord, "distance": dist_coord}
    dims = ("time", "distance")
    attrs = _get_dar_attrs(data_node, root, time_coord, dist_coord)
    return Patch(data=data, coords=coords, attrs=attrs, dims=dims)


def _get_default_attrs(root_node_attrs):
    """
    Return the required/default attributes which can be fetched from attributes.

    Note: missing time, distance absolute ranges. Downstream functions should handle
    this.
    """
    dmin, dmax, dstep, _ = _get_distance_start_stop_step(root_node_attrs)
    out = dict(
        dims="time,distance",
        distance_min=dmin,
        distance_max=dmax,
        d_distance=dstep,
    )
    _root_attrs = {
        "data_product": "data_type",
        "serial_number": "instrument_id",
        "data_product_units": "data_units",
    }

    for treble_name, out_name in _root_attrs.items():
        out[out_name] = getattr(root_node_attrs, treble_name)

    return out


def _get_distance_start_stop_step(root):
    """Get distance values for start, stop, step."""
    # TODO: At least for the v4 test file, sensing_range_start, sensing_range_stop,
    # nx, and dx are not consistent, meaning d_min + dx * nx != d_max
    # so I just used this method. We need to look more into this.
    attrs = getattr(root, "_v_attrs", root)
    start, step, samps = attrs.sensing_range_start, attrs.dx, attrs.nx
    stop = attrs.sensing_range_start + (attrs.nx - 1) * attrs.dx
    return start, stop, step, samps


def _get_distance_coord(root):
    """Get the distance coordinate."""
    start, stop, step, samps = _get_distance_start_stop_step(root)
    return get_coord(start=start, stop=stop + step, step=step, units="m")
