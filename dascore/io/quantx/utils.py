"""Utilities for Quantx."""

from typing import Optional

import numpy as np

from dascore.constants import timeable_types
from dascore.core import Patch
from dascore.core.schema import PatchFileSummary
from dascore.utils.misc import get_slice
from dascore.utils.time import to_datetime64

# --- Getting format/version


def _get_qunatx_version_str(hdf_fi) -> str:
    """
    Return the version string for Quantx file.
    """

    # define a few root attrs that act as a "fingerprint" for Quantx files
    expected_attrs = [
        "GaugeLength",
        "VendorCode",
        "MeasurementStartTime",
        "schemaVersion",
    ]

    acquisition_group = hdf_fi.get_node("/Acquisition")
    acquisition_attrs = acquisition_group._v_attrs

    is_quantx = all([hasattr(acquisition_attrs, x) for x in expected_attrs])
    has_optasense = getattr(acquisition_attrs, "VendorCode", None)
    if not (is_quantx and has_optasense):
        return ""
    return str(acquisition_attrs.schemaVersion.decode())


# --- Getting File summaries


def _get_scanned_time_min_max(data_node):
    """Get the min/max time from time array."""
    # time array is an int with microseconds as precision, so it is better
    # to not convert to float and potentially lose precision.
    time = data_node["RawDataTime"]
    t_len = len(time)
    # first try fast path by tacking first/last of time
    # Note: due to numpy #22197 we have to cast to python int
    tmin = np.datetime64(int(time[0]), "us")
    tmax = np.datetime64(int(time[-1]), "us")
    assert tmin < tmax, "Bad timeblock found, contact the developers."
    return tmin, tmax, t_len


def _get_extra_scan_attrs(self, file_version, path, data_node):
    """Get the extra attributes that go into summary information."""
    tmin, tmax, _ = _get_scanned_time_min_max(data_node)
    out = {
        "time_min": to_datetime64(tmin),
        "time_max": to_datetime64(tmax),
        "path": path,
        "file_format": self.name,
        "file_version": str(file_version),
    }
    return out


def _get_version_data_node(root):
    """Get the version, time, and data node from Quantx file."""
    version = str(root._v_attrs.schemaVersion.decode())
    if version == "2.0":
        data_type = "Raw[0]"
        data_node = root[data_type]
    else:
        raise NotImplementedError("Unknown Quantx version")
    return version, data_node


def _scan_quantx(self, fi, path):
    """Scan an Quantx file, return metadata."""
    root = fi.get_node("/Acquisition")
    root_attrs = root._v_attrs
    version, data_node = _get_version_data_node(root)
    out = _get_default_attrs(data_node, root_attrs)
    out.update(_get_extra_scan_attrs(self, version, path, data_node))
    return [PatchFileSummary.parse_obj(out)]


#
# --- Reading patch


def _get_start_stop(time_len, time_lims, file_tmin, dt):
    """Get the start/stop index along time axis."""
    # sst start index
    tmin = time_lims[0] or file_tmin
    tmax = time_lims[1] or dt * (time_len - 1) + file_tmin
    start_ind = int(np.round((tmin - file_tmin) / dt))
    stop_ind = int(np.round((tmax - file_tmin) / dt)) + 1
    # enforce upper limit on time end index.
    if stop_ind > time_len:
        stop_ind = time_len
    assert 0 <= start_ind < stop_ind
    return start_ind, stop_ind


def _get_dar_attrs(data_node, root, tar, dar):
    """Get the attributes for the Quantx data array (loaded)"""
    attrs = _get_default_attrs(data_node, root._v_attrs)
    attrs["time_min"] = tar.min()
    attrs["time_max"] = tar.max()
    attrs["distance_min"] = dar.min()
    attrs["distance_max"] = dar.max()
    return attrs


def _get_distance_array(root):
    """
    Return an array of distance along fiber values.
    """
    vattrs = root._v_attrs
    channel_numbers = np.arange(vattrs.NumberOfLoci) + vattrs.StartLocusIndex
    assert vattrs.SpatialSamplingIntervalUnit == b"m", "expecting distance in m"
    return channel_numbers * vattrs.SpatialSamplingInterval


def _read_quantx(
    root,
    time: Optional[tuple[timeable_types, timeable_types]] = None,
    distance: Optional[tuple[float, float]] = None,
) -> Patch:
    """
    Read an Quantx file.
    """
    # get time array
    time_lims = tuple(
        to_datetime64(x) if x is not None else None
        for x in (time if time is not None else (None, None))
    )
    _, data_node = _get_version_data_node(root)
    file_t_min, file_t_max, time_len = _get_scanned_time_min_max(data_node)
    dt = (file_t_max - file_t_min) / (time_len - 1)
    # get the start and stop along the time axis
    start_ind, stop_ind = _get_start_stop(time_len, time_lims, file_t_min, dt)
    req_t_min = file_t_min if start_ind == 0 else file_t_min + dt * start_ind
    # account for files that might not be full, adjust requested max time
    stop_ind = min(stop_ind, time_len)
    assert stop_ind > start_ind
    req_t_max = (
        time_lims[-1] if stop_ind < time_len else file_t_min + (stop_ind - 1) * dt
    )
    assert req_t_max > req_t_min
    # calculate time array, convert to datetime64
    time_ar = to_datetime64(file_t_min + np.arange(start_ind, stop_ind) * dt)
    time_inds = (start_ind, stop_ind)
    # get data and sliced distance coord
    dist_ar = _get_distance_array(root)
    dslice = get_slice(dist_ar, distance)
    dist_ar_trimmed = dist_ar[dslice]
    data = data_node.RawData[slice(*time_inds), dslice]
    coords = {"time": time_ar, "channel_number": dist_ar_trimmed}
    dims = ("time", "channel_number")
    attrs = _get_dar_attrs(data_node, root, time_ar, dist_ar_trimmed)
    return Patch(data=data, coords=coords, attrs=attrs, dims=dims)


def _get_default_attrs(data_node, root_node_attrs):
    """
    Return the required/default attributes which can be fetched from attributes.

    Note: missing time, distance absolute ranges. Downstream functions should handle
    this.
    """
    # breakpoint()
    out = dict(dims="time, channel_number", data_category="DAS")
    _root_attrs = {
        "SpatialSamplingInterval": "d_distance",
        "SpatialSamplingIntervalUnit": "distance_units",
        "uuid": "intrument_id",  # not 100% sure about this one
    }
    for treble_name, out_name in _root_attrs.items():
        out[out_name] = getattr(root_node_attrs, treble_name)
    # get calculated attributes
    timing = data_node.RawDataTime
    out["d_time"] = np.timedelta64(timing[1] - timing[0], "us")
    d_ind_start = root_node_attrs.StartLocusIndex
    d_ind_num = root_node_attrs.NumberOfLoci
    out["distance_min"] = d_ind_start * out["d_distance"]
    out["distance_max"] = (d_ind_start + d_ind_num) * out["d_distance"]
    # get attributes from data node
    out["data_units"] = data_node._v_attrs.RawDataUnit.decode()
    # when radians are part of the units this is a phase measurement
    if "rad" in out["data_units"]:
        out["data_type"] = "phase"
    return out
