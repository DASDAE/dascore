"""Utilities for Quantx."""
from __future__ import annotations

import numpy as np
from tables import NoSuchNodeError

import dascore as dc
from dascore.constants import timeable_types
from dascore.core import Patch
from dascore.core.coords import get_coord
from dascore.utils.time import to_datetime64
from dascore.utils.misc import maybe_get_attrs
from dascore.units import get_quantity

# --- Getting format/version


def _get_qunatx_version_str(hdf_fi) -> str:
    """Return the version string for Quantx file."""
    # define a few root attrs that act as a "fingerprint" for Quantx files
    expected_attrs = [
        "GaugeLength",
        "VendorCode",
        "MeasurementStartTime",
        "schemaVersion",
    ]
    try:
        acquisition_group = hdf_fi.get_node("/Acquisition")
    except NoSuchNodeError:
        return ""
    acquisition_attrs = acquisition_group._v_attrs

    is_quantx = all([hasattr(acquisition_attrs, x) for x in expected_attrs])
    has_vendor_code = getattr(acquisition_attrs, "VendorCode", None)
    if not (is_quantx and has_vendor_code):
        return ""
    return str(acquisition_attrs.schemaVersion.decode())


# --- Getting File summaries


def _get_time_coord(data_node):
    """Return the time coordinate."""
    # time array is an int with microseconds as precision, so it is better
    # to not convert to float and potentially lose precision.
    time = data_node["RawDataTime"]
    t_len = len(time)
    # first try fast path by tacking first/last of time
    # Note: due to numpy #22197 we have to cast to python int
    # also, We have only every encountered evenly sampled time data in this
    # format, but our experience is limited.
    tmin = to_datetime64(np.datetime64(int(time[0]), "us"))
    tmax = to_datetime64(np.datetime64(int(time[-1]), "us"))
    dt = (tmax - tmin) / (t_len - 1)
    return get_coord(start=tmin, stop=tmax + dt, step=dt, units="s")


def _get_dist_coord(root_node_attrs):
    """Get the distance coordinate."""
    d_step = root_node_attrs.SpatialSamplingInterval
    d_unit = get_quantity(root_node_attrs.SpatialSamplingIntervalUnit)
    d_ind_start = root_node_attrs.StartLocusIndex
    d_ind_num = root_node_attrs.NumberOfLoci
    d_min = d_ind_start * d_step
    d_max = (d_ind_start + d_ind_num) * d_step
    coord = get_coord(start=d_min, stop=d_max, step=d_step, units=d_unit)
    return coord


def _get_version_data_node(root):
    """Get the version, time, and data node from Quantx file."""
    version = str(root._v_attrs.schemaVersion.decode())
    if version == "2.0":
        data_type = "Raw[0]"
        data_node = root[data_type]
    else:
        raise NotImplementedError("Unknown Quantx version")
    return version, data_node


def _scan_quantx(self, fi):
    """Scan an Quantx file, return metadata."""
    root = fi.get_node("/Acquisition")
    root_attrs = root._v_attrs
    version, data_node = _get_version_data_node(root)
    out = _get_attrs(data_node, root_attrs)
    extras = dict(path=fi.filename, file_format=self.name, file_version=version)
    out.update(extras)
    return out


#
# --- Reading patch


def _read_quantx(
    root,
    time: tuple[timeable_types, timeable_types] | None = None,
    distance: tuple[float, float] | None = None,
    cls=dc.PatchAttrs,
) -> Patch:
    """Read a Quantx file."""
    _, data_node = _get_version_data_node(root)
    attrs = _get_attrs(data_node, root._v_attrs)
    t_coord, t_ind = attrs.pop("_t_coord").select(time)
    d_coord, d_ind = attrs.pop("_d_coord").select(distance)
    # get data and sliced distance coord
    data = data_node.RawData[t_ind, d_ind]
    coords = {"time": t_coord, "distance": d_coord}
    dims = ("time", "distance")
    return Patch(data=data, coords=coords, attrs=cls(**attrs), dims=dims)


def _get_attrs(data_node, root_node_attrs):
    """Return the required/default attributes which can be fetched from attributes."""
    out = dict(dims="time,distance", data_category="DAS")
    time_coord = _get_time_coord(data_node)
    dist_coord = _get_dist_coord(root_node_attrs)
    out.update(time_coord.get_attrs_dict("time"))
    out.update(dist_coord.get_attrs_dict("distance"))
    out["_t_coord"], out["_d_coord"] = time_coord, dist_coord
    _root_attrs = {
        "uuid": "instrument_id",  # not 100% sure about this one
        "RawDescription": "raw_description",
        "PulseWidth": "pulse_width",
        "PulseWidthUnits": "pulse_width_units",
        "GaugeLength": "gauge_length_units",
        "schemaVersion": "schema_version",
    }
    out.update(maybe_get_attrs(root_node_attrs, _root_attrs))
    # get calculated attributes
    # get attributes from data node
    out["data_units"] = data_node._v_attrs.RawDataUnit.decode()
    # when radians are part of the units this is a phase measurement
    if "rad" in out["data_units"]:
        out["data_type"] = "phase"
    return out
