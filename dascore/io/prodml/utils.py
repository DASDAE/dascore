"""Utilities for terra15."""

import numpy as np

import dascore as dc
from dascore.core.schema import PatchAttrs
from dascore.utils.misc import trim_attrs_get_inds

# --- Getting format/version


def _get_prodml_version_str(hdf_fi) -> str:
    """
    Return the version string for prodml file.
    """

    # define a few root attrs that act as a "fingerprint" for terra15 files

    acquisition = getattr(hdf_fi.root, "Acquisition", None)
    if acquisition is None:
        return ""
    acq_attrs = acquisition._v_attrs
    expected_attrs = (
        "AcquisitionDescription",
        "GaugeLength",
        "PulseRate",
        "PulseWidth",
        "NumberOfLoci",
        "schemaVersion",
        "uuid",
    )
    is_prodml = all([hasattr(acq_attrs, x) for x in expected_attrs])
    return str(acq_attrs.schemaVersion.decode()) if is_prodml else ""


def _get_raw_node_dict(acquisition_node):
    """Get a dict of {Raw[x]: node}"""
    out = {
        x._v_name: x
        for x in acquisition_node._f_iter_nodes()
        if x._v_name.startswith("Raw")
    }
    return dict(sorted(out.items()))


def _get_distances(acq):
    """Get the distance ranges and spacing."""
    user_settings = acq.Custom.UserSettings._v_attrs
    start = user_settings.StartDistance
    # Note: subtracting StopDistance from StartDistance doesn't equal the
    # length specified in MeasuredLength. If MeasuredLength is multiplied
    # by Custom/SystemSettings.FiberLengthMultiplier, however, it does. So,
    # to be consistent with the TDMS reader, we assume this is correct, which
    # means the SpatialResolution is multiplied by the modifier.
    stop = user_settings.StopDistance
    multiplier = acq.Custom.SystemSettings._v_attrs.FibreLengthMultiplier
    step = user_settings.SpatialResolution * multiplier
    out = dict(distance_min=start, distance_max=stop, d_distance=step)
    return out


def _get_time_info(node):
    """Get the time information from a Raw node."""
    time_attrs = node.RawDataTime._v_attrs
    start = np.datetime64(time_attrs.PartStartTime.decode().split("+")[0])
    end = np.datetime64(time_attrs.PartEndTime.decode().split("+")[0])
    step = (end - start) / (len(node.RawDataTime) - 1)
    out = dict(time_min=start, time_max=end, d_time=step)
    return out


def _get_data_unit_and_type(node):
    """Get the data type and units."""
    attrs = node._v_attrs
    out = dict(
        data_type=attrs.RawDescription.decode().lower().replace(" ", "_"),
        data_units=attrs.RawDataUnit.decode(),
    )
    return out


def _get_prodml_attrs(fi, extras=None, cls=PatchAttrs):
    """Scan a prodML file, return metadata."""
    base_info = dict(
        gauge_length=fi.root.Acquisition._v_attrs.GaugeLength,
    )
    acq = fi.root.Acquisition
    base_info.update(_get_distances(acq))
    raw_nodes = _get_raw_node_dict(acq)
    # Iterate each raw data node. I have only ever seen 1 in a file but since
    # it is indexed like Raw[0] there might be more.
    out = []
    for node in raw_nodes.values():
        info = dict(base_info)
        info.update(_get_data_unit_and_type(node))
        info.update(_get_time_info(node))
        info["dims"] = ["time", "distance"]
        if extras is not None:
            info.update(extras)
        out.append(cls(**info))
    return out


def _get_data_attr(attrs, node, time, distance):
    """Get a new attributes with adjusted time/distance and data array."""
    shape = node.RawData.shape
    time_inds, distance_inds = slice(None), slice(None)
    if time is not None:
        time_inds, attrs = trim_attrs_get_inds(attrs, shape[0], time=time)
    if distance is not None:
        distance_inds, attrs = trim_attrs_get_inds(attrs, shape[1], distance=distance)
    data = node.RawData[time_inds, distance_inds]
    return data, attrs


def _read_prodml(fi, distance=None, time=None):
    """Read the prodml values into a patch."""
    attr_list = _get_prodml_attrs(fi)
    nodes = list(_get_raw_node_dict(fi.root.Acquisition).values())
    out = []
    for attrs, node in zip(attr_list, nodes):
        data, new_attrs = _get_data_attr(attrs, node, time, distance)
        dims = ("time", "distance")  # dims are fixed for this file format
        if data.size:
            out.append(dc.Patch(data=data, attrs=new_attrs, dims=dims))
    return out
