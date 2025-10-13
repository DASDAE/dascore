"""Utilities for prodML."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import h5py
import numpy as np

import dascore as dc
from dascore.constants import VALID_DATA_TYPES
from dascore.core.coords import get_coord
from dascore.utils.misc import iterate, maybe_get_items, register_func, unbyte
from dascore.utils.models import UnitQuantity, UTF8Str

# --- Getting format/version

_EXPECTED_ATTRS = (
    "PulseRate",
    "PulseWidth",
    "NumberOfLoci",
    "schemaVersion",
    "uuid",
)

_ROOT_ATTRS = {
    "PulseWidth": "pulse_width",
    "PulseWidthUnits": "pulse_width_units",
    "PulseWidthUnit": "pulse_width_units",
    "PulseWidth.uom": "pulse_width_units",
    "PulseRate": "pulse_rate",
    "PulseRateUnit": "pulse_rate_units",
    "PulseRateUnits": "pulse_rate_units",
    "PulseRate.uom": "pulse_rate_units",
    "GaugeLength": "gauge_length",
    "GaugeLengthUnit": "gauge_length_units",
    "GaugeLengthUnits": "gauge_length_units",
    "GaugeLength.uom": "gauge_length_units",
    "schemaVersion": "schema_version",
}

_FBE_NODE_ATTRS = {
    "StartFrequency": "start_frequency",
    "EndFrequency": "end_frequency",
}

_FBE_PARENT_ATTRS = {
    "RawReference": "raw_reference",
    "TransformSize": "transform_size",
    "TransformType": "transform_type",
    "WindowFunction": "window_function",
    "WindowOverlap": "window_overlap",
    "WindowSize": "window_size",
}

_NODE_ATTRS_PROCESSORS = {}
_NODE_DATA_PROCESSORS = {}


# --- Patch Attrs from ProdML files.


class ProdMLRawPatchAttrs(dc.PatchAttrs):
    """Patch attrs for raw data contained in ProdML."""

    pulse_width: float = np.nan
    pulse_width_units: UnitQuantity | None = None
    gauge_length: float = np.nan
    gauge_length_units: UnitQuantity | None = None
    schema_version: UTF8Str = ""


class ProdMLFbePatchAttrs(ProdMLRawPatchAttrs):
    """Patch attrs for fbe (frequency band extracted) data in Prodml."""

    raw_reference: UTF8Str = ""
    transform_size: float = np.nan
    transform_type: UTF8Str = ""
    window_size: int | None = None
    window_function: UTF8Str = ""
    window_overlap: int | None = None
    start_frequency: float = 0
    end_frequency: float = np.inf


@dataclass
class _ProdMLNodeInfo:
    """An internal class for parsing the prodml node tree."""

    h5_file: h5py.File
    name: str
    patch_type: str
    node: h5py.Dataset | h5py.Group
    parent_node: h5py.Dataset | h5py.Group


def _get_prodml_version_str(hdf_fi) -> str:
    """Return the version string for prodml file."""
    # define a few root attrs that act as a "fingerprint" for prodML files
    acquisition = hdf_fi.get("Acquisition", None)
    if acquisition is None:
        return ""
    attrs = acquisition.attrs
    is_prodml = set(_EXPECTED_ATTRS).issubset(set(attrs))
    # in some prodml schemaVersion is str, in other float; this handles both.
    version_str = str(unbyte(attrs["schemaVersion"]))
    return version_str if is_prodml else ""


def _get_distance_coord(acq):
    """Get the distance ranges and spacing."""

    def get_distance_units(attrs):
        """Get distance units, works for both v2 and v3."""
        name_v2_0 = "SpatialSamplingIntervalUnit"
        name_v2_1 = "SpatialSamplingInterval.uom"
        ustr = attrs.get(name_v2_0, attrs.get(name_v2_1, ""))
        return unbyte(ustr)

    attrs = acq.attrs
    step = float(attrs["SpatialSamplingInterval"])
    num_dist_channels = int(attrs["NumberOfLoci"])
    start = int(attrs["StartLocusIndex"]) * step
    stop = start + num_dist_channels * step
    units = get_distance_units(attrs)
    return get_coord(start=start, stop=stop, units=units, step=step)


def _get_time_coord(node):
    """Get the time information from a Raw node."""
    time_names = [x for x in node.keys() if x.endswith("DataTime")]
    assert len(time_names) == 1, f"Found bad time information in prodml {node=}"
    time_name = time_names[0]
    time_array = node[time_name]
    array_len = len(time_array)
    assert array_len > 0, "Missing time array in ProdML file."
    time_attrs = time_array.attrs
    start_str = unbyte(time_attrs["PartStartTime"]).split("+")[0]
    start = dc.to_datetime64(start_str.rstrip("Z"))
    end_str = unbyte(time_attrs["PartEndTime"]).split("+")[0]
    end = dc.to_datetime64(end_str.rstrip("Z"))
    step = (end - start) / (array_len - 1)
    time_coord = get_coord(start=start, stop=end + step, step=step, units="s")
    # Sometimes the "PartEndTime" can be wrong. Check for this and try to
    # compensate. See #414.
    last = np.asarray(time_array[-1:]).astype("datetime64[us]")[0]
    tc_max = np.asarray(time_coord.max()).astype("datetime64[us]")
    diff = float(np.abs((tc_max - last) / step))
    # Note: just in case the time array is not in microseconds as it should
    # be, we prefer to use the iso 8601 strings in the 'PartStartTime' attrs
    # because they are less likely to get messed up. Therefore, we only
    # correct time coordinate from time array if the values are "close" but off.
    if 0 < diff < 10:
        time_array = time_array[:].astype("datetime64[us]")
        time_coord = get_coord(data=time_array)
    return time_coord


def _yield_data_nodes(fi) -> Iterator[_ProdMLNodeInfo]:
    """
    Iterate the data nodes contained in the prodml file and yield

    Return a tuple of (acq_node, parent_node, data_node).

    This accounts for raw data and processed data.
    """
    acq = fi["Acquisition"]
    raw_arrays = {"RawDataTime", "RawData"}
    # Get raw data.
    raws = {i: v for i, v in acq.items() if i.lower().startswith("raw")}
    # Get fbe from processed data.
    processed = acq.get("Processed", {})
    fbe = {i: v for i, v in processed.items() if i.lower().startswith("fbe")}
    # Iterate each node and yield back.
    for name, node in (fbe | raws).items():
        # Patch type is first 3 chars of data array name.
        patch_type = str(name.split("/")[-1].lower()[:3])
        if patch_type == "raw":
            # This is an empty data node.
            if not raw_arrays.issubset(set(node)):
                continue
            raw_info = _ProdMLNodeInfo(
                node=node,
                name=name,
                h5_file=fi,
                patch_type=patch_type,
                parent_node=acq,
            )
            yield raw_info
        # need to iterate into sub fbe because it can have many bands.
        if patch_type == "fbe":
            subs = {i: v for i, v in node.items() if i.lower().startswith("fbedata[")}
            for sub_name, sub_node in subs.items():
                fbe_info = _ProdMLNodeInfo(
                    node=sub_node,
                    name=sub_name,
                    h5_file=fi,
                    patch_type=patch_type,
                    parent_node=node,
                )
                yield fbe_info


def _get_data_unit_and_type(node):
    """Get the data type and units."""
    attrs = node.attrs
    attr_map = {
        "RawDescription": "data_type",
        "RawDataUnit": "data_units",
        "FbeDataUnit": "data_units",
    }
    out = maybe_get_items(attrs, attr_map)
    if (data_type := out.get("data_type")) is not None:
        clean = data_type.lower().replace(" ", "_")
        out["data_type"] = clean if clean in VALID_DATA_TYPES else ""
    return out


@register_func(_NODE_ATTRS_PROCESSORS, key="raw")
def _get_raw_node_attr_coords(node_info, d_coord, base_info):
    """Get the raw data information."""
    info = dict(base_info)
    t_coord = _get_time_coord(node_info.node)
    info.update(_get_data_unit_and_type(node_info.node))
    coords = dc.get_coord_manager(
        coords={"time": t_coord, "distance": d_coord},
        dims=_get_dims_from_attrs(node_info.node["RawData"].attrs),
    )
    return ProdMLRawPatchAttrs(**info), coords


@register_func(_NODE_ATTRS_PROCESSORS, key="fbe")
def _get_processed_node_attr_coords(node_info, d_coord, base_info):
    """Get information about the processed patches."""
    out = dict(base_info)
    t_coord = _get_time_coord(node_info.parent_node)
    out.update(_get_data_unit_and_type(node_info.node))
    out.update(maybe_get_items(node_info.node.attrs, _FBE_NODE_ATTRS))
    out.update(maybe_get_items(node_info.parent_node.attrs, _FBE_PARENT_ATTRS))
    # For some reason, the distance coords in raw and fbe data are not the
    # same. For now, we just assume the FBE distance is a subset of the raw
    # distance referenced by StartLocusIndex. It may be that the test file used
    # develop this code was malformed and we need to revise this.
    start = node_info.parent_node.attrs.get("StartLocusIndex", 0)
    total = node_info.parent_node.attrs["NumberOfLoci"]
    ind_1 = int(unbyte(start) if isinstance(start, bytes | np.bytes_) else start)
    count = int(unbyte(total) if isinstance(total, bytes | np.bytes_) else total)
    # Put the coords back together.
    distance = d_coord[ind_1 : ind_1 + count]
    coords = dc.get_coord_manager(
        coords={"time": t_coord, "distance": distance},
        dims=_get_dims_from_attrs(node_info.parent_node.attrs),
    )
    return ProdMLFbePatchAttrs(**out), coords


@register_func(_NODE_DATA_PROCESSORS, key="raw")
def _get_raw_data(node_info):
    """Get the data from a raw node."""
    node = node_info.node
    data_node = [x for x in list(node) if x.lower().endswith("data")]
    assert len(data_node) == 1, "more than one data node found."
    data = node[data_node[0]]
    return data


@register_func(_NODE_DATA_PROCESSORS, key="fbe")
def _get_fbe_data(node_info):
    """Get the data from a fbe node."""
    return node_info.node


def _yield_prodml_attrs_coords(fi, extras=None):
    """Scan a prodML file, return metadata."""
    acq = fi["Acquisition"]
    # Get the information common to all from root attributes.
    base_info = maybe_get_items(acq.attrs, _ROOT_ATTRS)
    base_info.update(extras if extras is not None else {})
    d_coord = _get_distance_coord(acq)
    # Iterate the raw and processed data and return results in a list.
    for node_info in _yield_data_nodes(fi):
        func = _NODE_ATTRS_PROCESSORS[node_info.patch_type]
        attr, coords = func(node_info, d_coord, base_info)
        yield (attr, coords)


def _get_dims_from_attrs(attrs):
    """Get the dimension names in the form of a tuple."""
    # we use distance rather than locus, setup mapping to rename this.
    map_ = {
        "locus": "distance",
        "Locus": "distance",
        "Time": "time",
        "Distance": "distance",
    }
    dims = unbyte(attrs.get("Dimensions", "time, distance"))
    if isinstance(dims, str):
        dims = dims.replace(",", " ")
        dims = tuple(map_.get(x, x) for x in dims.split())
    else:
        unbytes = [unbyte(x) for x in iterate(dims)]
        dims = tuple(map_.get(x, x) for x in unbytes)
    return dims


def _read_prodml(fi, distance=None, time=None):
    """Read the prodml values into a patch."""
    out = []
    acq = fi["Acquisition"]
    base_info = maybe_get_items(acq.attrs, _ROOT_ATTRS)
    d_coord = _get_distance_coord(acq)
    for info in _yield_data_nodes(fi):
        attr_func = _NODE_ATTRS_PROCESSORS[info.patch_type]
        attrs, cm = attr_func(info, d_coord, base_info)
        data = _NODE_DATA_PROCESSORS[info.patch_type](info)
        if time is not None or distance is not None:
            cm, data = cm.select(array=data, time=time, distance=distance)
        if data.size:
            out.append(dc.Patch(data=data, attrs=attrs, coords=cm))
    return out
