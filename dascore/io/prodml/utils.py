"""Utilities for prodML."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from uuid import UUID, uuid4

import h5py
import numpy as np

import dascore as dc
from dascore.constants import VALID_DATA_TYPES
from dascore.core.coords import get_coord
from dascore.exceptions import InvalidSpoolError, PatchError, UnitError
from dascore.io.utils import get_exact_coord
from dascore.units import convert_units, get_quantity_str
from dascore.utils.io import _normalize_source_patch_ids
from dascore.utils.misc import iterate, maybe_get_items, register_func, unbyte
from dascore.utils.models import UnitQuantity, UTF8Str

# --- Getting format/version

_EXPECTED_ATTRS = (
    "NumberOfLoci",
    "SpatialSamplingInterval",
    "StartLocusIndex",
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
    "AcquisitionId": "acquisition_id",
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
    if not is_prodml:
        return ""
    # in some prodml schemaVersion is str, in other float; this handles both.
    version_str = str(unbyte(attrs["schemaVersion"]))
    return version_str


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


def _get_time_coord(node, snap=True):
    """Get the time information from a Raw node."""
    time_names = [x for x in node.keys() if x.endswith("DataTime")]
    assert len(time_names) == 1, f"Found bad time information in prodml {node=}"
    time_name = time_names[0]
    time_array = node[time_name]
    array_len = len(time_array)
    assert array_len > 0, "Missing time array in ProdML file."
    if not snap:
        values = time_array[:].astype("datetime64[us]")
        return get_exact_coord(values, units="s")
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
    if str(out.get("data_units", "")).upper() == "UNKNOWN":
        out.pop("data_units")
    if (data_type := out.get("data_type")) is not None:
        clean = data_type.lower().replace(" ", "_")
        out["data_type"] = clean if clean in VALID_DATA_TYPES else ""
    return out


# --- Writing ProdML files.


def _fixed_string(value):
    """Return a fixed-length HDF5 byte string."""
    return np.bytes_(str(value).encode())


def _fixed_strings(values):
    """Return an array of fixed-length HDF5 byte strings."""
    encoded = [str(value).encode() for value in values]
    return np.asarray(encoded, dtype=f"S{max(map(len, encoded))}")


def _get_single_patch(spool):
    """Return the only Patch in a writable spool."""
    patches = [spool] if isinstance(spool, dc.Patch) else list(spool)
    if len(patches) != 1 or not isinstance(patches[0], dc.Patch):
        msg = "ProdML writing requires exactly one Patch."
        raise InvalidSpoolError(msg)
    return patches[0]


def _round_times_to_microseconds(coord):
    """Round absolute timestamps to nearest microsecond using integer math."""
    values = np.asarray(coord.values)
    if not np.issubdtype(values.dtype, np.datetime64) or len(values) < 2:
        msg = "ProdML writing requires at least two absolute time samples."
        raise PatchError(msg)
    if np.any(np.isnat(values)):
        raise PatchError("ProdML time coordinates cannot contain NaT.")
    nanoseconds = values.astype("datetime64[ns]").astype(np.int64)
    quotient, remainder = np.divmod(nanoseconds, 1_000)
    increment = np.where(nanoseconds >= 0, remainder >= 500, remainder > 500)
    microseconds = quotient + increment
    diffs = np.diff(microseconds)
    if np.any(diffs <= 0) or not np.all(diffs == diffs[0]):
        msg = (
            "ProdML time coordinates must remain increasing and evenly sampled "
            "after microsecond rounding."
        )
        raise PatchError(msg)
    if np.any(remainder):
        msg = "ProdML timestamps were rounded to microseconds; precision was lost."
        warnings.warn(msg, UserWarning, stacklevel=3)
    return microseconds.astype(np.int64), int(diffs[0])


def _get_distance_info(coord):
    """Validate a distance coordinate and return its ProdML sampling values."""
    values = np.asarray(coord.values)
    if not len(values) or values.dtype.kind not in "iuf":
        msg = "ProdML writing requires a nonempty numeric distance coordinate."
        raise PatchError(msg)
    values = values.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise PatchError("ProdML distance coordinates must be finite.")
    diffs = np.diff(values)
    step = coord.step
    if step is None:
        step = diffs[0] if len(diffs) else np.nan
    step = float(step)
    tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(values)))) * 10
    if (
        not np.isfinite(step)
        or step <= 0
        or not np.allclose(diffs, step, rtol=1e-9, atol=tolerance)
    ):
        msg = "ProdML distance coordinates must be increasing and evenly sampled."
        raise PatchError(msg)
    units = coord.unit_str
    if not units:
        raise PatchError("ProdML distance coordinates require length units.")
    try:
        convert_units(1.0, to_units="m", from_units=units)
    except UnitError as exc:
        msg = "ProdML distance coordinate units must represent length."
        raise PatchError(msg) from exc
    ratio = float(values[0] / step)
    start_locus = int(np.round(ratio))
    integer_tolerance = max(1e-9, abs(float(np.spacing(ratio))) * 4)
    if not np.isclose(ratio, start_locus, rtol=0, atol=integer_tolerance):
        msg = "ProdML requires distance_start / distance_step to be an integer."
        raise PatchError(msg)
    return step, units, start_locus


def _get_measure(attrs, name, units_name, target_units):
    """Return a complete positive measure, or None when it is unusable."""
    value = attrs.get(name)
    try:
        units = get_quantity_str(attrs.get(units_name))
        value = float(value)
        if not units or not np.isfinite(value) or value <= 0:
            return None
        convert_units(value, to_units=target_units, from_units=units)
    except (TypeError, ValueError, UnitError):
        return None
    return value, units


def _iso_time(microseconds):
    """Return an ISO-8601 UTC timestamp with microsecond precision."""
    value = np.datetime64(int(microseconds), "us")
    return _fixed_string(f"{np.datetime_as_string(value, unit='us')}+00:00")


def _prepare_prodml_write(spool):
    """Validate input and prepare all values needed to write ProdML."""
    patch = _get_single_patch(spool)
    if patch.data.ndim != 2 or set(patch.dims) != {"time", "distance"}:
        msg = "ProdML writing requires a 2D Patch with time and distance dimensions."
        raise PatchError(msg)
    data = np.asarray(patch.data)
    valid_dtype = data.dtype.kind in "iuf" and data.dtype.itemsize in {1, 2, 4, 8}
    valid_dtype &= data.dtype.kind != "f" or data.dtype.itemsize in {4, 8}
    if not data.size or not valid_dtype:
        msg = "ProdML writing supports nonempty standard integer or float dtypes."
        raise PatchError(msg)

    times, time_step_us = _round_times_to_microseconds(patch.get_coord("time"))
    distance_step, distance_units, start_locus = _get_distance_info(
        patch.get_coord("distance")
    )
    attrs = patch.attrs
    acquisition_id = str(attrs.acquisition_id or uuid4())
    try:
        acquisition_id = str(UUID(acquisition_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise PatchError("ProdML acquisition_id must be a valid UUID.") from exc

    facility = attrs.get("facility_id") or attrs.station or "UNKNOWN"
    service_company = attrs.get("service_company_name") or "UNKNOWN"
    data_units = get_quantity_str(attrs.data_units) or "UNKNOWN"
    string_values = (facility, service_company, data_units)
    if not all(isinstance(x, str) and 0 < len(x) <= 64 for x in string_values):
        msg = "ProdML facility, service company, and data unit must be String64 values."
        raise PatchError(msg)

    start_time, end_time = _iso_time(times[0]), _iso_time(times[-1])
    output_rate = 1_000_000.0 / time_step_us
    pulse_rate = _get_measure(attrs, "pulse_rate", "pulse_rate_units", "Hz")
    pulse_rate = pulse_rate or (output_rate, "Hz")
    num_loci = len(patch.get_coord("distance"))
    file_uuid, acquisition_uuid, raw_uuid = (str(uuid4()) for _ in range(3))
    acquisition_attrs = {
        "AcquisitionId": _fixed_string(acquisition_id),
        "FacilityId": _fixed_strings([facility]),
        "MaximumFrequency": output_rate / 2,
        "MaximumFrequency.uom": _fixed_string("Hz"),
        "MeasurementStartTime": start_time,
        "MinimumFrequency": 0.0,
        "MinimumFrequency.uom": _fixed_string("Hz"),
        "NumberOfLoci": num_loci,
        "PulseRate": pulse_rate[0],
        "PulseRate.uom": _fixed_string(pulse_rate[1]),
        "ServiceCompanyName": _fixed_string(service_company),
        "SpatialSamplingInterval": distance_step,
        "SpatialSamplingInterval.uom": _fixed_string(distance_units),
        "StartLocusIndex": start_locus,
        "TriggeredMeasurement": False,
        "schemaVersion": _fixed_string("2.1"),
        "uuid": _fixed_string(acquisition_uuid),
    }
    for name, units_name, target, hdf_name in (
        ("pulse_width", "pulse_width_units", "s", "PulseWidth"),
        ("gauge_length", "gauge_length_units", "m", "GaugeLength"),
    ):
        if measure := _get_measure(attrs, name, units_name, target):
            acquisition_attrs[hdf_name] = measure[0]
            acquisition_attrs[f"{hdf_name}.uom"] = _fixed_string(measure[1])

    raw_attrs = {
        "NumberOfLoci": num_loci,
        "OutputDataRate": output_rate,
        "OutputDataRate.uom": _fixed_string("Hz"),
        "RawDataUnit": _fixed_string(data_units),
        "StartLocusIndex": start_locus,
        "uuid": _fixed_string(raw_uuid),
    }
    if attrs.data_type:
        raw_attrs["RawDescription"] = _fixed_string(attrs.data_type)
    data_attrs = {
        "Count": data.size,
        "Dimensions": _fixed_strings(
            ["locus" if dim == "distance" else dim for dim in patch.dims]
        ),
        "PartEndTime": end_time,
        "PartStartTime": start_time,
        "StartIndex": 0,
    }
    time_attrs = {
        "Count": len(times),
        "EndTime": end_time,
        "PartEndTime": end_time,
        "PartStartTime": start_time,
        "StartIndex": 0,
        "StartTime": start_time,
        "Uom": _fixed_string("us"),
    }
    return (
        data,
        times,
        _fixed_string(file_uuid),
        acquisition_attrs,
        raw_attrs,
        data_attrs,
        time_attrs,
    )


def _write_prodml(spool, resource):
    """Write one raw Patch to a standalone ProdML 2.1 HDF5 file."""
    prepared = _prepare_prodml_write(spool)
    data, times, file_uuid, acq_attrs, raw_attrs, data_attrs, time_attrs = prepared
    h5_file = getattr(resource, "_handle", resource)
    for key in list(h5_file):
        del h5_file[key]
    h5_file.attrs.clear()
    h5_file.attrs["uuid"] = file_uuid
    acquisition = h5_file.create_group("Acquisition")
    acquisition.attrs.update(acq_attrs)
    raw = acquisition.create_group("Raw[0]")
    raw.attrs.update(raw_attrs)
    raw_data = raw.create_dataset("RawData", data=data)
    raw_data.attrs.update(data_attrs)
    raw_time = raw.create_dataset("RawDataTime", data=times)
    raw_time.attrs.update(time_attrs)


@register_func(_NODE_ATTRS_PROCESSORS, key="raw")
def _get_raw_node_attr_coords(node_info, d_coord, base_info, snap=True):
    """Get the raw data information."""
    info = dict(base_info)
    t_coord = _get_time_coord(node_info.node, snap=snap)
    info.update(_get_data_unit_and_type(node_info.node))
    info["dtype"] = str(node_info.node["RawData"].dtype)
    info["_source_patch_id"] = node_info.name
    coords = dc.get_coord_manager(
        coords={"time": t_coord, "distance": d_coord},
        dims=_get_dims_from_attrs(node_info.node["RawData"].attrs),
    )
    return ProdMLRawPatchAttrs(**info), coords


@register_func(_NODE_ATTRS_PROCESSORS, key="fbe")
def _get_processed_node_attr_coords(node_info, d_coord, base_info, snap=True):
    """Get information about the processed patches."""
    out = dict(base_info)
    t_coord = _get_time_coord(node_info.parent_node, snap=snap)
    out.update(_get_data_unit_and_type(node_info.node))
    out["dtype"] = str(node_info.node.dtype)
    out["_source_patch_id"] = node_info.name
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
    return ProdMLFbePatchAttrs.from_dict(out), coords


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


def _yield_prodml_attrs_coords(
    fi, extras=None, snap=True
) -> Iterator[tuple[dc.PatchAttrs, dc.CoordManager, str]]:
    """Scan a prodML file, return metadata."""
    acq = fi["Acquisition"]
    # Get the information common to all from root attributes.
    base_info = maybe_get_items(acq.attrs, _ROOT_ATTRS)
    base_info.update(extras if extras is not None else {})
    d_coord = _get_distance_coord(acq)
    # Iterate the raw and processed data and return results in a list.
    for node_info in _yield_data_nodes(fi):
        func = _NODE_ATTRS_PROCESSORS[node_info.patch_type]
        attr, coords = func(node_info, d_coord, base_info, snap=snap)
        yield attr, coords, node_info.name


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


def _read_prodml(fi, distance=None, time=None, source_patch_id=None):
    """Read the prodml values into a patch."""
    out = []
    acq = fi["Acquisition"]
    base_info = maybe_get_items(acq.attrs, _ROOT_ATTRS)
    d_coord = _get_distance_coord(acq)
    source_patch_ids = _normalize_source_patch_ids(source_patch_id)
    for info in _yield_data_nodes(fi):
        if source_patch_ids and info.name not in source_patch_ids:
            continue
        attr_func = _NODE_ATTRS_PROCESSORS[info.patch_type]
        attrs, cm = attr_func(info, d_coord, base_info)
        data = _NODE_DATA_PROCESSORS[info.patch_type](info)
        if time is not None or distance is not None:
            cm, data = cm.select(array=data, time=time, distance=distance)
        if data.size:
            out.append(dc.Patch(data=data, attrs=attrs, coords=cm))
    return out
