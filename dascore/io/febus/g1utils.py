"""
Utilities for working with Febus' G1 DSTS system files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import dascore as dc
from dascore.utils.misc import maybe_get_items, unbyte

_MTX_H5_DATASETS = frozenset(
    {"distances", "end_times", "mtx", "start_times", "temperatures"}
)
_MTX_H5_ATTRS = frozenset({"formatVersion", "freq_offset_abs", "freq_step"})
_MTX_DIMS = ("time", "distance", "frequency")
_MTX_ATTR_MAP = {
    "acq_res": "acq_res",
    "ampliPower": "ampli_power",
    "average": "average",
    "channel": "channel",
    "fiberBreak": "fiber_break",
    "fiberFrom": "fiber_from",
    "fiberLength": "fiber_length",
    "fiberTo": "fiber_to",
    "formatVersion": "format_version",
    "freq_fiber": "freq_fiber",
    "freq_offset": "freq_offset",
    "freq_offset_abs": "freq_offset_abs",
    "freq_ref": "freq_ref",
    "freq_step": "freq_step",
    "mode": "mode",
    "sampling_resolution": "sampling_resolution",
    "signal_size": "signal_size",
    "spatial_resolution": "spatial_resolution",
    "start_time": "start_time",
    "end_time": "end_time",
    "zoneCount": "zone_count",
    "zones": "zones",
    "febusDataKind": "febus_data_kind",
}

param_parse_dict = {
    "start time": lambda x: dc.to_datetime64(float(x[0])),
    "end time": lambda x: dc.to_datetime64(float(x[0])),
    "mode": lambda x: str(x[0]),
    "channel": lambda x: int(x[0]),
}

# Values in the attr dict to not
attr_exclude = frozenset({"start_time", "end_time", "fiberfrom", "fiberto"})


def _make_param_dict(resource):
    """Yield the parameters out from the header in the file."""
    resource.seek(0)  # reset resource position to iterate from start.
    out = {}
    for num, line in enumerate(resource):
        current = line.strip()
        split = current.split(";")
        # We have reached the end of the params.
        if not line.startswith("Param;") or len(split) < 3:
            out["_data_start_line"] = num
            break
        _, name, *vals = current.split(";")
        key_name = name.lower().replace(" ", "_")
        # Handle special parsing or just convert to float
        if name in param_parse_dict:
            out[key_name] = param_parse_dict[name](vals)
        else:
            out[key_name] = float(vals[0])
    suffix = Path(getattr(resource, "name", "")).suffix
    out["_spectra"] = True if suffix == ".mtx" else False
    out["data_units"] = "microstrain" if out["mode"] == "strain" else "celsius"
    return out


def _is_g1_file(resource) -> bool:
    """Get the format tuple for a potential febus G1 file or return False."""
    name = Path(getattr(resource, "name", "")).stem
    if len(split := name.split("_")) != 3:
        return False
    _, _, time_str = split
    try:
        # We only need to know that part of the date string is valid; later
        # we read the time stamps directory.
        _ = dc.to_datetime64(time_str.replace(".", ":")[:19])
    except (ValueError, IndexError):
        return False
    # Next read the first few lines, ensure they start with param;
    first_lines = []
    for _ in range(3):
        line = resource.readline()
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        first_lines.append(line)
    return all((x.startswith("Param;")) for x in first_lines)


def _get_coords(params):
    """Get coordinates from header info."""

    def _get_distance_coord(params):
        """The fiber from/to/sampling defines the length in m"""
        start = params["fiberfrom"]
        end = params["fiberto"]
        step = params["sampling_resolution"]
        return dc.get_coord(start=start, stop=end, step=step, units="m")

    def _get_time_coord(params):
        """The time coord is really just one value."""
        t1 = params["start_time"]
        t2 = params["end_time"]
        step = t2 - t1
        return dc.get_coord(start=t1, stop=t2, step=step)

    spectra = params["_spectra"]
    if spectra:
        msg = "DASCore cannot yet parse spectra Febus G1 files."
        raise NotImplementedError(msg)
    coords = dict(
        distance=_get_distance_coord(params),
        time=_get_time_coord(params),
    )
    return dc.get_coord_manager(coords, dims=tuple(coords))


def _get_attrs(params):
    """Get the attributes from header info."""
    return {i: v for (i, v) in params.items() if i not in attr_exclude}


def _get_g1_coords_and_attrs(resource):
    """Placeholder parser for g1 scan/read paths."""
    resource_name = Path(getattr(resource, "name", "")).stem
    name, _channel, _ = resource_name.split("_")
    params = _make_param_dict(resource)
    params["instrument_id"] = name
    params["data_type"] = params.pop("mode", None)
    params["data_category"] = "DSS"

    coords = _get_coords(params)
    attrs = _get_attrs(params)
    return coords, attrs


def _get_g1_patch(resource, attr_cls):
    """Get a patch from the g1 file."""
    coords, attrs = _get_g1_coords_and_attrs(resource)
    data_start_line = int(attrs.pop("_data_start_line", 0))
    resource.seek(0)
    data = np.loadtxt(resource, skiprows=data_start_line)
    data = np.asarray(data).reshape(coords.shape)
    attrs = attr_cls(**{i: v for i, v in attrs.items() if not i.startswith("_")})
    return dc.Patch(data=data, coords=coords, attrs=attrs)


def _get_mtx_mapped_attrs(resource, mapping):
    """Return mapped MTX attrs, unpacking scalar arrays and decoding bytes."""
    attrs = dict(resource.attrs)
    out = maybe_get_items(attrs, mapping, unpack_names=set(mapping))
    return {key: unbyte(value) for key, value in out.items()}


def _get_mtx_attrs(resource):
    """Return normalized Febus MTX HDF5 attributes."""
    attrs = _get_mtx_mapped_attrs(resource, _MTX_ATTR_MAP)
    data_type = attrs.pop("febus_data_kind", "brillouin_spectrum")
    attrs.update(
        {
            "data_category": "DSS",
            "data_type": data_type,
        }
    )
    return attrs


def _mtx_version(resource) -> str | bool:
    """Return True if the resource looks like a Febus MTX HDF5 file."""
    dataset_names = set(resource)
    has_datasets = _MTX_H5_DATASETS.issubset(dataset_names)
    if not has_datasets or not _MTX_H5_ATTRS.issubset(resource.attrs):
        return False
    attrs = _get_mtx_mapped_attrs(resource, {"formatVersion": "format_version"})
    format_version = str(attrs["format_version"])
    return format_version


def _get_mtx_frequency(resource):
    """Return the Brillouin frequency coordinate."""
    mtx = resource["mtx"]
    freq_len = mtx.shape[-1]
    attrs = _get_mtx_mapped_attrs(
        resource,
        {"freq_offset_abs": "freq_offset_abs", "freq_step": "freq_step"},
    )
    freq_start = float(attrs["freq_offset_abs"])
    freq_step = float(attrs["freq_step"])
    return dc.get_coord(
        start=freq_start,
        step=freq_step,
        shape=freq_len,
        units="MHz",
    )


def _get_mtx_coords(resource, dims=_MTX_DIMS):
    """Return the coordinate manager for a Febus MTX HDF5 file."""
    mtx = resource["mtx"]
    if mtx.ndim != 3:
        msg = f"_get_mtx_coords expected 'mtx' to be 3D, got {mtx.ndim}D."
        raise ValueError(msg)
    time = dc.get_coord(data=dc.to_datetime64(resource["start_times"][...]))
    distance = dc.get_coord(data=resource["distances"][...], units="m")
    frequency = _get_mtx_frequency(resource)
    temperature = dc.get_coord(data=resource["temperatures"][...], units="degC")
    coords = {
        "frequency": frequency,
        "time": time,
        "distance": distance,
        "temperature": ("time", temperature),
    }
    return dc.get_coord_manager(coords, dims=dims)


def _get_mtx_patch(resource, attr_cls, select_kwargs=None):
    """Read a Febus MTX HDF5 file into a patch."""
    select_kwargs = {} if select_kwargs is None else select_kwargs
    data_node = resource["mtx"]
    coords = _get_mtx_coords(resource)
    coords, data = coords.select(array=data_node, **select_kwargs)
    if 0 in coords.shape:  # Empty data; dont return
        return None
    attrs = _get_mtx_attrs(resource)
    data = np.asarray(data)
    return dc.Patch(data=data, coords=coords, attrs=attr_cls(**attrs))
