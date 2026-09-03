"""Utilities for DASVader JLD2 files."""

from __future__ import annotations

import h5py
import numpy as np
from h5py.h5r import Reference, dereference

import dascore as dc
from dascore.core.coords import get_coord
from dascore.exceptions import DASVaderCompatibilityError
from dascore.io.utils import build_patches, drop_blank_attrs
from dascore.utils.misc import maybe_get_items, unbyte

# Julia DateTime "instant" values (Dates.value) are milliseconds since
# 0000-12-31T00:00:00 (Rata Die epoch). Note that Dates.epochms2datetime(0)
# is 0000-01-01, which is *not* the epoch used by Dates.value/JLD2 storage.
# Therefore the Unix epoch offset for JLD2 DateTime integers is 62135683200000 ms.
_JULIA_EPOCH_MS = 62135683200000

attrs_map = {
    "GaugeLength": "gauge_length",
    "Hostname": "interrogator.name",
    "PipelineTracker": "pipeline_tracker",
    "PulseRateFreq": "pulse_rate_frequency",
    "SamplingRate": "sampling_rate",
    "AmpliPower": "amplitude_power",
    "PulseWidth": "pulse_width",
    "FiberLength": "fiber_length",
}

DATA_NAMES = {"data", "strainrate"}
EXPECTED = {"time", "htime", "offset"}


# --- Helpers


def _twice_precision_to_float(tp) -> float:
    """Convert JLD2 TwicePrecision values to a float."""
    return float(tp["hi"] + tp["lo"])


def _step_range_len_to_params(sr):
    """Convert JLD2 StepRangeLen to (start, step, length)."""
    ref = _twice_precision_to_float(sr["ref"])
    step = _twice_precision_to_float(sr["step"])
    length = int(sr["len"])
    offset = int(sr["offset"])
    start = ref + (1 - offset) * step
    return start, step, length


def _julia_ms_to_datetime64(ms_values):
    """Convert Julia DateTime milliseconds to numpy datetime64[ns]."""
    # Here we just prevent infinite loops.
    count = 0
    while isinstance(ms_values, np.void) and count < 10:
        ms_values = ms_values[0]
        count += 1
    unix_ms = np.asarray(ms_values, dtype="int64") - _JULIA_EPOCH_MS
    return dc.to_datetime64(unix_ms / 1_000)


def _raise_legacy_ref_error(h5, field_name: str) -> None:
    """Raise a clear error for legacy DASVader files with anonymous refs."""
    filename = getattr(h5, "filename", "<unknown>")
    version = f"h5py {h5py.__version__} / HDF5 {h5py.version.hdf5_version}"
    msg = (
        f"{filename} is a legacy DASVader JLD2 file with anonymous object "
        f"references in '{field_name}'. This file class is not supported by "
        f"the current HDF5 stack ({version}). Install a compatibility stack "
        f"such as h5py<3.16 with HDF5 1.14.x, then retry."
    )
    raise DASVaderCompatibilityError(msg)


def _dereference(h5, value, field_name: str):
    """Resolve an HDF5 reference or raise a clear compatibility error."""
    if not isinstance(value, Reference):
        return value
    try:
        return h5[value]
    except KeyError:
        # The high-level lookup fails for some references HDF5 can still
        # resolve directly, so try that before giving up.
        try:
            return h5py.Dataset(dereference(value, h5.id))
        except Exception:
            _raise_legacy_ref_error(h5, field_name)


# --- Metadata parsing


def _dataset_to_dict(atrib) -> dict:
    """
    Convert the compound dataset to a python dict.
    """

    def _resolve_ref(h5, value, field_name):
        """Resolve h5py references to concrete values."""
        if not isinstance(value, Reference):
            return value
        obj = _dereference(h5, value, field_name)
        out = obj[()]
        if isinstance(out, np.ndarray) and out.size == 1:
            return out.item()
        return out

    data = atrib[()]
    assert isinstance(data, np.void)
    h5 = atrib.file
    out = {}
    for name in data.dtype.names:
        val = _resolve_ref(h5, data[name], name)
        out[name] = unbyte(val)
    return out


def _get_attr_dict(atrib) -> dict:
    """Map DASVader attrib values to PatchAttrs fields."""
    attrs = _dataset_to_dict(atrib)
    attrs = maybe_get_items(attrs, attrs_map)
    drop_blank_attrs(attrs, ("interrogator.name",))
    attrs["data_category"] = "DAS"
    attrs["data_units"] = "nanostrain"
    return attrs


def _get_time_coord(h5, rec):
    """Build the time coordinate from htime or time struct."""
    time_struct = rec["time"]
    htime_node = _dereference(h5, rec["htime"], "htime")
    start_htime = _julia_ms_to_datetime64(htime_node[0])
    _start, _step, time_len = _step_range_len_to_params(time_struct)
    _end = _start + time_len * _step

    out = dc.get_coord(
        min=start_htime + dc.to_timedelta64(_start),
        max=start_htime + dc.to_timedelta64(_end),
        step=dc.to_timedelta64(_step),
    )
    return out


def _get_distance_coord(rec):
    """Build the distance coordinate from offset struct."""
    offset_struct = rec["offset"]
    dist_start, dist_step, dist_len = _step_range_len_to_params(offset_struct)
    return get_coord(start=dist_start, step=dist_step, shape=dist_len)


def _get_coord_manager(h5, rec):
    """Get the coordinate manager for the contained patch."""
    names = set(_get_reference_names(h5))
    time = _get_time_coord(h5, rec)
    dist = _get_distance_coord(rec)
    # The data axis is transposed based on if they are stored in "strainrate"
    # or "data" reference.
    return dc.get_coord_manager(
        {"time": time, "distance": dist},
        dims=_get_dims(names),
    )


def _get_dims(names) -> tuple[str, str]:
    """The stored dimension order, which the data reference's name says."""
    # The data axis is transposed based on if they are stored in
    # "strainrate" or "data" reference.
    return ("distance", "time") if "data" in names else ("time", "distance")


def _get_data_and_dims(h5, rec=None):
    """Resolve the data reference to its dataset, with its dimension order."""
    rec = h5["dDAS"][()] if rec is None else rec
    names = set(_get_reference_names(h5))
    data_name = "data" if "data" in names else next(iter(DATA_NAMES & names))
    return _dereference(h5, rec[data_name], data_name), _get_dims(names)


# --- Reading


def _get_reference_names(h5):
    """Return field names from ``h5.get("dDAS").dtype.names``.

    Returns a tuple of field names, or ``None`` if the dtype has no named
    fields.
    """
    return h5.get("dDAS").dtype.names


def _is_dasvader_jld2(h5) -> bool:
    """Return True if file contains DASVader JLD2 data."""
    try:
        dtype_names = _get_reference_names(h5)
    except AttributeError:
        return False
    # Certain refs that all dasvader files have.
    has_expected = EXPECTED.issubset(set(dtype_names))
    # Data name can change. Coerced to a bool so an empty intersection
    # returns False rather than the empty set the `and` would hand back.
    has_data = bool(DATA_NAMES & set(dtype_names))
    return has_data and has_expected


def _read_dasvader(h5, distance=None, time=None):
    """Read DASVader data into a Patch."""
    rec = h5["dDAS"][()]
    cm = _get_coord_manager(h5, rec)
    ref_names = set(_get_reference_names(h5))
    data, _ = _get_data_and_dims(h5, rec)
    attrs = (
        _get_attr_dict(_dereference(h5, rec["atrib"], "atrib"))
        if "atrib" in ref_names
        else {}
    )
    return build_patches(
        cm, data, attrs, selection={"time": time, "distance": distance}
    )
