"""Utilities for DASVader JLD2 files."""

from __future__ import annotations

import numpy as np
from h5py.h5r import Reference

import dascore as dc
from dascore.compat import array
from dascore.core.coords import get_coord
from dascore.utils.misc import maybe_get_items, unbyte

# Julia DateTime "instant" values (Dates.value) are milliseconds since
# 0000-12-31T00:00:00 (Rata Die epoch). Note that Dates.epochms2datetime(0)
# is 0000-01-01, which is *not* the epoch used by Dates.value/JLD2 storage.
# Therefore the Unix epoch offset for JLD2 DateTime integers is 62135683200000 ms.
_JULIA_EPOCH_MS = 62135683200000

attrs_map = {
    "GaugeLength": "gauge_length",
    "Hostname": "host_name",
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


# --- Metadata parsing


def _dataset_to_dict(atrib) -> dict:
    """
    Convert the compound dataset to a python dict.
    """

    def _resolve_ref(h5, value):
        """Resolve h5py references to concrete values."""
        if not isinstance(value, Reference):
            return value
        obj = h5[value]
        out = obj[()]
        if isinstance(out, np.ndarray) and out.size == 1:
            return out.item()
        return out

    data = atrib[()]
    assert isinstance(data, np.void)
    h5 = atrib.file
    out = {}
    for name in data.dtype.names:
        val = _resolve_ref(h5, data[name])
        out[name] = unbyte(val)
    return out


def _get_attr_dict(atrib) -> dict:
    """Map DASVader attrib values to PatchAttrs fields."""
    attrs = _dataset_to_dict(atrib)
    attrs = maybe_get_items(attrs, attrs_map)
    attrs["data_category"] = "DAS"
    return attrs


def _get_time_coord(h5, rec):
    """Build the time coordinate from htime or time struct."""
    time_struct = rec["time"]
    htime_node = h5[rec["htime"]]
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
    dims = ("distance", "time") if "data" in names else ("time", "distance")
    return dc.get_coord_manager(
        {"time": time, "distance": dist},
        dims=dims,
    )


# --- Reading


def _get_reference_names(h5):
    """Get a set of the reference names."""
    return h5.get("dDAS").dtype.names


def _is_dasvader_jld2(h5) -> bool:
    """Return True if file contains DASVader JLD2 data."""
    try:
        dtype_names = _get_reference_names(h5)
    except AttributeError:
        return False
    # Certain refs that all dasvader files have.
    has_expected = EXPECTED.issubset(set(dtype_names))
    # Data name can change.
    has_data = DATA_NAMES & set(dtype_names)
    return has_data and has_expected


def _read_dasvader(h5, distance=None, time=None):
    """Read DASVader data into a Patch."""
    rec = h5["dDAS"][()]
    cm = _get_coord_manager(h5, rec)
    # First, figure out what the data name is (this changes in different
    # dasvader iterations), but if we get to here it has at least one.
    ref_names = set(_get_reference_names(h5))
    data_name = "data" if "data" in ref_names else next(iter(DATA_NAMES & ref_names))
    # data is a reference here; need to resolve it with h5 File.
    data = h5[rec[data_name]]
    if distance is not None or time is not None:
        cm, data = cm.select(data, distance=distance, time=time)
    data = array(data)
    if not data.size:
        return []
    attrs = _get_attr_dict(h5[rec["atrib"]]) if "atrib" in ref_names else {}
    # attrs["coords"] = cm.to_summary_dict()
    # attrs["dims"] = cm.dims
    return [dc.Patch(data=data, coords=cm, attrs=dc.PatchAttrs(**attrs))]
