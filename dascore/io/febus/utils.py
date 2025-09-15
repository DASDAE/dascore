"""Utilities for Febus."""

from __future__ import annotations

from collections import namedtuple

import numpy as np

import dascore as dc
from dascore.core import get_coord, get_coord_manager
from dascore.core.coordmanager import CoordManager
from dascore.utils.misc import (
    _maybe_unpack,
    broadcast_for_index,
    maybe_get_items,
    unbyte,
)

# --- Getting format/version

_FebusSlice = namedtuple(
    "FebusSlice",
    ["group", "group_name", "source", "source_name", "zone", "zone_name", "data_name"],
)


def _flatten_febus_info(fi) -> tuple[_FebusSlice, ...]:
    """
    Given a febus file, return a tuple of named tuples with key info.

    This flattens the iteration nesting to a single layer.
    """
    out = []
    for group_name, group in fi.items():
        for source_name, source in group.items():
            for zone_name, zone in source.items():
                # Skip time dataset (we only want zone groups).
                if zone_name == "time":
                    continue
                # get dataset name (not always StrainRate for older data)
                possible_ds_names = list(zone.keys())
                assert len(possible_ds_names) == 1
                data_name = possible_ds_names[0]
                zlice = _FebusSlice(
                    group, group_name, source, source_name, zone, zone_name, data_name
                )
                out.append(zlice)
    return tuple(out)


def _get_febus_version_str(hdf_fi) -> str:
    """Return the version string for febus file."""
    # Define a few root attrs that act as a "fingerprint"
    # all Febus DAS files have folders that start with fa (I hope).
    # Edit: They do not. I have simply removed this requirement (#525).
    inst_keys = sorted(hdf_fi.keys())
    expected_source_attrs = {
        "AmpliPower",
        "Hostname",
        "WholeExtent",
        "SamplingRate",
    }
    is_febus = True
    # Version 1, or what I think is version one (eg Valencia PubDAS data)
    # did not include a Version attr in Source dataset, so we use that as
    # the default.
    version = "1"
    for inst_key in inst_keys:
        inst = hdf_fi[inst_key]
        source_keys = set(inst.keys())
        is_febus = is_febus and all(x.startswith("Source") for x in source_keys)
        for source_key in source_keys:
            source = inst[source_key]
            # If the version is set in a Source use that version.
            # Hopefully this is the file version...
            version = unbyte(source.attrs.get("Version", version)).split(".")[0]
            is_febus = is_febus and expected_source_attrs.issubset(set(source.attrs))
    if inst_keys and is_febus:
        return version
    return ""


def _get_febus_attrs(feb: _FebusSlice) -> dict:
    """Get non-coordinate attrs from febus slice."""
    zone_attrs = feb.zone.attrs
    attr_mapping = {
        "GaugeLength": "gauge_length",
        "PulseWidth": "pulse_width",
        "Version": "folog_a1_software_version",
    }
    out = maybe_get_items(zone_attrs, attr_mapping, unpack_names=set(attr_mapping))
    out["group"] = feb.group_name
    out["source"] = feb.source_name
    out["zone"] = feb.zone_name
    out["schema_version"] = out.get("folog_a1_software_version", "").split(".")[0]
    out["dims"] = ("time", "distance")
    return out


def _get_time_overlap_samples(feb, data_shape):
    """Determine the number of redundant samples in the time dimension."""
    time_step = feb.zone.attrs["Spacing"][1] / 1_000  # value in ms, convert to s.
    block_time = _maybe_unpack(1 / (feb.zone.attrs["BlockRate"] / 1_000))
    # Since the data have overlaps in each block's time dimension, we need to
    # trim the overlap off the time dimension to avoid having to merge blocks later.
    # However, sometimes the "BlockOverlap" is wrong, so we calculate it
    # manually here.
    expected_samples = int(np.round(block_time / time_step))
    excess_rows = data_shape[1] - expected_samples
    assert (
        excess_rows % 2 == 0
    ), "excess rows must be symmetric to distribute on both ends"
    return excess_rows


def _get_time_coord(feb):
    """Get the time coordinate contained in the febus slice."""
    time = feb.source["time"]
    # In older version time shape is different, always grab first element.
    first_slice = tuple(0 for _ in time.shape)
    t_0 = time[first_slice]
    # Data dimensions are block_index, time, distance
    data_shape = feb.zone[feb.data_name].shape
    n_blocks = data_shape[0]
    # Get spacing between time samples (in s) and the total time of each block.
    time_step = feb.zone.attrs["Spacing"][1] / 1_000  # value in ms, convert to s.
    excess_rows = _get_time_overlap_samples(feb, data_shape)
    total_time_rows = (data_shape[1] - excess_rows) * n_blocks
    # Get origin info, these are offsets from time to get to the first simple
    # of the block. These should always be non-positive.
    time_offset = feb.zone.attrs["Origin"][1] / 1_000  # also convert to s
    assert time_offset <= 0, "time offset must be non positive"
    # Get the start/stop indices for the zone. We assume zones never sub-slice
    # time (only distance) but assert that here.
    extent = feb.zone.attrs["Extent"]
    assert (extent[3] - extent[2] + 1) == data_shape[1], "Cant handle sub time zones"
    # Create time coord
    # Need to account for removing overlap times.
    total_start = t_0 + time_offset + (excess_rows // 2) * time_step
    total_end = total_start + total_time_rows * time_step
    time_coord = get_coord(
        start=dc.to_datetime64(total_start),
        stop=dc.to_datetime64(total_end),
        step=dc.to_timedelta64(time_step),
    )
    # Note: we have found some files in which the sampling rate is 1/3e-4
    # because we use datetime64 we lose some precision which has caused
    # slight differences in shape of the patch.
    out = time_coord.change_length(total_time_rows)
    return out


def _get_distance_coord(feb):
    """Get the distance coordinate associated with febus slice."""
    data_shape = feb.zone[feb.data_name].shape
    total_distance_inds = data_shape[2]
    # Get spacing between channels (in m)
    distance_step = feb.zone.attrs["Spacing"][0]
    # Get origin info, these are absolute for distance.
    distance_origin = feb.zone.attrs["Origin"][0]
    # Get the start/stop indices for the zone
    extent = feb.zone.attrs["Extent"]
    dist_ids = (extent[0], extent[1])
    # Create distance coord
    # Need to account for removing overlap times.
    start = dist_ids[0] * distance_step + distance_origin
    stop = start + total_distance_inds * distance_step
    dist_coord = get_coord(
        start=start,
        stop=stop,
        step=distance_step,
        units="m",
    )
    return dist_coord.change_length(total_distance_inds)


def _get_febus_coord_manager(feb: _FebusSlice) -> CoordManager:
    """Get a coordinate manager for febus slice."""
    coords = dict(
        time=_get_time_coord(feb),
        distance=_get_distance_coord(feb),
    )
    cm = get_coord_manager(coords=coords, dims=("time", "distance"))
    return cm


def _yield_attrs_coords(fi) -> tuple[dict, CoordManager]:
    """Scan a febus file, return metadata."""
    febuses = _flatten_febus_info(fi)
    for febus in febuses:
        attr = _get_febus_attrs(febus)
        cm = _get_febus_coord_manager(febus)
        yield attr, cm, febus


def _get_data_new_cm(cm, febus, distance=None, time=None):
    """
    Get the data from febus file, maybe filtering on time/distance.

    This is a bit more complicated since the febus data are stored in a 3d array,
    but we want a 2d output.
    """

    def _get_start_end_time_array(time_coord, total_time_rows, data_shape, time):
        """Get a 2d array where columns are start/end times for each block."""
        block_count = data_shape[0]
        block_duration = total_time_rows * time_coord.step
        start = (
            np.arange(block_count) * block_duration + time_coord.step
        ) + time_coord.min()
        end = start + block_duration
        return np.stack([start, end], axis=-1)

    def _get_time_filtered_data(data, t_start_end, time, total_slice, time_coord):
        """Get new data array filtered from time query."""
        assert len(time) == 2
        t1, t2 = time
        # block for which all data are needed.
        in_time = np.ones(len(t_start_end), dtype=bool)
        if t1 is not None and t1 is not ...:
            in_time = np.logical_and(in_time, ~(t_start_end[:, 1] < t1))
        if t2 is not None and t2 is not ...:
            in_time = np.logical_and(in_time, ~(t_start_end[:, 0] > t2))
        times = t_start_end[in_time]
        tmin = times[:, 0].min()
        tmax = times[:, 1].max()
        # get start/stop indexes for complete blocks
        start = np.argmax(in_time)
        stop = np.argmax(np.cumsum(in_time))
        total_slice[0] = slice(start, stop)
        # load data from disk.
        data_2d = data[tuple(total_slice)].reshape(-1, data.shape[-1])
        # Next, get mew time coord and slice.
        new_coord, time_slice = (
            get_coord(min=tmin, max=tmax, step=time_coord.step)
            .change_length(len(data_2d))
            .select((t1, t2))
        )
        return data_2d[time_slice], new_coord

    dist_coord, time_coord = cm.coord_map["distance"], cm.coord_map["time"]
    data = febus.zone[febus.data_name]
    data_shape = data.shape
    skip_rows = _get_time_overlap_samples(febus, data_shape) // 2
    # Need to handle case where excess_rows == 0
    data_slice = slice(skip_rows, -skip_rows if skip_rows else None)
    total_slice = list(broadcast_for_index(3, 1, data_slice))
    total_time_rows = data_shape[1] - 2 * skip_rows
    if distance:
        dist_coord, total_slice[2] = dist_coord.select(distance)
    if time:  # need to sub-select blocks to get data we are after.
        t_start_end = _get_start_end_time_array(
            time_coord, total_time_rows, data_shape, time
        )
        data, time_coord = _get_time_filtered_data(
            data, t_start_end, time, total_slice, time_coord
        )
    else:  # no need to mess with blocks, all time is selected
        data_3d = data[tuple(total_slice)]
        data = data_3d.reshape(-1, data_3d.shape[2])
    cm = get_coord_manager({"time": time_coord, "distance": dist_coord}, dims=cm.dims)
    return data, cm


def _read_febus(fi, distance=None, time=None, attr_cls=dc.PatchAttrs):
    """Read the febus values into a patch."""
    out = []
    for attr, cm, febus in _yield_attrs_coords(fi):
        data, new_cm = _get_data_new_cm(cm, febus, distance=distance, time=time)
        patch = dc.Patch(data=data, coords=new_cm, attrs=attr_cls(**attr))
        out.append(patch)
    return out
