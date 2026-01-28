"""Utilities for Febus."""

from __future__ import annotations

from collections import namedtuple
from functools import cache

import numpy as np

import dascore as dc
from dascore.core import get_coord, get_coord_manager
from dascore.core.coordmanager import CoordManager
from dascore.utils.misc import (
    _maybe_unpack,
    broadcast_for_index,
    maybe_get_items,
    tukey_fence,
    unbyte,
)

# --- Getting format/version

_FebusSlice = namedtuple(
    "FebusSlice",
    ["group", "group_name", "source", "source_name", "zone", "zone_name", "data_name"],
)


_FebusTime = namedtuple(
    "FebusTime",
    ["block_time", "time_step", "idx_start", "idx_stop"],
)


def _get_zone_time(feb):
    """
    Attempt to get time information for the current zone.

    The files are very inconsistent accross versions, so, to try to support
    as many febus files as possible, this function does a lot of heavy lifting.

    Danger: Here be dragons.
    """

    def get_data_index(block_pad, block_no_pad) -> tuple[int, int]:
        """
        Compute the index of data, removing redundancy.

        # Note: This function is based on a similar function in Febus'
        febus_optics_lib.
        """
        to_remove = block_pad - block_no_pad
        # Need to handle even/odd cases to see which samples to remove.
        assert to_remove >= 0
        qotient, reminder = divmod(to_remove, 2)
        start = int(max(0, qotient))
        end = int(block_pad - 1 - (reminder + qotient))
        return start, end

    zone = feb.zone
    block_time = _get_block_time(feb)
    extents, spacing = zone.attrs["Extent"], zone.attrs["Spacing"]
    overlap_attr = zone.attrs.get("Overlap", zone.attrs.get("BlockOverlap", 0))
    overlap = np.atleast_1d(_maybe_unpack(overlap_attr))[0]
    shape = feb.zone[feb.data_name].shape
    # We need to determine if this a v1 file (no version in attrs). See # 589
    # and # 587. This could perhaps be made more robust in the future.
    has_version = "Version" in feb.zone.attrs
    # When the file has a version, the spacing can be trusted, otherise
    # use spatial sampling.
    if has_version:
        block_pad = 1 + extents[3] - extents[2]
        dt = spacing[1] / 1_000 if block_pad != 0 else block_time
    else:
        # In these versions of the files the extents appear to be wrong, but
        # they don't have overlaps so we can just use the shape.
        dt = 1 / float(_maybe_unpack(zone.attrs["SamplingRate"]))
        block_pad = shape[1]
    # Apparently, if the extents are set to 0 the overlapping edges are still
    # in the file, otherwise they have been removed.
    # This does not, however, mean the block dimension match the actual
    # data length. We need to handle that issue separately.
    assert block_pad > 1
    overlaps_removed = extents[2] != 0
    if overlaps_removed:
        block_no_pad = int(round(block_time / dt))
    else:
        block_no_pad = int(round(block_pad / (1 + (overlap / 100)), 0))
    # Perform checks to make sure this is DAS data. If not, you need to use
    # the Febus parser. Just assert for now.
    missing_gauge = feb.zone.attrs.get("GaugeLength", None) is None
    flat = shape == 1
    msg = (
        "Complex Febus file found. Either contact the DASCore developers or "
        "use the python library made by Febus."
    )
    assert not (missing_gauge or flat), msg
    # Next determine where the data actually live.
    idx_start, idx_stop = get_data_index(
        block_pad,
        block_no_pad,
    )
    return _FebusTime(block_time, dt, idx_start, idx_stop)


@cache
def _get_block_time(feb):
    """Get the block time (time in seconds between each block)."""
    # Some files have this set. We haven't yet seen any files where this
    # values exists and is wrong, so we trust it (for now). This is probably
    # much faster than reading the whole time vector.
    br = _maybe_unpack(feb.zone.attrs.get("BlockRate", 0) / 1_000)
    if br > 0:
        return float(1 / br)
    # Otherwise we have to try to use the time vector. Here be dragons.
    time_shape = feb.source["time"].shape
    # Not sure why but time has the shape of [1, n] for some files and just
    # n for others. The first might imply different times for different
    # zones? We aren't set up to handle that, but we don't know if it can happen
    # so just assert here.
    assert np.max(time_shape) == np.prod(
        time_shape
    ), "Non flat 2d time vector is not supported by DASCore Febus reader."
    # Get the average time spacing in each block. These can vary a bit so
    # account for outliers.
    time = np.squeeze(feb.source["time"][:])
    d_time = time[1:] - time[:-1]
    tmin, tmax = tukey_fence(d_time)
    d_time = d_time[(d_time >= tmin) & (d_time <= tmax)]
    # After removing outliers, the mean seems to work better than the median
    # for the test files we have. There is still a concerning amount of
    # variability.
    return float(_maybe_unpack(np.mean(d_time)))


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


def _get_time_coord(feb):
    """Get the time coordinate contained in the febus slice."""
    time = feb.source["time"]
    # In older version time shape is different, always grab first element.
    first_slice = tuple(0 for _ in time.shape)
    t_0 = time[first_slice]
    # Number of time blocks in the data cube.
    shape = feb.zone[feb.data_name].shape
    num_blocks = shape[0]
    time_info = _get_zone_time(feb)
    time_step = time_info.time_step
    total_time_rows = time_info.idx_stop - time_info.idx_start + 1
    # Get origin info, these are offsets from time to get to the first simple
    # of the block. These should always be non-positive.
    time_offset = feb.zone.attrs["Origin"][1] / 1_000  # also convert to s
    assert time_offset <= 0, "time offset must be non positive"
    # Create time coord.
    # Need to account for removing overlap times. Also, time vector refers
    # to the center of the block, so this finds the first non-overlapping
    # sample.
    total_start = t_0 + time_offset + time_info.idx_start * time_step
    total_end = total_start + (total_time_rows * time_step) * num_blocks
    time_coord = get_coord(
        start=dc.to_datetime64(total_start),
        stop=dc.to_datetime64(total_end),
        step=dc.to_timedelta64(time_step),
    )
    # Note: we have found some files in which the sampling rate is 1/3e-4
    # because we use datetime64 we lose some precision which has caused
    # slight differences in shape of the patch.
    out = time_coord.change_length(total_time_rows * num_blocks)
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
        # get start/stop indexes for complete blocks
        start = np.argmax(in_time)
        stop = np.argmax(np.cumsum(in_time)) + (1 if len(times) else 0)
        total_slice[0] = slice(start, stop)
        # load data from disk.
        data_2d = data[tuple(total_slice)].reshape(-1, data.shape[-1])
        # Bail out early, no size on array.
        if not data_2d.size:
            return data_2d, time_coord.empty()
        # Next, get mew time coord and slice.
        tmin = times[:, 0].min()
        tmax = times[:, 1].max()
        new_coord, time_slice = (
            get_coord(min=tmin, max=tmax, step=time_coord.step)
            .change_length(len(data_2d))
            .select((t1, t2))
        )
        return data_2d[time_slice], new_coord

    time_info = _get_zone_time(febus)
    dist_coord, time_coord = cm.coord_map["distance"], cm.coord_map["time"]
    data = febus.zone[febus.data_name]
    data_shape = data.shape
    data_slice = slice(time_info.idx_start, time_info.idx_stop + 1)
    total_slice = list(broadcast_for_index(3, 1, data_slice))
    total_time_rows = time_info.idx_stop - time_info.idx_start + 1
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
        # Distance has been selected out (no distance remains)
        if not len(dist_coord):
            data = np.zeros((len(time_coord), len(dist_coord)), dtype=data_3d.dtype)
        else:
            data = data_3d.reshape(-1, data_3d.shape[2])
    cm = get_coord_manager({"time": time_coord, "distance": dist_coord}, dims=cm.dims)
    return data, cm


def _read_febus(fi, distance=None, time=None, attr_cls=dc.PatchAttrs):
    """Read the febus values into a patch."""
    out = []
    for attr, cm, febus in _yield_attrs_coords(fi):
        data, new_cm = _get_data_new_cm(cm, febus, distance=distance, time=time)
        if data.size:
            patch = dc.Patch(data=data, coords=new_cm, attrs=attr_cls(**attr))
            out.append(patch)
    return out
