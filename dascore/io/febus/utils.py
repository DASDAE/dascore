"""Utilities for Febus."""

from __future__ import annotations

import warnings
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


@cache
def _get_sample_spacing(feb: _FebusSlice, n_time_samps: int):
    """
    Determine the temporal sample spacing (in seconds).
    """
    # Note: This is a bit dicey, but we are trying to account for variability
    # seen in real Febus Files. In some files the zone Spacing attr indicates one
    # sample rate, while zone.attrs['SamplingRate'] indicates another. It
    # varies as to which one is actually right, so we try to figure that
    # out here.
    ts_1 = feb.zone.attrs["Spacing"][1] / 1_000  # value in ms, convert to s.
    # In most cases sample_rate is either bogus or in Hz. It isn't even mentioned
    # in some Febus documentation.
    ts_2 = _maybe_unpack(1.0 / feb.zone.attrs["SamplingRate"])
    # Get the block time. This doesn't account for overlap, so it wont be exact.
    block_time = _get_block_time(feb)
    # Get candidate times, return the closet to the block_time.
    ts_array = np.array([ts_1, ts_2])
    block_time_array = ts_array * n_time_samps
    return ts_array[np.argmin(np.abs(block_time_array - block_time))]


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


def _get_time_overlap_samples(feb, n_time_samps, tstep=None):
    """Determine the number of redundant samples in the time dimension."""
    tstep = tstep if tstep is not None else _get_sample_spacing(feb, n_time_samps)
    block_time = _get_block_time(feb)
    # Since the data have overlaps in each block's time dimension, we need to
    # trim the overlap off the time dimension to avoid having to merge blocks later.
    # However, sometimes the "BlockOverlap" is wrong, so we calculate it
    # manually here, rounding to nearest even number.
    expected_samples = int(np.round((block_time / tstep) / 2) * 2)
    excess_rows = n_time_samps - expected_samples
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
    # Number of time blocks in the data cube.
    shape = feb.zone[feb.data_name].shape
    n_time_samps = shape[1]
    n_blocks = shape[0]
    # Get spacing between time samples (in s) and the total time of each block.
    time_step = _get_sample_spacing(feb, n_time_samps)
    excess_rows = _get_time_overlap_samples(feb, n_time_samps, tstep=time_step)
    total_time_rows = (n_time_samps - excess_rows) * n_blocks
    # Get origin info, these are offsets from time to get to the first simple
    # of the block. These should always be non-positive.
    time_offset = feb.zone.attrs["Origin"][1] / 1_000  # also convert to s
    assert time_offset <= 0, "time offset must be non positive"
    # Get the start/stop indices for the zone. We assume zones never sub-slice
    # time (only distance). However, some files (eg Valencia) have an incorrect
    # value set here, so we only warn.
    extent = feb.zone.attrs["Extent"]
    if (extent[3] - extent[2] + 1) != n_time_samps:
        msg = (
            "It appears the Febus file extents specify a different range than "
            "found in the data array. Double check this is correct."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
    # Create time coord.
    # Need to account for removing overlap times. Also, time vector refers
    # to the center of the block, so this finds the first non-overlapping
    # sample.
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

    dist_coord, time_coord = cm.coord_map["distance"], cm.coord_map["time"]
    data = febus.zone[febus.data_name]
    data_shape = data.shape
    skip_rows = _get_time_overlap_samples(febus, data_shape[1]) // 2
    # This handles the case where excess_rows == 0
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
