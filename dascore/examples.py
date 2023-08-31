"""A module for loading examples."""
from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

import dascore as dc
import dascore.core
from dascore.exceptions import UnknownExample
from dascore.utils.docs import compose_docstring
from dascore.utils.downloader import fetch
from dascore.utils.misc import register_func
from dascore.utils.patch import get_default_patch_name
from dascore.utils.time import to_timedelta64

EXAMPLE_PATCHES = {}
EXAMPLE_SPOOLS = {}


@register_func(EXAMPLE_PATCHES, key="random_das")
def _random_patch(
    *,
    starttime="2017-09-18",
    start_distance=0,
    network="",
    station="",
    tag="random",
    shape=(300, 2_000),
    time_step=to_timedelta64(1 / 250),
    distance_step=1,
    time_array=None,
    dist_array=None,
):
    """Generate a random DAS Patch."""
    # get input data
    rand = np.random.RandomState(13)
    array = rand.random(shape)
    # create attrs
    t1 = np.atleast_1d(np.datetime64(starttime))[0]
    d1 = np.atleast_1d(start_distance)
    attrs = dict(
        distance_step=distance_step,
        time_step=to_timedelta64(time_step),
        category="DAS",
        time_min=t1,
        network=network,
        station=station,
        tag=tag,
        time_units="s",
        distance_units="m",
    )
    # need to pop out dim attrs if coordinates provided.
    if time_array is not None:
        attrs.pop("time_min")
        # need to keep time_step if time_array is len 1 to get coord range
        if len(time_array) > 1:
            attrs.pop("time_step")
    else:
        time_array = dascore.core.get_coord(
            values=t1 + np.arange(array.shape[1]) * attrs["time_step"],
            step=attrs["time_step"],
            units=attrs["time_units"],
        )
    if dist_array is not None:
        attrs.pop("distance_step")
    else:
        dist_array = dascore.core.get_coord(
            values=d1 + np.arange(array.shape[0]) * attrs["distance_step"],
            step=attrs["distance_step"],
            units=attrs["distance_units"],
        )
    coords = dict(distance=dist_array, time=time_array)
    # assemble and output.
    out = dict(data=array, coords=coords, attrs=attrs, dims=("distance", "time"))
    patch = dc.Patch(**out)
    return patch


@register_func(EXAMPLE_PATCHES, key="patch_with_null")
def _patch_with_null():
    """Create a patch which has nullish values."""
    patch = _random_patch()
    data = np.array(patch.data)
    data[data > 0.9] = np.NaN
    # also set the first row and column to NaN
    data[:, 0] = np.NaN
    data[0, :] = np.NaN
    return patch.new(data=data)


@register_func(EXAMPLE_PATCHES, key="wacky_dim_coords_patch")
def _wacky_dim_coord_patch():
    """Creates a patch with one Monotonic and one Array coord."""
    shape = (100, 1_000)
    # distance is neither monotonic nor evenly sampled.
    dist_ar = np.random.random(100) + np.arange(100) * 0.3
    # time is monotonic, not evenly sampled.
    time_ar = dc.to_datetime64(np.cumsum(np.random.random(1_000)))
    patch = _random_patch(shape=shape, dist_array=dist_ar, time_array=time_ar)
    # check attrs
    attrs = patch.attrs
    assert pd.isnull(attrs.coords["time"].step)
    assert pd.isnull(attrs.coords["time"].step)
    return patch


@register_func(EXAMPLE_PATCHES, key="sin_wav")
def _sin_wave_patch(
    sample_rate=44100,
    frequency: Sequence[float] | float = 100.0,
    time_min="2020-01-01",
    channel_count=3,
    duration=1,
    amplitude=10,
):
    """
    Return a Patch composed of simple 1 second sin waves.

    This is useful for debugging output to audio formats.

    Parameters
    ----------
    sample_rate
        The sample rate in Hz.
    frequency
        The frequency of the sin wave.
    time_min
        The start time in the metadata.
    channel_count
        The number of  distance channels to include.
    duration
        Duration of signal in seconds.
    amplitude
        The amplitude of the sin wave.
    """
    t_array = np.linspace(0.0, duration, sample_rate * duration)
    # Get time and distance coords
    distance = np.arange(1, channel_count + 1, 1)
    time = to_timedelta64(t_array) + np.datetime64(time_min)
    freqs = [frequency] if isinstance(frequency, float | int) else frequency
    # init empty data and add frequencies.
    data = np.zeros((len(time), len(distance)))
    for freq in freqs:
        sin_data = amplitude * np.sin(2.0 * np.pi * freq * t_array)
        data += sin_data[..., np.newaxis]
    patch = dc.Patch(
        data=data,
        coords={"time": time, "distance": distance},
        dims=("time", "distance"),
    )
    return patch


@register_func(EXAMPLE_PATCHES, key="random_patch_with_lat_lon")
def _random_patch_lat_lon():
    """Create a patch with latitude/longitude coords on distance dim."""
    random_patch = get_example_patch("random_das")
    dist = random_patch.coords["distance"]
    lat = np.arange(0, len(dist)) * 0.001 - 109.857952
    lon = np.arange(0, len(dist)) * 0.001 + 41.544654
    # add a single coord
    out = random_patch.update_coords(
        latitude=("distance", lat), longitude=("distance", lon)
    )
    return out


@register_func(EXAMPLE_PATCHES, key="example_event_1")
def _example_event_1():
    """
    Returns an example of a passive event recorded by DAS.

    This event is from @stanvek2022fracture.
    """
    path = fetch("example_dasdae_event_1.h5")
    return dc.spool(path)[0]


@register_func(EXAMPLE_SPOOLS, key="random_das")
def _random_spool(
    time_gap=0, length=3, starttime=np.datetime64("2020-01-03"), **kwargs
):
    """
    Generate several random patches in the spool.

    Parameters
    ----------
    time_gap
        The difference in time between each patch. Use a negative
        number to create overlap.
    length
        The number of patches to generate.
    starttime
        The starttime of the first patch. Subsequent patches have startimes
        after the endtime of the previous patch, plus the time_gap.
    **kwargs
        Passed to the [_random_patch](`dasocre.examples._random_patch`) function.
    """
    out = []
    for _ in range(length):
        patch = _random_patch(starttime=starttime, **kwargs)
        out.append(patch)
        diff = to_timedelta64(time_gap) + patch.attrs.coords["time"].step
        starttime = patch.attrs["time_max"] + diff
    return dc.spool(out)


@register_func(EXAMPLE_SPOOLS, key="diverse_das")
def _diverse_spool():
    """
    Create a spool with a diverse set of patches for testing.

    There are various gaps, tags, station names, etc.
    """
    spool_no_gaps = _random_spool()
    spool_no_gaps_different_network = _random_spool(network="das2")
    spool_big_gaps = _random_spool(time_gap=np.timedelta64(1, "s"), station="big_gaps")
    spool_overlaps = _random_spool(
        time_gap=-np.timedelta64(10, "ms"), station="overlaps"
    )
    time_step = spool_big_gaps[0].attrs.coords["time"].step
    dt = to_timedelta64(time_step / np.timedelta64(1, "s"))
    spool_small_gaps = _random_spool(time_gap=dt, station="smallg")
    spool_way_late = _random_spool(
        length=1, starttime=np.datetime64("2030-01-01"), station="wayout"
    )
    spool_new_tag = _random_spool(tag="some_tag", length=1)
    spool_way_early = _random_spool(
        length=1, starttime=np.datetime64("1989-05-04"), station="wayout"
    )

    all_spools = [
        spool_no_gaps,
        spool_no_gaps_different_network,
        spool_big_gaps,
        spool_overlaps,
        spool_small_gaps,
        spool_way_late,
        spool_new_tag,
        spool_way_early,
    ]

    return dc.spool([y for x in all_spools for y in x])


def spool_to_directory(spool, path=None, file_format="DASDAE", extention="hdf5"):
    """
    Write out each patch in a spool to a directory.

    Parameters
    ----------
    spool
        The spool to save to
    path
        The path to the directory, if None, create tempdir.
    file_format
        The file format for the saved files.
    """
    if path is None:
        path = Path(tempfile.mkdtemp())
        assert path.exists()
    for patch in spool:
        out_path = path / (f"{get_default_patch_name(patch)}.{extention}")
        patch.io.write(out_path, file_format=file_format)
    return path


@compose_docstring(examples=", ".join(list(EXAMPLE_PATCHES)))
def get_example_patch(example_name="random_das", **kwargs) -> dc.Patch:
    """
    Load an example Patch.

    Options are:
    {examples}

    Parameters
    ----------
    example_name
        The name of the example to load. Options are:
    **kwargs
        Passed to the corresponding functions to generate data.

    Raises
    ------
        UnknownExample if unregistered patch is requested.
    """
    if example_name not in EXAMPLE_PATCHES:
        msg = (
            f"No example patch registered with name {example_name} "
            f"Registered example patches are {list(EXAMPLE_PATCHES)}"
        )
        raise UnknownExample(msg)
    return EXAMPLE_PATCHES[example_name](**kwargs)


@compose_docstring(examples=", ".join(list(EXAMPLE_SPOOLS)))
def get_example_spool(example_name="random_das", **kwargs) -> dc.BaseSpool:
    """
    Load an example Spool.

    Options are:
    {examples}

    Parameters
    ----------
    example_name
        The name of the example to load. Options are:
    **kwargs
        Passed to the corresponding functions to generate data.

    Raises
    ------
        UnknownExample if unregistered patch is requested.
    """
    if example_name not in EXAMPLE_SPOOLS:
        msg = (
            f"No example spool registered with name {example_name} "
            f"Registered example spools are {list(EXAMPLE_SPOOLS)}"
        )
        raise UnknownExample(msg)
    return EXAMPLE_SPOOLS[example_name](**kwargs)
