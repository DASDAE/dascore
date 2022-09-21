"""
A small module for loading examples.
"""
import tempfile
from pathlib import Path

import numpy as np

import dascore as dc
from dascore.utils.misc import register_func
from dascore.utils.patch import get_default_patch_name
from dascore.utils.time import to_datetime64, to_timedelta64

EXAMPLE_PATCHES = {}
EXAMPLE_SPOOLS = {}


def get_example_patch(example_name="random_das", **kwargs):
    """
    Load an example Patch.

    kwargs are passed to the corresponding functions to generate data.
    """
    return EXAMPLE_PATCHES[example_name](**kwargs)


def get_example_spool(example_name="random_das", **kwargs):
    """
    Load an example spool.

    kwargs are passed to the corresponding functions to generate data.
    """
    return EXAMPLE_SPOOLS[example_name](**kwargs)


@register_func(EXAMPLE_PATCHES, key="random_das")
def _random_patch(starttime="2017-09-18", network="", station="", tag="random"):
    """Generate a random DAS Patch"""
    rand = np.random.RandomState(13)
    array = rand.random(size=(300, 2_000))
    t1 = np.datetime64(starttime)
    attrs = dict(
        d_distance=1,
        d_time=to_timedelta64(1 / 250),
        category="DAS",
        id="test_data1",
        time_min=t1,
        network=network,
        station=station,
        tag=tag,
    )
    coords = dict(
        distance=np.arange(array.shape[0]) * attrs["d_distance"],
        time=np.arange(array.shape[1]) * attrs["d_time"],
    )
    out = dict(data=array, coords=coords, attrs=attrs)
    return dc.Patch(**out)


@register_func(EXAMPLE_PATCHES, key="sin_wav")
def sin_wave_patch(
    sample_rate=44100,
    frequency=100,
    time_min="2020-01-01",
    channel_count=3,
    duration=1,
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

    """
    t_array = np.linspace(0.0, duration, sample_rate * duration)
    sin_data = 10 * np.sin(2.0 * np.pi * frequency * t_array)
    data = np.stack([sin_data] * channel_count).T
    time = to_timedelta64(t_array) + np.datetime64(time_min)
    distance = np.arange(1, channel_count + 1, 1)

    patch = dc.Patch(
        data=data,
        coords={"time": time, "distance": distance},
        dims=("time", "distance"),
        attrs={
            "time_min": to_datetime64(time_min),
            "d_time": 1 / sample_rate,
            "distance_min": 1,
            "d_distance": 1,
            "distance_max": 3,
        },
    )
    return patch


@register_func(EXAMPLE_SPOOLS, key="random_das")
def _random_spool(
    d_time=0,
    length=3,
    starttime=np.datetime64("2020-01-03"),
    network="",
    station="",
    tag="random",
):
    """
    Generate several random patches in the spool.

    Parameters
    ----------
    d_time
        The difference in time between each patch.
    length
    """
    out = []
    for _ in range(length):
        patch = _random_patch(
            starttime=starttime, network=network, station=station, tag=tag
        )
        out.append(patch)
        starttime = patch.attrs["time_max"] + to_timedelta64(d_time)
    return dc.spool(out)


@register_func(EXAMPLE_SPOOLS, key="diverse_das")
def _diverse_spool():
    """
    Create a spool with a diverse set of patches for testing.

    There are various gaps, tags, station names, etc.
    """
    spool_no_gaps = _random_spool()
    spool_no_gaps_different_network = _random_spool(network="das2")
    spool_big_gaps = _random_spool(d_time=np.timedelta64(1, "s"), station="big_gaps")
    spool_overlaps = _random_spool(d_time=-np.timedelta64(10, "ms"), station="overlaps")
    dt = to_timedelta64(spool_big_gaps[0].attrs["d_time"] / np.timedelta64(1, "s"))
    spool_small_gaps = _random_spool(d_time=dt, station="small_gaps")
    spool_way_late = _random_spool(
        length=1, starttime=np.datetime64("2030-01-01"), station="way_out"
    )
    spool_new_tag = _random_spool(tag="some_tag", length=1)
    spool_way_early = _random_spool(
        length=1, starttime=np.datetime64("1989-05-04"), station="way_out"
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
