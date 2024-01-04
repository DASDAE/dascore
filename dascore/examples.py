"""A module for loading examples."""
from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

import dascore as dc
import dascore.core
from dascore.exceptions import UnknownExampleError
from dascore.utils.docs import compose_docstring
from dascore.utils.downloader import fetch
from dascore.utils.misc import register_func
from dascore.utils.patch import get_default_patch_name
from dascore.utils.time import to_timedelta64

EXAMPLE_PATCHES = {}
EXAMPLE_SPOOLS = {}


@register_func(EXAMPLE_PATCHES, key="random_das")
def random_patch(
    *,
    time_min="2017-09-18",
    time_step=to_timedelta64(1 / 250),
    time_array=None,
    distance_min=0,
    distance_step=1,
    dist_array=None,
    network="",
    station="",
    tag="random",
    shape=(300, 2_000),
):
    """
    Generate a random DAS Patch.

    Parameters
    ----------
    time_min
        The time the patch starts.
    time_step
        The step between time samples.
    time_array
        If not None, an array for time coordinate and`time_min` and
        `time_step` will not be used.
    distance_min
        The start of the distance coordinate.
    distance_step
        The spacing between distance samples.
    dist_array
        If not None, an array of distance values and `distance_min` and
        `distance_step` will not be used.
    network
        The network code.
    station
        The station designation.
    tag
        The patch tag
    shape
        The shape pf data array.
    """
    # get input data
    rand = np.random.RandomState(13)
    array = rand.random(shape)
    # create attrs
    t1 = np.atleast_1d(np.datetime64(time_min))[0]
    d1 = np.atleast_1d(distance_min)
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
            data=t1 + np.arange(array.shape[1]) * attrs["time_step"],
            step=attrs["time_step"],
            units=attrs["time_units"],
        )
    if dist_array is not None:
        attrs.pop("distance_step")
    else:
        dist_array = dascore.core.get_coord(
            data=d1 + np.arange(array.shape[0]) * attrs["distance_step"],
            step=attrs["distance_step"],
            units=attrs["distance_units"],
        )
    coords = dict(distance=dist_array, time=time_array)
    # assemble and output.
    out = dict(data=array, coords=coords, attrs=attrs, dims=("distance", "time"))
    patch = dc.Patch(**out)
    return patch


@register_func(EXAMPLE_PATCHES, key="patch_with_null")
def patch_with_null(**kwargs):
    """
    A patch which has nullish values.

    Parameters
    ----------
    **kwargs
        Parameters passed to [`random_patch`](`dascore.examples.random_patch`).
    """
    patch = random_patch(**kwargs)
    data = np.array(patch.data)
    data[data > 0.9] = np.NaN
    # also set the first row and column to NaN
    data[:, 0] = np.NaN
    data[0, :] = np.NaN
    return patch.new(data=data)


@register_func(EXAMPLE_PATCHES, key="random_patch_with_lat_lon")
def random_patch_lat_lon(**kwargs):
    """
    Create a patch with latitude/longitude coords on distance dim.

    Parameters
    ----------
    **kwargs
        Parameters passed to [`random_patch`](`dascore.examples.random_patch`).
    """
    patch = random_patch(**kwargs)
    dist = patch.coords.get_array("distance")
    lat = np.arange(0, len(dist)) * 0.001 - 109.857952
    lon = np.arange(0, len(dist)) * 0.001 + 41.544654
    # add a single coord
    out = patch.update_coords(latitude=("distance", lat), longitude=("distance", lon))
    return out


@register_func(EXAMPLE_PATCHES, key="wacky_dim_coords_patch")
def wacky_dim_coord_patch():
    """
    A patch with one Monotonic and one Array coord.
    """
    shape = (100, 1_000)
    # distance is neither monotonic nor evenly sampled.
    dist_ar = np.random.random(100) + np.arange(100) * 0.3
    # time is monotonic, not evenly sampled.
    time_ar = dc.to_datetime64(np.cumsum(np.random.random(1_000)))
    patch = random_patch(shape=shape, dist_array=dist_ar, time_array=time_ar)
    # check attrs
    attrs = patch.attrs
    assert pd.isnull(attrs.coords["time"].step)
    assert pd.isnull(attrs.coords["time"].step)
    return patch


@register_func(EXAMPLE_PATCHES, key="sin_wav")
def sin_wave_patch(
    sample_rate=44100,
    frequency: Sequence[float] | float = 100.0,
    time_min="2020-01-01",
    channel_count=3,
    duration=1,
    amplitude=10,
):
    """
    A Patch composed of sine waves.

    Parameters
    ----------
    sample_rate
        The sample rate in Hz.
    frequency
        The frequency of the sin wave. If a sequence is provided multiple
        sine waves will be generated at each frequency.
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


@register_func(EXAMPLE_PATCHES, key="example_event_1")
def example_event_1():
    """
    An induced event recorded on a borehole fiber  from @stanvek2022fracture.
    """
    path = fetch("example_dasdae_event_1.h5")
    return dc.spool(path)[0]


@register_func(EXAMPLE_PATCHES, key="example_event_2")
def example_event_2():
    """
    [`example_event_1`](`dascore.examples.example_event_1`) with pre-processing.
    """
    path = fetch("example_dasdae_event_1.h5")
    patch = dc.spool(path)[0].update_attrs(data_type="strain_rate")
    # We convert time to relative time in seconds to match the figure in
    # the publication.
    delta_time = patch.coords.get_array("time") - patch.coords.min("time")
    out = (
        patch.update_coords(time=delta_time / np.timedelta64(1, "s"))
        .set_units("1/s", distance="m", time="s")
        .taper(time=0.05)
        .pass_filter(time=(..., 300))
    )
    return out


@register_func(EXAMPLE_PATCHES, key="ricker_moveout")
def ricker_moveout(
    frequency=15,
    peak_time=0.25,
    duration=1.5,
    time_step=0.002,
    distance_step=10,
    channel_count=10,
    source_channel=0,
    velocity=100,
):
    """
    A patch of a ricker wavelet with some apparent velocity.

    Parameters
    ----------
    frequency
        The center frequency of the wavelet in Hz.
    peak_time
        The peak time of the first ricker wavelet in seconds.
    duration
        The total duration of the time coordinate in seconds.
    time_step
        The time dimension time step.
    distance_step
        The distance dimension sampling interval.
    channel_count
        The total number of channels (number of distance).
    source_channel
        The index of the source.
    velocity
        The aparent velocity in m/s.

    Notes
    -----
    Based on https://github.com/lijunzh/ricker/
    """

    def _ricker(time, delay):
        # shift time vector to account for different peak times.
        new_time = time - delay
        f = frequency
        # get amplitude and exp term of ricker
        const = 1 - 2 * np.pi**2 * f**2 * new_time**2
        exp = np.exp(-(np.pi**2) * f**2 * new_time**2)
        return const * exp

    time = np.arange(0, duration + time_step, time_step)
    distance = np.arange(channel_count) * distance_step
    assert source_channel < len(distance)
    source_distance = distance[source_channel]
    data = np.zeros((len(time), len(distance)))
    # iterate each distance channel and update data
    for ind, dist in enumerate(distance):
        dist_to_source = np.abs(dist - source_distance)
        time_delay = peak_time + (dist_to_source / velocity)
        data[:, ind] = _ricker(time, time_delay)

    coords = {"time": to_timedelta64(time), "distance": distance}
    dims = ("time", "distance")
    return dc.Patch(data=data, coords=coords, dims=dims)


@register_func(EXAMPLE_PATCHES, key="dispersion_event")
def dispersion_event():
    """
    A synthetic shot record that exhibits dispersion.
    """
    path = fetch("dispersion_event.h5")
    return dc.spool(path)[0]


@register_func(EXAMPLE_SPOOLS, key="random_das")
def random_spool(time_gap=0, length=3, time_min=np.datetime64("2020-01-03"), **kwargs):
    """
    Several random patches in the spool.

    Parameters
    ----------
    time_gap
        The difference in time between each patch. Use a negative
        number to create overlap.
    length
        The number of patches to generate.
    time_min
        The start time of the first patch. Subsequent patches have start times
        after the end time of the previous patch, plus the time_gap.
    **kwargs
        Passed to the [_random_patch](`dascore.examples.random_patch`) function.
    """
    out = []
    for _ in range(length):
        patch = random_patch(time_min=time_min, **kwargs)
        out.append(patch)
        diff = to_timedelta64(time_gap) + patch.attrs.coords["time"].step
        time_min = patch.attrs["time_max"] + diff
    return dc.spool(out)


@register_func(EXAMPLE_SPOOLS, key="diverse_das")
def diverse_spool():
    """
    A spool with a diverse set of patch metadata for testing.

    There are various gaps, tags, station names, etc.
    """
    spool_no_gaps = random_spool()
    spool_no_gaps_different_network = random_spool(network="das2")
    spool_big_gaps = random_spool(time_gap=np.timedelta64(1, "s"), station="big_gaps")
    spool_overlaps = random_spool(
        time_gap=-np.timedelta64(10, "ms"), station="overlaps"
    )
    time_step = spool_big_gaps[0].attrs.coords["time"].step
    dt = to_timedelta64(time_step / np.timedelta64(1, "s"))
    spool_small_gaps = random_spool(time_gap=dt, station="smallg")
    spool_way_late = random_spool(
        length=1, time_min=np.datetime64("2030-01-01"), station="wayout"
    )
    spool_new_tag = random_spool(tag="some_tag", length=1)
    spool_way_early = random_spool(
        length=1, time_min=np.datetime64("1989-05-04"), station="wayout"
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
    ```{python}
    #| echo: false
    #| output: asis
    from dascore.examples import EXAMPLE_PATCHES

    from dascore.utils.docs import objs_to_doc_df

    df = objs_to_doc_df(EXAMPLE_PATCHES)
    print(df.to_markdown(index=False, stralign="center"))
    ```

    Parameters
    ----------
    example_name
        The name of the example to load. Options are:
    **kwargs
        Passed to the corresponding functions to generate data.

    Raises
    ------
        (`UnknownExampleError`)['dascore.examples.UnknownExampleError`] if
        unregistered patch is requested.
    """
    if example_name not in EXAMPLE_PATCHES:
        msg = (
            f"No example patch registered with name {example_name} "
            f"Registered example patches are {list(EXAMPLE_PATCHES)}"
        )
        raise UnknownExampleError(msg)
    return EXAMPLE_PATCHES[example_name](**kwargs)


@compose_docstring(examples=", ".join(list(EXAMPLE_SPOOLS)))
def get_example_spool(example_name="random_das", **kwargs) -> dc.BaseSpool:
    """
    Load an example Spool.

    Supported example spools are:
    ```{python}
    #| echo: false
    #| output: asis
    from dascore.examples import EXAMPLE_SPOOLS

    from dascore.utils.docs import objs_to_doc_df

    df = objs_to_doc_df(EXAMPLE_SPOOLS)
    print(df.to_markdown(index=False, stralign="center"))
    ```

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
        raise UnknownExampleError(msg)
    return EXAMPLE_SPOOLS[example_name](**kwargs)
