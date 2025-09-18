"""A module for loading examples."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import chirp as spy_chirp

import dascore as dc
import dascore.core
from dascore.compat import random_state
from dascore.exceptions import UnknownExampleError
from dascore.utils.docs import compose_docstring
from dascore.utils.downloader import fetch
from dascore.utils.misc import iterate, register_func
from dascore.utils.patch import get_patch_names
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
    data[data > 0.9] = np.nan
    # also set the first row and column to NaN
    data[:, 0] = np.nan
    data[0, :] = np.nan
    return patch.new(data=data)


@register_func(EXAMPLE_PATCHES, key="random_patch_with_lat_lon")
def random_patch_lat_lon(**kwargs):
    """
    Create a patch with latitude/longitude coords on distance dimension.

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


@register_func(EXAMPLE_PATCHES, key="random_patch_with_xyz")
def random_patch_xyz(**kwargs):
    """
    Create a patch with x, y, and z coords on distance dimension.

    Parameters
    ----------
    **kwargs
        Parameters passed to [`random_patch`](`dascore.examples.random_patch`).
    """
    patch = random_patch(**kwargs)
    dist = patch.coords.get_array("distance")
    x = np.arange(0, len(dist)) * 5
    y = np.arange(0, len(dist)) * 5
    z = np.zeros_like(dist)
    # add a single coord
    out = patch.update_coords(x=("distance", x), y=("distance", y), z=("distance", z))
    return out


@register_func(EXAMPLE_PATCHES, key="wacky_dim_coords_patch")
def wacky_dim_coord_patch():
    """
    A patch with one Monotonic and one Array coord.
    """
    shape = (100, 1_000)
    # distance is neither monotonic nor evenly sampled.
    dist_ar = random_state.random(100) + np.arange(100) * 0.3
    # time is monotonic, not evenly sampled.
    time_ar = dc.to_datetime64(np.cumsum(random_state.random(1_000)))
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
    amplitude: Sequence[float] | float = 10.0,
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
        Signal duration in seconds.
    amplitude
        The amplitude of the sin wave. If a sequence is provided it represents
        the amplitude of each frequency.
    """
    t_array = np.linspace(0.0, duration, int(sample_rate * duration))
    # Get time and distance coords
    distance = np.arange(1, channel_count + 1, 1)
    time = to_timedelta64(t_array) + np.datetime64(time_min)
    freqs = np.atleast_1d(frequency)
    amps = np.broadcast_to(np.atleast_1d(amplitude), shape=freqs.shape)
    # init empty data and add frequencies.
    data = np.zeros((len(time), len(distance)))
    for amp, freq in zip(amps, freqs):
        sin_data = amp * np.sin(2.0 * np.pi * freq * t_array)
        data += sin_data[..., np.newaxis]
    patch = dc.Patch(
        data=data,
        coords={"time": time, "distance": distance},
        dims=("time", "distance"),
    )
    return patch


@register_func(EXAMPLE_PATCHES, key="chirp")
def chirp(
    sample_rate=150,
    time_min="2020-01-01",
    channel_count: int = 1,
    duration: float = 10.0,
    f0: float = 5.0,
    t1: float | None = None,
    f1: float = 25.0,
    method="linear",
    phi: float = 0.0,
    **kwargs,
):
    """
    Create a patch from a chirp signal.

    Simply uses scipy.signal.chirp under the hood.

    Parameters
    ----------
    sample_rate
        The sample rate in Hz.
    time_min
        The start time in the metadata.
    channel_count
        The number of  distance channels to include.
    duration
        The duration, in seconds, of the signal.
    f0
        The frequency of the chirp at the start of the signal.
    f1
        The frequency of the chirp at the end of the signal.
    t1
        The time (relative from signal start) corresponding to f1. If None,
        use the end of the signal.
    method
        The kind of the frequency sweep. See scipy.signal.chirp for
        more details.
    phi
        Phase offset in degrees.
    **kwargs
        Passed directly to scipy.signal.chirp.
    """
    t_array = np.linspace(0.0, duration, int(sample_rate * duration))
    t1 = t1 if t1 is not None else np.max(t_array)
    array = spy_chirp(t_array, f0=f0, t1=t1, f1=f1, method=method, phi=phi, **kwargs)
    # Get time and distance coords
    distance = np.arange(1, channel_count + 1, 1)
    time = to_timedelta64(t_array) + np.datetime64(time_min)
    data = np.array([array for _ in range(len(distance))])
    patch = dc.Patch(
        data=data,
        coords={"time": time, "distance": distance},
        dims=("distance", "time"),
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
        .set_units("strain/s", distance="m", time="s")
        .taper(time=0.05)
        .pass_filter(time=(..., 300))
    )
    return out


@register_func(EXAMPLE_PATCHES, key="deformation_rate_event_1")
def deformation_rate_event_1():
    """
    An event recorded in an underground mine by a Terra15 unit.
    """
    path = fetch("deformation_rate_event_1.hdf5")
    return dc.spool(path)[0]


@register_func(EXAMPLE_PATCHES, key="forge_dss")
def forge_dss():
    """
    A DSS file from the Forge dataset collected by Neubrex.

    https://gdr.openei.org/submissions/1565
    """
    path = fetch("neubrex_dss_forge.h5")
    return dc.spool(path)[0]


@register_func(EXAMPLE_PATCHES, key="forge_dts")
def forge_dts():
    """
    A DTS file from the Forge dataset collected by Neubrex.

    https://gdr.openei.org/submissions/1565
    """
    path = fetch("neubrex_dts_forge.h5")
    return dc.spool(path)[0]


@register_func(EXAMPLE_PATCHES, key="nd_patch")
def nd_patch(dim_count=3, coord_lens=10):
    """
    Make an N dimensional Patch.

    Parameters
    ----------
    dim_count
        The number of dimensions.
    coord_lens
        The length of the coordinates.
    """
    ran = np.random.RandomState(42)
    dims = tuple(f"dim_{i + 1}" for i in range(dim_count))
    coords = {d: np.arange(coord_lens) for d in dims}
    shape = tuple(len(coords[d]) for d in dims)
    data = ran.randn(*shape)
    return dc.Patch(data=data, coords=coords, dims=dims)


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
        The apparent velocity in m/s.

    Notes
    -----
    Based on https://github.com/lijunzh/ricker/.
    """

    def _ricker(time, delay):
        # shift time vector to account for different peak times.
        delay = 0 if not np.isfinite(delay) else delay
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
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = dist_to_source / velocity
        actual_shift = shift if np.isfinite(shift) else 0
        time_delay = peak_time + actual_shift
        data[:, ind] = _ricker(time, time_delay)

    coords = {"time": to_timedelta64(time), "distance": distance}
    dims = ("time", "distance")
    return dc.Patch(data=data, coords=coords, dims=dims)


@register_func(EXAMPLE_PATCHES, key="delta_patch")
def delta_patch(
    dim="time",
    shape=(10, 200),
    time_min="2020-01-01",
    time_step=1 / 250,
    distance_min=0,
    distance_step=1,
    patch=None,
):
    """
    Create a delta function patch (zeros everywhere except for
    a unit value at the center) along the specified dimension.
    The returned delta patch has single coordinate(s) along the
    other dimensions.

    Parameters
    ----------
    dim : str
        The dimension at the center of which to place the unit value.
        Typically ``"time"`` or ``"distance"``.
    shape : tuple of int
        The shape of the data as (distance, time). Defaults to (10, 200).
        This is used only if no existing ``patch`` is provided.
    time_min : str or datetime64
        The start time of the patch.
    time_step : float
        The time step in seconds between samples.
    distance_min : float
        The minimum distance coordinate.
    distance_step : float
        The distance step in meters between samples.
    patch : dascore.Patch
        If provided, creates the delta patch based on this existing patch.
        Default is None.
    """
    if patch is None:
        if dim not in ["time", "distance"]:
            raise ValueError(
                "In case no patch is provided, the delta patch will be "
                "a 2D patch with 'time' and 'distance' dimensions."
            )

        dims = ("distance", "time")
        dist_len, time_len = shape

        # Create coordinates
        time_step_td = to_timedelta64(time_step)
        t0 = np.datetime64(time_min)
        time_coord = dascore.core.get_coord(
            data=t0 + np.arange(time_len) * time_step_td, step=time_step_td, units="s"
        )
        dist_coord = dascore.core.get_coord(
            data=distance_min + np.arange(dist_len) * distance_step,
            step=distance_step,
            units="m",
        )

        coords = {"distance": dist_coord, "time": time_coord}
        attrs = dict(
            time_min=t0,
            time_step=time_step_td,
            distance_min=distance_min,
            distance_step=distance_step,
            category="DAS",
            network="",
            station="",
            tag="delta",
            time_units="s",
            distance_units="m",
        )

        # Depending on the selected dimension, place a line of ones at the midpoint
        used_dims = tuple(iterate(dim))
        unused_dims = set(dims) - set(used_dims)

        # Get data with ones centered on selected dimensions.
        index = tuple(
            shape[dims.index(dimension)] // 2 if dimension in used_dims else 0
            for dimension in dims
        )
        data = np.zeros((dist_len, time_len))
        data[index] = 1
        delta_patch = dc.Patch(data=data, coords=coords, dims=dims, attrs=attrs)
        return delta_patch.select(**{x: 0 for x in unused_dims}, samples=True)
    else:
        used_dims = tuple(iterate(dim))
        unused_dims = set(patch.dims) - set(used_dims)
        patch = patch.select(**{x: 0 for x in unused_dims}, samples=True)

        # Get data with ones centered on selected dimensions.
        shape = patch.shape
        index = tuple(
            shape[patch.get_axis(dimension)] // 2 if dimension in used_dims else 0
            for dimension in patch.dims
        )
        data = np.zeros_like(patch.data)
        data[index] = 1
        return patch.update(data=data)


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


@register_func(EXAMPLE_SPOOLS, key="random_directory_das")
def random_directory_spool(path=None, **kwargs):
    """
    Create a random spool, then save to specified path.

    Parameters
    ----------
    path
        If provided, the path to save the directory spool. If None, use
        a temporary path.

    kwargs are passed to [`random_spool`](`dascore.examples.random_spool`)
    """
    spool = random_spool(**kwargs)
    path = spool_to_directory(spool, path)
    return dc.spool(path)


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


def spool_to_directory(spool, path=None, file_format="DASDAE", extension="hdf5"):
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
        name = get_patch_names(patch).iloc[0]
        out_path = path / (f"{name}.{extension}")
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

    Using an entry from the data_registry file is also supported.
    If multiple patches are contained in the specified file, only the
    first is returned. Data registry files are:
    ```{python}
    #| echo: false
    #| output: asis
    from dascore.utils.downloader import get_registry_df
    print(get_registry_df()[['name']].to_markdown(index=False, stralign="center"))
    ```

    Parameters
    ----------
    example_name
        The name of the example to load. Options are listed above.
    **kwargs
        Passed to the corresponding functions to generate data.

    Raises
    ------
        (`UnknownExampleError`)['dascore.examples.UnknownExampleError`] if
        unregistered patch is requested.
    """
    if example_name not in EXAMPLE_PATCHES:
        # Allow the example name to be a data registry entry.
        with suppress(ValueError):
            return dc.spool(fetch(example_name))[0]
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

    Using an entry from the data_registry file is also supported.
    These include:
    ```{python}
    #| echo: false
    #| output: asis
    from dascore.utils.downloader import get_registry_df
    print(get_registry_df()[['name']].to_markdown(index=False, stralign="center"))
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
    if example_name not in EXAMPLE_SPOOLS:
        # Allow the example spool to be a data registry file.
        with suppress(ValueError):
            return dc.spool(fetch(example_name))
        msg = (
            f"No example spool registered with name {example_name} "
            f"Registered example spools are {list(EXAMPLE_SPOOLS)}"
        )
        raise UnknownExampleError(msg)
    return EXAMPLE_SPOOLS[example_name](**kwargs)
