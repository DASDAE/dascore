"""A module for loading examples."""

from __future__ import annotations

import io
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

import numpy as np
import pandas as pd

import dascore as dc
import dascore.core
from dascore.compat import random_state
from dascore.config import config_context
from dascore.core.inventory import (
    Acquisition,
    CouplingCondition,
    DistanceMap,
    FiberArray,
    FiberSegment,
    Geometry,
    Interrogator,
    Inventory,
    Network,
    OpticalPath,
    OpticalPathLabel,
)
from dascore.exceptions import UnknownExampleError
from dascore.utils.downloader import fetch
from dascore.utils.imports import lazy_import
from dascore.utils.misc import iterate, register_func
from dascore.utils.patch import get_patch_names
from dascore.utils.time import to_timedelta64

spy_chirp = lazy_import("scipy.signal", "chirp")

EXAMPLE_PATCHES = {}
EXAMPLE_SPOOLS = {}
EXAMPLE_INVENTORIES = {}


def _load_example_patch_from_file(path: str | Path) -> dc.Patch:
    """Load the first patch from an example file without spool indirection."""
    with config_context(allow_dasdae_format_unpickle=True):
        return dc.read(path)[0]


@register_func(EXAMPLE_PATCHES, key="random_das")
def random_patch(
    *,
    time_min="2017-09-18",
    time_step=to_timedelta64(1 / 250),
    time_array=None,
    distance_min=0,
    distance_step=1,
    dist_array=None,
    acquisition_key="",
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
    acquisition_key
        The inventory identity of the data source
        (network.fiber_array.location.acquisition).
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
    time_step = to_timedelta64(time_step)
    attrs = dict(
        category="DAS",
        acquisition_key=acquisition_key,
        tag=tag,
    )
    if time_array is None:
        time_array = dascore.core.get_coord(
            data=t1 + np.arange(array.shape[1]) * time_step,
            step=time_step,
            units="s",
        )
    if dist_array is None:
        dist_array = dascore.core.get_coord(
            data=d1 + np.arange(array.shape[0]) * distance_step,
            step=distance_step,
            units="m",
        )
    coords = dict(distance=dist_array, time=time_array)
    # assemble and output.
    return dc.Patch(data=array, coords=coords, attrs=attrs, dims=("distance", "time"))


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
    time_coord = patch.coords.coord_map["time"]
    assert pd.isnull(time_coord.step)
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
    return _load_example_patch_from_file(path)


@register_func(EXAMPLE_PATCHES, key="example_event_2")
def example_event_2():
    """
    [`example_event_1`](`dascore.examples.example_event_1`) with pre-processing.
    """
    path = fetch("example_dasdae_event_1.h5")
    patch = _load_example_patch_from_file(path).update_attrs(data_type="strain_rate")
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
    return _load_example_patch_from_file(path)


@register_func(EXAMPLE_PATCHES, key="forge_dss")
def forge_dss():
    """
    A DSS file from the Forge dataset collected by Neubrex.

    https://gdr.openei.org/submissions/1565
    """
    path = fetch("neubrex_dss_forge.h5")
    return _load_example_patch_from_file(path)


@register_func(EXAMPLE_PATCHES, key="febus_dss_mine_tight")
def febus_dss_mine_1():
    """
    DSS file from a tight-buffered fiber at a mine with Febus interrogator
    """
    return _load_example_patch_from_file(fetch("dss_ug_mine_tight.h5"))


@register_func(EXAMPLE_PATCHES, key="febus_dss_mine_loose")
def febus_dss_mine_2():
    """
    DSS file from a loose-buffered fiber at a mine with Febus interrogator
    """
    return _load_example_patch_from_file(fetch("dss_ug_mine_loose.h5"))


@register_func(EXAMPLE_PATCHES, key="forge_dts")
def forge_dts():
    """
    A DTS file from the Forge dataset collected by Neubrex.

    https://gdr.openei.org/submissions/1565
    """
    path = fetch("neubrex_dts_forge.h5")
    return _load_example_patch_from_file(path)


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
            category="DAS",
            acquisition_key="",
            tag="delta",
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
    return _load_example_patch_from_file(path)


@register_func(EXAMPLE_SPOOLS, key="random_das")
def random_spool(
    time_gap=0, length=3, time_min=np.datetime64("2020-01-03"), var=0, **kwargs
):
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
    var
        How much the patch lengths vary, in percent. Zero makes every patch
        the same length; a positive value draws each from a normal
        distribution that wide, as an archive of real files has.
    **kwargs
        Passed to the [_random_patch](`dascore.examples.random_patch`) function.
    """
    shape = kwargs.pop("shape", (300, 2_000))
    samples = shape[-1]
    if var > 0:
        # Seeded, since an example which differs run to run is not one.
        draws = np.random.default_rng(42).normal(samples, samples * var / 100, length)
        lengths = np.clip(draws, 1, None).astype(int)
    else:
        lengths = np.full(length, samples, dtype=int)
    out = []
    for count in lengths:
        patch = random_patch(time_min=time_min, shape=(*shape[:-1], count), **kwargs)
        out.append(patch)
        diff = to_timedelta64(time_gap) + patch.coords.step("time")
        time_min = patch.coords.max("time") + diff
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

    There are various gaps, tags, acquisition keys, etc.
    """
    spool_no_gaps = random_spool()
    spool_no_gaps_different_source = random_spool(acquisition_key="DAS2.R2D1..RAW")
    spool_big_gaps = random_spool(time_gap=np.timedelta64(1, "s"), tag="big_gaps")
    spool_overlaps = random_spool(time_gap=-np.timedelta64(10, "ms"), tag="overlaps")
    time_step = spool_big_gaps[0].coords.step("time")
    dt = to_timedelta64(time_step / np.timedelta64(1, "s"))
    spool_small_gaps = random_spool(time_gap=dt, tag="smallg")
    spool_way_late = random_spool(
        length=1, time_min=np.datetime64("2030-01-01"), tag="wayout"
    )
    spool_new_tag = random_spool(tag="some_tag", length=1)
    spool_way_early = random_spool(
        length=1, time_min=np.datetime64("1989-05-04"), tag="wayout"
    )

    all_spools = [
        spool_no_gaps,
        spool_no_gaps_different_source,
        spool_big_gaps,
        spool_overlaps,
        spool_small_gaps,
        spool_way_late,
        spool_new_tag,
        spool_way_early,
    ]

    return dc.spool([y for x in all_spools for y in x])


@register_func(EXAMPLE_SPOOLS, key="sparse_dss")
def sparse_dss_spool():
    """
    Two months of a sparsely sampled DSS deployment.

    One patch per day of hourly samples along 20 channels, for a
    temperature and a strain acquisition which start and end at
    different times and lose different days to outages. Sampled once an
    hour, so the whole thing is a few hundred kilobytes -- small enough
    to build in memory, long enough to draw on a calendar.

    A day short of its 24 samples leaves a hole after it, so the
    acquisitions cover their spans by different amounts.
    """
    hour = to_timedelta64(np.timedelta64(1, "h"))
    day_one = np.datetime64("2024-01-01")
    runs = {
        # tag: (days it ran, days it was down, days it cut short)
        # Days 17-20 are the site's own outage, so both lose them.
        "temperature": (range(60), {17, 18, 19, 20, 33}, {8: 18, 41: 12}),
        "strain": (range(9, 50), {17, 18, 19, 20, 28, 29, 30}, {41: 12}),
    }
    patches = []
    for tag, (days, down, short) in runs.items():
        for day in days:
            if day in down:
                continue
            samples = short.get(day, 24)
            patches.append(
                random_patch(
                    time_min=day_one + np.timedelta64(day, "D"),
                    time_step=hour,
                    distance_step=5,
                    shape=(20, samples),
                    tag=tag,
                )
            )
    return dc.spool(patches)


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
    extension
        The file extension given to each saved file.
    """
    if path is None:
        path = Path(tempfile.mkdtemp())
        assert path.exists()
    for patch in spool:
        name = get_patch_names(patch).iloc[0]
        out_path = path / (f"{name}.{extension}")
        patch.io.write(out_path, file_format=file_format)
    return path


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
            return _load_example_patch_from_file(fetch(example_name))
        msg = (
            f"No example patch registered with name {example_name} "
            f"Registered example patches are {list(EXAMPLE_PATCHES)}"
        )
        raise UnknownExampleError(msg)
    return EXAMPLE_PATCHES[example_name](**kwargs)


def get_example_spool(example_name="random_das", **kwargs) -> dc.Spool:
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


def inventory_patch_pair():
    """
    Return a patch and an inventory which resolves it.

    The patch is the random DAS example carrying the acquisition key of the
    inventory's one acquisition. That acquisition places its 300 channels on
    an optical path through a measured two-point distance map, so the path's
    geometry, coupling, and labels project onto the patch. Used by the
    enrich documentation and tests.
    """
    patch = random_patch(acquisition_key="DAS.R2D1..RAW")
    distance = patch.get_coord("distance")
    # The interrogator's own axis starts at its channel 0; the path axis
    # starts 100 m later, at the far end of the lead-in cable.
    acquisition = Acquisition(
        code="RAW",
        location_code="",
        data_type="velocity",
        data_category="DAS",
        gauge_length=10.0,
        spatial_interval=1.0,
        sample_rate=1.0 / dc.to_float(patch.get_coord("time").step),
        pulse_width=1e-8,
        interrogator=Interrogator(
            manufacturer="Fake Interrogators", model="FI-1", serial_number="sn-1"
        ),
        distance_map=DistanceMap(
            instrument_distance=(float(distance.min()), float(distance.max())),
            distance=(100.0, 100.0 + float(distance.max() - distance.min())),
        ),
    )
    path = OpticalPath(
        name="main",
        location_code="",
        optical_components=(FiberSegment(name="cable", optical_length=500.0),),
        geometry=(
            Geometry(
                name="trench",
                distance=(100.0, 400.0),
                # The canonical axis names, so the segment states the CRS's
                # axes whatever this inventory's CRS happens to call them.
                coordinates={
                    "x": (-117.0, -117.0),
                    "y": (40.0, 40.1),
                    "z": (1500.0, 1500.0),
                },
            ),
        ),
        coupling=(
            CouplingCondition(
                start_distance=100.0,
                end_distance=250.0,
                coupling_type="trench",
                medium="soil",
            ),
        ),
        labels=(
            OpticalPathLabel(
                start_distance=100.0, end_distance=200.0, group="zone", value="north"
            ),
            OpticalPathLabel(
                start_distance=200.0, end_distance=400.0, group="zone", value="south"
            ),
            OpticalPathLabel(start_distance=150.0, end_distance=300.0, group="noisy"),
        ),
    )
    inventory = Inventory(
        networks=(
            Network(
                code="DAS",
                fiber_arrays=(
                    FiberArray(
                        code="R2D1", acquisitions=(acquisition,), optical_paths=(path,)
                    ),
                ),
            ),
        )
    ).check()
    return patch, inventory


@register_func(EXAMPLE_INVENTORIES, key="random_das")
def random_das_inventory() -> Inventory:
    """A single-path inventory which resolves the random_das example patch."""
    return inventory_patch_pair()[1]


def get_example_inventory(example_name="random_das", **kwargs) -> Inventory:
    """
    Load an example Inventory.

    Supported example inventories are:
    ```{python}
    #| echo: false
    #| output: asis
    from dascore.examples import EXAMPLE_INVENTORIES

    from dascore.utils.docs import objs_to_doc_df

    df = objs_to_doc_df(EXAMPLE_INVENTORIES)
    print(df.to_markdown(index=False, stralign="center"))
    ```

    Parameters
    ----------
    example_name
        The name of the example to load. Options are listed above.
    **kwargs
        Passed to the corresponding functions to generate the inventory.

    Raises
    ------
    (`UnknownExampleError`)['dascore.examples.UnknownExampleError`] if an
        unregistered inventory is requested.

    Examples
    --------
    >>> import dascore as dc
    >>> inventory = dc.get_example_inventory("tunnel")
    >>> len(inventory.networks)
    1
    """
    if example_name not in EXAMPLE_INVENTORIES:
        msg = (
            f"No example inventory registered with name {example_name} "
            f"Registered example inventories are {list(EXAMPLE_INVENTORIES)}"
        )
        raise UnknownExampleError(msg)
    return EXAMPLE_INVENTORIES[example_name](**kwargs)


# --- The tunnel inventory -------------------------------------------------
#
# This builds the deployment the tunnel recipe walks through, and is the
# single definition of it: the recipe displays these very files rather
# than composing its own, so the page and the example cannot drift apart.

_TUNNEL_RESOURCES = {
    "telemetry-cable": (
        "object_type: Cable\n"
        "name: tunnel telemetry cable\n"
        "manufacturer: Corning\n"
        "model: MIC tight-buffered 4F OS2\n"
        "fiber_count: 4\n"
        "description: The run in from the instrument room.\n"
    ),
    "connecting-cable": (
        "object_type: Cable\n"
        "name: tunnel connecting cable\n"
        "manufacturer: Corning\n"
        "model: MIC tight-buffered 4F OS2\n"
        "fiber_count: 4\n"
        "description: The links between boxes, couplers, and borehole heads.\n"
    ),
    "borehole-cable": (
        "object_type: Cable\n"
        "name: borehole sensing cable\n"
        "manufacturer: Nerve Sensors\n"
        "model: Epsilon\n"
        "fiber_count: 4\n"
        "description: Rock-coupled downhole cable with an armored pigtail.\n"
    ),
    "trench-cable": (
        "object_type: Cable\n"
        "name: helically wound trench cable\n"
        "manufacturer: Silixa\n"
        "model: HWC\n"
        "fiber_count: 1\n"
    ),
    "das-interrogator": (
        "object_type: Interrogator\n"
        "name: tunnel DAS interrogator\n"
        "manufacturer: Sintela\n"
        "model: Onyxia\n"
        "instrument_type: DAS interrogator\n"
    ),
    "repair-cord": (
        "object_type: Cable\nname: trench repair patch cord\nfiber_count: 1\n"
    ),
    "repair-box": (
        "object_type: Enclosure\nname: trench repair box\nenclosure_type: box\n"
    ),
}

# One enclosure per housing, because a resource_id names an asset rather than
# a kind: box A and box E are two boxes, and each borehole has its own
# turnaround down the hole.
for _label, _name in [
    ("splice-box-a", "splice box at A"),
    ("splice-box-e", "splice box at E"),
    ("turnaround-1", "borehole 1 turnaround housing"),
    ("turnaround-2", "borehole 2 turnaround housing"),
    ("turnaround-3", "borehole 3 turnaround housing"),
]:
    _kind = "box" if "splice" in _label else "housing"
    _TUNNEL_RESOURCES[_label] = (
        f"object_type: Enclosure\nname: tunnel {_name}\nenclosure_type: {_kind}\n"
    )

_TUNNEL_COMPONENTS = """\
sequence,object_type,optical_length,name,container
1,FiberSegment,1500.0,telemetry lead-in,telemetry-cable
2,Splice,0.0,splice at box A,splice-box-a
3,FiberSegment,2.5,drop into the trench,trench-cable
4,FiberSegment,25.0,trench B to the coil,trench-cable
5,FiberSegment,10.0,cable coil at C,trench-cable
6,FiberSegment,25.0,trench from the coil to D,trench-cable
7,FiberSegment,2.5,rise out of the trench,trench-cable
8,Splice,0.0,splice at box E,splice-box-e
9,FiberSegment,15.0,link E to borehole 3,connecting-cable
10,FiberSegment,20.0,borehole 3 down,borehole-cable
11,Splice,0.0,borehole 3 turnaround,turnaround-3
12,FiberSegment,20.0,borehole 3 up,borehole-cable
13,FiberSegment,15.0,link borehole 3 to coupler G,connecting-cable
14,Connector,0.0,coupler G,
15,FiberSegment,15.0,link coupler G to borehole 2,connecting-cable
16,FiberSegment,20.0,borehole 2 down,borehole-cable
17,Splice,0.0,borehole 2 turnaround,turnaround-2
18,FiberSegment,20.0,borehole 2 up,borehole-cable
19,FiberSegment,15.0,link borehole 2 to coupler H,connecting-cable
20,Connector,0.0,coupler H,
21,FiberSegment,15.0,link coupler H to borehole 1,connecting-cable
22,FiberSegment,20.0,borehole 1 down,borehole-cable
23,Splice,0.0,borehole 1 turnaround,turnaround-1
24,FiberSegment,20.0,borehole 1 up,borehole-cable
25,FiberSegment,15.0,link borehole 1 back to box A,connecting-cable
26,Terminator,0.0,path end,
"""

# The surveyed waypoints, lettered as the recipe's drawing letters them.
_TUNNEL_A = (100.00, 100.00, 0.0)
_TUNNEL_B = (100.00, 97.79, -0.5)
_TUNNEL_C = (122.15, 97.79, -0.5)
_TUNNEL_D = (144.30, 97.79, -0.5)
_TUNNEL_E = (144.30, 100.00, 0.0)
_TUNNEL_HEADS = {
    1: (108.00, 100.00, 0.0),
    2: (126.00, 100.00, 0.0),
    3: (142.00, 100.00, 0.0),
}
_TUNNEL_DEPTH = 20.0
# The trench cable is wound helically, so a meter of fiber covers cos(phi)
# of a meter of tunnel.
_TUNNEL_WIND = 0.886
_TUNNEL_TRENCH = (
    "drop into the trench",
    "trench B to the coil",
    "trench from the coil to D",
    "rise out of the trench",
)
_TUNNEL_REPAIRED_TRENCH = (
    "drop into the trench",
    "trench B to the break",
    "trench from the break to the coil",
    "trench from the coil to D",
    "rise out of the trench",
)


def _tunnel_spans(components_csv):
    """Map each component's name to the optical interval it covers."""
    frame = pd.read_csv(io.StringIO(components_csv))
    end = frame["optical_length"].cumsum()
    return dict(zip(frame["name"], zip(end - frame["optical_length"], end)))


def _tunnel_bottom(number):
    """The bottom of a borehole is its head, straight down."""
    x, y, _ = _TUNNEL_HEADS[number]
    return (x, y, -_TUNNEL_DEPTH)


def _tunnel_runs(repaired=False):
    """Which component runs between which two surveyed waypoints."""
    if repaired:
        # The patch cord is coiled in a splice box, so the trench is
        # surveyed up to the break and again from it, and the two meters
        # between get no position at all.
        brk = (100.00 + 15.0 * _TUNNEL_WIND, 97.79, -0.5)
        runs = [
            ("drop into the trench", _TUNNEL_A, _TUNNEL_B),
            ("trench B to the break", _TUNNEL_B, brk),
            ("trench from the break to the coil", brk, _TUNNEL_C),
            ("trench from the coil to D", _TUNNEL_C, _TUNNEL_D),
            ("rise out of the trench", _TUNNEL_D, _TUNNEL_E),
        ]
    else:
        runs = [
            ("drop into the trench", _TUNNEL_A, _TUNNEL_B),
            ("trench B to the coil", _TUNNEL_B, _TUNNEL_C),
            ("trench from the coil to D", _TUNNEL_C, _TUNNEL_D),
            ("rise out of the trench", _TUNNEL_D, _TUNNEL_E),
        ]
    for number in (3, 2, 1):
        runs.append(
            (f"borehole {number} down", _TUNNEL_HEADS[number], _tunnel_bottom(number))
        )
        runs.append(
            (f"borehole {number} up", _tunnel_bottom(number), _TUNNEL_HEADS[number])
        )
    return runs


def _tunnel_geometry(at, runs):
    """Turn each straight run into the two control points which place it."""
    rows = []
    for name, start, end in runs:
        first, last = at[name]
        rows.append((name, first, *start))
        rows.append((name, last, *end))
    frame = pd.DataFrame(rows, columns=["segment", "distance", "x", "y", "z"])
    return frame.to_csv(index=False)


def _tunnel_coupling(at, trench_parts):
    """Buried in the trench, coiled at C, cemented in the boreholes."""
    rows: list[tuple] = [
        (*at[name], "trench", "soil", "direct_burial", 0.5) for name in trench_parts
    ]
    rows.append((*at["cable coil at C"], "coiled", "soil", "", 0.5))
    rows.extend(
        (
            at[f"borehole {number} down"][0],
            at[f"borehole {number} up"][1],
            "outside_borehole_casing",
            "rock",
            "cemented",
            "",
        )
        for number in (3, 2, 1)
    )
    frame = pd.DataFrame(
        rows,
        columns=[
            "start_distance",
            "end_distance",
            "coupling_type",
            "medium",
            "attachment",
            "depth",
        ],
    )
    return frame.to_csv(index=False)


def _tunnel_labels(at, trench_parts):
    """Which section a channel is in, and which borehole if it is in one."""
    rows: list[tuple] = [(*at[name], "section", "trench") for name in trench_parts]
    rows.append((*at["cable coil at C"], "section", "coil"))
    for number in (3, 2, 1):
        span = (at[f"borehole {number} down"][0], at[f"borehole {number} up"][1])
        rows.append((*span, "section", "borehole"))
        rows.append((*span, "borehole", number))
    frame = pd.DataFrame(
        rows, columns=["start_distance", "end_distance", "group", "value"]
    )
    return frame.to_csv(index=False)


def _tunnel_repaired_components():
    """One row becomes five where the contractor cut the trench cable."""
    rows = pd.read_csv(io.StringIO(_TUNNEL_COMPONENTS)).to_dict("records")
    index = next(
        i for i, row in enumerate(rows) if row["name"] == "trench B to the coil"
    )
    rows[index : index + 1] = [
        dict(
            object_type="FiberSegment",
            optical_length=15.0,
            name="trench B to the break",
            container="trench-cable",
        ),
        dict(
            object_type="Splice",
            optical_length=0.0,
            name="repair splice near side",
            container="repair-box",
        ),
        dict(
            object_type="FiberSegment",
            optical_length=2.0,
            name="repair patch cord",
            container="repair-cord",
        ),
        dict(
            object_type="Splice",
            optical_length=0.0,
            name="repair splice far side",
            container="repair-box",
        ),
        dict(
            object_type="FiberSegment",
            optical_length=10.0,
            name="trench from the break to the coil",
            container="trench-cable",
        ),
    ]
    frame = pd.DataFrame(rows)
    frame["sequence"] = range(1, len(frame) + 1)
    return frame.to_csv(index=False)


def tunnel_inventory_files(repaired: bool = True) -> dict[str, str]:
    """
    Return the tunnel inventory as a mapping of file name to file text.

    This is the authoring directory the tunnel recipe writes, as data. It
    is exposed so the recipe can display the same files the example
    loads, rather than composing a second copy which could drift.

    Parameters
    ----------
    repaired
        Whether to include the epoch added when the trench cable was
        repaired. False is the deployment as first installed, which is
        what the recipe shows before it gets to the repair.
    """
    array = "fiber_arrays/XT.TUN1"
    path, epoch = f"{array}/path.00", f"{array}/path.00@2024-09-01"
    at = _tunnel_spans(_TUNNEL_COMPONENTS)
    repaired_csv = _tunnel_repaired_components()
    repaired_at = _tunnel_spans(repaired_csv)
    files = {
        "inventory.yaml": (
            "object_type: Inventory\n"
            "coordinate_reference_system:\n"
            "  authority: local\n"
            "  code: tunnel\n"
            "  name: tunnel engineering grid\n"
            "  coordinate_labels: [x, y, z]\n"
            "  units: [meter, meter, meter]\n"
        ),
        f"{array}/attrs.yaml": ("object_type: FiberArray\nname: tunnel fiber array\n"),
        "acquisitions/XT.TUN1.00.DAS.yaml": (
            "object_type: Acquisition\n"
            "data_category: DAS\n"
            "data_type: strain_rate\n"
            "data_units: 1/s\n"
            "interrogator: das-interrogator\n"
            "gauge_length: 10.0\n"
            "spatial_interval: 1.0\n"
            "sample_rate: 250.0\n"
            "distance_map:\n"
            "  instrument_distance: [0.0, 2000.0]\n"
            "  distance: [0.0, 2000.0]\n"
        ),
        f"{path}/attrs.yaml": "object_type: OpticalPath\n",
        f"{path}/optical_components.csv": _TUNNEL_COMPONENTS,
        f"{path}/geometry.csv": _tunnel_geometry(at, _tunnel_runs()),
        f"{path}/coupling.csv": _tunnel_coupling(at, _TUNNEL_TRENCH),
        f"{path}/labels.csv": _tunnel_labels(at, _TUNNEL_TRENCH),
        f"{epoch}/attrs.yaml": "object_type: OpticalPath\n",
        f"{epoch}/optical_components.csv": repaired_csv,
        f"{epoch}/geometry.csv": _tunnel_geometry(
            repaired_at, _tunnel_runs(repaired=True)
        ),
        f"{epoch}/coupling.csv": _tunnel_coupling(repaired_at, _TUNNEL_REPAIRED_TRENCH),
        f"{epoch}/labels.csv": _tunnel_labels(repaired_at, _TUNNEL_REPAIRED_TRENCH),
    }
    for name, text in _TUNNEL_RESOURCES.items():
        files[f"resources/{name}.yaml"] = text
    if not repaired:
        # The repair is later hardware, so before it happens neither its
        # epoch nor the resources it introduced exist yet.
        files = {
            name: text
            for name, text in files.items()
            if epoch not in name and "repair-" not in name
        }
    return files


def write_tunnel_inventory(path, repaired: bool = True) -> Path:
    """Write the tunnel inventory's authoring directory and return it."""
    path = Path(path)
    for name, text in tunnel_inventory_files(repaired=repaired).items():
        file_path = path / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(text)
    return path


@register_func(EXAMPLE_INVENTORIES, key="tunnel")
def tunnel_inventory() -> Inventory:
    """The tunnel deployment the tunnel recipe builds, read from its files."""
    directory = Path(tempfile.mkdtemp()) / "tunnel_inventory"
    return dc.inventory(write_tunnel_inventory(directory))
