"""pytest configuration for dascore."""
from __future__ import annotations

import os
import shutil
from contextlib import suppress
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import tables as tb
import tables.parameters

import dascore as dc
import dascore.examples as ex
from dascore.clients.dirspool import DirectorySpool
from dascore.constants import SpoolType
from dascore.core import Patch
from dascore.examples import get_example_patch
from dascore.io.core import read
from dascore.io.indexer import DirectoryIndexer
from dascore.utils.downloader import fetch
from dascore.utils.misc import register_func

test_data_path = Path(__file__).parent.absolute() / "test_data"

# A list to register functions that return general spools or patches
# These are to be used for running many patches/spools through
# Generic tests.
SPOOL_FIXTURES = []
PATCH_FIXTURES = []


# --- Pytest configuration


def pytest_addoption(parser):
    """Add pytest command options."""
    parser.addoption(
        "--integration",
        action="store_true",
        dest="run_integration",
        default=False,
        help="Run integration tests",
    )


def pytest_collection_modifyitems(config, items):
    """Configure pytest command line options."""
    marks = {}
    if not config.getoption("--integration"):
        msg = "needs --integration option to run"
        marks["integration"] = pytest.mark.skip(reason=msg)

    for item in items:
        marks_to_apply = set(marks)
        item_marks = set(item.keywords)
        for mark_name in marks_to_apply & item_marks:
            item.add_marker(marks[mark_name])


def pytest_sessionstart(session):
    """
    Hook to run before any other tests.

    Used to ensure a non-visual backend is used so plots don't pop up
    and to set debug hook to True to avoid showing progress bars,
    except when explicitly being tested.
    """
    # If running in CI make sure to turn off matplotlib.
    if os.environ.get("CI", False):
        matplotlib.use("Agg")

    # need to set nodes to 32 to avoid crash on p3.11. See pytables#977.
    tables.parameters.NODE_CACHE_SLOTS = 32

    # Ensure debug is set. This disables progress bars which disrupt debugging.
    dc._debug = True


@pytest.fixture(scope="session", autouse=True)
def swap_index_map_path(tmp_path_factory):
    """For all tests cases, use a temporary index file."""
    tmp_map_path = tmp_path_factory.mktemp("cache_paths") / "cache_paths.json"
    setattr(DirectoryIndexer, "index_map_path", tmp_map_path)


# --- Patch Fixtures


@pytest.fixture(scope="session")
def terra15_das_example_path():
    """Return the path to the example terra15 file."""
    out = fetch("terra15_das_1_trimmed.hdf5")
    assert out.exists()
    return out


@pytest.fixture(scope="class")
def terra15_v5_path():
    """Get the path to terra15 V5 file, download if not cached."""
    return fetch("terra15_v5_test_file.hdf5")


@pytest.fixture(scope="class")
def terra15_v6_path():
    """Get the path to terra15 V5 file, download if not cached."""
    return fetch("terra15_v6_test_file.hdf5")


@pytest.fixture(scope="session")
def prodml_v2_0_example_path():
    """Return the path to the prodml v2.0 file."""
    out = fetch("prodml_2.0.h5")
    assert out.exists()
    return out


@pytest.fixture(scope="session")
def prodml_v2_1_example_path():
    """Return the path to the prodml v2.1 file."""
    out = fetch("prodml_2.1.h5")
    assert out.exists()
    return out


@pytest.fixture(scope="session")
def idas_h5_example_path():
    """Return the path to the example terra15 file."""
    out = fetch("iDAS005_hdf5_example.626.h5")
    assert out.exists()
    return out


@pytest.fixture(scope="session")
def brady_hs_das_dts_coords_path():
    """Return the path to the brady Hotspot DAS/DTS coords file."""
    out = fetch("brady_hs_DAS_DTS_coords.csv")
    assert out.exists()
    return out


# --- Patch fixtures


@pytest.fixture(scope="session")
@register_func(PATCH_FIXTURES)
def terra15_das_patch(terra15_das_example_path) -> Patch:
    """Read the terra15 data, return contained DataArray."""
    out = read(terra15_das_example_path, "terra15")[0]
    attr_time = out.attrs["time_max"]
    coortime_step = out.coords.coord_map["time"].max()
    assert attr_time == coortime_step
    return out


@pytest.fixture(scope="session")
@register_func(PATCH_FIXTURES)
def prodml_v2_0_patch(prodml_v2_0_example_path) -> Patch:
    """Read the prodML v2.0 patch."""
    out = read(prodml_v2_0_example_path, "prodml")[0]
    return out


@pytest.fixture(scope="session")
@register_func(PATCH_FIXTURES)
def prodml_v2_1_patch(prodml_v2_1_example_path) -> Patch:
    """Read the prodML v2.1 patch."""
    out = read(prodml_v2_1_example_path, "prodml")[0]
    return out


@pytest.fixture(scope="session")
@register_func(PATCH_FIXTURES)
def random_patch() -> Patch:
    """Init a random array."""
    return get_example_patch("random_das")


@pytest.fixture(scope="session")
@register_func(PATCH_FIXTURES)
def random_dft_patch(random_patch) -> Patch:
    """Return the random patch with dft applied."""
    return random_patch.dft("time")


@pytest.fixture(scope="class")
@register_func(PATCH_FIXTURES)
def random_patch_with_lat_lon(random_patch):
    """Get a random patch with added lat/lon coordinates."""
    out = dc.get_example_patch("random_patch_with_lat_lon")
    return out


@pytest.fixture(scope="class")
@register_func(PATCH_FIXTURES)
def multi_dim_coords_patch(random_patch):
    """A patch with a multiple dimensional coord."""
    quality = np.ones(random_patch.shape)
    out = random_patch.update_coords(quality=(random_patch.dims, quality))
    return out


@pytest.fixture(scope="session")
@register_func(PATCH_FIXTURES)
def random_patch_many_coords(random_patch):
    """Get a random patch with many different coordinates."""
    shapes = random_patch.coord_shapes
    patch = random_patch.update_coords(
        lat=("distance", np.random.random(shapes["distance"])),
        time2=("time", np.random.random(shapes["time"])),
        quality=(random_patch.dims, np.random.random(random_patch.shape)),
    )
    return patch


@pytest.fixture(scope="session")
@register_func(PATCH_FIXTURES)
def event_patch_1():
    """Fetch event patch 1."""
    return dc.get_example_patch("example_event_1")


@pytest.fixture(scope="session")
@register_func(PATCH_FIXTURES)
def event_patch_2():
    """Fetch event patch 2."""
    return dc.get_example_patch("example_event_2")


@pytest.fixture(scope="session")
@register_func(PATCH_FIXTURES)
def dispersion_patch():
    """Fetch dispersion event."""
    return dc.get_example_patch("dispersion_event")


@pytest.fixture(scope="session")
@register_func(PATCH_FIXTURES)
def correlation_patch(random_patch):
    """
    Get a patch which is the result of correlation.

    This is useful because the lag_time dimension is a timedelta64.
    """
    return random_patch.correlate(distance=0, samples=True)


@pytest.fixture(scope="class")
@register_func(PATCH_FIXTURES)
def range_patch_3d():
    """Return a 3D patch for testing."""
    data = np.broadcast_to(np.arange(10)[:, None, None], (10, 10, 10))
    coords = {
        "time": np.arange(10),
        "distance": np.arange(10),
        "smell": np.arange(10),
    }
    patch = dc.Patch(data=data, coords=coords, dims=tuple(coords))
    return patch


@pytest.fixture(scope="session")
@register_func(PATCH_FIXTURES)
def wacky_dim_patch():
    """Fetch event patch 1."""
    return dc.get_example_patch("wacky_dim_coords_patch")


@pytest.fixture(scope="class", params=PATCH_FIXTURES)
def patch(request):
    """A meta-fixtures for collecting all patches used in testing."""
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="class")
def one_file_dir(tmp_path_factory, random_patch):
    """Create a directory with a single DAS file."""
    out = Path(tmp_path_factory.mktemp("one_file_file_spool"))
    spool = dc.spool(random_patch)
    return ex.spool_to_directory(spool, path=out)


@pytest.fixture(scope="class")
def two_patch_directory(tmp_path_factory, terra15_das_example_path, random_patch):
    """Create a directory of DAS files for testing."""
    # first copy in a terra15 file
    dir_path = tmp_path_factory.mktemp("bank_basic")
    shutil.copy(terra15_das_example_path, dir_path)
    # save a random patch
    random_patch.io.write(dir_path / "random.hdf5", "dasdae")
    return dir_path


@pytest.fixture(scope="class")
def diverse_spool_directory(diverse_spool):
    """Save the diverse spool contents to a directory."""
    out = ex.spool_to_directory(diverse_spool)
    yield out
    if out.is_dir():
        shutil.rmtree(out)


@pytest.fixture(scope="class")
def adjacent_spool_directory(tmp_path_factory, adjacent_spool_no_overlap):
    """Create a directory of adjacent patches."""
    # create a directory with several patch files in it.
    dir_path = Path(tmp_path_factory.mktemp("data"))
    for num, patch in enumerate(adjacent_spool_no_overlap):
        path = dir_path / f"{num}_patch.hdf5"
        dc.write(patch, path, file_format="dasdae")
    return dir_path


# --- Spool fixtures


@pytest.fixture()
@register_func(SPOOL_FIXTURES)
def terra15_das_spool(terra15_das_example_path) -> SpoolType:
    """Return the spool of Terra15 Das Array."""
    return read(terra15_das_example_path, file_format="terra15")


@pytest.fixture(scope="session")
@register_func(SPOOL_FIXTURES)
def terra15_das_unfinished_path() -> Path:
    """Return the spool of Terra15 Das Array."""
    out = fetch("terra15_das_unfinished.hdf5")
    assert out.exists()
    return out


@pytest.fixture(scope="class")
@register_func(SPOOL_FIXTURES)
def random_spool() -> SpoolType:
    """Init a random array."""
    from dascore.examples import get_example_spool

    return get_example_spool("random_das")


@pytest.fixture(scope="class")
@register_func(SPOOL_FIXTURES)
def adjacent_spool_no_overlap(random_patch) -> dc.BaseSpool:
    """
    Create a spool with several patches within one time sample but not
    overlapping.
    """
    pa1 = random_patch
    t2 = random_patch.attrs["time_max"]
    time_step = random_patch.attrs["time_step"]

    pa2 = random_patch.update_attrs(time_min=t2 + time_step)
    t3 = pa2.attrs["time_max"]

    pa3 = pa2.update_attrs(time_min=t3 + time_step)

    expectetime_step = pa3.attrs["time_max"] - pa1.attrs["time_min"]
    actual_time = pa3.coords.max("time") - pa1.coords.min("time")
    assert expectetime_step == actual_time
    return dc.spool([pa2, pa1, pa3])


@pytest.fixture(scope="class")
@register_func(SPOOL_FIXTURES)
def one_file_directory_spool(one_file_dir):
    """Create a directory with a single DAS file."""
    return DirectorySpool(one_file_dir).update()


@pytest.fixture(scope="class")
@register_func(SPOOL_FIXTURES)
def diverse_spool():
    """Create a spool with a diverse set of patches for testing."""
    return ex.diverse_spool()


@pytest.fixture(scope="class")
@register_func(SPOOL_FIXTURES)
def diverse_directory_spool(diverse_spool_directory):
    """Save the diverse spool contents to a directory."""
    out = dc.spool(diverse_spool_directory).update()
    return out


@pytest.fixture(scope="class")
@register_func(SPOOL_FIXTURES)
def basic_file_spool(two_patch_directory):
    """Return a DAS bank on basic_bank_directory."""
    out = DirectorySpool(two_patch_directory).update()
    return out.update()


@pytest.fixture(scope="class")
@register_func(SPOOL_FIXTURES)
def terra15_file_spool(terra15_v5_path):
    """A file spool for terra15."""
    return dc.spool(terra15_v5_path)


@pytest.fixture(scope="class")
@register_func(SPOOL_FIXTURES)
def memory_spool_dim_1_patches():
    """
    Memory spool with patches that have length 1 in one dimension.
    Related to #171.
    """
    spool = dc.get_example_spool(
        "random_das",
        time_step=0.999767552,
        shape=(100, 1),
        length=10,
        time_min="2023-06-13T15:38:00.49953408",
    )
    return spool


@pytest.fixture(scope="class")
@register_func(SPOOL_FIXTURES)
def all_examples_spool(terra15_das_example_path):
    """Create a spool from all the examples."""
    parent = terra15_das_example_path.parent
    spool = dc.spool(parent)
    try:
        spool = spool.update()
    except Exception:
        with suppress(FileNotFoundError):
            spool.indexer.index_path.unlink()  # delete index if problems found
        spool = spool.update()  # then re-index
    return spool


@pytest.fixture(scope="class")
@register_func(SPOOL_FIXTURES)
def memory_spool_small_dt_differences(random_spool):
    """Create a memory spool with slightly different time_steps."""
    out = []
    for num, patch in enumerate(random_spool):
        dt = patch.attrs.time_step + num * np.timedelta64(1, "ns")
        new = patch.update_attrs(time_step=dt)
        out.append(new)
    spool = dc.spool(out)
    assert len(out) == len(spool)
    return spool


@pytest.fixture(scope="class", params=SPOOL_FIXTURES)
def spool(request):
    """A meta-fixtures for collecting all spools used in testing."""
    return request.getfixturevalue(request.param)


# --- Misc. test fixtures


@pytest.fixture(scope="session")
def generic_hdf5(tmp_path_factory):
    """
    Create a generic hdf5 file (not das). This is useful for ensuring formatters
    recognize differences in HDF5 files.
    """
    tmp_path = Path(tmp_path_factory.mktemp("generic_h5"))
    parent = tmp_path / "sum"
    parent.mkdir()
    path = parent / "simple.hdf5"

    with tb.open_file(str(path), "w") as fi:
        group = fi.create_group("/", "bob")
        fi.create_carray(group, "data", obj=np.random.rand(10))
    return path


@pytest.fixture(scope="session")
def dummy_text_file(tmp_path_factory):
    """Return a text file with silliness in it."""
    parent = tmp_path_factory.mktemp("dummy")
    path = parent / "hello.txt"
    path.write_text("Clearly not a hdf5 file. Or is it?")
    return path


@pytest.fixture(scope="session")
def brady_hs_das_dts_coords():
    """Return a pandas dataframe with X,Y,Z coordinates."""
    path = fetch("brady_hs_DAS_DTS_coords.csv")
    coord_table = pd.read_csv(path)
    coord_table = coord_table.iloc[51:]
    coord_table = coord_table.astype(float)
    return coord_table
