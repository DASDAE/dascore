"""pytest configuration for dascore."""

from __future__ import annotations

import gc
import os
import shutil
import threading
import warnings
from contextlib import contextmanager
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import pytest

import dascore as dc
import dascore.examples as ex
import dascore.utils.remote_io as remote_io
from dascore.compat import random_state
from dascore.config import get_config, set_config
from dascore.constants import SpoolType
from dascore.core import Patch
from dascore.core.spool import Spool
from dascore.examples import get_example_patch, get_example_spool
from dascore.io.core import read
from dascore.utils.coordmanager import merge_coord_managers
from dascore.utils.downloader import fetch
from dascore.utils.misc import register_func

test_data_path = Path(__file__).parent.absolute() / "test_data"

# A list to register functions that return patches, for running many of
# them through generic tests (the `patch` meta-fixture below).
PATCH_FIXTURES = []

# By default DASCore only issues a warning once per line. This ensures
# they get issued every time so tests around warning behavior aren't flaky.
warnings.filterwarnings("default", category=UserWarning)


# A filesystem which has neither kind of link raises pathlib's
# UnsupportedOperation (a NotImplementedError) rather than an OSError;
# emscripten is one.
_NO_LINK = (OSError, NotImplementedError)


def _link_or_copy(source: Path, dest: Path) -> None:
    """Populate one file path using the cheapest available local copy."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        dest.hardlink_to(source)
        return
    except _NO_LINK:
        pass
    try:
        dest.symlink_to(source)
        return
    except _NO_LINK:
        pass
    shutil.copy2(source, dest)


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
    markexpr = getattr(config.option, "markexpr", "") or ""
    # Skip slow tests by default
    if "slow" not in markexpr:
        msg = "needs -m slow to run"
        marks["slow"] = pytest.mark.skip(reason=msg)

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

    # Test-time debug defaults are applied by fixture to avoid state leakage.


@contextmanager
def _permanent_config(**overrides):
    """Set process-wide config for the block, restoring the prior config.

    Test fixtures use the permanent base (not a scoped ``config_context``) so
    overrides are visible to worker threads and forked processes the tests
    spawn, and so they do not shadow a test's own ``set_config`` calls.
    """
    previous = get_config()
    set_config(**overrides)
    try:
        yield
    finally:
        set_config(previous)


@pytest.fixture(scope="session")
def permanent_config():
    """Return the context manager broad-scoped fixtures use to set config."""
    return _permanent_config


@pytest.fixture(scope="session")
def run_in_threads():
    """
    Return a helper which runs func(index) in several threads at once.

    A barrier releases every thread together, so concurrency tests do not
    need sleeps. The timeouts turn a deadlock into a failure rather than a
    hung test run, and anything a worker raises is re-raised here rather
    than being printed while the test carries on with a None result.
    """

    def _run(func, count=4, timeout=60):
        barrier = threading.Barrier(count, timeout=timeout)
        results = [None] * count
        errors = []

        def worker(index):
            try:
                barrier.wait()
                results[index] = func(index)
            except BaseException as error:  # re-raised in the calling thread
                errors.append(error)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout)
            assert not thread.is_alive(), "thread never finished; possible deadlock"
        if errors:
            # The first failure is re-raised as-is rather than aggregated:
            # ExceptionGroup rejects BaseException members such as pytest's
            # own skip/fail.
            raise errors[0]
        return results

    return _run


@pytest.fixture(autouse=True)
def use_test_config():
    """Run tests with debug mode enabled unless overridden locally."""
    with _permanent_config(debug=True):
        yield


@pytest.fixture(autouse=True)
def gc_pause_is_not_leaked():
    """
    Blame the test which strands the remote-read gc pause, and repair it.

    Remote HDF5 handles disable automatic collection process-wide (see
    `dascore.utils.remote_io.pause_gc`). Repairing here keeps one leak from
    cascading into every later test, which would bury the real failure.
    """
    was_enabled = gc.isenabled()
    yield
    depth = remote_io._gc_pause_depth
    is_enabled = gc.isenabled()
    if not depth and is_enabled == was_enabled:
        return
    remote_io._gc_pause_depth = 0
    if was_enabled:
        gc.enable()
    else:
        gc.disable()
    pytest.fail(
        f"test changed collection state: pause depth={depth}, "
        f"gc enabled={is_enabled}, expected={was_enabled}"
    )


@pytest.fixture(scope="session", autouse=True)
def allow_legacy_dasdae_coord_unpickle():
    """Test fixtures may rely on trusted historical DASDAE coord payloads."""
    with _permanent_config(allow_dasdae_format_unpickle=True):
        yield


@pytest.fixture(scope="session", autouse=True)
def swap_index_map_path(tmp_path_factory):
    """For all tests cases, use a temporary index-map directory."""
    tmp_map_dir = tmp_path_factory.mktemp("cache_paths") / "path_map"
    with _permanent_config(directory_index_map_dir=tmp_map_dir):
        yield


# --- Coordinate fixtures

COORD_MANAGERS = []

COORDS = {
    "time": dc.to_datetime64(np.arange(10, 100, 10)),
    "distance": dc.get_coord(data=np.arange(0, 1_000, 10)),
}
DIMS = ("time", "distance")


@pytest.fixture(scope="class")
@register_func(COORD_MANAGERS)
def cm_basic():
    """The simplest coord manager."""
    return dc.get_coord_manager(COORDS, DIMS)


@pytest.fixture(scope="class")
@register_func(COORD_MANAGERS)
def cm_with_units(cm_basic):
    """The simplest coord manager."""
    return cm_basic.set_units(time="s", distance="m")


@pytest.fixture(scope="class")
# @register_func(COORD_MANAGERS)
def cm_basic_degenerate(cm_basic):
    """A degenerate coord manager on time axis."""
    time_coord = cm_basic.coord_map["time"]
    degenerate = time_coord.empty()
    return cm_basic.update(time=degenerate)


@pytest.fixture(scope="class")
@register_func(COORD_MANAGERS)
def cm_multidim() -> dc.CoordManager:
    """The simplest coord manager with several coords added."""
    coords = {
        "time": dc.to_datetime64(np.arange(10, 110, 10)),
        "distance": dc.get_coord(data=np.arange(0, 1000, 10)),
        "quality": (("time", "distance"), np.ones((10, 100))),
        "latitude": ("distance", random_state.rand(100)),
    }
    dims = ("time", "distance")

    return dc.get_coord_manager(coords, dims)


@pytest.fixture(scope="class")
@register_func(COORD_MANAGERS)
def cm_degenerate_time(cm_multidim) -> dc.CoordManager:
    """A coordinate manager with len 1 time array."""
    new_time = dc.to_datetime64(["2017-09-18T01:00:01"])
    out = cm_multidim.update(time=new_time)
    return out


@pytest.fixture(scope="class")
@register_func(COORD_MANAGERS)
def cm_wacky_dims() -> dc.CoordManager:
    """A coordinate manager with non evenly sampled dims."""
    patch = dc.get_example_patch("wacky_dim_coords_patch")
    return patch.coords


@pytest.fixture(scope="class")
@register_func(COORD_MANAGERS)
def cm_dt_small_diff(memory_spool_small_dt_differences):
    """A list of coordinate managers with differences in dt merged."""
    spool = memory_spool_small_dt_differences
    coords = [x.coords for x in spool]
    out = merge_coord_managers(coords, dim="time")
    return out


@pytest.fixture(scope="class")
@register_func(COORD_MANAGERS)
def cm_non_associated_coord(cm_basic):
    """A cm with coordinates that are not associated with a dimension."""
    new = cm_basic.update(
        bob=(None, np.arange(10)),
        bill=((), np.arange(100)),
    )
    return new


@pytest.fixture(scope="class")
def cm_non_coord_dim():
    """A cm with a dimension that has a partial (no coordinate)."""
    coords = {"time": 10, "distance": np.arange(5)}
    dims = ("time", "distance")
    out = dc.get_coord_manager(coords=coords, dims=dims)
    return out


@pytest.fixture(scope="class", params=COORD_MANAGERS)
def coord_manager(request) -> dc.CoordManager:
    """Meta fixture for aggregating coordinates."""
    return request.getfixturevalue(request.param)


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
    attr_time = out.summary.get_coord_summary("time").max
    coord_time_max = out.coords.coord_map["time"].max()
    assert attr_time == coord_time_max
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
def random_patch_with_xyz(random_patch):
    """Get a random patch with added x, y, and z coordinates."""
    out = dc.get_example_patch("random_patch_with_xyz")
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
        lat=("distance", random_state.random(shapes["distance"])),
        time2=("time", random_state.random(shapes["time"])),
        quality=(random_patch.dims, random_state.random(random_patch.shape)),
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


@pytest.fixture(scope="session")
def one_file_dir(tmp_path_factory, random_patch):
    """Create a directory with a single DAS file."""
    out = Path(tmp_path_factory.mktemp("one_file_file_spool"))
    spool = dc.spool(random_patch)
    return ex.spool_to_directory(spool, path=out)


@pytest.fixture(scope="session")
def random_directory_spool(tmp_path_factory):
    """A directory with a few patch files."""
    path = Path(tmp_path_factory.mktemp("one_file_file_spool"))
    return dc.examples.random_directory_spool(path=path)


@pytest.fixture(scope="session")
def two_patch_directory(tmp_path_factory, terra15_das_example_path, random_patch):
    """Create a directory of DAS files for testing."""
    # first copy in a terra15 file
    dir_path = tmp_path_factory.mktemp("bank_basic")
    shutil.copy(terra15_das_example_path, dir_path)
    # save a random patch
    random_patch.io.write(dir_path / "random.hdf5", "dasdae")
    return dir_path


@pytest.fixture(scope="session")
def diverse_spool_directory(diverse_spool, tmp_path_factory):
    """Save the diverse spool contents to a directory.

    Pytest owns the directory's lifetime: an explicit rmtree teardown
    raced lazily-finalized SQLite index connections on Windows
    (WinError 32), so no teardown here.
    """
    out = tmp_path_factory.mktemp("diverse_spool_dir")
    return ex.spool_to_directory(diverse_spool, path=out)


# --- Spool fixtures


@pytest.fixture(scope="session")
def terra15_das_unfinished_path() -> Path:
    """Return the spool of Terra15 Das Array."""
    out = fetch("terra15_das_unfinished.hdf5")
    assert out.exists()
    return out


@pytest.fixture(scope="session")
def random_spool() -> SpoolType:
    """Init a random array."""
    return get_example_spool("random_das")


@pytest.fixture(scope="session")
def adjacent_spool_no_overlap(random_patch) -> dc.BaseSpool:
    """
    Create a spool with several patches within one time sample but not
    overlapping.
    """
    pa1 = random_patch
    time_coord = random_patch.get_coord("time")
    t2 = time_coord.max()
    time_step = time_coord.step

    pa2 = random_patch.new(coords=random_patch.coords.update(time_min=t2 + time_step))
    t3 = pa2.get_coord("time").max()

    pa3 = pa2.new(coords=pa2.coords.update(time_min=t3 + time_step))

    expected_time = pa3.get_coord("time").max() - pa1.get_coord("time").min()
    actual_time = pa3.coords.max("time") - pa1.coords.min("time")
    assert expected_time == actual_time
    return dc.spool([pa2, pa1, pa3])


@pytest.fixture(scope="session")
def one_file_directory_spool(one_file_dir):
    """Create a directory with a single DAS file."""
    return Spool.from_directory(one_file_dir).update()


@pytest.fixture(scope="session")
def diverse_spool():
    """Create a spool with a diverse set of patches for testing."""
    return ex.diverse_spool()


@pytest.fixture(scope="session")
def diverse_directory_spool(diverse_spool_directory):
    """Save the diverse spool contents to a directory."""
    out = dc.spool(diverse_spool_directory).update()
    yield out
    # release the SQLite index handle so Windows can clean the temp dir
    out.indexer.close()


@pytest.fixture(scope="session")
def basic_file_spool(two_patch_directory):
    """Return a DAS bank on basic_bank_directory."""
    out = Spool.from_directory(two_patch_directory).update().update()
    yield out
    out.indexer.close()


@pytest.fixture(scope="class")
def terra15_file_spool(terra15_v5_path):
    """A file spool for terra15."""
    return dc.spool(terra15_v5_path)


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def all_examples_spool(tmp_path_factory, terra15_das_example_path):
    """Create a spool from all the example files."""
    # Indexing the example files where they sit would write an index into the
    # download cache, which every test process shares. Links cost nothing and
    # give the index a directory of its own.
    source = terra15_das_example_path.parent
    directory = Path(tmp_path_factory.mktemp("all_examples"))
    for path in source.rglob("*"):
        # Skip the index (and anything else hidden) a previous run may have
        # left in the cache: a hard link to it is that same file.
        if path.is_file() and not path.name.startswith("."):
            _link_or_copy(path, directory / path.relative_to(source))
    return dc.spool(directory).update()


@pytest.fixture(scope="session")
def memory_spool_small_dt_differences(random_spool):
    """Create a memory spool with slightly different time_steps."""
    out = []
    for num, patch in enumerate(random_spool):
        dt = patch.get_coord("time").step + num * np.timedelta64(1, "ns")
        new = patch.new(coords=patch.coords.update(time_step=dt))
        out.append(new)
    spool = dc.spool(out)
    assert len(out) == len(spool)
    return spool


@pytest.fixture(scope="session")
def spool_with_non_coords():
    """Return a spool which has some non-coordinate patches inside."""
    patches = list(dc.examples.get_example_spool(length=3))
    patches += [x.mean("time") for x in patches]
    return dc.spool(patches)


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

    with h5py.File(str(path), "w") as fi:
        group = fi.create_group("bob")
        group.create_dataset("data", data=random_state.rand(10))
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
