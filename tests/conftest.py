"""
pytest configuration for dascore
"""
import shutil
from pathlib import Path

import numpy as np
import pytest
import tables as tb

import dascore as dc
import dascore.examples as ex
from dascore.clients.dirspool import DirectorySpool
from dascore.constants import SpoolType
from dascore.core import Patch
from dascore.io.core import read
from dascore.utils.downloader import fetch
from dascore.utils.misc import register_func

test_data_path = Path(__file__).parent.absolute() / "test_data"

# A list to register functions that return general spools or patches
# These are to be used for running many different patches/spools through
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
    parser.addoption(
        "--gui", action="store_true", default=False, help="only run gui tests"
    )


def pytest_collection_modifyitems(config, items):
    """Configure pytest command line options."""
    # skip workbench gui tests unless --gui is specified.
    run_guis = config.getoption("--gui")
    skip = pytest.mark.skip(reason="only run manual tests when --gui is used")
    for item in items:
        # skip all gui tests if gui flag is not set
        if not run_guis and "gui" in item.keywords:
            item.add_marker(skip)
        # skip all non-gui tests if gui flag is set
        if run_guis and "gui" not in item.keywords:
            item.add_marker(skip)

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
    import matplotlib

    import dascore as dc

    matplotlib.use("Agg")
    dc._debug = True


# --- Patch Paths


@pytest.fixture(scope="session")
def terra15_das_example_path():
    """Return the path to the example terra15 file."""
    out = fetch("terra15_das_1_trimmed.hdf5")
    assert out.exists()
    return out


@pytest.fixture(scope="session")
@register_func(SPOOL_FIXTURES)
def terra15_das_unfinished_path() -> Path:
    """Return the spool of Terra15 Das Array"""
    out = fetch("terra15_das_unfinished.hdf5")
    assert out.exists()
    return out


@pytest.fixture(scope="class")
def terra15_v5_path():
    """Get the path to terra15 V5 file, download if not cached."""
    return fetch("terra15_v5_test_file.hdf5")


@pytest.fixture()
@register_func(SPOOL_FIXTURES)
def terra15_das_spool(terra15_das_example_path) -> SpoolType:
    """Return the spool of Terra15 Das Array"""
    return read(terra15_das_example_path, file_format="terra15")


# --- Patch fixtures


@pytest.fixture(scope="session")
@register_func(PATCH_FIXTURES)
def terra15_das_patch(terra15_das_example_path) -> Patch:
    """Read the terra15 data, return contained DataArray"""
    out = read(terra15_das_example_path, "terra15")[0]
    attr_time = out.attrs["time_max"]
    coord_time = out.coords["time"].max()
    assert attr_time == coord_time
    return out


@pytest.fixture(scope="session")
@register_func(PATCH_FIXTURES)
def random_patch() -> Patch:
    """Init a random array."""
    from dascore.examples import get_example_patch

    return get_example_patch("random_das")


@pytest.fixture(scope="class")
@register_func(PATCH_FIXTURES)
def random_patch_with_lat_lon(random_patch):
    """Get a random patch with added lat/lon coordinates."""
    return dc.get_example_patch("random_patch_with_lat_lon")


@pytest.fixture(scope="class")
@register_func(PATCH_FIXTURES)
def multi_dim_coords_patch(random_patch):
    """A patch with a multiple dimensional coord."""
    dist = random_patch.coords["distance"]
    time = random_patch.coords["time"]
    quality = np.ones((len(dist), len(time)))
    out = random_patch.assign_coords(quality=(("distance", "time"), quality))
    return out


@pytest.fixture(scope="class", params=PATCH_FIXTURES)
def patch(request):
    """A meta-fixtures for collecting all patches used in testing."""
    return request.getfixturevalue(request.param)


# --- Spool setup fixtures


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


@pytest.fixture(scope="session")
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
    d_time = random_patch.attrs["d_time"]

    pa2 = random_patch.update_attrs(time_min=t2 + d_time)
    t3 = pa2.attrs["time_max"]

    pa3 = pa2.update_attrs(time_min=t3 + d_time)

    expected_time = pa3.attrs["time_max"] - pa1.attrs["time_min"]
    actual_time = pa3.coords["time"].max() - pa1.coords["time"].min()
    assert expected_time == actual_time
    return dc.spool([pa2, pa1, pa3])


@pytest.fixture(scope="class")
@register_func(SPOOL_FIXTURES)
def one_file_file_spool(one_file_dir):
    """Create a directory with a single DAS file."""
    return DirectorySpool(one_file_dir).update()


@pytest.fixture(scope="class")
@register_func(SPOOL_FIXTURES)
def diverse_spool():
    """Create a spool with a diverse set of patches for testing."""
    return ex._diverse_spool()


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
    out = DirectorySpool(two_patch_directory)
    return out.update()


@pytest.fixture(scope="class", params=SPOOL_FIXTURES)
def spool(request):
    """A meta-fixtures for collecting all spools used in testing."""
    return request.getfixturevalue(request.param)


# --- Misc. test fixtures


@pytest.fixture
def generic_hdf5(tmp_path):
    """
    Create a generic hdf5 file (not das). This is useful for ensuring formatters
    recognize differences in HDF5 files.
    """
    parent = tmp_path / "sum"
    parent.mkdir()
    path = parent / "simple.hdf5"

    with tb.open_file(str(path), "w") as fi:
        group = fi.create_group("/", "bob")
        fi.create_carray(group, "data", obj=np.random.rand(10))
    return path


@pytest.fixture(scope="class")
def dummy_text_file(tmp_path_factory):
    """Return a text file with silliness in it."""
    parent = tmp_path_factory.mktemp("dummy")
    path = parent / "hello.txt"
    path.write_text("Clearly not a hdf5 file. Or is it?")
    return path
