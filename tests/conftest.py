"""
pytest configuration for fios
"""
from pathlib import Path

import pytest

import fios
from fios.core import Patch, Stream
from fios.io.terra15.ver2 import _read_terra15_v2
from fios.utils.downloader import fetch
from fios.utils.misc import register_func

test_data_path = Path(__file__).parent.absolute() / "test_data"

STREAM_FIXTURES = []
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


# --- Test fixtures


@pytest.fixture(scope="session")
def terra15_path():
    """Return the path to the example terra15 file."""
    out = fetch("terra15_v2_das_1_trimmed.hdf5")
    assert out.exists()
    return out


@pytest.fixture()
@register_func(STREAM_FIXTURES)
def terra15_das_stream(terra15_path) -> Stream:
    """Return the stream of Terra15 Das Array"""
    return _read_terra15_v2(terra15_path)


@pytest.fixture(scope="session")
@register_func(PATCH_FIXTURES)
def terra15_das_patch(terra15_path) -> Patch:
    """Read the terra15 data, return contained DataArray"""
    return _read_terra15_v2(terra15_path)[0]


@pytest.fixture(scope="session")
@register_func(PATCH_FIXTURES)
def random_patch() -> Patch:
    """Init a random array."""
    from fios.examples import get_example_patch

    return get_example_patch("random_das")


@pytest.fixture(scope="session")
@register_func(STREAM_FIXTURES)
def random_stream() -> Stream:
    """Init a random array."""
    from fios.examples import get_example_stream

    return get_example_stream("random_das")


@pytest.fixture(scope="session", params=PATCH_FIXTURES)
def patch(request):
    """A meta-fixtures for collecting all patches used in testing."""
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="class")
def dummy_text_file(tmp_path_factory):
    """Return a text file with silliness in it."""
    parent = tmp_path_factory.mktemp("dummy")
    path = parent / "hello.txt"
    path.write_text("Clearly not a hdf5 file. Or is it?")
    return path


@pytest.fixture()
def adjacent_stream_no_overlap(random_patch) -> fios.Stream:
    """
    Create a stream with several patches within one time sample but not
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

    return fios.Stream([pa2, pa1, pa3])
