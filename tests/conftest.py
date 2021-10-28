"""
pytest configuration for dfs
"""
from pathlib import Path

import numpy as np
import pytest

from dfs.core import Trace2D, Stream
from dfs.io import _read_terra15_v2
from dfs.utils.downloader import fetch
from dfs.utils.misc import register_func


test_data_path = Path(__file__).parent.absolute() / "test_data"

STREAM_FIXTURES = []
ARRAY_FIXTURES = []


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
@register_func(ARRAY_FIXTURES)
def terra15_das_array(terra15_path) -> Trace2D:
    """Read the terra15 data, return contained DataArray"""
    return _read_terra15_v2(terra15_path)[0]


@pytest.fixture(scope="session")
def random_das_dict():
    """A dict of all the components for making a DAS array."""
    rand = np.random.RandomState(13)
    array = rand.random(size=(300, 2_000))
    attrs = dict(dx=1, dt=1 / 250.0, category="DAS", id="test_data1")
    out = dict(
        data=array,
        time=np.arange(array.shape[1]) * attrs["dt"],
        distance=np.arange(array.shape[0]) * attrs["dx"],
        attrs=attrs,
    )
    return out


@pytest.fixture(scope="session")
@register_func(ARRAY_FIXTURES)
def random_das_array(random_das_dict) -> Trace2D:
    """Init a random array."""
    return Trace2D(**random_das_dict)


@pytest.fixture(scope="class")
def dummy_text_file(tmp_path_factory):
    """Return a text file with silliness in it."""
    parent = tmp_path_factory.mktemp("dummy")
    path = parent / "hello.txt"
    path.write_text("Clearly not a hdf5 file. Or is it?")
    return path


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
