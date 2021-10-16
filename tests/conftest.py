"""
pytest configuration for dfs
"""
from pathlib import Path

import pytest

from dfs.io import read_terra15
from dfs.utils.downloader import fetch

test_data_path = Path(__file__).parent.absolute() / "test_data"


@pytest.fixture(scope="session")
def terra15_das():
    path = fetch("terra15_v2_das_1_trimmed.hdf5")
    assert path.exists()
    return read_terra15(path)


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
