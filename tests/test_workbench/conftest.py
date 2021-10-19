"""
Tests configuration for vizualization tests.
"""
import pytest

from dfs.workbench import WorkBench


@pytest.fixture(scope="session")
def das_work_bench(terra15_das_array):
    """Return an instance of WorkBench w/ terra15 DAS."""
    return WorkBench(terra15_das_array)
