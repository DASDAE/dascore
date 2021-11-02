"""
Tests configuration for vizualization tests.
"""
import pytest

import fios


@pytest.fixture(scope="session")
def das_work_bench(terra15_das_array):
    """Return an instance of WorkBench w/ terra15 DAS."""
    return fios.workbench.WorkBench(terra15_das_array)
