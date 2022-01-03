"""
Tests configuration for vizualization tests.
"""
import pytest

import dascore


@pytest.fixture(scope="session")
def das_work_bench(terra15_das_patch):
    """Return an instance of WorkBench w/ terra15 DAS."""
    return dascore.workbench.WorkBench(terra15_das_patch)
