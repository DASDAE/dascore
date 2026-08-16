"""
Fixtures for testing dascore's utilities.

Patch data can be backed by any array library implementing the array API
standard, so the backend fixtures here run a test once per backend without
the test naming any of them. See "Testing array backends" in
docs/contributing/testing.qmd.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import dascore as dc

# Array backends dascore is tested against, and how to build one of their
# arrays. array_api_strict implements the standard natively and rejects any
# numpy leak; dask does not implement it at all and is only usable through
# array-api-compat's wrapper. Both are test dependencies, but some
# environments (eg wasm) install dascore without the test extras.
BACKENDS = {
    "array_api_strict": ("array_api_strict", "asarray"),
    "dask": ("dask.array", "from_array"),
}


@pytest.fixture(params=sorted(BACKENDS), scope="class")
def backend(request) -> str:
    """
    The name of the array backend under test.

    Parametrized, so asking for this fixture, or for one which needs it,
    runs the test against every backend in BACKENDS. The import happens
    here rather than at the top of the module so an environment without
    the test extras skips these tests instead of the whole file.
    """
    module_name, _ = BACKENDS[request.param]
    pytest.importorskip(module_name)
    return request.param


@pytest.fixture(scope="class")
def to_array(backend):
    """Return a function which moves an array to the backend under test."""
    module_name, func_name = BACKENDS[backend]
    return getattr(importlib.import_module(module_name), func_name)


@pytest.fixture(scope="class")
def to_backend(to_array):
    """Return a function which moves a patch's data to the backend."""

    def _to_backend(patch: dc.Patch) -> dc.Patch:
        """Return the patch with its data on the array backend."""
        return patch.new(data=to_array(np.asarray(patch.data)))

    return _to_backend
