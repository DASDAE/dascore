"""Fixtures for testing dascore's utilities."""

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
    """The name of the array backend under test."""
    module_name, _ = BACKENDS[request.param]
    pytest.importorskip(module_name)
    return request.param


@pytest.fixture(scope="class")
def to_backend(backend):
    """Return a function which moves a patch's data to the backend."""
    module_name, func_name = BACKENDS[backend]
    factory = getattr(importlib.import_module(module_name), func_name)

    def _to_backend(patch: dc.Patch) -> dc.Patch:
        """Return the patch with its data on the array backend."""
        return patch.new(data=factory(np.asarray(patch.data)))

    return _to_backend
