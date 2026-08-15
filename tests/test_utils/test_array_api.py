"""Tests for array API utilities and array backend support."""

from __future__ import annotations

import importlib
import pkgutil
import sys
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from typing import NamedTuple

import numpy as np
import pytest

import dascore as dc
from dascore.utils.array_api import (
    ARRAY_API_BACKEND,
    asarray_like,
    backend_name,
    is_numpy,
    to_numpy,
)
from dascore.utils.misc import suppress_warnings
from dascore.utils.patch import _get_backend_name
from dascore.warnings import NumpyFallbackWarning


@pytest.fixture(scope="module")
def xps():
    """The reference implementation of the array API standard."""
    # It is a test dependency, but some environments (eg wasm) install
    # dascore without the test extras.
    return pytest.importorskip("array_api_strict")


@contextmanager
def warnings_as_errors():
    """A context manager which raises rather than warns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


@pytest.fixture(scope="class")
def backend_patch(random_patch, to_backend) -> dc.Patch:
    """Return a patch whose data are on the array backend under test."""
    return to_backend(random_patch)


class _DeviceArray:
    """An array-like which must be moved to the host to become numpy."""

    def __init__(self, array):
        self._array = array
        self.shape = array.shape
        self.dtype = array.dtype

    def __array__(self, dtype=None, copy=None):
        msg = "Implicit conversion to a numpy array is not allowed."
        raise TypeError(msg)

    def to_device(self, device, stream=None):
        """Return the host array, like cupy's asnumpy does."""
        assert device == "cpu"
        return self._array


class TestBackendName:
    """Tests for getting the name of an array's backend."""

    def test_numpy(self):
        """Numpy arrays report the numpy backend."""
        assert backend_name(np.array([1, 2])) == "numpy"

    def test_strict(self, xps):
        """Other backends report their top-level package name."""
        assert backend_name(xps.asarray([1, 2])) == "array_api_strict"


class TestIsNumpy:
    """Tests for detecting numpy arrays."""

    def test_numpy(self):
        """Numpy arrays are numpy arrays."""
        assert is_numpy(np.array([1, 2]))

    def test_not_numpy(self, xps):
        """Arrays from other backends are not."""
        assert not is_numpy(xps.asarray([1, 2]))


class TestToNumpy:
    """Tests for converting arrays to numpy."""

    def test_numpy_returns_same_array(self):
        """Numpy arrays are returned without a copy."""
        array = np.array([1, 2])
        assert to_numpy(array) is array

    def test_other_backend(self, xps):
        """Other backends are converted to numpy arrays."""
        out = to_numpy(xps.asarray([1.0, 2.0]))
        assert isinstance(out, np.ndarray)
        assert np.allclose(out, [1.0, 2.0])

    def test_device_array(self):
        """Arrays which refuse implicit conversion are moved to the host."""
        array = np.array([1.0, 2.0])
        out = to_numpy(_DeviceArray(array))
        assert isinstance(out, np.ndarray)
        assert np.allclose(out, array)


class TestAsArrayLike:
    """Tests for converting arrays back to another backend."""

    def test_numpy_like(self, xps):
        """A numpy template returns a numpy array."""
        out = asarray_like(xps.asarray([1.0, 2.0]), np.array([1.0]))
        assert isinstance(out, np.ndarray)

    def test_other_backend_like(self, xps):
        """A non-numpy template returns that backend's array."""
        out = asarray_like(np.array([1.0, 2.0]), xps.asarray([1.0]))
        assert backend_name(out) == "array_api_strict"


class TestPatchBackends:
    """Tests for patches backed by a non-numpy array library."""

    def test_patch_keeps_backend(self, backend_patch, backend):
        """Creating a patch does not convert the data to numpy."""
        assert backend_name(backend_patch.data) == backend

    def test_array_api_function_keeps_backend(self, backend_patch, backend):
        """Functions written against the standard don't warn or convert."""
        with warnings_as_errors():
            out = backend_patch.transpose()
        assert backend_name(out.data) == backend
        assert out.dims == backend_patch.dims[::-1]

    def test_squeeze_keeps_backend(self, backend_patch, backend):
        """Squeeze also works on any backend."""
        with suppress_warnings(NumpyFallbackWarning):
            patch = backend_patch.select(distance=0, samples=True)
        with warnings_as_errors():
            out = patch.squeeze()
        assert backend_name(out.data) == backend
        assert "distance" not in out.dims

    def test_numpy_only_function_warns(self, backend_patch, backend):
        """Numpy-only functions warn but preserve the input backend."""
        with pytest.warns(NumpyFallbackWarning, match="detrend"):
            out = backend_patch.detrend("time")
        assert backend_name(out.data) == backend
        assert out.shape == backend_patch.shape

    def test_to_numpy_array(self, backend_patch):
        """Patches from any backend convert to numpy arrays."""
        array = np.asarray(backend_patch)
        assert isinstance(array, np.ndarray)
        assert array.shape == backend_patch.shape

    def test_str(self, backend_patch):
        """Patches from any backend have a string representation."""
        assert "Patch" in str(backend_patch)


def test_suppress_fallback_warning(random_patch, to_backend, backend):
    """The fallback warning can be silenced like any other dascore warning."""
    patch = to_backend(random_patch)
    with suppress_warnings(NumpyFallbackWarning):
        out = patch.detrend("time")
    assert backend_name(out.data) == backend


def test_backend_name_of_non_array():
    """Objects which don't carry array data dispatch to numpy."""
    assert _get_backend_name(object()) == "numpy"


class _ArrayLike:
    """An array-like which numpy can consume but which isn't standard."""

    def __init__(self, array):
        self._array = array
        self.shape = array.shape
        self.dtype = array.dtype

    def __array__(self, dtype=None, copy=None):
        return self._array


class TestNonStandardArrayLike:
    """Tests for patch data which doesn't implement the array API."""

    @pytest.fixture(scope="class")
    def array_like_patch(self, random_patch) -> dc.Patch:
        """A patch whose data only implement __array__."""
        data = _ArrayLike(np.asarray(random_patch.data))
        return random_patch.new(data=data)

    def test_namespace_is_numpy(self, array_like_patch):
        """Such arrays are handled by numpy, so they report numpy."""
        assert backend_name(array_like_patch.data) == "numpy"

    def test_numpy_function(self, array_like_patch):
        """Numpy-only functions work as they did before dispatch existed."""
        with warnings_as_errors():
            out = array_like_patch.detrend("time")
        assert isinstance(out.data, np.ndarray)

    def test_array_api_function(self, array_like_patch):
        """So do functions written against the standard."""
        with warnings_as_errors():
            out = array_like_patch.transpose()
        assert out.dims == array_like_patch.dims[::-1]


def _identity(patch):
    """Return the patch unchanged."""
    return patch


class _Case(NamedTuple):
    """How to exercise one patch function on a non-numpy backend."""

    call: Callable
    setup: Callable = _identity


# Every patch function which declares the array API backend needs an entry
# here, which is also the inventory of what has been converted so far.
# setup runs on the numpy patch, before it is moved to another backend.
ARRAY_API_CASES = {
    "dascore.proc.coords.transpose": _Case(call=lambda patch: patch.transpose()),
    "dascore.proc.coords.make_broadcastable_to": _Case(
        call=lambda patch: patch.make_broadcastable_to((patch.shape[0], 3)),
        setup=lambda patch: patch.mean("time"),
    ),
    "dascore.proc.coords.squeeze": _Case(
        call=lambda patch: patch.squeeze(),
        setup=lambda patch: patch.select(distance=0, samples=True),
    ),
}


def _get_patch_functions() -> dict[str, Callable]:
    """Return every patch function dascore defines, keyed by qualified name."""
    for module in pkgutil.walk_packages(dc.__path__, "dascore."):
        importlib.import_module(module.name)
    out = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith("dascore"):
            continue
        for obj in vars(module).values():
            if callable(obj) and isinstance(getattr(obj, "backends", None), dict):
                out[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return out


PATCH_FUNCTIONS = _get_patch_functions()
ARRAY_API_FUNCTIONS = {
    i: v for i, v in PATCH_FUNCTIONS.items() if ARRAY_API_BACKEND in v.backends
}


class TestArrayApiPatchFunctions:
    """Every patch function which declares the array API must work on it."""

    def test_patch_functions_found(self):
        """The discovery finds dascore's patch functions."""
        assert len(PATCH_FUNCTIONS) > 50
        assert "dascore.proc.detrend.detrend" in PATCH_FUNCTIONS

    def test_every_function_has_a_case(self):
        """Declaring the array API backend requires proving it works."""
        missing = sorted(set(ARRAY_API_FUNCTIONS) - set(ARRAY_API_CASES))
        assert not missing, f"add an ARRAY_API_CASES entry for: {missing}"

    def test_no_stale_cases(self):
        """Cases for functions which no longer declare the array API."""
        stale = sorted(set(ARRAY_API_CASES) - set(ARRAY_API_FUNCTIONS))
        assert not stale, f"remove the ARRAY_API_CASES entry for: {stale}"

    @pytest.mark.parametrize("name", sorted(ARRAY_API_CASES))
    def test_backend_preserved(self, name, random_patch, to_backend, backend):
        """The function runs on another backend, unconverted, with no warning."""
        case = ARRAY_API_CASES[name]
        numpy_patch = case.setup(random_patch)
        patch = to_backend(numpy_patch)
        with warnings_as_errors():
            out = case.call(patch)
        # A function which returns its input proves nothing about the backend.
        assert out is not patch
        assert backend_name(out.data) == backend
        # The whole patch must match what the numpy implementation returns.
        expected = case.call(numpy_patch)
        array = np.asarray(out.data)
        assert array.dtype == expected.data.dtype
        assert out.dims == expected.dims
        assert out.coords == expected.coords
        assert out.attrs == expected.attrs
        assert np.allclose(array, np.asarray(expected.data), equal_nan=True)


class TestRegisterBackend:
    """Tests for naming the backend an implementation is registered under."""

    @pytest.fixture
    def patch_func(self):
        """A patch function with only a numpy implementation."""

        @dc.patch_function()
        def func(patch):
            return patch

        return func

    def test_string(self, patch_func):
        """A backend can be named with a string."""
        patch_func.register("array_api_strict")(_identity)
        assert "array_api_strict" in patch_func.backends

    def test_namespace(self, patch_func, xps):
        """It can also be named with the array namespace itself."""
        patch_func.register(xps)(_identity)
        assert "array_api_strict" in patch_func.backends

    def test_array(self, patch_func, xps):
        """Or with an example array."""
        patch_func.register(xps.asarray([1.0]))(_identity)
        assert "array_api_strict" in patch_func.backends

    def test_decorator_argument(self):
        """The decorator's backend argument accepts the same forms."""

        @dc.patch_function(backend=np)
        def func(patch):
            return patch

        assert set(func.backends) == {"numpy"}
