"""Tests for array API utilities and array backend support."""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import numpy as np
import pytest

import dascore as dc
from dascore.utils.array_api import (
    asarray_like,
    backend_name,
    is_numpy,
    to_numpy,
)
from dascore.utils.misc import suppress_warnings
from dascore.utils.patch import _get_backend_name
from dascore.warnings import NumpyFallbackWarning

# array_api_strict is a test dependency, but some environments (eg wasm)
# install dascore without the test extras.
xps = pytest.importorskip("array_api_strict")


@contextmanager
def warnings_as_errors():
    """A context manager which raises rather than warns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


@pytest.fixture(scope="module")
def strict_patch(random_patch) -> dc.Patch:
    """Return a patch whose data are backed by array_api_strict."""
    return random_patch.new(data=xps.asarray(np.asarray(random_patch.data)))


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

    def test_strict(self):
        """Other backends report their top-level package name."""
        assert backend_name(xps.asarray([1, 2])) == "array_api_strict"


class TestIsNumpy:
    """Tests for detecting numpy arrays."""

    def test_numpy(self):
        """Numpy arrays are numpy arrays."""
        assert is_numpy(np.array([1, 2]))

    def test_not_numpy(self):
        """Arrays from other backends are not."""
        assert not is_numpy(xps.asarray([1, 2]))


class TestToNumpy:
    """Tests for converting arrays to numpy."""

    def test_numpy_returns_same_array(self):
        """Numpy arrays are returned without a copy."""
        array = np.array([1, 2])
        assert to_numpy(array) is array

    def test_other_backend(self):
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

    def test_numpy_like(self):
        """A numpy template returns a numpy array."""
        out = asarray_like(xps.asarray([1.0, 2.0]), np.array([1.0]))
        assert isinstance(out, np.ndarray)

    def test_other_backend_like(self):
        """A non-numpy template returns that backend's array."""
        out = asarray_like(np.array([1.0, 2.0]), xps.asarray([1.0]))
        assert backend_name(out) == "array_api_strict"


class TestPatchBackends:
    """Tests for patches backed by a non-numpy array library."""

    def test_patch_keeps_backend(self, strict_patch):
        """Creating a patch does not convert the data to numpy."""
        assert backend_name(strict_patch.data) == "array_api_strict"

    def test_array_api_function_keeps_backend(self, strict_patch):
        """Functions written against the standard don't warn or convert."""
        with warnings_as_errors():
            out = strict_patch.transpose()
        assert backend_name(out.data) == "array_api_strict"
        assert out.dims == strict_patch.dims[::-1]

    def test_squeeze_keeps_backend(self, strict_patch):
        """Squeeze also works on any backend."""
        with suppress_warnings(NumpyFallbackWarning):
            patch = strict_patch.select(distance=0, samples=True)
        with warnings_as_errors():
            out = patch.squeeze()
        assert backend_name(out.data) == "array_api_strict"
        assert "distance" not in out.dims

    def test_numpy_only_function_warns(self, strict_patch):
        """Numpy-only functions warn but preserve the input backend."""
        with pytest.warns(NumpyFallbackWarning, match="detrend"):
            out = strict_patch.detrend("time")
        assert backend_name(out.data) == "array_api_strict"
        assert out.shape == strict_patch.shape

    def test_to_numpy_array(self, strict_patch):
        """Patches from any backend convert to numpy arrays."""
        array = np.asarray(strict_patch)
        assert isinstance(array, np.ndarray)
        assert array.shape == strict_patch.shape

    def test_str(self, strict_patch):
        """Patches from any backend have a string representation."""
        assert "Patch" in str(strict_patch)


def test_suppress_fallback_warning(random_patch):
    """The fallback warning can be silenced like any other dascore warning."""
    patch = random_patch.new(data=xps.asarray(np.asarray(random_patch.data)))
    with suppress_warnings(NumpyFallbackWarning):
        out = patch.detrend("time")
    assert backend_name(out.data) == "array_api_strict"


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
