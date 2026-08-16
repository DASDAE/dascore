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
import pandas as pd
import pytest

import dascore as dc
from dascore.utils.array_api import (
    ARRAY_API_BACKEND,
    asarray_like,
    backend_name,
    can_nan_reduce,
    is_numpy,
    nan_reduce,
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


class _StuckArray:
    """An array-like which cannot be moved to the host at all."""

    def __init__(self, array):
        self.shape = array.shape
        self.dtype = array.dtype
        self.device = "nowhere"

    def __array__(self, dtype=None, copy=None):
        msg = "Implicit conversion to a numpy array is not allowed."
        raise TypeError(msg)

    def to_device(self, device, stream=None):
        """Refuse, as a device with no host transfer would."""
        msg = f"cannot move to {device}"
        raise ValueError(msg)


class _DeviceArray:
    """An array-like which must be moved to the host to become numpy."""

    def __init__(self, array):
        self._array = array
        self.shape = array.shape
        self.dtype = array.dtype
        self.device = "elsewhere"

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

    def test_device_array_on_another_device(self, xps):
        """A backend which names its own devices still converts."""
        device = xps.__array_namespace_info__().devices()[1]
        array = xps.asarray(np.array([1.0, 2.0]), device=device)
        assert np.allclose(to_numpy(array), [1.0, 2.0])

    def test_keeps_the_real_error(self):
        """A failure which has nothing to do with devices keeps its own."""
        dask_array = pytest.importorskip("dask.array")

        def _boom(block):
            """Fail the way a user's function would."""
            msg = "the user's function raised"
            raise ValueError(msg)

        array = dask_array.from_array(np.arange(4.0)).map_blocks(_boom, dtype=float)
        with pytest.raises(ValueError, match="the user's function raised"):
            to_numpy(array)

    def test_stuck_array(self):
        """An array which cannot reach the host keeps its own error."""
        with pytest.raises(TypeError, match="Implicit conversion"):
            to_numpy(_StuckArray(np.array([1.0, 2.0])))


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


def _assert_coords_match(out, expected):
    """
    Assert two patches have the same coordinates.

    Compared value by value rather than with ==, since a coordinate can
    hold nulls, eg after a pad which doesn't expand it, and a null is not
    equal to itself.
    """
    assert out.coords.dims == expected.coords.dims
    assert set(out.coords.coord_map) == set(expected.coords.coord_map)
    for name in expected.coords.coord_map:
        left, right = out.get_array(name), expected.get_array(name)
        assert left.dtype == right.dtype
        null = pd.isnull(left)
        assert np.array_equal(null, pd.isnull(right))
        assert np.array_equal(left[~null], right[~null])


def _identity(patch):
    """Return the patch unchanged."""
    return patch


class _Case(NamedTuple):
    """
    How to exercise one patch function on a non-numpy backend.

    Each entry in calls is run as its own test, so covering another
    argument combination costs one line.
    """

    calls: tuple[Callable, ...]
    setup: Callable = _identity


# Every patch function which declares the array API backend needs an entry
# here, which is also the inventory of what has been converted so far. The
# test below collects the patch functions dascore defines and fails if one
# declares the array API backend without an entry.
#
# setup runs on the numpy patch, before it is moved to another backend, so
# a function which is still numpy backed can be used to build the input
# without its own conversion warning muddying the case.
ARRAY_API_CASES = {
    "dascore.proc.aggregate.all": _Case(calls=(lambda patch: patch.all("time"),)),
    "dascore.proc.aggregate.any": _Case(calls=(lambda patch: patch.any("time"),)),
    "dascore.proc.aggregate.max": _Case(calls=(lambda patch: patch.max("time"),)),
    "dascore.proc.aggregate.mean": _Case(calls=(lambda patch: patch.mean("time"),)),
    "dascore.proc.aggregate.min": _Case(calls=(lambda patch: patch.min("time"),)),
    "dascore.proc.aggregate.std": _Case(calls=(lambda patch: patch.std("time"),)),
    "dascore.proc.aggregate.sum": _Case(calls=(lambda patch: patch.sum("time"),)),
    "dascore.proc.basic.abs": _Case(calls=(lambda patch: patch.abs(),)),
    "dascore.proc.basic.angle": _Case(
        calls=(lambda patch: patch.angle(),),
        setup=lambda patch: patch.dft("time"),
    ),
    "dascore.proc.basic.conj": _Case(
        calls=(lambda patch: patch.conj(),),
        setup=lambda patch: patch.dft("time"),
    ),
    "dascore.proc.basic.demean": _Case(calls=(lambda patch: patch.demean("time"),)),
    "dascore.proc.basic.flip": _Case(
        calls=(
            lambda patch: patch.flip("time"),
            lambda patch: patch.flip(*patch.dims),
            lambda patch: patch.flip("time", flip_coords=False),
        ),
    ),
    "dascore.proc.basic.full": _Case(
        calls=(lambda patch: patch.full(1.0), lambda patch: patch.full(0)),
    ),
    "dascore.proc.basic.imag": _Case(
        calls=(lambda patch: patch.imag(),),
        setup=lambda patch: patch.dft("time"),
    ),
    "dascore.proc.basic.normalize": _Case(
        calls=(
            lambda patch: patch.normalize("time", norm="l1"),
            lambda patch: patch.normalize("time", norm="l2"),
            lambda patch: patch.normalize("time", norm="max"),
            lambda patch: patch.normalize("time", norm="bit"),
        ),
    ),
    "dascore.proc.basic.real": _Case(
        calls=(lambda patch: patch.real(),),
        setup=lambda patch: patch.dft("time"),
    ),
    "dascore.proc.basic.roll": _Case(
        calls=(
            lambda patch: patch.roll(time=5, samples=True),
            lambda patch: patch.roll(time=5, samples=True, update_coord=True),
        ),
    ),
    "dascore.proc.basic.standardize": _Case(
        calls=(lambda patch: patch.standardize("time"),),
    ),
    "dascore.proc.coords.update_coords": _Case(
        calls=(
            lambda patch: patch.update_coords(distance=patch.get_array("distance") + 1),
        ),
    ),
    "dascore.proc.coords.make_broadcastable_to": _Case(
        calls=(lambda patch: patch.make_broadcastable_to((patch.shape[0], 3)),),
        setup=lambda patch: patch.mean("time"),
    ),
    "dascore.proc.coords.squeeze": _Case(
        calls=(lambda patch: patch.squeeze(),),
        setup=lambda patch: patch.select(distance=0, samples=True),
    ),
    "dascore.proc.coords.transpose": _Case(calls=(lambda patch: patch.transpose(),)),
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


# One test per argument combination, per backend.
_CASE_CALLS = [
    (name, index)
    for name, case in sorted(ARRAY_API_CASES.items())
    for index in range(len(case.calls))
]

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

    @pytest.mark.parametrize("name,index", _CASE_CALLS)
    def test_backend_preserved(self, name, index, random_patch, to_backend, backend):
        """The function runs on another backend, unconverted, with no warning."""
        case = ARRAY_API_CASES[name]
        call = case.calls[index]
        numpy_patch = case.setup(random_patch)
        patch = to_backend(numpy_patch)
        with warnings_as_errors():
            out = call(patch)
        # A function which returns its input proves nothing about the backend.
        assert out is not patch
        assert backend_name(out.data) == backend
        # The whole patch must match what the numpy implementation returns.
        expected = call(numpy_patch)
        array = np.asarray(out.data)
        assert array.dtype == expected.data.dtype
        assert out.dims == expected.dims
        assert np.allclose(array, np.asarray(expected.data), equal_nan=True)
        _assert_coords_match(out, expected)
        # History records the arguments, whose repr differs by backend.
        exclude = {"history"}
        assert out.attrs.model_dump(exclude=exclude) == expected.attrs.model_dump(
            exclude=exclude
        )
        assert len(out.attrs.history) == len(expected.attrs.history)


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


class TestDevices:
    """Tests for patches whose data are not on the backend's default device."""

    @pytest.fixture(scope="class")
    def device(self, xps):
        """A device which is not the default one."""
        # array_api_strict has fake devices for exactly this purpose.
        return xps.__array_namespace_info__().devices()[1]

    @pytest.fixture(scope="class")
    def device_patch(self, random_patch, xps, device) -> dc.Patch:
        """A patch whose data live on a non-default device."""
        data = xps.asarray(np.asarray(random_patch.data), device=device)
        return random_patch.new(data=data)

    @pytest.mark.parametrize(
        "call",
        [
            lambda patch: patch.pad(time=(1, 2), samples=True),
            lambda patch: patch.fillna(0),
            lambda patch: patch.full(1.0),
            lambda patch: patch.normalize("time", norm="bit"),
            lambda patch: patch.where(patch.data > 0),
            lambda patch: patch.mean("time"),
            lambda patch: patch.std("time"),
            lambda patch: patch.min("time"),
            lambda patch: patch.demean("time"),
        ],
    )
    def test_output_device(self, call, device_patch, device):
        """Data allocated by a patch function land on the input's device."""
        with suppress_warnings(NumpyFallbackWarning):
            out = call(device_patch)
        assert out.data.device == device


class TestNanReduce:
    """Tests for reductions which ignore nan values."""

    names = ("min", "max", "mean", "std", "sum")

    @pytest.fixture(scope="class")
    def numpy_array(self):
        """An array with a scattered nan, and a slice of nothing but nans."""
        array = np.linspace(-2, 2, 24).reshape(4, 6)
        array[1, 2] = np.nan
        array[3, :] = np.nan
        return array

    @pytest.mark.parametrize("name", names)
    @pytest.mark.parametrize("axis", [0, 1, None])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_matches_numpy(self, name, axis, keepdims, numpy_array, to_array):
        """The reductions agree with numpy, including on all-nan slices."""
        array = to_array(numpy_array)
        with suppress_warnings(RuntimeWarning):
            expected = getattr(np, f"nan{name}")(
                numpy_array, axis=axis, keepdims=keepdims
            )
        out = np.asarray(nan_reduce(name, array, axis=axis, keepdims=keepdims))
        assert out.shape == expected.shape
        assert np.allclose(out, expected, equal_nan=True)

    @pytest.mark.parametrize("name", names)
    def test_no_nans(self, name, to_array):
        """The reductions agree with numpy when there is nothing to skip."""
        array = np.linspace(1, 5, 12).reshape(3, 4)
        out = np.asarray(nan_reduce(name, to_array(array), axis=1))
        assert np.allclose(out, getattr(np, f"nan{name}")(array, axis=1))

    @pytest.mark.parametrize("name", names)
    @pytest.mark.parametrize(
        "dtype", ["bool", "int64", "float32", "float64", "complex128"]
    )
    def test_dtypes_match_numpy(self, name, dtype, to_array):
        """Each reduction matches numpy's value and dtype for each dtype."""
        array = np.array([1, 0, 3, 2], dtype=dtype).reshape(2, 2)
        with suppress_warnings():
            expected = getattr(np, f"nan{name}")(array, axis=0)
            out = np.asarray(nan_reduce(name, to_array(array), axis=0))
        assert out.dtype == expected.dtype
        assert np.allclose(out, expected, equal_nan=True)

    @pytest.mark.parametrize("name", names)
    @pytest.mark.parametrize(
        "values",
        [
            [np.inf, np.inf],
            [np.inf, 1.0],
            [np.nan, np.inf],
            [1 + 1j, 1 - 1j],
            [1 + 1j, np.nan],
        ],
    )
    def test_hard_values(self, name, values, to_array):
        """Values where numpy's answer is easy to get wrong."""
        array = np.array(values)
        with suppress_warnings():
            expected = np.asarray(getattr(np, f"nan{name}")(array))
            out = np.asarray(nan_reduce(name, to_array(array)))
        assert out.dtype == expected.dtype
        assert np.allclose(out, expected, equal_nan=True)

    @pytest.mark.parametrize("name", ["min", "max"])
    def test_infinities(self, name, to_array):
        """Infinities are values like any other, not a missing-data marker."""
        array = np.array([np.nan, np.inf, -np.inf])
        out = np.asarray(nan_reduce(name, to_array(array)))
        assert out == getattr(np, f"nan{name}")(array)

    @pytest.mark.parametrize("name", names)
    def test_empty(self, name, to_array):
        """Reducing nothing does what numpy does, including refusing to."""
        array = np.array([], dtype="float64")
        with suppress_warnings(RuntimeWarning):
            # Neither numpy nor the standard has an identity for min or max.
            if name in {"min", "max"}:
                with pytest.raises(ValueError):
                    np.asarray(nan_reduce(name, to_array(array)))
                return
            expected = getattr(np, f"nan{name}")(array)
            out = np.asarray(nan_reduce(name, to_array(array)))
        assert np.allclose(out, expected, equal_nan=True)

    def test_unknown_name(self):
        """A reduction dascore doesn't have is an error, not a std."""
        with pytest.raises(ValueError, match="not a reduction"):
            nan_reduce("median", np.array([1.0, 2.0]))

    def test_can_nan_reduce(self, to_array, backend):
        """Only reductions the standard defines for a dtype can be done."""
        assert can_nan_reduce("mean", to_array(np.array([1.0, 2.0])))
        # dask has its own nan reductions, so it can do them all.
        booleans = to_array(np.array([True, False]))
        assert can_nan_reduce("min", booleans) == (backend == "dask")

    @pytest.mark.parametrize("name", names)
    def test_integer_data(self, name, to_array):
        """Integer data have no nans to skip, but must still reduce."""
        array = np.arange(12, dtype="int64").reshape(3, 4)
        out = np.asarray(nan_reduce(name, to_array(array), axis=0))
        assert np.allclose(out, getattr(np, f"nan{name}")(array, axis=0))
