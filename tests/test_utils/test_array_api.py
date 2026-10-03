"""Tests for array API utilities and array backend support."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from contextlib import contextmanager
from typing import NamedTuple

import numpy as np
import pytest
from scipy.signal import hilbert as sp_hilbert

import dascore as dc
from dascore.core.coords import concat_coords
from dascore.utils.array_api import (
    array_namespace,
    asarray_like,
    backend_name,
    device,
    is_numpy,
    nan_reduce,
    to_numpy,
)
from dascore.utils.misc import suppress_warnings
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
        patch = backend_patch.select(distance=0, samples=True)
        with warnings_as_errors():
            out = patch.squeeze()
        assert backend_name(out.data) == backend
        assert "distance" not in out.dims

    def test_numpy_only_function_does_not_warn(self, backend_patch):
        """Nothing converts or warns for a plain numpy-only function."""
        # stalta is not a processor; its rolling means go through numpy or
        # pandas, which give numpy back.
        with warnings_as_errors():
            out = backend_patch.stalta(time=(0.01, 0.1))
        assert out.shape == backend_patch.shape

    def test_numpy_kernel_falls_back_with_a_warning(self, backend_patch):
        """A processor's numpy kernel runs on numpy copies, and says so."""
        with pytest.warns(NumpyFallbackWarning, match="detrend"):
            out = backend_patch.detrend("time")
        assert backend_name(out.data) == backend_name(backend_patch.data)
        assert out.shape == backend_patch.shape

    def test_to_numpy_array(self, backend_patch):
        """Patches from any backend convert to numpy arrays."""
        array = np.asarray(backend_patch)
        assert isinstance(array, np.ndarray)
        assert array.shape == backend_patch.shape

    def test_str(self, backend_patch):
        """Patches from any backend have a string representation."""
        assert "Patch" in str(backend_patch)


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
        """Numpy-only functions handle them, since numpy consumes them."""
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


def _make_complex(patch):
    """Return the patch with complex data, so a conjugate means something."""
    data = np.asarray(patch.data)
    return patch.new(data=(data + 1j * data[::-1]).astype("complex128"))


class _Case(NamedTuple):
    """How to exercise one patch function on a non-numpy backend."""

    call: Callable
    setup: Callable = _identity


# The patch functions whose bodies are written to the array API standard, and
# so run on any backend. Nothing on a patch function declares that, so this
# inventory is hand-kept: add an entry when you convert one.
# setup runs on the numpy patch, before it is moved to another backend.
ARRAY_API_CASES = {
    "dascore.proc.coords.transpose": _Case(call=lambda patch: patch.transpose()),
    "dascore.proc.coords.rename_coords": _Case(
        call=lambda patch: patch.rename_coords(time="t")
    ),
    "dascore.proc.basic.abs": _Case(call=lambda patch: patch.abs()),
    # conj and real hand a real patch straight back, which says nothing
    # about the backend, so they are given something to actually do.
    "dascore.proc.basic.conj": _Case(
        call=lambda patch: patch.conj(), setup=_make_complex
    ),
    "dascore.proc.basic.imag": _Case(call=lambda patch: patch.imag()),
    "dascore.proc.basic.real": _Case(
        call=lambda patch: patch.real(), setup=_make_complex
    ),
    "dascore.proc.aggregate.all": _Case(call=lambda patch: patch.all("time")),
    "dascore.proc.aggregate.any": _Case(call=lambda patch: patch.any("time")),
    "dascore.proc.aggregate.max": _Case(call=lambda patch: patch.max("time")),
    "dascore.proc.aggregate.mean": _Case(call=lambda patch: patch.mean("time")),
    "dascore.proc.aggregate.min": _Case(call=lambda patch: patch.min("time")),
    "dascore.proc.aggregate.std": _Case(call=lambda patch: patch.std("time")),
    "dascore.proc.aggregate.sum": _Case(call=lambda patch: patch.sum("time")),
    "dascore.proc.basic.demean": _Case(call=lambda patch: patch.demean("time")),
    "dascore.proc.basic.normalize": _Case(
        call=lambda patch: patch.normalize("time", norm="l2"),
    ),
    "dascore.proc.basic.standardize": _Case(
        call=lambda patch: patch.standardize("time"),
    ),
    "dascore.proc.coords.make_broadcastable_to": _Case(
        call=lambda patch: patch.make_broadcastable_to((patch.shape[0], 3)),
        setup=lambda patch: patch.mean("time"),
    ),
    "dascore.proc.coords.squeeze": _Case(
        call=lambda patch: patch.squeeze(),
        setup=lambda patch: patch.select(distance=0, samples=True),
    ),
}


class TestArrayApiPatchFunctions:
    """The patch functions listed as written to the standard must work on it."""

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

    # Every pair but (keepdims=True, axis=0): what keepdims does to a
    # reduction over the first axis, axis=1 already says. Keep the rest --
    # min/max over axis 1 with keepdims is the only cell which notices the
    # mask shape at array_api.py's all-nan check.
    @pytest.mark.parametrize("name", names)
    @pytest.mark.parametrize(
        ("axis", "keepdims"),
        [(0, False), (1, False), (None, False), (1, True), (None, True)],
    )
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

    @pytest.mark.parametrize("name", names)
    def test_integer_data(self, name, to_array):
        """Integer data have no nans to skip, but must still reduce."""
        array = np.arange(12, dtype="int64").reshape(3, 4)
        out = np.asarray(nan_reduce(name, to_array(array), axis=0))
        assert np.allclose(out, getattr(np, f"nan{name}")(array, axis=0))


def _units(patch, units="rad"):
    """Return the patch with data units a strain conversion can take."""
    return patch.update_attrs(data_units=units, gauge_length=10)


def _signed(patch):
    """Return the patch with negative values, NaNs and infinities in it."""
    data = np.asarray(patch.data) - 0.5
    data[0, :3] = [np.nan, np.inf, -np.inf]
    return patch.new(data=data)


def _single(patch):
    """Return the patch in single precision."""
    return patch.new(data=np.asarray(patch.data, dtype=np.float32))


def _velocity(patch):
    """Return the patch labelled as velocity, which strain rate needs."""
    return patch.update_attrs(data_type="velocity")


def _shifted(patch):
    """Return the patch with a shift in samples for each distance."""
    return patch.update_coords(shift=("distance", np.arange(patch.shape[0]) % 5))


def _tiled(patch):
    """Return the patch cut into a stack of tiles along time."""
    return patch.tile_apply(np.positive, mode="stack", time=64, samples=True)


def _transformed(patch):
    """Return the patch transformed along time."""
    return patch.dft("time")


def _spectrum(patch):
    """Return the one-sided spectrum of the patch along time."""
    return patch.dft("time", real=True)


def _windowed(patch):
    """Return the short-time transform of the patch along time."""
    return patch.stft(time=64, samples=True)


def _holed(patch):
    """Return the first 190 times of the patch with ten samples missing."""
    coord = patch.get_coord("time")
    holed = concat_coords(coord[:100], coord[110:200])
    return patch.isel(time=slice(0, 190)).update_coords(time=holed)


# One call per operation converted to a processor, and the numpy patch it is
# given (on each backend too); the results must match numpy's.
_CONVERTED = {
    "angle": (lambda p: p.angle(), _signed),
    "demedian": (lambda p: p.demedian("time"), None),
    "fillna": (lambda p: p.fillna(0), _signed),
    "fillna_no_inf": (lambda p: p.fillna(0, include_inf=False), _signed),
    "full": (lambda p: p.full(2.0), None),
    "flip": (lambda p: p.flip("time"), None),
    "roll": (lambda p: p.roll(time=3, samples=True), None),
    "squeeze": (lambda p: p.isel(time=slice(0, 1)).squeeze(), None),
    "squeeze_one": (
        lambda p: p.isel(time=slice(0, 1)).append_dims("new").squeeze("time"),
        None,
    ),
    "append_dims": (lambda p: p.append_dims(new=2), None),
    "make_broadcastable_to": (
        lambda p: p.append_dims("new").make_broadcastable_to((*p.shape, 3)),
        None,
    ),
    "drop_coords": (
        lambda p: p.update_coords(extra=("time", np.ones(p.shape[1]))).drop_coords(
            "extra"
        ),
        None,
    ),
    "drop_private_coords": (lambda p: p.drop_private_coords(), None),
    "update_coords": (lambda p: p.update_coords(time_min=0), None),
    "set_units": (lambda p: p.set_units("m/s"), None),
    "convert_units": (lambda p: p.set_units("m/s").convert_units("mm/s"), None),
    "convert_units_offset": (lambda p: p.set_units("degC").convert_units("K"), None),
    "simplify_units": (lambda p: p.set_units("km/s").simplify_units(), None),
    "detrend": (lambda p: p.detrend("time"), None),
    "sobel_filter": (lambda p: p.sobel_filter("time"), None),
    "hilbert": (lambda p: p.hilbert("time"), None),
    "hilbert_single": (lambda p: p.hilbert("time"), _single),
    "envelope": (lambda p: p.envelope("time"), None),
    "kurtosis": (lambda p: p.kurtosis(time=8, samples=True), None),
    "radians_to_strain": (lambda p: _units(p, "mrad").radians_to_strain(), None),
    "pass_filter": (lambda p: p.pass_filter(time=(None, 100)), None),
    "notch_filter": (lambda p: p.notch_filter(time=60, q=30), None),
    "median_filter": (lambda p: p.median_filter(time=3, samples=True), None),
    "savgol_filter": (lambda p: p.savgol_filter(2, time=5, samples=True), None),
    "gaussian_filter": (lambda p: p.gaussian_filter(time=2, samples=True), None),
    "wiener_filter": (lambda p: p.wiener_filter(time=5, samples=True), None),
    "hampel_filter": (lambda p: p.hampel_filter(time=5, samples=True), None),
    "taper": (lambda p: p.taper(time=0.1), None),
    "taper_single": (lambda p: p.taper(time=(None, 0.2), window_type="ramp"), _single),
    "taper_range": (lambda p: p.taper_range(time=(5, 10, 20, 30), samples=True), None),
    "line_mute": (lambda p: p.line_mute(time=(0, 0.01), smooth=0.1), None),
    "slope_mute": (lambda p: p.slope_mute((1, 3)), None),
    "pow_coord": (lambda p: p.pow_coord(time=2, distance=1), None),
    "pow_coord_single": (lambda p: p.pow_coord(time=2), _single),
    "differentiate": (lambda p: p.differentiate("time"), None),
    "integrate": (lambda p: p.integrate("distance", definite=True), None),
    "velocity_to_strain_rate": (lambda p: p.velocity_to_strain_rate(), _velocity),
    "velocity_to_strain_rate_edgeless": (
        lambda p: p.velocity_to_strain_rate_edgeless(step_multiple=3),
        _velocity,
    ),
    "phase_weighted_stack": (lambda p: p.phase_weighted_stack("distance"), None),
    "pad": (lambda p: p.pad(time=(2, 3), samples=True), None),
    "decimate": (lambda p: p.decimate(time=4), None),
    "interpolate": (lambda p: p.interpolate(time=p.get_array("time")[1::3]), None),
    "resample": (lambda p: p.resample(time=0.005), None),
    "align_to_coord": (
        lambda p: p.align_to_coord(time="shift", samples=True),
        _shifted,
    ),
    "correlate_shift": (lambda p: p.correlate_shift("time"), None),
    "reassemble": (lambda p: p.reassemble(), _tiled),
    "fill_gaps": (lambda p: p.fill_gaps("time"), _holed),
    "dft": (lambda p: p.dft("time"), None),
    "idft": (lambda p: p.idft(), _transformed),
    "stft": (lambda p: p.stft(time=64, samples=True), None),
    "istft": (lambda p: p.istft(), _windowed),
    "spectral_centroid": (lambda p: p.spectral_centroid(), _spectrum),
    "median_frequency": (lambda p: p.median_frequency(), _spectrum),
    "spectral_peak_frequency": (lambda p: p.spectral_peak_frequency(), _spectrum),
    "spectral_peak_amplitude": (lambda p: p.spectral_peak_amplitude(), _spectrum),
    "spectral_entropy": (lambda p: p.spectral_entropy(), _spectrum),
    "spectral_kurtosis": (lambda p: p.spectral_kurtosis(), _spectrum),
    "spectral_flatness": (lambda p: p.spectral_flatness(), _spectrum),
}


# The converted operations which have only a numpy kernel, so fall back.
_NUMPY_ONLY = {
    "demedian",
    "detrend",
    "sobel_filter",
    "kurtosis",
    "pass_filter",
    "notch_filter",
    "median_filter",
    "savgol_filter",
    "gaussian_filter",
    "wiener_filter",
    "hampel_filter",
    "differentiate",
    "integrate",
    "velocity_to_strain_rate",
    "decimate",
    "interpolate",
    "resample",
    "align_to_coord",
    "reassemble",
    "fill_gaps",
    "stft",
    "istft",
    "spectral_centroid",
    "median_frequency",
    "spectral_peak_frequency",
    "spectral_peak_amplitude",
    "spectral_entropy",
    "spectral_kurtosis",
    "spectral_flatness",
}


class TestConvertedProcessorsOnBackends:
    """Every converted operation runs on every backend, or warns it falls back."""

    @pytest.mark.parametrize("name", sorted(_CONVERTED))
    def test_runs_or_falls_back(self, random_patch, to_backend, name):
        """
        Native kernels are silent, numpy kernels warn, the backend comes back,
        and the values, dtype and shape are numpy's.
        """
        call, prepare = _CONVERTED[name]
        numpy_patch = prepare(random_patch) if prepare else random_patch
        patch = to_backend(numpy_patch)
        operation = name.split("_single")[0].split("_offset")[0]
        operation = {"fillna_no_inf": "fillna", "squeeze_one": "squeeze"}.get(
            operation, operation
        )
        cls = getattr(dc.Patch, operation).__processor__
        backend = backend_name(patch.data)
        # A kernel registered for the backend (dask's lazy median) is native.
        registered = backend in cls.__dict__.get("_kernels", {})
        if operation not in _NUMPY_ONLY or registered:
            with warnings_as_errors():
                out = call(patch)
        else:
            with pytest.warns(NumpyFallbackWarning, match=cls.name) as record:
                out = call(patch)
            # One fallback, pointing at the caller rather than into dascore.
            fallbacks = [x for x in record if x.category is NumpyFallbackWarning]
            assert len(fallbacks) == 1
            assert fallbacks[0].filename == __file__
        assert backend_name(out.data) == backend
        expected = call(numpy_patch)
        values = np.asarray(out.data)
        assert values.dtype == np.asarray(expected.data).dtype
        assert out.dims == expected.dims
        # Single precision agrees to its own rounding, not float64's.
        single = values.dtype in (np.float32, np.complex64)
        atol = 1e-5 if single else 1e-8
        assert np.allclose(values, expected.data, atol=atol, equal_nan=True)


# Aggregations whose aggregator may go through numpy (the standard has no
# median or take, and idxmax indexes coordinate values with numpy), plus a
# native mean as control.
_AGGREGATIONS = {
    "aggregate": lambda p: p.aggregate("time", method="median"),
    "aggregate_mean": lambda p: p.aggregate("time", method="mean"),
    "median": lambda p: p.median("time"),
    "first": lambda p: p.first("time"),
    "last": lambda p: p.last("distance", dim_reduce="squeeze"),
    "idxmax": lambda p: p.idxmax("distance"),
    "idxmin": lambda p: p.idxmin("distance", dim_reduce="squeeze"),
}


class TestAggregationsOnBackends:
    """Aggregations run natively, or warn they fall back, on every backend."""

    @pytest.mark.parametrize("name", sorted(_AGGREGATIONS))
    def test_runs_or_falls_back(self, random_patch, to_backend, backend, name):
        """The backend comes back, and the values are numpy's."""
        call = _AGGREGATIONS[name]
        patch = to_backend(random_patch)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            out = call(patch)
        # Dask runs numpy's median and take itself; the indexing never does.
        native = name == "aggregate_mean" or (
            backend == "dask" and not name.startswith("idx")
        )
        assert [x.category for x in record] == (
            [] if native else [NumpyFallbackWarning]
        )
        # The warning points at the caller, not into dascore.
        assert all(x.filename == __file__ for x in record)
        assert backend_name(out.data) == backend_name(patch.data)
        expected = call(random_patch)
        assert out.dims == expected.dims
        assert np.allclose(np.asarray(out.data), expected.data, equal_nan=True)


class TestArrayApiKernelBranches:
    """The array API kernels' branches, on a backend which is not numpy."""

    def test_angle_of_complex_data(self, backend_patch):
        """The phase of complex data is atan2 of its parts."""
        xp = array_namespace(backend_patch.data)
        data = xp.astype(backend_patch.data, xp.complex128) * (1 + 2j)
        out = backend_patch.new(data=data).angle()
        positive = np.asarray(backend_patch.data) > 0
        assert np.allclose(np.asarray(out.data)[positive], np.arctan2(2, 1))

    def test_pad_writes_the_constant(self, backend_patch):
        """The padded samples hold the constant, both ends of both axes."""
        out = backend_patch.pad(
            time=(1, 2), distance=(0, 1), samples=True, constant_values=7
        )
        values = np.asarray(out.data)
        assert values.shape == (301, 2003)
        assert (values[:, :1] == 7).all() and (values[:, -2:] == 7).all()
        assert (values[-1] == 7).all()
        assert np.array_equal(values[:-1, 1:-2], np.asarray(backend_patch.data))

    def test_pad_with_a_numpy_scalar(self, backend_patch):
        """A numpy scalar fill is written, on the patch's backend."""
        out = backend_patch.pad(time=1, samples=True, constant_values=np.int64(7))
        assert backend_name(out.data) == backend_name(backend_patch.data)
        assert (np.asarray(out.data)[:, 0] == 7).all()

    def test_decimate_unfiltered_is_native(self, random_patch, backend_patch):
        """Striding needs no numpy: no warning, and the backend kept."""
        expected = random_patch.decimate(time=4, filter_type=None)
        with warnings_as_errors():
            for copy in (True, False):
                out = backend_patch.decimate(time=4, filter_type=None, copy=copy)
                assert backend_name(out.data) == backend_name(backend_patch.data)
                assert np.array_equal(np.asarray(out.data), expected.data)

    @pytest.mark.parametrize("undo_weighting", (True, False))
    def test_correlate_shift_of_integers(
        self, random_patch, to_backend, undo_weighting
    ):
        """Integers shift as numpy shifts them, dtype and promotion included."""
        ints = np.arange(random_patch.size).reshape(random_patch.shape)
        numpy_patch = random_patch.new(data=ints)
        patch = to_backend(numpy_patch)
        out = patch.correlate_shift("time", undo_weighting=undo_weighting)
        expected = numpy_patch.correlate_shift("time", undo_weighting=undo_weighting)
        assert backend_name(out.data) == backend_name(patch.data)
        assert np.asarray(out.data).dtype == expected.dtype
        assert np.array_equal(np.asarray(out.data), expected.data)

    @pytest.mark.parametrize("real", (True, False))
    def test_dft_of_integers(self, random_patch, to_backend, real):
        """Integers transform as numpy transforms them, dtype included."""
        ints = np.arange(random_patch.size).reshape(random_patch.shape) % 7
        numpy_patch = random_patch.new(data=ints)
        patch = to_backend(numpy_patch)
        out = patch.dft("time", real=real)
        expected = numpy_patch.dft("time", real=real)
        assert backend_name(out.data) == backend_name(patch.data)
        assert np.asarray(out.data).dtype == expected.dtype
        assert np.allclose(np.asarray(out.data), expected.data)

    def test_fillna_fills(self, backend_patch):
        """Non-finite values are replaced by the value."""
        xp = array_namespace(backend_patch.data)
        data = xp.where(backend_patch.data > 0.5, xp.nan, backend_patch.data)
        out = backend_patch.new(data=data).fillna(-1.0)
        numpy_out = np.asarray(out.data)
        assert np.isfinite(numpy_out).all()
        assert (numpy_out[np.asarray(backend_patch.data) > 0.5] == -1).all()

    def test_fillna_of_finite_data_is_a_no_op(self, backend_patch):
        """With nothing to fill the patch comes back untouched."""
        out = backend_patch.fillna(-1.0)
        assert out is backend_patch

    def test_hilbert_of_odd_length(self, backend_patch):
        """Odd and even lengths weight the spectrum differently."""
        odd = backend_patch.isel(time=slice(0, 99))
        out = odd.hilbert("time")
        expected = sp_hilbert(np.asarray(odd.data), axis=odd.get_axis("time"))
        assert np.allclose(np.asarray(out.data), expected)

    def test_hilbert_keeps_single_precision(self, backend_patch):
        """float32 in, complex64 out; the envelope float32."""
        xp = array_namespace(backend_patch.data)
        single = backend_patch.new(data=xp.astype(backend_patch.data, xp.float32))
        assert single.hilbert("time").data.dtype == xp.complex64
        assert single.envelope("time").data.dtype == xp.float32

    def test_hilbert_refuses_complex(self, backend_patch):
        """As scipy does: the analytic signal is of real data."""
        xp = array_namespace(backend_patch.data)
        data = xp.astype(backend_patch.data, xp.complex128)
        with pytest.raises(ValueError, match="must be real"):
            backend_patch.new(data=data).hilbert("time")

    def test_phase_weighted_stack_of_integers(self, backend_patch):
        """Integer data are stacked in float, as numpy does."""
        xp = array_namespace(backend_patch.data)
        ints = backend_patch.new(data=xp.astype(backend_patch.data * 100, xp.int32))
        out = ints.phase_weighted_stack("distance")
        expected = dc.Patch(
            data=np.asarray(ints.data), coords=ints.coords, dims=ints.dims
        ).phase_weighted_stack("distance")
        assert np.allclose(np.asarray(out.data), expected.data)

    def test_full_with_a_numpy_scalar(self, backend_patch):
        """A numpy scalar fill keeps its dtype, on the patch's backend."""
        out = backend_patch.full(np.float32(2))
        assert backend_name(out.data) == backend_name(backend_patch.data)
        assert np.asarray(out.data).dtype == np.float32


class TestDaskChunks:
    """A dask array chunked along the transformed axis."""

    def test_hilbert_across_chunks(self, random_patch):
        """The chunks along the axis are joined before the transform."""
        da = pytest.importorskip("dask.array")
        data = np.asarray(random_patch.data)
        chunked = random_patch.new(data=da.from_array(data, chunks=(50, 100)))
        expected = np.asarray(random_patch.hilbert("time").data)
        assert np.allclose(np.asarray(chunked.hilbert("time").data), expected)


class TestDevices:
    """Arrays built by a kernel live on the data's device."""

    def test_fillna_and_full_keep_the_device(self, random_patch):
        """array_api_strict refuses to mix devices, so this would raise."""
        xp = pytest.importorskip("array_api_strict")
        other = xp.__array_namespace_info__().devices()[1]
        data = xp.asarray(np.where(np.asarray(random_patch.data) > 0.5, np.nan, 1.0))
        patch = random_patch.new(data=xp.asarray(data, device=other))
        assert device(patch.fillna(2.0).data) == other
        assert device(patch.full(2.0).data) == other


class TestDaskLaziness:
    """Operations dask implements itself stay lazy."""

    def test_demedian_stays_lazy(self, random_patch):
        """No numpy copy, no warning, and the chunks kept."""
        da = pytest.importorskip("dask.array")
        data = da.from_array(np.asarray(random_patch.data), chunks=(50, 100))
        with warnings_as_errors():
            out = random_patch.new(data=data).demedian("time")
        assert isinstance(out.data, da.Array)
        assert np.allclose(np.asarray(out.data), random_patch.demedian("time").data)


class TestFallbackWarningLocation:
    """The fallback warning points at the caller, not at dascore."""

    def test_points_at_the_call(self, random_patch):
        """Whichever way the operation is reached."""
        xp = pytest.importorskip("array_api_strict")
        patch = random_patch.new(data=xp.asarray(np.asarray(random_patch.data)))
        with pytest.warns(NumpyFallbackWarning) as record:
            patch.detrend("time")
        assert record[0].filename == __file__
