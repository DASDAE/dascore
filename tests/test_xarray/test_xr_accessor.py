"""Tests for the `.dc` accessor, which puts patch methods on a DataArray."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import PatchConversionError
from dascore.warnings import NumpyFallbackWarning


@pytest.fixture(scope="module")
def patch():
    """One ordinary patch."""
    return dc.get_example_patch()


@pytest.fixture(scope="module")
def data_array(patch):
    """That patch as a DataArray, which registers the accessor."""
    pytest.importorskip("xarray")
    return patch.io.to_xarray()


@pytest.fixture()
def lazy_data_array(patch):
    """The same patch backed by a dask array."""
    pytest.importorskip("xarray")
    da = pytest.importorskip("dask.array")
    lazy = patch.new(data=da.from_array(patch.data, chunks=(100, -1)))
    return lazy.io.to_xarray()


class TestForwarding:
    """Every public name a patch has is reachable through the accessor."""

    def test_a_method_answers_what_the_patch_answers(self, patch, data_array):
        """The route through xarray changes nothing about the result."""
        import xarray as xr  # noqa: PLC0415

        out = data_array.dc.abs()
        assert isinstance(out, xr.DataArray)
        assert np.array_equal(out.values, patch.abs().data)

    def test_arguments_reach_the_method(self, patch, data_array):
        """A method's own arguments are passed through untouched."""
        out = data_array.dc.pass_filter(time=(1, 10))
        assert np.allclose(out.values, patch.pass_filter(time=(1, 10)).data)

    def test_a_data_array_argument_is_a_patch(self, patch, data_array):
        """A method taking another patch takes another DataArray here."""
        out = data_array.dc.add(data_array)
        assert np.allclose(out.values, patch.data * 2)

    def test_a_property_answers_for_itself(self, patch, data_array):
        """There is no call to convert arguments for, and no patch back."""
        assert data_array.dc.dims == patch.dims
        assert data_array.dc.shape == patch.shape

    def test_a_result_which_is_no_patch_comes_back_as_it_is(self, data_array):
        """An array, a spool or a figure is what the caller asked for."""
        values = data_array.dc.get_array("time")
        assert isinstance(values, np.ndarray)
        assert not hasattr(values, "dims")

    def test_the_patch_itself_is_available(self, patch, data_array):
        """Converting back is the one thing the accessor adds of its own."""
        assert data_array.dc.to_patch() == patch

    def test_a_name_no_patch_has(self, data_array):
        """The error names what was asked for, and where it was looked for."""
        with pytest.raises(AttributeError, match="not_a_patch_method"):
            data_array.dc.not_a_patch_method

    def test_a_private_name_is_not_forwarded(self, data_array):
        """The accessor forwards a patch's public surface, not its insides."""
        with pytest.raises(AttributeError):
            data_array.dc._data

    def test_completion_offers_the_patch_names(self, data_array):
        """Whatever can be forwarded should be offered."""
        names = dir(data_array.dc)
        assert "pass_filter" in names and "abs" in names
        assert not any(x.startswith("_") for x in names)

    def test_it_says_what_it_is(self, data_array):
        """A repr which names the shape it wraps."""
        text = repr(data_array.dc)
        assert "DASCore" in text and "300x2000" in text


class TestLaziness:
    """A dask-backed array is passed through as it stands."""

    def test_an_operation_on_the_arrays_backend_stays_lazy(self, lazy_data_array):
        """Nothing is computed on the way in or the way out."""
        import dask.array as da  # noqa: PLC0415

        assert isinstance(lazy_data_array.data, da.Array)
        out = lazy_data_array.dc.abs()
        assert isinstance(out.data, da.Array)

    def test_the_values_are_the_same_either_way(self, patch, lazy_data_array):
        """Laziness is about when the work happens, not what it produces."""
        out = lazy_data_array.dc.abs().compute()
        assert np.array_equal(np.asarray(out.values), patch.abs().data)


class TestMemoryLookAhead:
    """What happens when an operation would materialize more than fits."""

    @staticmethod
    def _sizer():
        from dascore.xarray.accessor import _too_large_to_materialize  # noqa: PLC0415

        return _too_large_to_materialize

    def test_an_array_already_in_memory_is_never_too_large(self, patch):
        """Materializing what is already materialized costs nothing."""
        assert self._sizer()(patch.data) is None

    def test_a_lazy_array_which_fits_is_not_too_large(self, monkeypatch):
        """Room to spare, so nothing is refused."""
        import dask.array as da  # noqa: PLC0415

        import dascore.xarray.accessor as accessor  # noqa: PLC0415

        monkeypatch.setattr(accessor, "_available_memory", lambda: 10**9)
        assert self._sizer()(da.zeros((10, 10))) is None

    def test_a_lazy_array_which_does_not_fit_states_its_size(self, monkeypatch):
        """The size is what the message reports, so it is what is returned."""
        import dask.array as da  # noqa: PLC0415

        import dascore.xarray.accessor as accessor  # noqa: PLC0415

        monkeypatch.setattr(accessor, "_available_memory", lambda: 8)
        array = da.zeros((10, 10))
        assert self._sizer()(array) == array.nbytes

    def test_an_unknowable_budget_refuses_nothing(self, monkeypatch):
        """Without a memory figure there is no look-ahead to do."""
        import dask.array as da  # noqa: PLC0415

        import dascore.xarray.accessor as accessor  # noqa: PLC0415

        monkeypatch.setattr(accessor, "_available_memory", lambda: None)
        assert self._sizer()(da.zeros((10, 10))) is None

    def test_the_budget_is_what_is_free_less_some_headroom(self, monkeypatch):
        """An array is not the only thing memory has to hold."""
        import dascore.xarray.accessor as accessor  # noqa: PLC0415

        class _Stub:
            @staticmethod
            def virtual_memory():
                return type("_M", (), {"available": 1000})

        monkeypatch.setattr(accessor, "optional_import", lambda name: _Stub)
        assert accessor._available_memory() == int(1000 * accessor._MEMORY_HEADROOM)

    def test_no_way_to_ask_means_no_look_ahead(self, monkeypatch):
        """Psutil is not a dependency, so its absence is ordinary."""
        import dascore.xarray.accessor as accessor  # noqa: PLC0415

        def _missing(name):
            raise ImportError(name)

        monkeypatch.setattr(accessor, "optional_import", _missing)
        assert accessor._available_memory() is None

    @staticmethod
    def _announcing_method(monkeypatch):
        """Give Patch a method which announces a fallback, as some do."""

        def method(self):
            warnings.warn("falling back", NumpyFallbackWarning, stacklevel=1)
            return self.new(data=np.asarray(self.data))

        monkeypatch.setattr(dc.Patch, "announce_fallback", method, raising=False)

    def test_a_method_which_would_convert_a_huge_array_is_refused(
        self, monkeypatch, lazy_data_array
    ):
        """The array does not fit, so the conversion is not attempted."""
        import dascore.xarray.accessor as accessor  # noqa: PLC0415

        self._announcing_method(monkeypatch)
        monkeypatch.setattr(accessor, "_available_memory", lambda: 1)
        with pytest.raises(PatchConversionError, match="exceed what memory"):
            lazy_data_array.dc.announce_fallback()

    def test_the_same_method_runs_when_the_array_fits(
        self, monkeypatch, lazy_data_array
    ):
        """Converting is only refused for want of room to convert into."""
        import dascore.xarray.accessor as accessor  # noqa: PLC0415

        self._announcing_method(monkeypatch)
        monkeypatch.setattr(accessor, "_available_memory", lambda: 10**12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumpyFallbackWarning)
            out = lazy_data_array.dc.announce_fallback()
        assert isinstance(out.values, np.ndarray)

    def test_an_array_in_memory_is_never_refused(self, monkeypatch, data_array):
        """It is already where a conversion would put it."""
        import dascore.xarray.accessor as accessor  # noqa: PLC0415

        self._announcing_method(monkeypatch)
        monkeypatch.setattr(accessor, "_available_memory", lambda: 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumpyFallbackWarning)
            assert data_array.dc.announce_fallback() is not None

    def test_a_call_which_announces_a_fallback_is_refused(self):
        """The warning comes before the conversion, so raising stops it."""
        from dascore.xarray.accessor import _guarded  # noqa: PLC0415

        def call():
            warnings.warn("falling back", NumpyFallbackWarning, stacklevel=1)
            return "converted"

        with pytest.raises(PatchConversionError, match="exceed what memory"):
            _guarded(call, "some_method", 10**12)

    def test_a_call_which_stays_on_its_backend_is_not_refused(self):
        """No warning, no conversion, nothing to stop."""
        from dascore.xarray.accessor import _guarded  # noqa: PLC0415

        assert _guarded(lambda: "lazy", "some_method", 10**12) == "lazy"

    def test_a_lazy_operation_runs_however_little_memory_is_free(
        self, monkeypatch, lazy_data_array
    ):
        """The look-ahead refuses a conversion, not an operation."""
        import dask.array as da  # noqa: PLC0415

        import dascore.xarray.accessor as accessor  # noqa: PLC0415

        monkeypatch.setattr(accessor, "_available_memory", lambda: 1)
        assert isinstance(lazy_data_array.dc.abs().data, da.Array)


class TestRegistration:
    """Where the accessor comes from."""

    def test_a_conversion_brings_it(self, patch):
        """Anything DASCore hands back carries the accessor."""
        pytest.importorskip("xarray")
        assert hasattr(patch.io.to_xarray(), "dc")

    def test_registering_twice_is_not_an_error(self):
        """Importing the module again must not raise or warn."""
        from dascore.xarray.accessor import register  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            register()

    def test_a_tree_node_reaches_it_through_its_variable(self, random_spool):
        """A Dataset holds several arrays, so a variable is named first."""
        pytest.importorskip("xarray")
        pytest.importorskip("dask")
        tree = random_spool.io.to_xarray()
        node = next(x for x in tree.subtree if "data" in x.dataset)
        assert hasattr(node.dataset["data"], "dc")
