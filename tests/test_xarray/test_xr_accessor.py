"""Tests for the `.dc` accessor, which puts patch methods on a DataArray."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import PatchConversionError
from dascore.utils.misc import suppress_warnings
from dascore.warnings import NumpyFallbackWarning

# Importing the accessor registers it, which needs xarray; without it
# there is no accessor to test rather than a failure to report.
pytest.importorskip("xarray")


@pytest.fixture(scope="module")
def patch():
    """One ordinary patch."""
    return dc.get_example_patch()


@pytest.fixture(scope="module")
def data_array(patch):
    """That patch as a DataArray with materialized labels."""
    return patch.io.to_xarray(lazy_coords=False)


@pytest.fixture()
def lazy_data_array(patch):
    """The same patch backed by a dask array."""
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

    def test_a_keyword_data_array_argument_is_a_patch(self, patch, data_array):
        """A keyword is converted like a position; a patch method needs one.

        `add` builds a patch out of what it is given, so an unconverted
        DataArray does not merely give a different answer, it refuses.
        """
        out = data_array.dc.add(other=data_array)
        assert np.allclose(out.values, patch.data * 2)

    def test_a_property_stating_a_patch_states_a_data_array(self, data_array):
        """`T` is a property whose value is a patch, so it converts too."""
        import xarray as xr  # noqa: PLC0415

        transposed = data_array.dc.T
        assert isinstance(transposed, xr.DataArray)
        assert transposed.dims == data_array.dims[::-1]

    def test_a_namespace_is_reachable(self, patch, data_array):
        """A patch resolves its namespaces itself, so asking the patch finds them."""
        assert type(data_array.dc.io).__name__ == type(patch.io).__name__
        assert "io" in dir(data_array.dc)

    def test_a_name_only_the_class_has_is_not_forwarded(self, data_array):
        """`mro` is a name of the type, not of a patch."""
        with pytest.raises(AttributeError, match="mro"):
            data_array.dc.mro

    def test_a_ufunc_keeps_the_methods_it_carries(self, patch, data_array):
        """`add` and its kind are objects with `reduce` and `accumulate`."""
        assert hasattr(data_array.dc.add, "reduce")
        out = data_array.dc.add.reduce("time")
        expected = patch.add.reduce("time")
        assert np.allclose(out.values, expected.data)

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
        # what the accessor adds of its own is offered too
        assert "to_patch" in names
        assert not any(x.startswith("_") for x in names)

    def test_it_says_what_it_is(self, data_array):
        """A repr which names the shape it wraps."""
        text = repr(data_array.dc)
        assert "DASCore" in text and "300x2000" in text


class TestLaziness:
    """A dask-backed array is passed through as it stands."""

    @pytest.fixture()
    def tree_leaf(self, random_spool):
        """One segment of a tree, whose time coordinate is served lazily."""
        pytest.importorskip("dask")
        tree = random_spool.io.to_xarray()
        return next(x for x in tree.subtree if "data" in x.dataset)["data"]

    def test_an_operation_on_the_arrays_backend_stays_lazy(self, lazy_data_array):
        """Nothing is computed on the way in or the way out."""
        da = pytest.importorskip("dask.array")

        assert isinstance(lazy_data_array.data, da.Array)
        out = lazy_data_array.dc.abs()
        assert isinstance(out.data, da.Array)

    def test_a_result_keeps_a_lazy_coordinate_lazy(self, random_spool):
        """Converting back out must not spell out what came in stated."""
        pytest.importorskip("dask")
        tree = random_spool.io.to_xarray()
        leaf = next(x for x in tree.subtree if "data" in x.dataset)["data"]
        assert type(leaf.xindexes["time"]).__name__ == "CoordIndex"
        out = leaf.dc.abs()
        assert type(out.xindexes["time"]).__name__ == "CoordIndex"

    def test_an_ordinary_array_keeps_its_eager_index(self, data_array):
        """Whether a coordinate is spelled out belongs to the object.

        An eager index is what xarray aligns arithmetic on, so a
        conversion must not quietly swap one for a lazy index.
        """
        assert type(data_array.dc.abs().xindexes["time"]).__name__ == "PandasIndex"

    def test_ordinary_arithmetic_still_works(self, data_array):
        """Which is what replacing the index used to break."""
        assert (data_array.dc.abs() - data_array).shape == data_array.shape
        pair = data_array + data_array.isel(time=slice(1, 3))
        assert pair.sizes["time"] == 2

    def test_a_moved_coordinate_is_served_where_it_now_is(self, tree_leaf):
        """A method which trims it is served lazily at its new bounds."""
        values = np.asarray(tree_leaf["time"].values)
        out = tree_leaf.dc.select(time=(None, values[100]))
        assert type(out.xindexes["time"]).__name__ == "CoordIndex"
        assert np.array_equal(np.asarray(out["time"].values), values[:101])

    def test_a_property_keeps_it_lazy_too(self, tree_leaf):
        """A property's value is converted like a method's result."""
        assert type(tree_leaf.dc.T.xindexes["time"]).__name__ == "CoordIndex"

    def test_a_coordinate_riding_a_dimension_is_left_alone(self, tree_leaf):
        """Only a coordinate which defines its dimension can index it.

        A lazy index is built for a dimension, so offering to serve a
        coordinate riding one would build an index for a dimension that
        coordinate does not define.
        """
        values = np.asarray(tree_leaf["time"].values)
        with_aux = tree_leaf.assign_coords(aux_time=("time", values))
        out = with_aux.dc.abs()
        assert out.coords["aux_time"].dims == ("time",)
        assert type(out.xindexes["time"]).__name__ == "CoordIndex"

    def test_the_values_are_the_same_either_way(self, patch, lazy_data_array):
        """Laziness is about when the work happens, not what it produces."""
        out = lazy_data_array.dc.abs().compute()
        assert np.array_equal(np.asarray(out.values), patch.abs().data)


class TestLazyCoordinates:
    """A coordinate the tree states rather than stores stays stated."""

    @pytest.fixture()
    def tree_leaf(self, random_spool):
        """One segment of a tree, whose time coordinate is served lazily."""
        pytest.importorskip("dask")
        tree = random_spool.io.to_xarray()
        return next(x for x in tree.subtree if "data" in x.dataset)["data"]

    def test_the_index_is_lazy_to_begin_with(self, tree_leaf):
        """Otherwise the test below would prove nothing."""
        assert type(tree_leaf.xindexes["time"]).__name__ == "CoordIndex"

    def test_converting_does_not_spell_out_the_labels(self, tree_leaf, monkeypatch):
        """The index hands back its coordinate, not every sample.

        The transform computes labels, so a conversion that never calls it keeps them
        lazy. The returned coordinate alone cannot prove this because evenly spaced
        materialized values also infer a range.
        """
        from dascore.core.coords import CoordRange  # noqa: PLC0415
        from dascore.xarray.index import CoordTransform  # noqa: PLC0415

        def _refuse(self, dim_positions):
            raise AssertionError("the labels were materialized")

        monkeypatch.setattr(CoordTransform, "forward", _refuse)
        coord = tree_leaf.dc.to_patch().get_coord("time")
        assert isinstance(coord, CoordRange)

    def test_the_coordinate_is_the_same_either_way(self, tree_leaf):
        """Staying lazy must not change which samples it names."""
        coord = tree_leaf.dc.to_patch().get_coord("time")
        assert np.array_equal(coord.values, tree_leaf["time"].values)

    @staticmethod
    def _refuse_to_materialize(monkeypatch):
        """Make spelling out a temporal label an error, either side of it.

        Either xarray's transform or the patch range can materialize labels. Refuse
        both paths; numeric coordinates remain unaffected.
        """
        from dascore.core.coords import CoordRange  # noqa: PLC0415
        from dascore.xarray.index import CoordTransform  # noqa: PLC0415

        def _refuse_forward(self, dim_positions):
            raise AssertionError("the labels were computed by the transform")

        original = CoordRange.values

        @property
        def _refuse_values(self):
            if np.issubdtype(self.dtype, np.datetime64) or np.issubdtype(
                self.dtype, np.timedelta64
            ):
                raise AssertionError("the range spelled out its labels")
            return original.__get__(self, CoordRange)

        monkeypatch.setattr(CoordTransform, "forward", _refuse_forward)
        monkeypatch.setattr(CoordRange, "values", _refuse_values)

    def test_a_forwarded_call_never_spells_out_the_labels(self, tree_leaf, monkeypatch):
        """Not on the way in, and not on the way back out.

        Building labels before restoring the range still costs eight bytes per sample,
        so the conversion must preserve the lazy representation throughout.
        """
        self._refuse_to_materialize(monkeypatch)
        out = tree_leaf.dc.abs()
        assert type(out.xindexes["time"]).__name__ == "CoordIndex"

    def test_a_forwarded_property_never_spells_them_out_either(
        self, tree_leaf, monkeypatch
    ):
        """A property states a patch by the same conversion a call does."""
        self._refuse_to_materialize(monkeypatch)
        assert type(tree_leaf.dc.T.xindexes["time"]).__name__ == "CoordIndex"

    def test_a_renamed_coordinate_is_spelled_out(self, tree_leaf):
        """Laziness follows the name, and a rename is a new name.

        A renamed coordinate has no record that it arrived lazily, so it materializes
        its labels.
        """
        out = tree_leaf.dc.rename_coords(time="t")
        assert type(out.xindexes["t"]).__name__ == "PandasIndex"
        assert np.array_equal(
            np.asarray(out["t"].values), np.asarray(tree_leaf["time"].values)
        )

    def test_an_eager_dimension_beside_a_lazy_one_stays_eager(self):
        """Which is why the names are carried, not one flag for the array.

        An eager index is what xarray aligns on, so making a coordinate
        lazy because a coordinate beside it is refuses the arithmetic
        which worked before the call. A correlation's lag is a second
        temporal dimension, and is eligible to be served lazily without
        having arrived that way.
        """
        from dascore.core.coords import get_coord  # noqa: PLC0415
        from dascore.xarray.patch import patch_to_xarray  # noqa: PLC0415

        start = np.datetime64("2020-01-01")
        time = get_coord(start=start, step=np.timedelta64(4, "ms"), shape=(6,))
        lag = get_coord(
            start=np.timedelta64(0, "s"), step=np.timedelta64(1, "s"), shape=(4,)
        )
        patch = dc.Patch(
            data=np.arange(24.0).reshape(6, 4),
            dims=("time", "lag"),
            coords={"time": time, "lag": lag},
        )
        # only `time` arrived stated, though `lag` could be served the same way
        array = patch_to_xarray(patch, lazy_coords={"time"})
        assert type(array.xindexes["lag"]).__name__ == "PandasIndex"
        out = array.dc.abs()
        assert type(out.xindexes["time"]).__name__ == "CoordIndex"
        assert type(out.xindexes["lag"]).__name__ == "PandasIndex"
        # which is what a lazy index on `lag` would refuse to align
        assert (out - array).shape == array.shape

    @staticmethod
    def _offsets(tree_leaf):
        """One value a channel, with no time coordinate of its own."""
        xr = pytest.importorskip("xarray")

        return xr.DataArray(
            np.arange(tree_leaf.sizes["distance"]) * 1.0,
            dims=("distance",),
            coords={"distance": tree_leaf["distance"].values},
        )

    @pytest.mark.parametrize("keyword", [False, True])
    def test_an_argument_says_how_its_coordinates_arrived(
        self, tree_leaf, monkeypatch, keyword
    ):
        """The coordinate the result takes is the argument's, so it decides.

        Adding per-channel offsets to a long acquisition takes that
        acquisition's time coordinate whichever of the two is called on;
        reading it from the receiver alone spells out every label in the
        ordering where the receiver has no time coordinate at all.
        """
        offsets = self._offsets(tree_leaf)
        self._refuse_to_materialize(monkeypatch)
        out = offsets.dc.add(other=tree_leaf) if keyword else offsets.dc.add(tree_leaf)
        assert type(out.xindexes["time"]).__name__ == "CoordIndex"

    def test_either_order_states_it_the_same_way(self, tree_leaf):
        """The same sum, so the same coordinate, however it is written."""
        offsets = self._offsets(tree_leaf)
        first, second = tree_leaf.dc.add(offsets), offsets.dc.add(tree_leaf)
        assert type(first.xindexes["time"]).__name__ == "CoordIndex"
        assert type(first.xindexes["time"]) is type(second.xindexes["time"])
        assert np.allclose(first.transpose(*second.dims).values, second.values)

    def test_a_coordinate_which_arrived_both_ways_is_spelled_out(self, tree_leaf):
        """An eager index is what xarray aligns on, so it is not taken away.

        One of the two stated the coordinate and the other spelled it
        out; the result keeps what the stricter of them can align on,
        the same way round either way.
        """
        spelled_out = tree_leaf.dc.to_patch().io.to_xarray(lazy_coords=False)
        assert type(spelled_out.xindexes["time"]).__name__ == "PandasIndex"
        for out in (spelled_out.dc.add(tree_leaf), tree_leaf.dc.add(spelled_out)):
            assert type(out.xindexes["time"]).__name__ == "PandasIndex"

    @pytest.mark.parametrize("reverse", [False, True])
    def test_an_unindexed_coordinate_does_not_force_eager(self, tree_leaf, reverse):
        """Only an eager index requires the result to preserve eager alignment."""
        patch = tree_leaf.dc.to_patch()
        unindexed = patch.io.to_xarray(lazy_coords=False).drop_indexes("time")
        first, second = (unindexed, tree_leaf) if reverse else (tree_leaf, unindexed)
        out = first.dc.add(second)
        assert type(out.xindexes["time"]).__name__ == "CoordIndex"
        assert (out - unindexed).shape == tree_leaf.shape

    def test_a_duration_is_served_like_a_time(self):
        """A lag says its units in its dtype as a stamp does."""
        from dascore.core.coords import get_coord  # noqa: PLC0415
        from dascore.xarray.patch import patch_to_xarray  # noqa: PLC0415

        lag = get_coord(
            start=np.timedelta64(0, "s"), step=np.timedelta64(1, "s"), shape=(4,)
        )
        patch = dc.Patch(data=np.arange(4.0), dims=("lag",), coords={"lag": lag})
        array = patch_to_xarray(patch, lazy_coords={"lag"})
        assert type(array.xindexes["lag"]).__name__ == "CoordIndex"
        assert np.array_equal(np.asarray(array["lag"].values), lag.values)

    def test_a_coordinate_which_no_longer_names_a_dimension(self, tree_leaf):
        """An index labels a dimension, so a coordinate which stops
        defining one is spelled out beside it rather than refused.
        """
        size = tree_leaf.sizes["time"]
        with_sample = tree_leaf.assign_coords(sample=("time", np.arange(size)))
        out = with_sample.dc.set_dims(time="sample")
        renamed = tuple("sample" if x == "time" else x for x in tree_leaf.dims)
        assert out.dims == renamed
        assert out.coords["time"].dims == ("sample",)


class TestAuxiliaryCoordinates:
    """A coordinate riding a dimension it is not named for."""

    def test_a_temporal_coordinate_on_another_dimension_converts(self, patch):
        """Its index would be built for a dimension it does not define."""
        aux = patch.update_coords(aux_time=("time", patch.get_array("time")))
        out = aux.io.to_xarray()
        assert out.coords["aux_time"].dims == ("time",)
        assert np.array_equal(
            np.asarray(out["aux_time"].values), patch.get_array("time")
        )


class TestUnits:
    """A coordinate's units survive both conversions."""

    def test_units_come_back(self, patch):
        """Otherwise a method which converts them has nothing to convert."""
        with_units = patch.set_units(distance="m")
        back = dc.io.xarray_to_patch(with_units.io.to_xarray())
        assert (
            back.get_coord("distance").units == with_units.get_coord("distance").units
        )

    def test_a_forwarded_conversion_converts(self, data_array):
        """The point of carrying them: `convert_units` changes the values."""
        before = np.asarray(data_array["distance"].values)
        out = data_array.dc.set_units(distance="m").dc.convert_units(distance="km")
        assert np.allclose(np.asarray(out["distance"].values), before / 1000)

    def test_a_temporal_coordinate_states_no_units(self, data_array):
        """A datetime says its units in its dtype, and xarray spends that
        attribute on saying how to serialize it.
        """
        assert "units" not in data_array["time"].attrs

    def test_a_rebuilt_range_states_a_scalar_step(self, patch, data_array):
        """A step is divided into, which a zero-dimensional array breaks."""
        step = dc.io.xarray_to_patch(data_array).get_coord("time").step
        assert np.ndim(step) == 0 and not isinstance(step, np.ndarray)
        assert step == patch.get_coord("time").step


class TestEmptySelections:
    """A selection which keeps nothing is still convertible."""

    def test_an_empty_time_window_converts(self, random_spool):
        """A range needs somewhere to go; an empty coordinate has nowhere."""
        pytest.importorskip("dask")
        tree = random_spool.io.to_xarray()
        leaf = next(x for x in tree.subtree if "data" in x.dataset)["data"]
        out = leaf.isel(time=slice(0, 0)).dc.to_patch()
        assert out.shape[leaf.dims.index("time")] == 0

    def test_a_property_of_an_empty_selection(self, random_spool):
        """Anything forwarded has to convert first, so this failed too."""
        pytest.importorskip("dask")
        tree = random_spool.io.to_xarray()
        leaf = next(x for x in tree.subtree if "data" in x.dataset)["data"]
        assert 0 in leaf.isel(time=slice(0, 0)).dc.shape


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
        da = pytest.importorskip("dask.array")

        import dascore.xarray.accessor as accessor  # noqa: PLC0415

        monkeypatch.setattr(accessor, "_available_memory", lambda: 10**9)
        assert self._sizer()(da.zeros((10, 10))) is None

    def test_a_lazy_array_which_does_not_fit_states_its_size(self, monkeypatch):
        """The size is what the message reports, so it is what is returned."""
        da = pytest.importorskip("dask.array")

        import dascore.xarray.accessor as accessor  # noqa: PLC0415

        monkeypatch.setattr(accessor, "_available_memory", lambda: 8)
        array = da.zeros((10, 10))
        assert self._sizer()(array) == array.nbytes

    def test_an_unknowable_budget_refuses_nothing(self, monkeypatch):
        """Without a memory figure there is no look-ahead to do."""
        da = pytest.importorskip("dask.array")

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
        with suppress_warnings(NumpyFallbackWarning):
            out = lazy_data_array.dc.announce_fallback()
        assert isinstance(out.values, np.ndarray)

    def test_an_array_in_memory_is_never_refused(self, monkeypatch, data_array):
        """It is already where a conversion would put it."""
        import dascore.xarray.accessor as accessor  # noqa: PLC0415

        self._announcing_method(monkeypatch)
        monkeypatch.setattr(accessor, "_available_memory", lambda: 1)
        with suppress_warnings(NumpyFallbackWarning):
            assert data_array.dc.announce_fallback() is not None

    def test_a_bare_array_argument_is_weighed(self, monkeypatch):
        """A conversion converts a raw array argument as it does a patch's."""
        import dascore.xarray.accessor as accessor  # noqa: PLC0415

        da = pytest.importorskip("dask.array")
        monkeypatch.setattr(accessor, "_available_memory", lambda: 8)
        array = da.zeros((10, 10))
        assert accessor._largest_to_materialize([array]) == array.nbytes

    def test_an_argument_too_large_refuses_the_call(
        self, monkeypatch, data_array, lazy_data_array
    ):
        """A conversion converts the arguments, so they are weighed too."""
        import dascore.xarray.accessor as accessor  # noqa: PLC0415

        self._announcing_method(monkeypatch)

        def method(self, other):
            warnings.warn("falling back", NumpyFallbackWarning, stacklevel=1)
            return self

        monkeypatch.setattr(dc.Patch, "announce_with_arg", method, raising=False)
        monkeypatch.setattr(accessor, "_available_memory", lambda: 1)
        # the receiver is in memory; the argument is the lazy one
        with pytest.raises(PatchConversionError, match="exceed what memory"):
            data_array.dc.announce_with_arg(lazy_data_array)

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
        da = pytest.importorskip("dask.array")

        import dascore.xarray.accessor as accessor  # noqa: PLC0415

        monkeypatch.setattr(accessor, "_available_memory", lambda: 1)
        assert isinstance(lazy_data_array.dc.abs().data, da.Array)


class TestRegistration:
    """Where the accessor comes from."""

    def test_a_conversion_brings_it(self, patch):
        """Anything DASCore hands back carries the accessor."""
        assert hasattr(patch.io.to_xarray(), "dc")

    def test_registering_twice_is_not_an_error(self):
        """Importing the module again must not raise or warn."""
        from dascore.xarray.accessor import register  # noqa: PLC0415

        with suppress_warnings(action="error"):
            register()

    def test_a_tree_node_reaches_it_through_its_variable(self, random_spool):
        """A Dataset holds several arrays, so a variable is named first."""
        pytest.importorskip("dask")
        tree = random_spool.io.to_xarray()
        node = next(x for x in tree.subtree if "data" in x.dataset)
        assert hasattr(node.dataset["data"], "dc")
