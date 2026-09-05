"""Patch indexing contracts compared directly with xarray."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import CoordArray, concat_coords, get_coord
from dascore.units import m, s
from dascore.utils.array_api import to_numpy

xr = pytest.importorskip("xarray")


@pytest.fixture()
def patch():
    """Three dimensions with scalar, 1D, and multidimensional coordinates."""
    data = np.arange(120).reshape(5, 6, 4)
    time = np.datetime64("2020-01-01", "ns") + np.arange(6) * np.timedelta64(1, "s")
    coords = {
        "distance": get_coord(data=np.arange(5) * 2, units="m"),
        "time": time,
        "component": np.array(["a", "b", "c", "d"]),
        "latitude": ("distance", np.linspace(40, 41, 5)),
        "quality": (("time", "distance"), np.arange(30).reshape(6, 5)),
        "cube": (("component", "distance", "time"), data.transpose(2, 0, 1)),
        "instrument": ((), np.asarray("sensor")),
    }
    return dc.Patch(data=data, coords=coords, dims=("distance", "time", "component"))


def assert_matches(patch, operation, indexers, **kwargs):
    """Compare data, dimension order, and every coordinate with xarray."""
    reference = patch.io.to_xarray()
    expected = getattr(reference, operation)(indexers, **kwargs)
    actual = getattr(patch, operation)(indexers, **kwargs)
    xr.testing.assert_equal(actual.io.to_xarray(), expected)
    assert actual.shape == expected.shape
    assert actual.dims == expected.dims
    assert actual.summary.shape == expected.shape
    assert len(str(actual))
    return actual


class TestIsel:
    """Orthogonal positional indexing, including scalar dimension removal."""

    @pytest.mark.parametrize("drop", [False, True])
    @pytest.mark.parametrize(
        "indexers",
        [
            {},
            {"distance": 0},
            {"distance": -5},
            {"distance": np.asarray(-1)},
            {"distance": [3]},
            {"distance": [3, 1, 3, -5]},
            {"distance": np.array([4, 0], dtype=np.uint64)},
            {"distance": [False, True, False, True, False]},
            {"distance": []},
            {"distance": slice(1, 5, 2)},
            {"distance": slice(None, None, -1)},
            {"distance": slice(-20, 40)},
            {"distance": slice(30, 40)},
            {"distance": slice(3, 1)},
            {"time": slice(4, None, -2)},
            {"distance": [3, 1], "time": [5, 1, 5]},
            {"distance": 2, "time": [4, 0], "component": [2, 0, 2]},
            {"distance": [2, 0], "time": 1, "component": [2, 0]},
            {"distance": 2, "time": 1, "component": 0},
        ],
    )
    def test_xarray(self, patch, indexers, drop):
        """Selections match xarray, including dependent coordinate reduction."""
        assert_matches(patch, "isel", indexers, drop=drop)

    @pytest.mark.parametrize(
        "value",
        [
            5,
            -6,
            [0, 5],
            [-6],
            1.2,
            [1.2],
            None,
            True,
            [True, False],
            np.ones((2, 2), dtype=int),
            slice(None, None, 0),
            slice(0.5, 2),
            np.array([], dtype=float),
        ],
    )
    def test_errors(self, patch, value):
        """Invalid positional indexers fail with the same exception category."""
        with pytest.raises((TypeError, IndexError, ValueError)) as expected:
            patch.io.to_xarray().isel(distance=value)
        with pytest.raises(type(expected.value)):
            patch.isel(distance=value)

    @pytest.mark.parametrize("missing", ["raise", "warn", "ignore"])
    def test_missing_dims(self, patch, missing):
        """Unknown dimensions follow xarray's explicit policy."""
        kwargs = dict(missing_dims=missing, unknown=1)
        if missing == "raise":
            with pytest.raises(ValueError):
                patch.isel(**kwargs)
        elif missing == "warn":
            with pytest.warns(UserWarning):
                assert patch.isel(**kwargs) is patch
        else:
            assert patch.isel(**kwargs) is patch

    def test_invalid_missing_policy(self, patch):
        """Misspelled policies must not silently ignore dimension names."""
        with pytest.raises(ValueError, match="missing_dims"):
            patch.isel(missing_dims="typo")

    def test_slice_view(self, patch):
        """Basic slicing shares numpy data instead of copying the entire patch."""
        assert np.shares_memory(patch.data, patch.isel(time=slice(1, 3)).data)


class TestSel:
    """Label lookup and inclusive ranges match xarray/pandas."""

    @pytest.mark.parametrize("drop", [False, True])
    @pytest.mark.parametrize(
        "indexers",
        [
            {},
            {"distance": 4},
            {"distance": [8, 2, 8]},
            {"distance": []},
            {"distance": slice(2, 6)},
            {"distance": slice(1, 7)},
            {"distance": slice(None, 4)},
            {"distance": slice(8, None, -2)},
            {"distance": slice(20, 30)},
            {"component": "b"},
            {"component": ["c", "a", "c"]},
            {"component": slice("b", "c")},
            {"distance": [False, True, False, True, False]},
            {"distance": [8, 2], "component": ["d", "a", "d"]},
            {
                "distance": 2,
                "time": np.datetime64("2020-01-01T00:00:02"),
                "component": "a",
            },
            {"time": "2020-01"},
            {"time": slice("2020-01-01T00:00:01", "2020-01-01T00:00:03")},
            {"time": ["2020-01-01T00:00:03", "2020-01-01T00:00:00"]},
        ],
    )
    def test_xarray(self, patch, indexers, drop):
        """Labels, datetime strings, and arrays retain xarray's meaning."""
        assert_matches(patch, "sel", indexers, drop=drop)

    @pytest.mark.parametrize("value", [1, [2, 3], [2, 200]])
    def test_missing_labels(self, patch, value):
        """A missing label raises rather than returning a partial selection."""
        with pytest.raises(KeyError):
            patch.io.to_xarray().sel(distance=value)
        with pytest.raises(KeyError):
            patch.sel(distance=value)

    @pytest.mark.parametrize("value", [1, [1, 3, 1], -1, 9])
    @pytest.mark.parametrize("tolerance", [None, 1, 2])
    def test_nearest(self, patch, value, tolerance):
        """Nearest matches, including ties and edge values, match xarray."""
        assert_matches(
            patch, "sel", {"distance": value}, method="nearest", tolerance=tolerance
        )

    def test_nearest_too_far(self, patch):
        """Tolerance failures raise rather than silently dropping requests."""
        for obj in (patch, patch.io.to_xarray()):
            with pytest.raises(KeyError):
                obj.sel(distance=[2, 3], method="nearest", tolerance=0.4)

    def test_nearest_time(self, patch):
        """Datetime nearest selection interprets tolerance as a duration."""
        assert_matches(
            patch,
            "sel",
            {"time": np.datetime64("2020-01-01T00:00:01.4")},
            method="nearest",
            tolerance=np.timedelta64(500, "ms"),
        )

    @pytest.mark.parametrize("option", [{"method": "nearest"}, {"tolerance": 1}])
    def test_slice_inexact(self, patch, option):
        """Inexact lookup options cannot be applied to label slices."""
        for obj in (patch, patch.io.to_xarray()):
            with pytest.raises(NotImplementedError):
                obj.sel(distance=slice(2, 6), **option)

    @pytest.mark.parametrize(
        "coords",
        [
            [8, 6, 4, 2, 0],
            [0, 2, 3, 7, 8],
            [4, 0, 8, 2, 6],
            [0, 2, 2, 6, 8],
            [2, 0, 8, 2, 6],
        ],
    )
    @pytest.mark.parametrize("value", [2, slice(2, 8)])
    def test_coordinate_order(self, patch, coords, value):
        """Descending, irregular, and duplicate labels use pandas semantics."""
        patch = patch.update_coords(distance=CoordArray(values=np.asarray(coords)))
        try:
            patch.io.to_xarray().sel(distance=value)
        except (KeyError, pd.errors.InvalidIndexError) as exc:
            with pytest.raises(type(exc)):
                patch.sel(distance=value)
        else:
            assert_matches(patch, "sel", {"distance": value})

    def test_duplicate_array_error(self, patch):
        """Array label lookup requires the same unique index xarray requires."""
        patch = patch.update_coords(distance=np.array([0, 2, 2, 6, 8]))
        for obj in (patch, patch.io.to_xarray()):
            with pytest.raises(pd.errors.InvalidIndexError):
                obj.sel(distance=[2, 6])

    def test_string_exact(self, patch):
        """String labels are exact, even when they contain glob characters."""
        with pytest.raises(KeyError):
            patch.sel(component="*")


class TestIntegration:
    """DASCore units, backend data, serialization, and coordinate types."""

    @pytest.mark.parametrize("method", ["sel", "isel"])
    def test_call_forms(self, patch, method):
        """Dictionary and keyword forms work, and ambiguous input fails."""
        func = getattr(patch, method)
        assert func({"distance": [2, 0]}).equals(func(distance=[2, 0]))
        with pytest.raises(ValueError):
            func({"distance": 2}, time=0)
        with pytest.raises(TypeError):
            func([2, 0])
        with pytest.raises(ValueError):
            func(latitude=40)
        with pytest.raises(TypeError, match="unlabelled"):
            func(distance=xr.DataArray([0, 2], dims="points"))

    def test_units(self, patch):
        """Label and tolerance quantities convert to the coordinate's units."""
        out = patch.sel(distance=400 * dc.units.cm)
        assert out.equals(patch.sel(distance=4))
        out = patch.sel(distance=slice(200 * dc.units.cm, 6 * m))
        assert out.equals(patch.sel(distance=slice(2, 6)))
        out = patch.sel(
            distance=310 * dc.units.cm, method="nearest", tolerance=100 * dc.units.cm
        )
        assert out.equals(patch.sel(distance=4))
        assert out.get_coord("distance").units == patch.get_coord("distance").units

    def test_time_tolerance_units(self, patch):
        """Unit-bearing time tolerances are durations, not timestamps."""
        target = np.datetime64("2020-01-01T00:00:01.4")
        out = patch.sel(time=target, method="nearest", tolerance=0.5 * s)
        assert out.equals(patch.isel(time=1))

    @pytest.mark.parametrize("drop", [False, True])
    def test_scalar_roundtrip(self, patch, drop):
        """A scalar Patch stays scalar through updates, pickle, and export."""
        out = patch.isel(distance=1, time=2, component=3, drop=drop)
        assert out.shape == () and out.size == 1
        assert out.update().shape == ()
        assert pickle.loads(pickle.dumps(out)).equals(out)
        assert (out + 1).shape == ()
        assert get_coord_manager().shape == (0,)
        assert dc.Patch().shape == (0,)

    def test_partial_dimension(self):
        """Unlabelled dimensions have xarray's positional fallback."""
        patch = dc.Patch(data=np.arange(12).reshape(3, 4), dims=("x", "y"), coords={})
        for method in ("sel", "isel"):
            assert_matches(patch, method, {"x": [2, 0], "y": slice(1, 3)})
            assert_matches(patch, method, {"x": 1})

    def test_segmented(self, patch):
        """Selections work across gaps in a compact segmented coordinate."""
        coord = concat_coords(
            get_coord(start=0, stop=3, step=1), get_coord(start=5, stop=7, step=1)
        )
        patch = patch.update_coords(distance=coord)
        assert_matches(patch, "isel", {"distance": [4, 0, 3]})
        assert_matches(patch, "sel", {"distance": slice(1, 6)})

    @pytest.mark.parametrize("backend", ["dask.array", "array_api_strict"])
    def test_backend(self, patch, backend):
        """Multiple array indexers retain the data backend and its laziness."""
        xp = pytest.importorskip(backend)
        data = xp.asarray(patch.data)
        other = patch.update(data=data)
        out = other.isel(distance=[3, 0], time=1, component=[2, 0, 2])
        expected = patch.isel(distance=[3, 0], time=1, component=[2, 0, 2])
        assert type(out.data) is type(data)
        np.testing.assert_array_equal(to_numpy(out.data), expected.data)


class TestIndexingBoundaries:
    """Less common indexing contracts that matter for real labelled data."""

    def test_float_precision(self, patch):
        """Lookup rounds probes to the float coordinate precision, as xarray does."""
        values = np.array([0.1, 0.2, 0.31, 0.4, 0.5], dtype=np.float32)
        patch = patch.update_coords(distance=CoordArray(values=values))
        assert_matches(patch, "sel", {"distance": [0.1, 0.31, 0.1]})

    def test_nonmonotonic_datetime_slice(self, patch):
        """A datetime slice requiring an array indexer raises like xarray."""
        patch = patch.isel(time=[4, 0, 3, 1, 5, 2])
        window = slice("2020-01-01T00:00:01.000000000", "2020-01-01T00:00:04.000000000")
        for obj in (patch, patch.io.to_xarray()):
            with pytest.raises(KeyError):
                obj.sel(time=window)

    def test_unsupported_method(self, patch):
        """Unsupported filling methods are rejected explicitly."""
        with pytest.raises(ValueError, match="method"):
            patch.sel(distance=3, method="pad")

    def test_unsigned_overflow(self, patch):
        """Huge unsigned indices must not wrap into valid negative indices."""
        with pytest.raises(IndexError):
            patch.isel(distance=np.array([2**64 - 1], dtype=np.uint64))

    def test_unlabelled_nearest(self):
        """Nearest selection requires a dimension with actual labels."""
        patch = dc.Patch(data=np.arange(4), dims=("x",), coords={})
        for obj in (patch, patch.io.to_xarray()):
            with pytest.raises(ValueError):
                obj.sel(x=1, method="nearest")

    def test_coord_manager_without_data(self, patch):
        """Coordinates can be indexed on their own with the same scalar rules."""
        coords, data = patch.coords.isel(distance=1, time=[4, 0])
        assert data is None
        assert coords == patch.isel(distance=1, time=[4, 0]).coords

    def test_scalar_constructor(self):
        """Scalar arrays disambiguate an otherwise empty coordinate manager."""
        empty = get_coord_manager()
        scalar = dc.Patch(data=np.asarray(7), coords=empty, dims=())
        assert scalar.shape == ()
        assert scalar.coords != empty
        assert scalar.coords.new().shape == ()
        assert scalar.coords.update().shape == ()
        labelled = scalar.coords.update(station=((), np.asarray("X")))
        assert labelled.drop_coords("station")[0].shape == ()
        assert labelled.set_dims().shape == ()
        assert not scalar.append_dims(x=2).coords.scalar
        restored = type(scalar.coords).model_validate(scalar.coords.model_dump())
        assert restored.shape == ()
