"""Tests for the lazy xarray index over DASCore coordinates."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.core.coords import CoordRange, CoordSegmented, concat_coords, get_coord

xr = pytest.importorskip("xarray")
da = pytest.importorskip("dask.array")

from dascore.xarray.index import CoordIndex, CoordTransform, is_servable  # noqa: E402
from dascore.xarray.patch import patch_to_xarray, xarray_to_patch  # noqa: E402

T0 = np.datetime64("2020-01-01", "ns")
ONE_MS = np.timedelta64(1_000_000, "ns")
HOUR = np.timedelta64(3_600_000_000_000, "ns")

MS = get_coord(start=T0, step=ONE_MS, shape=(100,))
COORDS = {
    "ms": MS,
    "hourly": get_coord(start=T0, step=HOUR, shape=(48,)),
    "descending": MS[::-1],
    "fraction": get_coord(start=T0, step=(1, 1024), shape=(3000,)),
    "duration": get_coord(start=np.timedelta64(0, "s"), step=ONE_MS, shape=(50,)),
    "float": get_coord(start=-3.3, step=1.0209, shape=(50,)),
    "int": get_coord(start=10, step=3, shape=(40,)),
    "segmented": concat_coords(MS[:30], MS[50:]),
    "float_segmented": concat_coords(
        get_coord(start=0.0, step=0.5, shape=(20,)),
        get_coord(start=20.0, step=0.5, shape=(20,)),
    ),
}


def _pair(coord, name="x", data=None):
    """The same array labeled lazily and eagerly by one coordinate."""
    data = np.arange(len(coord)) if data is None else data
    index = CoordIndex.from_coord(name, coord)
    lazy = xr.DataArray(data, dims=(name,), coords=xr.Coordinates.from_xindex(index))
    eager = xr.DataArray(data, dims=(name,), coords={name: coord.values})
    return lazy, eager


def _assert_same(lazy_call, eager_call):
    """The lazy array answers as the eager one does, raising alike."""
    try:
        expected = eager_call()
    except Exception as err:
        with pytest.raises(type(err)):
            lazy_call()
        return
    out = lazy_call()
    assert out.sizes == expected.sizes
    np.testing.assert_array_equal(np.asarray(out.values), expected.values)
    for name in expected.coords:
        got, want = out[name].values, expected[name].values
        if want.dtype.kind == "f":
            # a slice of a float range is labeled as DASCore labels it,
            # which can differ from the parent's labels in the last bits
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
        else:
            np.testing.assert_array_equal(got, want)


def _between(values, i):
    """A label a third of the way from sample i to sample i + 1."""
    gap = values[i + 1] - values[i]
    return values[i] + (gap // 3 if values.dtype.kind in "iumM" else gap / 3)


def _queries(values):
    """Label selections to ask of both arrays, as sel keyword dicts."""
    off = _between(values, 7)
    step = values[1] - values[0]
    far = values[0] - 5 * step
    tiny = step // 10 if values.dtype.kind in "iumM" else step / 10
    tiny = tiny or 1
    return [
        dict(x=values[7]),
        dict(x=values[[2, 8]]),
        dict(x=off),
        dict(x=off, method="nearest"),
        dict(x=values[[2, 8]] + (off - values[7]), method="nearest"),
        dict(x=far, method="nearest"),
        dict(x=off, method="nearest", tolerance=abs(tiny)),
        dict(x=off, method="nearest", tolerance=abs(step)),
        dict(x=slice(values[3], values[9])),
        dict(x=slice(off, values[20])),
        dict(x=slice(None, values[5])),
        dict(x=slice(values[5], None)),
        dict(x=slice(values[3], values[21], 3)),
        dict(x=slice(far, values[4])),
        dict(x=slice(values[9], values[3])),
    ]


class TestSelParity:
    """Label selection answers as a materialized pandas index answers."""

    @pytest.mark.parametrize("name", list(COORDS))
    def test_queries(self, name):
        """Scalars, arrays, nearest, tolerance, and slices, on every kind."""
        lazy, eager = _pair(COORDS[name])
        for query in _queries(COORDS[name].values):
            _assert_same(lambda q=query: lazy.sel(**q), lambda q=query: eager.sel(**q))

    @pytest.mark.parametrize(
        "label",
        [
            "2020-01-01",
            "2020-01-01T05",
            "2019-06",
            "not-a-date",
            ["not-a-date"],
            0,
            [0, 1],
            slice("2020-01-01", "2020-01-02"),
            slice("2020-01-01T10", None),
        ],
    )
    def test_datetime_strings(self, label):
        """Partial strings name their periods, as pandas reads them."""
        lazy, eager = _pair(COORDS["hourly"])
        _assert_same(lambda: lazy.sel(x=label), lambda: eager.sel(x=label))

    def test_one_sample_period_keeps_dimension(self):
        """A coarse string over one sample keeps the dimension."""
        coord = get_coord(start=T0 - HOUR, step=HOUR, shape=(30,))
        lazy, eager = _pair(coord)
        for label in ("2019-12-31", "2019-12-31T23"):
            _assert_same(lambda q=label: lazy.sel(x=q), lambda q=label: eager.sel(x=q))

    def test_dataarray_labels_keep_their_dims(self):
        """Vectorized labels select along their own dimension."""
        lazy, eager = _pair(MS)
        values = MS.values
        for label in (
            xr.DataArray(values[[2, 8]], dims="z"),
            xr.Variable("z", values[[4, 9]]),
            xr.DataArray(values[[[1, 2], [3, 4]]], dims=("a", "b")),
        ):
            _assert_same(lambda q=label: lazy.sel(x=q), lambda q=label: eager.sel(x=q))
        out = lazy.sel(x=xr.DataArray(values[[2, 8]], dims="z"))
        assert out.dims == ("z",)

    def test_method_with_slice_raises(self):
        """Neither index takes a method with a slice."""
        lazy, _ = _pair(MS)
        with pytest.raises(NotImplementedError):
            lazy.sel(x=slice(MS.values[1], None), method="nearest")

    def test_sel_reads_no_labels(self, monkeypatch):
        """Selecting on a range looks up the queried labels only."""
        lazy, _ = _pair(MS)

        def _refuse(self):
            raise AssertionError("the coordinate spelled out its labels")

        monkeypatch.setattr(CoordRange, "values", property(_refuse))
        values = MS._get_index_values(np.arange(len(MS)))
        assert lazy.sel(x=slice(values[3], values[9])).sizes["x"] == 7
        assert lazy.sel(x=values[5]).values == 5
        assert lazy.isel(x=slice(2, 50, 3)).sel(x=values[5]).values == 5

    def test_segmented_sel_keeps_no_labels(self, monkeypatch):
        """A segmented coordinate evaluates labels for a lookup, never keeps them."""
        lazy, _ = _pair(concat_coords(MS[:30], MS[50:]))

        def _refuse(self):
            raise AssertionError("the coordinate spelled out and cached its labels")

        monkeypatch.setattr(CoordSegmented, "values", property(_refuse))
        out = lazy.sel(x=slice(MS.values[20], MS.values[60]))
        np.testing.assert_array_equal(out.values, np.r_[20:30, 30:41])


class TestIsel:
    """Positional selection keeps a lazy index where it can."""

    @pytest.mark.parametrize("name", list(COORDS))
    @pytest.mark.parametrize(
        "indexer",
        [
            slice(3, 20),
            slice(2, 50, 3),
            slice(None, None, -1),
            slice(30, 5, -2),
            slice(5, 5),
            [1, 5, 9],
            [-1, 0],
            np.arange(10) % 3 == 0,
            3,
        ],
    )
    def test_labels_match(self, name, indexer):
        """Every indexer labels the samples it selects, as the eager array does."""
        lazy, eager = _pair(COORDS[name])
        if isinstance(indexer, np.ndarray):
            indexer = np.resize(indexer, len(COORDS[name]))
        out, expected = lazy.isel(x=indexer), eager.isel(x=indexer)
        _assert_same(lambda: out, lambda: expected)
        if np.ndim(indexer):
            # a label lookup still works on the result
            label = expected["x"].values[0]
            _assert_same(lambda: out.sel(x=label), lambda: expected.sel(x=label))

    @pytest.mark.parametrize(
        "indexer", [slice(3, 20), slice(2, 50, 3), slice(None, None, -1)]
    )
    def test_slices_stay_lazy(self, indexer):
        """A slice of a range is a range: the index stays lazy."""
        lazy, _ = _pair(MS)
        index = lazy.isel(x=indexer).xindexes["x"]
        assert isinstance(index, CoordIndex)
        assert index.coordinate == MS[indexer]

    def test_segmented_slice_stays_lazy(self):
        """A contiguous slice of a segmented coordinate is served lazily too."""
        lazy, _ = _pair(COORDS["segmented"])
        assert isinstance(lazy.isel(x=slice(10, 60)).xindexes["x"], CoordIndex)
        assert not isinstance(lazy.isel(x=slice(10, 60, 2)).xindexes["x"], CoordIndex)

    def test_fancy_indexing_materializes(self):
        """Fancy indexing keeps a materialized index over the picks."""
        lazy, _ = _pair(MS)
        assert type(lazy.isel(x=[1, 5]).xindexes["x"]).__name__ == "PandasIndex"

    def test_vectorized_isel_onto_another_dimension(self):
        """An indexer on a new dimension moves the labels there."""
        lazy, eager = _pair(MS)
        picks = xr.DataArray([1, 5], dims="sample")
        out, expected = lazy.isel(x=picks), eager.isel(x=picks)
        assert out["x"].dims == expected["x"].dims == ("sample",)
        assert "x" not in out.xindexes
        np.testing.assert_array_equal(out["x"].values, expected["x"].values)
        grid = xr.DataArray([[1, 5]], dims=("a", "b"))
        assert "x" not in lazy.isel(x=grid).xindexes


class TestTransform:
    """The transform contract xarray relies on."""

    @pytest.mark.parametrize("name", list(COORDS))
    def test_forward_matches_values(self, name):
        """Labels at any positions are the coordinate's own."""
        coord = COORDS[name]
        transform = CoordTransform("x", coord)
        positions = np.array([0, 3, len(coord) - 1, 3])
        out = transform.forward({"x": positions})["x"]
        np.testing.assert_array_equal(out, coord.values[positions])
        rounded = transform.forward({"x": np.array([2.4, 2.6])})["x"]
        np.testing.assert_array_equal(rounded, coord.values[[2, 3]])

    def test_reverse_returns_nearest_float_positions(self):
        """The reverse transform serves float positions of nearest samples."""
        transform = CoordTransform("x", MS)
        out = transform.reverse({"x": MS.values[[7]] + ONE_MS // 3})["x"]
        assert out.dtype == np.float64
        assert out[0] == 7.0

    def test_equality_compares_labels(self):
        """Transforms are equal when their labels are, units aside."""
        distance = get_coord(start=0.0, step=0.5, shape=(20,), units="m")
        same = CoordTransform("x", distance)
        assert same.equals(CoordTransform("x", distance.set_units("ft")))
        assert not same.equals(CoordTransform("x", distance[1:]))
        assert not same.equals(CoordTransform("y", distance))
        assert not same.equals(CoordTransform("x", distance.update(min=1.0)))
        assert not same.equals(CoordTransform("x", MS[:20]))

    def test_pickle_roundtrip(self):
        """The index ships in a dask graph, so it must pickle."""
        index = CoordIndex.from_coord("x", COORDS["fraction"])
        loaded = pickle.loads(pickle.dumps(index))
        assert loaded.equals(index)
        assert loaded.coordinate == index.coordinate


class TestEligibility:
    """Which coordinates are served lazily."""

    @pytest.mark.parametrize("name", list(COORDS))
    def test_ranges_and_segments_are_served(self, name):
        """Every range and segmented coordinate, any dtype or direction."""
        assert is_servable(COORDS[name])

    def test_others_are_not(self):
        """Arrays and a zero step, whose labels repeat, are not."""
        assert not is_servable(get_coord(data=np.array([0.0, 1.0, 5.0])))
        assert not is_servable(get_coord(start=0, step=0, shape=(4,)))
        with pytest.raises(AssertionError, match="not servable"):
            CoordIndex.from_coord("x", get_coord(data=np.array([0.0, 1.0, 5.0])))


class TestRename:
    """Renaming keeps the lazy index working under the new name."""

    def test_rename_dim_and_coord(self):
        """Label access works after rename, under the new name."""
        lazy, _ = _pair(MS)
        renamed = lazy.rename({"x": "t"})
        assert isinstance(renamed.xindexes["t"], CoordIndex)
        out = renamed.sel(t=slice(MS.values[3], MS.values[5]))
        np.testing.assert_array_equal(out.values, [3, 4, 5])
        assert renamed.xindexes["t"].rename({"other": "o"}, {}) is renamed.xindexes["t"]


class TestConcat:
    """Concatenation stays lazy when the coordinates chain."""

    def _parts(self, *pieces):
        return [_pair(MS[piece], data=np.arange(100)[piece])[0] for piece in pieces]

    def test_contiguous_concat_is_one_range(self):
        """Abutting slices merge back into the range they came from."""
        out = xr.concat(self._parts(slice(0, 50), slice(50, 100)), dim="x")
        assert out.xindexes["x"].coordinate == MS
        np.testing.assert_array_equal(out.values, np.arange(100))

    def test_gapped_concat_is_segmented(self):
        """A gap stays lazy as a segmented coordinate, labels unchanged."""
        out = xr.concat(self._parts(slice(0, 30), slice(60, 100)), dim="x")
        index = out.xindexes["x"]
        assert isinstance(index.coordinate, CoordSegmented)
        np.testing.assert_array_equal(
            out["x"].values, np.r_[MS.values[:30], MS.values[60:]]
        )

    def test_descending_concat_stays_lazy(self):
        """Reversed parts chain in their own direction."""
        parts = [_pair(MS[::-1][s])[0] for s in (slice(0, 50), slice(50, 100))]
        out = xr.concat(parts, dim="x")
        assert out.xindexes["x"].coordinate == MS[::-1]

    @pytest.mark.parametrize(
        "pieces",
        [
            (slice(50, 100), slice(0, 50)),  # out of order
            (slice(0, 60), slice(40, 100)),  # overlapping
        ],
    )
    def test_unchained_concat_materializes(self, pieces):
        """Parts which do not chain in order keep their labels, materialized."""
        parts = self._parts(*pieces)
        out = xr.concat(parts, dim="x")
        assert type(out.xindexes["x"]).__name__ == "PandasIndex"
        expected = np.concatenate([x["x"].values for x in parts])
        np.testing.assert_array_equal(out["x"].values, expected)

    def test_concat_with_a_materialized_part(self):
        """A materialized part makes the result materialized."""
        lazy, _ = _pair(MS[:50])
        later = _pair(MS[50:])[1]
        out = xr.concat([lazy, later], dim="x")
        np.testing.assert_array_equal(out["x"].values, MS.values)
        assert type(out.xindexes["x"]).__name__ == "PandasIndex"

    def test_concat_with_positions_reorders(self):
        """An explicit permutation materializes in that order."""
        first = CoordIndex.from_coord("x", MS[:50])
        second = CoordIndex.from_coord("x", MS[50:])
        out = CoordIndex.concat(
            [first, second], "x", positions=[range(50, 100), range(0, 50)]
        )
        np.testing.assert_array_equal(out.index.values[:50], MS.values[50:])


class TestAlignment:
    """Arithmetic between lazy arrays aligns as between eager ones."""

    def test_equal_labels_need_no_join(self):
        """Arrays whose coordinates label alike stay lazy."""
        distance = get_coord(start=0.0, step=0.5, shape=(20,), units="m")
        first = _pair(distance)[0]
        second = _pair(distance.set_units("ft"))[0]
        out = first + second
        assert isinstance(out.xindexes["x"], CoordIndex)

    @pytest.mark.parametrize("join", ["inner", "outer"])
    def test_offset_arrays_join(self, join):
        """Offset arrays join on their labels, as materialized arrays do."""
        lazy = [_pair(MS[s], data=np.arange(100.0)[s])[0] for s in (slice(0, 60),)]
        lazy.append(_pair(MS[40:], data=np.arange(40.0, 100.0))[0])
        lazy.append(_pair(MS[20:90], data=np.arange(20.0, 90.0))[0])
        eager = [
            xr.DataArray(x.values, dims="x", coords={"x": x["x"].values}) for x in lazy
        ]
        out = xr.align(*lazy, join=join)
        expected = xr.align(*eager, join=join)
        for got, want in zip(out, expected):
            _assert_same(lambda g=got: g, lambda w=want: w)

    def test_materialized_index_refuses_to_align(self):
        """Xarray matches indexes by type; lazy_coords=False is the way out."""
        lazy, eager = _pair(MS)
        with pytest.raises(xr.AlignmentError):
            lazy + eager


class TestScale:
    """The reason this index exists: metadata-cost construction."""

    def test_billion_sample_coordinate_is_free(self):
        """Three billion samples build and select without materializing labels."""
        n = 3_000_000_000
        coord = get_coord(start=T0, step=ONE_MS, shape=(n,))
        index = CoordIndex.from_coord("time", coord)
        array = xr.DataArray(
            da.zeros((n,), chunks=10_000_000),
            dims=("time",),
            coords=xr.Coordinates.from_xindex(index),
        )
        backing = type(array["time"].variable._data).__name__
        assert "CoordinateTransform" in backing
        # the stop names its full second, so the slice keeps 2000 ms samples
        sub = array.sel(time=slice("2020-01-05", "2020-01-05T00:00:01"))
        assert sub.sizes["time"] == 2000
        assert sub["time"].values[0] == np.datetime64("2020-01-05", "ns")
        assert isinstance(sub.xindexes["time"], CoordIndex)

    def test_billion_sample_segments_are_free(self):
        """A gapped merge of long runs builds without materializing labels."""
        n = 1_000_000_000
        first = get_coord(start=T0, step=ONE_MS, shape=(n,))
        second = get_coord(start=first.max() + 10 * ONE_MS, step=ONE_MS, shape=(n,))
        coord = concat_coords(first, second)
        array = xr.DataArray(
            da.zeros((2 * n,), chunks=100_000_000),
            dims=("time",),
            coords=xr.Coordinates.from_xindex(CoordIndex.from_coord("time", coord)),
        )
        sub = array.isel(time=slice(n - 2, n + 2))
        assert isinstance(sub.xindexes["time"].coordinate, CoordSegmented)
        np.testing.assert_array_equal(
            sub["time"].values, coord._get_index_values(np.arange(n - 2, n + 2))
        )


class TestPandasBridge:
    """The pandas spellings materialize on demand."""

    def test_to_pandas_index(self):
        """.indexes serves a real pandas index."""
        lazy, eager = _pair(COORDS["segmented"])
        index = lazy.indexes["x"]
        assert isinstance(index, pd.Index)
        np.testing.assert_array_equal(index.values, eager["x"].values)
        assert len(lazy.to_dataframe(name="v")) == len(COORDS["segmented"])

    def test_repr_names_the_coordinate(self):
        """The index says which coordinate it serves."""
        lazy, _ = _pair(COORDS["segmented"])
        text = repr(lazy.xindexes["x"])
        assert "CoordSegmented" in text
        assert str(len(COORDS["segmented"])) in text
        assert "CoordIndex" in repr(lazy)


class TestPatchRoundTrip:
    """A patch's coordinates travel through xarray as themselves."""

    @pytest.mark.parametrize("name", list(COORDS))
    def test_round_trip_is_exact(self, name):
        """The coordinate comes back equal, exact grid and segments included."""
        coord = COORDS[name]
        patch = dc.Patch(data=np.arange(len(coord)), dims=("x",), coords={"x": coord})
        array = patch_to_xarray(patch)
        assert isinstance(array.xindexes["x"], CoordIndex)
        assert xarray_to_patch(array).get_coord("x") == coord

    def test_attrs_units_win(self):
        """Units stated beside a lazy coordinate are the coordinate's units."""
        coord = get_coord(start=0.0, step=0.5, shape=(20,), units="m")
        patch = dc.Patch(data=np.arange(20), dims=("x",), coords={"x": coord})
        array = patch_to_xarray(patch)
        assert array["x"].attrs["units"] == "1 m"
        array["x"].attrs["units"] = "ft"
        back = xarray_to_patch(array).get_coord("x")
        assert back == coord.set_units("ft")

    @pytest.mark.parametrize("lazy", [False, ()])
    def test_opt_out(self, lazy):
        """lazy_coords=False (or no names) materializes every coordinate."""
        patch = dc.get_example_patch()
        array = patch_to_xarray(patch, lazy_coords=lazy)
        kinds = {type(x).__name__ for x in array.xindexes.values()}
        assert kinds == {"PandasIndex"}
        assert xarray_to_patch(array) == patch

    def test_named_coordinates_only(self):
        """Names choose which dimensions are served lazily."""
        array = patch_to_xarray(dc.get_example_patch(), lazy_coords={"time"})
        assert isinstance(array.xindexes["time"], CoordIndex)
        assert type(array.xindexes["distance"]).__name__ == "PandasIndex"
