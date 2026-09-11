"""Tests for the lazy xarray index over DASCore coordinates."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.core.coords import (
    CoordArray,
    CoordRange,
    CoordSegmented,
    concat_coords,
    get_coord,
)

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
        np.testing.assert_array_equal(out[name].values, expected[name].values)


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
        dict(x=slice(values[21], values[3], -3)),
        dict(x=slice(values[3], values[9], 0)),
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
            slice(np.datetime64("NaT", "ns"), T0 + 5 * HOUR),
            slice(np.datetime64("2500-01-01"), None),
            slice(None, np.datetime64("1500-01-01")),
            np.datetime64("2500-01-01"),
            np.datetime64("2020-01-01T05", "h"),
            xr.DataArray("2020-01-01"),
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

    @pytest.mark.parametrize("label", [0, [0, 1], 5.0])
    def test_numbers_do_not_name_stamps(self, label):
        """A number is no timestamp, even on a grid starting at the epoch."""
        epoch = get_coord(start=np.datetime64(0, "ns"), step=ONE_MS, shape=(10,))
        lazy, eager = _pair(epoch)
        _assert_same(lambda: lazy.sel(x=label), lambda: eager.sel(x=label))

    def test_integers_past_float_precision(self):
        """An integer grid beyond 2**53 answers exactly."""
        lazy, eager = _pair(get_coord(start=2**53, step=1, shape=(100,)))
        for label in (2**53 + 5, [2**53 + 5, 2**53 + 7]):
            _assert_same(lambda q=label: lazy.sel(x=q), lambda q=label: eager.sel(x=q))

    def test_repeats_anywhere_refuse_arrays(self):
        """A repeating segment far from the query still refuses array lookups."""
        coord = concat_coords(
            get_coord(start=0, step=1, shape=(5,)),
            get_coord(start=10, step=0, shape=(3,)),
            get_coord(start=20, step=1, shape=(5,)),
        )
        lazy, eager = _pair(coord)
        for query in (dict(x=[1, 21]), dict(x=21, method="nearest")):
            _assert_same(lambda q=query: lazy.sel(**q), lambda q=query: eager.sel(**q))

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
            xr.DataArray(np.arange(len(MS)) % 3 == 0, dims="x"),
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

    @pytest.mark.parametrize("name", ["ms", "fraction", "int", "descending"])
    def test_integer_grids_skip_pandas(self, name, monkeypatch):
        """Scalars and ascending slices on integer grids resolve arithmetically."""
        from dascore.utils import indexing  # noqa: PLC0415

        coord = COORDS[name]
        lazy, _ = _pair(coord)

        def _refuse(*args, **kwargs):
            raise AssertionError("resolved through pandas")

        values = coord.values
        monkeypatch.setattr(indexing, "_label_index", _refuse)
        assert lazy.sel(x=values[5]).values == 5
        if coord.sorted:
            out = lazy.sel(x=slice(values[3], values[9], 2))
            np.testing.assert_array_equal(out.values, [3, 5, 7, 9])

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
            [1.5],
            [1000],
        ],
    )
    def test_labels_match(self, name, indexer):
        """Every indexer labels the samples it selects, as the eager array does."""
        lazy, eager = _pair(COORDS[name])
        if isinstance(indexer, np.ndarray):
            indexer = np.resize(indexer, len(COORDS[name]))
        try:
            expected = eager.isel(x=indexer)
        except Exception as err:
            with pytest.raises(type(err)):
                lazy.isel(x=indexer)
            return
        out = lazy.isel(x=indexer)
        _assert_same(lambda: out, lambda: expected)
        if np.ndim(indexer) and len(expected["x"]):
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

    def test_float_slices_keep_their_labels(self):
        """A float range's slice is materialized, keeping the labels it selects."""
        coord = COORDS["float"]
        lazy, _ = _pair(coord)
        sub = lazy.isel(x=slice(3, 20))
        assert sub.sel(x=coord.values[5]).values == 5
        assert (lazy + sub).sizes["x"] == 17

    def test_segmented_slice_stays_lazy(self):
        """A contiguous slice of a segmented coordinate is served lazily too."""
        lazy, _ = _pair(COORDS["segmented"])
        assert isinstance(lazy.isel(x=slice(10, 60)).xindexes["x"], CoordIndex)
        strided = lazy.isel(x=slice(10, 60, 2)).xindexes["x"]
        assert isinstance(strided.coordinate, CoordArray)

    def test_fancy_indexing_holds_its_picks(self):
        """Fancy indexing holds the picked labels, and still aligns with a slice."""
        lazy, eager = _pair(MS)
        index = lazy.isel(x=[1, 5]).xindexes["x"]
        assert isinstance(index, CoordIndex)
        assert isinstance(index.coordinate, CoordArray)
        _assert_same(
            lambda: lazy.isel(x=[0, 1, 2]) + lazy.isel(x=slice(0, 3)),
            lambda: eager.isel(x=[0, 1, 2]) + eager.isel(x=slice(0, 3)),
        )

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


class TestRenames:
    """Renaming a dimension or its coordinate keeps them apart."""

    @pytest.mark.parametrize("name", ["ms", "segmented"])
    def test_rename_dims_and_vars(self, name):
        """Selection and slicing follow each new name, as eager arrays do."""
        lazy, eager = (x.to_dataset(name="v") for x in _pair(COORDS[name]))
        label = COORDS[name].values[3]
        for rename, key in ((dict(x="y"), "dims"), (dict(x="z"), "vars")):
            got = getattr(lazy, f"rename_{key}")(**rename)
            want = getattr(eager, f"rename_{key}")(**rename)
            coord_name, dim = ("x", "y") if key == "dims" else ("z", "x")
            _assert_same(
                lambda g=got, n=coord_name: g.sel({n: label})["v"],
                lambda w=want, n=coord_name: w.sel({n: label})["v"],
            )
            sliced = got.isel({dim: slice(2, 9)})
            assert sliced[coord_name].dims == (dim,)
            np.testing.assert_array_equal(
                sliced[coord_name].values,
                want.isel({dim: slice(2, 9)})[coord_name].values,
            )


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
        """Parts which do not chain in order keep their labels, held as they are."""
        parts = self._parts(*pieces)
        out = xr.concat(parts, dim="x")
        assert isinstance(out.xindexes["x"].coordinate, CoordArray)
        expected = np.concatenate([x["x"].values for x in parts])
        np.testing.assert_array_equal(out["x"].values, expected)

    def test_concat_with_a_materialized_part(self):
        """A materialized part makes the result materialized."""
        lazy, _ = _pair(MS[:50])
        later = _pair(MS[50:])[1]
        out = xr.concat([lazy, later], dim="x")
        np.testing.assert_array_equal(out["x"].values, MS.values)
        assert type(out.xindexes["x"]).__name__ == "PandasIndex"

    def test_concat_after_a_materialized_part(self):
        """A materialized first part concatenates with lazy ones."""
        lazy, eager = _pair(MS[:10])
        _assert_same(
            lambda: xr.concat([lazy.isel(x=[0, 1]), lazy.isel(x=slice(2, None))], "x"),
            lambda: xr.concat(
                [eager.isel(x=[0, 1]), eager.isel(x=slice(2, None))], "x"
            ),
        )

    def test_concat_after_a_pandas_index(self):
        """A plain pandas index first reads the lazy parts' materialized form."""
        eager = _pair(MS[:50])[1]
        lazy = _pair(MS[50:])[0]
        out = xr.concat([eager, lazy], dim="x")
        assert type(out.xindexes["x"]).__name__ == "PandasIndex"
        np.testing.assert_array_equal(out["x"].values, MS.values)

    def test_float_parts_keep_their_labels(self):
        """Float parts never fuse into a range that would relabel them."""
        coord = COORDS["float"]
        parts = [_pair(coord[s])[0] for s in (slice(0, 20), slice(20, 50))]
        out = xr.concat(parts, dim="x")
        expected = np.concatenate([x["x"].values for x in parts])
        np.testing.assert_array_equal(out["x"].values, expected)

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

    def test_exact_join_across_representations(self):
        """A slice and the same samples picked by position are the same labels."""
        lazy, eager = _pair(MS)
        _assert_same(
            lambda: xr.align(
                lazy.isel(x=slice(0, 3)), lazy.isel(x=[0, 1, 2]), join="exact"
            )[0],
            lambda: xr.align(
                eager.isel(x=slice(0, 3)), eager.isel(x=[0, 1, 2]), join="exact"
            )[0],
        )
        other = lazy.isel(x=[0, 1, 3])
        with pytest.raises(xr.AlignmentError):
            xr.align(lazy.isel(x=slice(0, 3)), other, join="exact")

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

    @pytest.mark.parametrize("method", [None, "nearest"])
    def test_reindex_like_takes_method(self, method):
        """Reindexing to another lazy array honours method, as eager does."""
        lazy, eager = _pair(MS, data=np.arange(len(MS)) * 1.0)
        # a coarser grid off this one's samples, so only nearest finds them
        coarse = get_coord(start=T0 + ONE_MS // 3, step=10 * ONE_MS, shape=(10,))
        lazy_to, eager_to = _pair(coarse)
        _assert_same(
            lambda: lazy.reindex_like(lazy_to, method=method),
            lambda: eager.reindex_like(eager_to, method=method),
        )

    def test_materialized_index_refuses_to_align(self):
        """Xarray matches indexes by type; lazy_coords=False is the way out."""
        lazy, eager = _pair(MS)
        with pytest.raises(xr.AlignmentError):
            lazy + eager


class TestFromVariables:
    """An ordinary array takes this index type through set_xindex."""

    @pytest.mark.parametrize("name", ["ms", "descending", "float", "segmented"])
    def test_labels_kept_exactly(self, name):
        """The labels an array holds come back unchanged, as a range when even."""
        _, eager = _pair(COORDS[name])
        out = eager.drop_indexes("x").set_xindex("x", CoordIndex)
        assert isinstance(out.xindexes["x"], CoordIndex)
        np.testing.assert_array_equal(out["x"].values, eager["x"].values)
        _assert_same(
            lambda: out.sel(x=eager["x"].values[3]),
            lambda: eager.sel(x=eager["x"].values[3]),
        )

    def test_aligns_with_a_lazy_array_without_reading_it(self, monkeypatch):
        """Equal labels align without evaluating the lazy side's labels."""
        lazy, eager = _pair(MS)
        converted = eager.drop_indexes("x").set_xindex("x", CoordIndex)
        original = CoordRange._get_index_values

        def _bounded(self, indices):
            assert np.size(indices) < 10, "the lazy labels were read"
            return original(self, indices)

        monkeypatch.setattr(CoordRange, "_get_index_values", _bounded)
        out = lazy + converted
        assert isinstance(out.xindexes["x"], CoordIndex)

    def test_text_labels_stay_text(self):
        """Picked text labels keep their own coordinate class."""
        text = xr.DataArray(np.arange(3), dims="x", coords={"x": ["a", "b", "c"]})
        out = text.drop_indexes("x").set_xindex("x", CoordIndex)
        picked = out.isel(x=[0, 1])
        assert type(picked.xindexes["x"].coordinate).__name__ == "CoordString"
        joined = xr.concat([picked, out.isel(x=[2])], "x")
        np.testing.assert_array_equal(joined["x"].values, ["a", "b", "c"])

    def test_units_and_refusals(self):
        """Units come from the variable; several or 2-d variables are refused."""
        distance = xr.DataArray(
            np.arange(5.0),
            dims="d",
            coords={"d": ("d", np.arange(5.0), {"units": "m"})},
        )
        out = distance.drop_indexes("d").set_xindex("d", CoordIndex)
        assert out.xindexes["d"].coordinate.units == dc.get_quantity("m")
        pair = xr.Dataset(coords={"a": ("x", [1, 2]), "b": ("x", [3, 4])})
        with pytest.raises(ValueError, match="one coordinate"):
            pair.set_xindex(["a", "b"], CoordIndex)
        grid = xr.Dataset(coords={"g": (("x", "y"), [[1, 2], [3, 4]])})
        with pytest.raises(ValueError, match="one-dimensional"):
            grid.set_xindex("g", CoordIndex)


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

    def test_segmented_selection_reads_few_labels(self, monkeypatch):
        """Selecting on a long segmented coordinate evaluates a handful of labels."""
        n = 1_000_000_000
        first = get_coord(start=T0, step=ONE_MS, shape=(n,))
        second = get_coord(start=first.max() + 10 * ONE_MS, step=ONE_MS, shape=(n,))
        coord = concat_coords(first, second)
        index = CoordIndex.from_coord("x", coord)
        lazy = xr.DataArray(
            da.zeros((2 * n,), chunks=100_000_000),
            dims=("x",),
            coords=xr.Coordinates.from_xindex(index),
        )
        original = CoordRange._get_index_values

        def _bounded(self, indices):
            assert np.size(indices) < 1000, "the selection evaluated every label"
            return original(self, indices)

        monkeypatch.setattr(CoordRange, "_get_index_values", _bounded)
        label = second.min() + 5 * ONE_MS
        assert int(lazy.sel(x=label)["x"].values == label)
        sub = lazy.sel(x=slice(first.max() - ONE_MS, label))
        assert sub.sizes["x"] == 8

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
        array = patch_to_xarray(patch, lazy_coords=True)
        assert isinstance(array.xindexes["x"], CoordIndex)
        assert xarray_to_patch(array).get_coord("x") == coord

    def test_attrs_units_win(self):
        """Units stated beside a lazy coordinate are the coordinate's units."""
        coord = get_coord(start=0.0, step=0.5, shape=(20,), units="m")
        patch = dc.Patch(data=np.arange(20), dims=("x",), coords={"x": coord})
        array = patch_to_xarray(patch, lazy_coords=True)
        assert array["x"].attrs["units"] == "1 m"
        array["x"].attrs["units"] = "ft"
        back = xarray_to_patch(array).get_coord("x")
        assert back == coord.set_units("ft")

    @pytest.mark.parametrize("lazy", [{}, {"lazy_coords": False}, {"lazy_coords": ()}])
    def test_materialized_by_default(self, lazy):
        """By default (or with no names) every coordinate is materialized."""
        patch = dc.get_example_patch()
        array = patch_to_xarray(patch, **lazy)
        kinds = {type(x).__name__ for x in array.xindexes.values()}
        assert kinds == {"PandasIndex"}
        assert xarray_to_patch(array) == patch

    def test_all_serves_ranges_only(self):
        """lazy_coords=True leaves a coordinate holding its labels materialized."""
        coord = get_coord(data=np.array([0.0, 1.0, 5.0]))
        patch = dc.Patch(data=np.arange(3), dims=("x",), coords={"x": coord})
        array = patch_to_xarray(patch, lazy_coords=True)
        assert type(array.xindexes["x"]).__name__ == "PandasIndex"
        named = patch_to_xarray(patch, lazy_coords={"x"})
        assert isinstance(named.xindexes["x"], CoordIndex)

    def test_named_coordinates_only(self):
        """Names choose which dimensions are served lazily."""
        array = patch_to_xarray(dc.get_example_patch(), lazy_coords={"time"})
        assert isinstance(array.xindexes["time"], CoordIndex)
        assert type(array.xindexes["distance"]).__name__ == "PandasIndex"
