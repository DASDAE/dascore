"""Tests for the lazy evenly sampled temporal xarray index."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from dascore.core.coords import get_coord

pytest.importorskip("xarray")
pytest.importorskip("dask")


ONE_MS = np.timedelta64(1_000_000, "ns")


@pytest.fixture(scope="module")
def small_index():
    """An index over 100 samples of millisecond data."""
    from dascore.xarray.index import TemporalRangeIndex  # noqa: PLC0415

    coord = get_coord(
        min=np.datetime64("2020-01-01", "ns"),
        max=np.datetime64("2020-01-01", "ns") + 100 * ONE_MS,
        step=ONE_MS,
    )
    return TemporalRangeIndex.from_coord("time", coord)


@pytest.fixture(scope="module")
def small_array(small_index):
    """A dask-backed DataArray labeled by the small index."""
    import dask.array as da  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415

    coords = xr.Coordinates.from_xindex(small_index)
    data = da.arange(100, chunks=25)
    return xr.DataArray(data, dims=("time",), coords=coords)


class TestTemporalRangeTransform:
    """Unit tests for the position/label arithmetic."""

    def test_forward_matches_coord_values(self, small_index):
        """Lazily served labels equal the materialized coordinate's."""
        coord = get_coord(
            min=np.datetime64("2020-01-01", "ns"),
            max=np.datetime64("2020-01-01", "ns") + 100 * ONE_MS,
            step=ONE_MS,
        )
        served = small_index.transform.forward({"time": np.arange(100)})["time"]
        np.testing.assert_array_equal(served, coord.values)

    def test_reverse_exact_at_large_offsets(self):
        """A label a year of nanoseconds out lands on its exact sample.

        Float inversion loses sample precision past 2**53 ns (~104
        days); the integer divmod must not.
        """
        from dascore.xarray.index import (  # noqa: PLC0415
            TemporalRangeIndex,
            TemporalRangeTransform,
        )

        step_ns = 1  # nanosecond sampling: the worst case
        size = 2**53 + 10
        start = np.datetime64("2020-01-01", "ns").astype("int64")
        t = TemporalRangeTransform("time", size, int(start), step_ns, "datetime64[ns]")
        # 2**53 + 1 has no float64 representation, so a float inversion
        # cannot return it; the exact integer path must.
        pos = 2**53 + 1
        label = t.forward({"time": np.array([pos])})["time"]
        quot, rem = TemporalRangeIndex(t)._exact_positions(label)
        assert int(quot[0]) == pos and int(rem[0]) == 0

    def test_timedelta_dtype(self):
        """A timedelta64 coordinate round-trips through the transform."""
        from dascore.xarray.index import TemporalRangeIndex  # noqa: PLC0415

        coord = get_coord(min=np.timedelta64(0, "ns"), max=10 * ONE_MS, step=ONE_MS)
        index = TemporalRangeIndex.from_coord("offset", coord)
        served = index.transform.forward({"offset": np.arange(len(coord))})["offset"]
        np.testing.assert_array_equal(served, coord.values)

    def test_equals(self, small_index):
        """Equality keys on start, step, size, and dtype."""
        from dascore.xarray.index import (  # noqa: PLC0415
            TemporalRangeIndex,
            TemporalRangeTransform,
        )

        t = small_index.transform
        same = TemporalRangeIndex(
            TemporalRangeTransform("time", 100, t.start_ns, t.step_ns, t.dtype)
        )
        assert t.equals(same.transform)
        changed = [
            TemporalRangeTransform("time", 100, t.start_ns + 1, t.step_ns, t.dtype),
            TemporalRangeTransform("time", 100, t.start_ns, t.step_ns + 1, t.dtype),
            TemporalRangeTransform("time", 99, t.start_ns, t.step_ns, t.dtype),
            TemporalRangeTransform(
                "time", 100, t.start_ns, t.step_ns, "timedelta64[ns]"
            ),
        ]
        for wrong in changed:
            assert not t.equals(wrong)

    def test_pickle_roundtrip(self, small_index):
        """The index ships in a dask graph, so it must pickle."""
        loaded = pickle.loads(pickle.dumps(small_index))
        assert loaded.transform.equals(small_index.transform)


class TestSel:
    """Label selection resolves arithmetically."""

    def test_slice_inclusive_endpoints(self, small_array):
        """Both on-grid endpoints stay, as pandas label slicing keeps."""
        t = small_array["time"].values
        out = small_array.sel(time=slice(t[10], t[20]))
        assert out.sizes["time"] == 11
        np.testing.assert_array_equal(out.compute().values, np.arange(10, 21))

    def test_slice_off_grid_endpoints(self, small_array):
        """Off-grid endpoints keep only samples inside the interval."""
        t = small_array["time"].values
        half = ONE_MS // 2
        out = small_array.sel(time=slice(t[10] + half, t[20] + half))
        np.testing.assert_array_equal(out.compute().values, np.arange(11, 21))

    def test_slice_beyond_span_clamps(self, small_array):
        """Overhanging endpoints clamp to the edges; disjoint ones empty."""
        t = small_array["time"].values
        wide = small_array.sel(time=slice(t[0] - 5 * ONE_MS, t[-1] + 5 * ONE_MS))
        assert wide.sizes["time"] == 100
        before = small_array.sel(time=slice(t[0] - 9 * ONE_MS, t[0] - ONE_MS))
        assert before.sizes["time"] == 0
        after = small_array.sel(time=slice(t[-1] + ONE_MS, t[-1] + 9 * ONE_MS))
        assert after.sizes["time"] == 0
        # far enough out that naive int64 subtraction could overflow
        century = small_array.sel(time=slice("1800-01-01", t[10]))
        assert century.sizes["time"] == 11

    def test_slice_step_strides_samples(self, small_array):
        """An integer slice step strides the selected samples."""
        t = small_array["time"].values
        out = small_array.sel(time=slice(t[10], t[20], 3))
        np.testing.assert_array_equal(out.compute().values, np.arange(10, 21, 3))

    def test_slice_step_validated(self, small_array):
        """A non-integer or negative slice step is refused with a reason."""
        t = small_array["time"].values
        with pytest.raises(ValueError, match="stride"):
            small_array.sel(time=slice(t[0], t[20], ONE_MS))
        with pytest.raises(ValueError, match="stride"):
            small_array.sel(time=slice(t[20], t[0], -1))

    def test_slice_open_ends(self, small_array):
        """Open-ended slices span to the array's edges."""
        t = small_array["time"].values
        assert small_array.sel(time=slice(None, t[5])).sizes["time"] == 6
        assert small_array.sel(time=slice(t[95], None)).sizes["time"] == 5
        assert small_array.sel(time=slice(None, None)).sizes["time"] == 100

    def test_slice_with_string_labels(self, small_array):
        """Strings parse like any other datetime spelling."""
        out = small_array.sel(
            time=slice("2020-01-01T00:00:00.010", "2020-01-01T00:00:00.020")
        )
        assert out.sizes["time"] == 11

    def test_scalar_exact_by_default(self, small_array):
        """An on-grid scalar selects; an off-grid one raises, as pandas."""
        t = small_array["time"].values
        assert small_array.sel(time=t[7]).compute().values == 7
        with pytest.raises(KeyError, match="sample grid"):
            small_array.sel(time=t[7] + ONE_MS // 4)

    def test_scalar_nearest_rounds_both_ways(self, small_array):
        """method="nearest" picks the closer sample on either side."""
        t = small_array["time"].values
        low = small_array.sel(time=t[7] + ONE_MS // 4, method="nearest")
        high = small_array.sel(time=t[7] + (6 * ONE_MS) // 10, method="nearest")
        assert low.compute().values == 7
        assert high.compute().values == 8

    def test_scalar_out_of_span_raises(self, small_array):
        """A label outside the sampled span never clips to an edge."""
        t = small_array["time"].values
        with pytest.raises(KeyError, match="outside"):
            small_array.sel(time=t[0] - ONE_MS, method="nearest")
        with pytest.raises(KeyError, match="outside"):
            small_array.sel(time=t[-1] + ONE_MS, method="nearest")
        # centuries away (but ns-representable) must not wrap into a sample
        with pytest.raises(KeyError):
            small_array.sel(time=np.datetime64("1800-01-01", "ns"), method="nearest")

    def test_nat_labels_raise(self, small_array):
        """NaT never resolves to a sample, scalar or slice endpoint."""
        with pytest.raises(ValueError, match="NaT"):
            small_array.sel(time=np.datetime64("NaT", "ns"), method="nearest")
        with pytest.raises(ValueError, match="NaT"):
            small_array.sel(time=slice(np.datetime64("NaT", "ns"), None))

    def test_unsupported_method_raises(self, small_array):
        """pad/ffill would silently answer differently; refuse them."""
        t = small_array["time"].values
        with pytest.raises(ValueError, match="method"):
            small_array.sel(time=t[7], method="pad")

    def test_dataarray_label_keeps_dims(self, small_array):
        """A DataArray label routes through vectorized nearest lookup."""
        import xarray as xr  # noqa: PLC0415

        t = small_array["time"].values
        label = xr.DataArray(t[[2, 8]], dims="z")
        out = small_array.sel(time=label)
        assert out.dims == ("z",)
        np.testing.assert_array_equal(out.compute().values, [2, 8])

    def test_array_labels(self, small_array):
        """An array of on-grid labels resolves per label."""
        t = small_array["time"].values
        out = small_array.sel(time=t[[3, 40, 99]])
        np.testing.assert_array_equal(out.compute().values, [3, 40, 99])
        with pytest.raises(KeyError, match="sample grid"):
            small_array.sel(time=t[[3, 40]] + ONE_MS // 3)

    def test_tolerance_refused(self, small_array):
        """Tolerance is not implemented; say so rather than ignore it."""
        t = small_array["time"].values
        with pytest.raises(ValueError, match="tolerance"):
            small_array.sel(time=t[0], tolerance=ONE_MS)


class TestIsel:
    """Positional selection keeps the index lazy where it can."""

    def test_contiguous_slice_keeps_lazy_index(self, small_array):
        """A plain slice yields a new lazy index with a shifted start."""
        from dascore.xarray.index import TemporalRangeIndex  # noqa: PLC0415

        out = small_array.isel(time=slice(10, 20))
        index = out.xindexes["time"]
        assert isinstance(index, TemporalRangeIndex)
        assert index.transform.start_ns == small_array["time"].values[10].astype(
            "int64"
        )
        np.testing.assert_array_equal(
            out["time"].values, small_array["time"].values[10:20]
        )

    def test_strided_slice_scales_step(self, small_array):
        """A strided slice multiplies the lazy step."""
        from dascore.xarray.index import TemporalRangeIndex  # noqa: PLC0415

        out = small_array.isel(time=slice(0, 18, 4))
        index = out.xindexes["time"]
        assert isinstance(index, TemporalRangeIndex)
        assert index.transform.step_ns == 4 * ONE_MS.astype("int64")
        assert out.sizes["time"] == 5  # ceil(18 / 4): the span does not divide
        np.testing.assert_array_equal(
            out["time"].values, small_array["time"].values[0:18:4]
        )

    def test_fancy_indexing_materializes(self, small_array):
        """Arbitrary positions cannot stay a range; labels still right."""
        from dascore.xarray.index import TemporalRangeIndex  # noqa: PLC0415

        out = small_array.isel(time=[1, 5, 30])
        assert not isinstance(out.xindexes.get("time"), TemporalRangeIndex)
        np.testing.assert_array_equal(
            out["time"].values, small_array["time"].values[[1, 5, 30]]
        )

    def test_negative_stride_materializes(self, small_array):
        """A reversed view is not an ascending range; fall back."""
        from dascore.xarray.index import TemporalRangeIndex  # noqa: PLC0415

        out = small_array.isel(time=slice(None, None, -1))
        assert not isinstance(out.xindexes.get("time"), TemporalRangeIndex)
        np.testing.assert_array_equal(
            out["time"].values, small_array["time"].values[::-1]
        )


class TestScale:
    """The reason this index exists: metadata-cost construction."""

    def test_billion_sample_coordinate_is_free(self):
        """Three billion samples build without materializing labels."""
        import dask.array as da  # noqa: PLC0415
        import xarray as xr  # noqa: PLC0415

        from dascore.xarray.index import (  # noqa: PLC0415
            TemporalRangeIndex,
            TemporalRangeTransform,
        )

        n = 3_000_000_000
        start = np.datetime64("2020-01-01", "ns").astype("int64")
        index = TemporalRangeIndex(
            TemporalRangeTransform("time", n, int(start), 1_000_000, "datetime64[ns]")
        )
        coords = xr.Coordinates.from_xindex(index)
        arr = xr.DataArray(
            da.zeros((n,), chunks=10_000_000), dims=("time",), coords=coords
        )
        # the labels are served, not stored: the lazy transform adapter
        backing = type(arr["time"].variable._data).__name__
        assert "CoordinateTransform" in backing
        # partial strings name whole periods, as pandas reads them: the
        # stop names its full second, so the slice keeps 2000 ms samples
        sub = arr.sel(time=slice("2020-01-05", "2020-01-05T00:00:01"))
        assert sub.sizes["time"] == 2000
        assert sub["time"].values[0] == np.datetime64("2020-01-05", "ns")


class TestLazyEligibility:
    """Which dascore coordinates may be served lazily."""

    def test_descending_temporal_coord_refused(self):
        """A descending range is not an ascending transform; materialize."""
        from dascore.xarray.spool import _lazy_temporal_index  # noqa: PLC0415

        start = np.datetime64("2020-01-01", "ns")
        coord = get_coord(start=start + 10 * ONE_MS, stop=start - ONE_MS, step=-ONE_MS)
        assert _lazy_temporal_index("time", coord) is None


class TestRename:
    """Renaming keeps the lazy index working under the new name."""

    def test_rename_dim_and_coord(self, small_array):
        """Label access works after ds.rename, under the new name only."""
        from dascore.xarray.index import TemporalRangeIndex  # noqa: PLC0415

        renamed = small_array.rename({"time": "t"})
        assert isinstance(renamed.xindexes["t"], TemporalRangeIndex)
        t = small_array["time"].values
        out = renamed.sel(t=slice(t[3], t[5]))
        np.testing.assert_array_equal(out.compute().values, [3, 4, 5])


class TestConcat:
    """Segment concatenation stays lazy when the grids chain."""

    def _labeled(self, start, n):
        import dask.array as da  # noqa: PLC0415
        import xarray as xr  # noqa: PLC0415

        from dascore.xarray.index import (  # noqa: PLC0415
            TemporalRangeIndex,
            TemporalRangeTransform,
        )

        start_ns = int(np.datetime64(start, "ns").astype("int64"))
        index = TemporalRangeIndex(
            TemporalRangeTransform(
                "time", n, start_ns, int(ONE_MS.astype("int64")), "datetime64[ns]"
            )
        )
        coords = xr.Coordinates.from_xindex(index)
        return xr.DataArray(da.arange(n, chunks=n), dims=("time",), coords=coords)

    def test_contiguous_concat_stays_lazy(self):
        """Abutting segments merge into one lazy index."""
        import xarray as xr  # noqa: PLC0415

        from dascore.xarray.index import TemporalRangeIndex  # noqa: PLC0415

        first = self._labeled("2020-01-01", 50)
        second = self._labeled("2020-01-01T00:00:00.050", 30)
        out = xr.concat([first, second], dim="time")
        assert isinstance(out.xindexes["time"], TemporalRangeIndex)
        assert out.sizes["time"] == 80
        np.testing.assert_array_equal(
            out["time"].values,
            np.concatenate([first["time"].values, second["time"].values]),
        )

    def test_gapped_concat_materializes(self):
        """A gap between segments cannot stay one range; labels do not lie."""
        import xarray as xr  # noqa: PLC0415

        from dascore.xarray.index import TemporalRangeIndex  # noqa: PLC0415

        first = self._labeled("2020-01-01", 50)
        second = self._labeled("2020-01-01T00:01:00", 30)
        out = xr.concat([first, second], dim="time")
        assert not isinstance(out.xindexes["time"], TemporalRangeIndex)
        np.testing.assert_array_equal(
            out["time"].values,
            np.concatenate([first["time"].values, second["time"].values]),
        )


class TestPandasBridge:
    """The pandas spellings materialize on demand instead of raising."""

    def test_to_pandas_index(self, small_array):
        """.indexes and friends serve a real DatetimeIndex."""
        import pandas as pd  # noqa: PLC0415

        index = small_array.indexes["time"]
        assert isinstance(index, pd.Index)
        np.testing.assert_array_equal(index.values, small_array["time"].values)
        frame = small_array.to_dataframe(name="x")
        assert len(frame) == 100


class TestCoverageEdges:
    """Small contracts the main paths do not reach."""

    def test_forward_rounds_float_positions(self, small_index):
        """Xarray hands float positions; they round to the nearest sample."""
        out = small_index.transform.forward({"time": np.array([2.4, 2.6])})["time"]
        expected = small_index.transform.forward({"time": np.array([2, 3])})["time"]
        np.testing.assert_array_equal(out, expected)

    def test_rename_of_other_coord_is_noop(self, small_index):
        """Renaming a different coordinate leaves the index untouched."""
        assert small_index.rename({"distance": "d"}, {}) is small_index

    def test_concat_with_positions_reorders(self, small_index):
        """An explicit position permutation materializes in that order."""
        from dascore.xarray.index import (  # noqa: PLC0415
            TemporalRangeIndex,
            TemporalRangeTransform,
        )

        t = small_index.transform
        second = TemporalRangeIndex(
            TemporalRangeTransform(
                "time", 100, t.start_ns + 100 * t.step_ns, t.step_ns, t.dtype
            )
        )
        out = TemporalRangeIndex.concat(
            [small_index, second],
            "time",
            positions=[range(100, 200), range(0, 100)],
        )
        merged = out.to_pandas_index()
        np.testing.assert_array_equal(
            merged.values[:100], second.to_pandas_index().values
        )


class TestPandasParity:
    """The lazy index answers as a materialized DatetimeIndex answers."""

    @pytest.fixture(scope="class")
    def hourly_pair(self):
        """The same 48-hour hourly array, lazily and eagerly labeled."""
        import dask.array as da  # noqa: PLC0415
        import xarray as xr  # noqa: PLC0415

        from dascore.xarray.index import (  # noqa: PLC0415
            TemporalRangeIndex,
            TemporalRangeTransform,
        )

        hour = np.timedelta64(3_600_000_000_000, "ns")
        start = np.datetime64("2020-01-01", "ns")
        index = TemporalRangeIndex(
            TemporalRangeTransform(
                "time",
                48,
                int(start.astype("int64")),
                int(hour.astype("int64")),
                "datetime64[ns]",
            )
        )
        lazy = xr.DataArray(
            da.arange(48, chunks=48),
            dims=("time",),
            coords=xr.Coordinates.from_xindex(index),
        )
        eager = xr.DataArray(
            np.arange(48), dims=("time",), coords={"time": lazy["time"].values}
        )
        return lazy, eager

    def test_partial_string_scalar_names_period(self, hourly_pair):
        """A day-string selects the whole day, exactly as pandas does."""
        lazy, eager = hourly_pair
        out = lazy.sel(time="2020-01-01")
        expected = eager.sel(time="2020-01-01")
        assert out.sizes == expected.sizes
        np.testing.assert_array_equal(out.compute().values, expected.values)

    def test_partial_string_slice_endpoints(self, hourly_pair):
        """Slice endpoints name their whole periods, as pandas reads them."""
        lazy, eager = hourly_pair
        out = lazy.sel(time=slice("2020-01-01", "2020-01-02"))
        expected = eager.sel(time=slice("2020-01-01", "2020-01-02"))
        assert out.sizes["time"] == expected.sizes["time"] == 48

    def test_exact_resolution_string_is_scalar(self, hourly_pair):
        """A string naming exactly one sample selects it as a scalar."""
        lazy, _ = hourly_pair
        out = lazy.sel(time="2020-01-01T05:00:00")
        assert out.ndim == 0
        assert out.compute().values == 5

    def test_empty_period_raises(self, hourly_pair):
        """A period holding no samples raises rather than answers."""
        lazy, _ = hourly_pair
        with pytest.raises(KeyError, match="period"):
            lazy.sel(time="2019-06")

    def test_out_of_ns_range_label_raises(self, hourly_pair):
        """A label the ns range cannot represent raises, never wraps."""
        lazy, _ = hourly_pair
        with pytest.raises(Exception, match=r"bounds|Out of"):
            lazy.sel(time=slice("2500-01-01", None))

    def test_method_with_slice_raises(self, hourly_pair):
        """Pandas rejects method with a slice; so does the lazy index."""
        lazy, _ = hourly_pair
        with pytest.raises(ValueError, match="slice"):
            lazy.sel(time=slice("2020-01-01", None), method="nearest")

    def test_dataarray_label_validates(self, hourly_pair):
        """Vectorized labels get the same exact-grid and bounds checks."""
        import xarray as xr  # noqa: PLC0415

        lazy, _ = hourly_pair
        t = lazy["time"].values
        off = xr.DataArray(t[[2, 8]] + np.timedelta64(1, "m"), dims="z")
        with pytest.raises(KeyError, match="sample grid"):
            lazy.sel(time=off)
        near = lazy.sel(time=off, method="nearest")
        np.testing.assert_array_equal(near.compute().values, [2, 8])
        before = xr.DataArray(t[[0]] - np.timedelta64(2, "h"), dims="z")
        with pytest.raises(KeyError, match="outside"):
            lazy.sel(time=before, method="nearest")

    def test_fancy_isel_keeps_label_selection(self, hourly_pair):
        """Fancy isel materializes an index instead of dropping it."""
        lazy, _ = hourly_pair
        sub = lazy.isel(time=[1, 5, 9])
        assert "time" in sub.xindexes
        t = lazy["time"].values
        out = sub.sel(time=t[5])
        assert out.compute().values == 5

    def test_reversed_isel_keeps_label_selection(self, hourly_pair):
        """A reversed view keeps working label lookup too."""
        lazy, _ = hourly_pair
        sub = lazy.isel(time=slice(None, None, -1))
        t = lazy["time"].values
        assert sub.sel(time=t[5]).compute().values == 5

    def test_one_sample_period_keeps_dimension(self):
        """A coarse string over one sample keeps the dimension, as pandas does."""
        import xarray as xr  # noqa: PLC0415

        from dascore.core.coords import get_coord  # noqa: PLC0415
        from dascore.xarray.index import TemporalRangeIndex  # noqa: PLC0415

        # an hourly segment whose first calendar day holds one sample
        coord = get_coord(
            start=np.datetime64("2019-12-31T23:00:00"),
            step=np.timedelta64(1, "h"),
            shape=(30,),
        )
        index = TemporalRangeIndex.from_coord("time", coord)
        lazy = xr.DataArray(
            np.arange(30), dims=("time",), coords=xr.Coordinates.from_xindex(index)
        )
        eager = xr.DataArray(
            np.arange(30), dims=("time",), coords={"time": coord.values}
        )
        out, expected = lazy.sel(time="2019-12-31"), eager.sel(time="2019-12-31")
        assert out.sizes == expected.sizes == {"time": 1}
        # a string at the index's own resolution is a single stamp
        assert (
            lazy.sel(time="2019-12-31T23").ndim
            == eager.sel(time="2019-12-31T23").ndim
            == 0
        )
        with pytest.raises(KeyError):
            lazy.sel(time="2020-01-01T05:30")

    def test_numeric_labels_raise(self, hourly_pair):
        """Pandas never reads a number as a stamp; neither does the lazy index."""
        lazy, eager = hourly_pair
        for label in (0, 5.0, [0, 1]):
            with pytest.raises(KeyError):
                eager.sel(time=label)
            with pytest.raises(KeyError, match="Numeric"):
                lazy.sel(time=label)

    def test_negative_fancy_isel_labels_from_the_end(self, hourly_pair):
        """A negative position labels the sample it selects, as the data does."""
        lazy, eager = hourly_pair
        out, expected = lazy.isel(time=[-1, 0]), eager.isel(time=[-1, 0])
        np.testing.assert_array_equal(out["time"].values, expected["time"].values)
        np.testing.assert_array_equal(out.compute().values, expected.values)

    def test_boolean_isel_keeps_labels(self, hourly_pair):
        """A boolean mask selects and labels the same samples."""
        lazy, eager = hourly_pair
        mask = np.arange(48) % 7 == 0
        out, expected = lazy.isel(time=mask), eager.isel(time=mask)
        np.testing.assert_array_equal(out["time"].values, expected["time"].values)
        assert "time" in out.xindexes

    def test_vectorized_isel_onto_another_dimension(self, hourly_pair):
        """An indexer on a new dimension moves the labels there and drops the index."""
        import xarray as xr  # noqa: PLC0415

        lazy, eager = hourly_pair
        picks = xr.DataArray([1, 5], dims="sample")
        out, expected = lazy.isel(time=picks), eager.isel(time=picks)
        assert out["time"].dims == expected["time"].dims == ("sample",)
        assert "time" not in out.xindexes
        np.testing.assert_array_equal(out["time"].values, expected["time"].values)
        assert (
            out.isel(sample=1)["time"].values == expected.isel(sample=1)["time"].values
        )


class TestInternalEdges:
    """Direct contracts the integration paths do not reach."""

    def test_reverse_returns_float_positions(self, small_index):
        """The transform contract's reverse serves float positions."""
        t = small_index.transform
        label = t.forward({"time": np.array([7])})["time"]
        out = t.reverse({"time": label})["time"]
        assert out.dtype == np.float64
        assert out[0] == 7.0

    def test_period_bounds_edges(self, small_index):
        """Non-datetime, unparseable, and NaT strings name no period."""
        from dascore.xarray.index import (  # noqa: PLC0415
            TemporalRangeIndex,
            TemporalRangeTransform,
        )

        td_index = TemporalRangeIndex(
            TemporalRangeTransform("offset", 10, 0, 1_000_000, "timedelta64[ns]")
        )
        assert td_index._period_bounds("0 days") is None
        assert small_index._period_bounds("not a date at all") is None
        assert small_index._period_bounds("nat") is None

    def test_resolution_follows_start_and_step(self):
        """The inferred resolution is the finest unit start or step carries."""
        from dascore.xarray.index import (  # noqa: PLC0415
            TemporalRangeIndex,
            TemporalRangeTransform,
        )

        day = 86_400 * 10**9
        midnight = int(np.datetime64("2020-01-01", "ns").astype("int64"))
        daily = TemporalRangeIndex(
            TemporalRangeTransform("time", 5, midnight, day, "datetime64[ns]")
        )
        assert daily._resolution == 6  # day
        hourly = TemporalRangeIndex(
            TemporalRangeTransform("time", 5, midnight, 3_600 * 10**9, "datetime64[ns]")
        )
        assert hourly._resolution == 5  # hour, from the step
        # one sample at midnight has day resolution whatever the step says
        lone = TemporalRangeIndex(
            TemporalRangeTransform("time", 1, midnight, 1_000, "datetime64[ns]")
        )
        assert lone._resolution == 6
        # a start off the hour drives the resolution below the step's
        odd = TemporalRangeIndex(
            TemporalRangeTransform("time", 5, midnight + 1, day, "datetime64[ns]")
        )
        assert odd._resolution == 0  # nanosecond

    def test_variable_label(self, small_array):
        """A bare Variable label resolves like a DataArray one."""
        from xarray import Variable  # noqa: PLC0415

        t = small_array["time"].values
        out = small_array.sel(time=Variable("z", t[[4, 9]]))
        np.testing.assert_array_equal(out.compute().values, [4, 9])
