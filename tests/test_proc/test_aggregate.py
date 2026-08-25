"""Tests for performing aggregations."""

from __future__ import annotations

import warnings

import numpy as np
import numpy.testing as npt
import pytest

import dascore
import dascore as dc
from dascore.core.coords import _is_translation_equivariant, _reduce_time_like
from dascore.exceptions import ParameterError
from dascore.proc.aggregate import _AGG_FUNCS
from dascore.utils.misc import broadcast_for_index


class TestReduceTimeLike:
    """Tests for time-like coordinate reductions."""

    def test_translation_equivariant_with_no_valid_data(self):
        """Empty or all-null data cannot disprove translation equivariance."""
        data = np.array([np.nan, np.nan])
        assert _is_translation_equivariant(np.nanmean, data)

    def test_translation_equivariant_when_reducer_raises(self):
        """Reducers which cannot be checked are treated as equivariant."""

        def reducer(_array):
            raise ValueError

        assert _is_translation_equivariant(reducer, np.array([1.0, 2.0]))

    def test_translation_equivariant_when_comparison_raises(self, monkeypatch):
        """Comparison failures default to treating reducers as equivariant."""

        def raise_type_error(*_args, **_kwargs):
            raise TypeError

        monkeypatch.setattr(np, "allclose", raise_type_error)
        assert _is_translation_equivariant(np.nanmean, np.array([1.0, 2.0]))

    def test_all_nat_datetime_reduction_returns_typed_nat(self):
        """All-null datetime reductions use the datetime NaT fallback."""
        data = np.array([np.datetime64("NaT", "ns")] * 3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = _reduce_time_like(np.nanmean, data)
        assert out.dtype == np.dtype("datetime64[ns]")
        assert np.isnat(out[0])

    def test_all_nat_timedelta_reduction_returns_typed_nat(self):
        """All-null timedelta reductions use the timedelta NaT fallback."""

        def reducer(array):
            if np.issubdtype(np.asarray(array).dtype, np.timedelta64):
                raise TypeError
            return np.nanmean(array)

        data = np.array([np.timedelta64("NaT", "ns")] * 3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = _reduce_time_like(reducer, data)
        assert out.dtype == np.dtype("timedelta64[ns]")
        assert np.isnat(out[0])

    def test_timedelta_sum_fallback_does_not_add_reference(self):
        """Non-equivariant timedelta reducers should not add the reference value."""

        def reducer(array):
            if np.issubdtype(np.asarray(array).dtype, np.timedelta64):
                raise TypeError
            return np.nansum(array)

        data = np.array([np.timedelta64(10, "ns"), np.timedelta64(11, "ns")])
        out = _reduce_time_like(reducer, data)
        assert out == np.array([np.timedelta64(1, "ns")])


class TestBasicAggregations:
    """Sanity checks for basic aggregations."""

    def test_first(self, random_patch):
        """Ensure aggregations can occur."""
        out = random_patch.aggregate(dim="distance", method="first")
        assert out.ndim == random_patch.ndim
        axis = random_patch.get_axis("distance")
        inds = broadcast_for_index(len(random_patch.data.shape), axis, 0)
        assert np.allclose(random_patch.data[inds].flatten(), out.data.flatten())

    def test_last(self, random_patch):
        """Ensure aggregations can occur."""
        out = random_patch.aggregate(dim="time", method="last")
        axis = random_patch.get_axis("time")
        inds = broadcast_for_index(len(random_patch.data.shape), axis, -1)
        assert np.allclose(random_patch.data[inds].flatten(), out.data.flatten())

    def test_no_dim(self, random_patch):
        """Ensure no dimension argument behaves like numpy."""
        out = random_patch.aggregate(method="mean")
        expected = np.mean(random_patch.data, keepdims=True)
        npt.assert_allclose(out.data, expected, rtol=1e-8, atol=0)

    def test_dtype_of_coord_unchanged(self, random_patch):
        """The dtype of the coord should not change."""
        out = random_patch.aggregate(dim="time", method="median")
        coord_new = out.get_coord("time")
        coord_old = random_patch.get_coord("time")
        assert coord_old.dtype == coord_new.dtype

    @pytest.mark.parametrize("method", list(_AGG_FUNCS))
    def test_named_aggregations(self, random_patch, method):
        """Simply run the named aggregations."""
        patch = getattr(random_patch, method)(dim="distance")
        assert isinstance(patch, dc.Patch)

    def test_dim_reduce_squeeze(self, random_patch):
        """Ensure the old dimension can be squeezed out."""
        out = random_patch.aggregate(dim="time", method="mean", dim_reduce="squeeze")
        assert "time" not in out.dims
        assert out.ndim == 1

    def test_multi_dim_reduce_squeeze_3d(self):
        """Squeezing multiple dims should not use stale axis numbers."""
        data = np.arange(27, dtype=float).reshape(3, 3, 3)
        coords = {
            "distance": np.arange(3),
            "time": np.arange(3),
            "face": np.arange(3),
        }
        patch = dc.Patch(data=data, coords=coords, dims=("distance", "time", "face"))
        out = patch.aggregate(
            dim=("distance", "time"), method="mean", dim_reduce="squeeze"
        )
        expected = np.mean(
            patch.data,
            axis=(patch.get_axis("distance"), patch.get_axis("time")),
        )
        assert out.dims == ("face",)
        npt.assert_allclose(out.data, expected)

    def test_dim_reduce_squeeze_all_dims_raises(self, random_patch):
        """Squeeze should not create an unsupported scalar patch."""
        msg = "at least one dimension"
        with pytest.raises(ParameterError, match=msg):
            random_patch.aggregate(
                dim=("distance", "time"), method="mean", dim_reduce="squeeze"
            )

    def test_dim_reduce_mean(self, random_patch):
        """Ensure the mean value can be left on the coord."""
        out = random_patch.aggregate(dim="time", method="mean", dim_reduce="mean")
        new_time = out.get_coord("time")
        assert len(new_time) == 1

    def test_dim_reduce_mean_time_delta(self, random_patch):
        """Ensure the mean value can be left on the coord."""
        time = random_patch.get_coord("time")
        dt = dc.to_timedelta64(time.values)
        patch = random_patch.update_coords(time=dt)
        out = patch.aggregate(dim="time", method="mean", dim_reduce="mean")
        new_time = out.get_coord("time")
        assert len(new_time) == 1

    def test_dim_reduce_mean_preserves_nanosecond_offsets(self):
        """Time-like coord reductions preserve ns offsets far from epoch."""
        start = np.datetime64("2020-01-01T00:00:00.123456789", "ns")
        offsets = np.arange(5).astype("timedelta64[ns]")
        time = start + offsets
        patch = dc.Patch(data=np.ones(5), coords={"time": time}, dims=("time",))

        out = patch.aggregate(dim="time", method="mean", dim_reduce="mean")

        new_time = out.get_coord("time")
        assert len(new_time) == 1
        assert new_time.values[0] == start + np.timedelta64(2, "ns")

    @pytest.mark.parametrize(
        ("method", "expected"),
        [("mean", np.timedelta64(2, "ns")), ("sum", np.timedelta64(8, "ns"))],
    )
    def test_dim_reduce_timedelta_nat_skips_nulls(self, method, expected):
        """Timedelta coord reductions skip NaT like numeric nan reducers."""
        time = np.arange(5).astype("timedelta64[ns]")
        time[2] = np.timedelta64("NaT", "ns")
        patch = dc.Patch(data=np.ones(5), coords={"time": time}, dims=("time",))

        out = patch.aggregate(dim="time", method="mean", dim_reduce=method)

        new_time = out.get_coord("time")
        assert len(new_time) == 1
        assert new_time.values[0] == expected

    def test_invalid_dim_reduce(self, random_patch):
        """Ensure an invalid dim_reduce argument raises."""
        msg = "dim_reduce must be"
        with pytest.raises(ParameterError, match=msg):
            random_patch.aggregate(dim="time", dim_reduce="invalid")

    def test_dim_reduce_first(self, random_patch):
        """Ensure first takes the first value"""
        out = random_patch.aggregate(dim="time", method="mean", dim_reduce="first")
        new_time = out.get_coord("time")
        assert len(new_time) == 1
        assert new_time[0] == out.get_array("time")[0]

    def test_dim_reduce_distance(self, random_patch):
        """Ensure non-time dims also work."""
        out = random_patch.aggregate(dim="distance", method="mean", dim_reduce=np.var)
        assert "distance" in out.dims
        expected = np.var(random_patch.get_array("distance"))
        assert out.get_coord("distance").values == expected

    def test_any(self, random_patch):
        """Ensure any works."""
        out = (random_patch > 0.5).any(dim="time")
        assert isinstance(out, dascore.Patch)
        assert np.issubdtype(out.dtype, np.bool_)

    def test_mean_monotonic_time_with_associated_coord(self, random_patch):
        """Regression for #635: monotonic time reduction should drop sibling coords."""
        ntime = len(random_patch.get_coord("time"))
        base = dc.to_datetime64("2024-01-01")
        offsets = np.cumsum(np.arange(ntime) + 1).astype("timedelta64[ns]")
        patch = random_patch.update_coords(
            time=base + offsets,
            auxiliary=("time", np.arange(ntime)),
        )

        out = patch.mean("time")

        assert out.shape == (len(patch.get_coord("distance")), 1)
        assert out.coords.shape == out.shape
        assert "auxiliary" not in out.coords

    def test_mean_single_sample_monotonic_time_keeps_value(self, random_patch):
        """Regression for #635: 1-sample monotonic coords should keep their value."""
        time = np.array([dc.to_datetime64("2024-01-01T00:00:00")])
        patch = random_patch.select(time=(0, 1), samples=True).update_coords(time=time)

        out = patch.mean("time")

        assert out.shape == patch.shape
        assert out.coords.shape == patch.coords.shape
        assert out.get_coord("time").values[0] == time[0]


class TestApplyOperators:
    """Ensure aggregated patches can be used as operators for arithmetic."""

    def test_complete_reduction(self, random_patch):
        """Ensure a patch with complete reduction works."""
        agg = random_patch.min(None)
        assert np.all(agg.data == np.min(random_patch.data))
        # Ensure broadcasting works with reduced data
        out = random_patch - agg
        assert isinstance(out, dc.Patch)
        assert np.allclose(random_patch.data - agg.data, out.data)

    def test_single_reduction(self, random_patch):
        """Ensure a single patch reduced also works with broadcasting."""
        agg = random_patch.first("time")
        out1 = random_patch + agg
        assert isinstance(out1, dc.Patch)


class TestIdxMaxMin:
    """Tests for idxmax and idxmin."""

    @pytest.fixture(scope="class")
    def nan_patch(self, random_patch):
        """A patch whose first channel is NaN for every time."""
        data = np.asarray(random_patch.data).astype(float).copy()
        axis = random_patch.get_axis("distance")
        index = broadcast_for_index(data.ndim, axis, 0)
        data[index] = np.nan
        return random_patch.new(data=data)

    @pytest.mark.parametrize("dim", ["time", "distance"])
    def test_matches_numpy(self, random_patch, dim):
        """The returned values are the coord values numpy's argmax picks."""
        axis = random_patch.get_axis(dim)
        values = random_patch.get_coord(dim).values
        for name, arg in (("idxmax", np.argmax), ("idxmin", np.argmin)):
            out = getattr(random_patch, name)(dim, dim_reduce="squeeze")
            expected = values[arg(random_patch.data, axis=axis)]
            assert np.array_equal(np.asarray(out.data), expected)

    def test_squeeze_drops_dimension(self, random_patch):
        """dim_reduce='squeeze' removes the reduced dimension."""
        out = random_patch.idxmax("time", dim_reduce="squeeze")
        assert "time" not in out.dims
        assert out.dims == ("distance",)

    def test_empty_keeps_dimension(self, random_patch):
        """The default keeps the reduced dimension with length one."""
        out = random_patch.idxmax("time")
        assert out.dims == random_patch.dims
        assert out.shape[random_patch.get_axis("time")] == 1

    def test_data_takes_coord_dtype_and_units(self, random_patch):
        """Output data are coordinate values, so they carry its dtype/units."""
        coord = random_patch.get_coord("time")
        out = random_patch.idxmax("time")
        assert out.data.dtype == coord.dtype
        assert out.attrs.data_units == coord.units

    def test_partial_nan_ignored(self, random_patch):
        """NaN samples are skipped rather than winning the comparison."""
        data = np.asarray(random_patch.data).astype(float).copy()
        data[0, :3] = np.nan
        patch = random_patch.new(data=data)
        out = patch.idxmax("time", dim_reduce="squeeze")
        values = random_patch.get_coord("time").values
        assert np.asarray(out.data)[0] == values[np.nanargmax(data[0])]

    def test_all_nan_slice_is_null(self, nan_patch):
        """A slice with no valid sample yields NaT for a time coordinate."""
        out = nan_patch.idxmax("time", dim_reduce="squeeze")
        values = np.asarray(out.data)
        assert np.isnat(values[0])
        assert not np.isnat(values[1:]).any()

    def test_all_nan_slice_upcasts_int_coord(self, random_patch):
        """An integer coordinate widens to float so it can hold the null."""
        coord = random_patch.get_coord("distance")
        assert np.issubdtype(coord.dtype, np.integer)
        # Blank one whole time sample so reducing distance has nothing to pick.
        data = np.asarray(random_patch.data).astype(float).copy()
        axis = random_patch.get_axis("time")
        data[broadcast_for_index(data.ndim, axis, 0)] = np.nan
        out = random_patch.new(data=data).idxmin("distance", dim_reduce="squeeze")
        values = np.asarray(out.data)
        assert np.issubdtype(values.dtype, np.floating)
        assert np.isnan(values[0])
        assert not np.isnan(values[1:]).any()

    def test_ties_take_first(self):
        """Equal values resolve to the first along the dimension, as numpy does."""
        data = np.ones((2, 4))
        patch = dc.Patch(
            data=data,
            coords={"distance": np.arange(2), "time": np.arange(4)},
            dims=("distance", "time"),
        )
        out = patch.idxmax("time", dim_reduce="squeeze")
        assert np.array_equal(np.asarray(out.data), np.zeros(2))

    @pytest.mark.parametrize("dim", [None, ["time"], ("time", "distance")])
    def test_non_string_dim_raises(self, random_patch, dim):
        """These reduce exactly one dimension, named by a string."""
        with pytest.raises(ParameterError, match="single dimension"):
            random_patch.idxmax(dim)

    def test_bad_dim_raises(self, random_patch):
        """A dimension the patch does not have is an error."""
        with pytest.raises(Exception):
            random_patch.idxmax("not_a_dim")

    def test_recovers_a_known_slope(self):
        """A linear moveout is recovered exactly by idxmax."""
        n_dist, n_time, shift = 8, 50, 2
        data = np.zeros((n_dist, n_time))
        for i in range(n_dist):
            data[i, 5 + i * shift] = 1.0
        patch = dc.Patch(
            data=data,
            coords={"distance": np.arange(n_dist), "time": np.arange(n_time)},
            dims=("distance", "time"),
        )
        out = patch.idxmax("time", dim_reduce="squeeze")
        expected = 5 + np.arange(n_dist) * shift
        assert np.array_equal(np.asarray(out.data), expected)
