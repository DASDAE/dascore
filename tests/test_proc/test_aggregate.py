"""Tests for performing aggregations."""

from __future__ import annotations

import warnings

import numpy as np
import numpy.testing as npt
import pytest

import dascore
import dascore as dc
from dascore.core.coords import _is_translation_equivariant, _reduce_time_like
from dascore.exceptions import CoordError, ParameterError
from dascore.proc.aggregate import _AGG_FUNCS
from dascore.utils.misc import broadcast_for_index
from dascore.warnings import NumpyFallbackWarning


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


def _row(data, time_values):
    """Build a one channel patch from a single row of data."""
    return dc.Patch(
        data=np.asarray(data).reshape(1, -1),
        coords={"distance": np.arange(1), "time": np.asarray(time_values)},
        dims=("distance", "time"),
    )


class TestIdxMaxMin:
    """Tests for idxmax and idxmin."""

    @pytest.fixture(scope="class")
    def dead_channel_patch(self, random_patch):
        """A patch whose first channel is NaN at every time."""
        data = np.asarray(random_patch.data).astype(float).copy()
        index = broadcast_for_index(data.ndim, random_patch.get_axis("distance"), 0)
        data[index] = np.nan
        return random_patch.new(data=data)

    @pytest.mark.parametrize("dim", ["time", "distance"])
    @pytest.mark.parametrize("name,arg", [("idxmax", np.argmax), ("idxmin", np.argmin)])
    def test_matches_numpy(self, random_patch, dim, name, arg):
        """The values are the coord values numpy's arg reduction picks."""
        values = random_patch.get_coord(dim).values
        out = getattr(random_patch, name)(dim, dim_reduce="squeeze")
        expected = values[arg(random_patch.data, axis=random_patch.get_axis(dim))]
        assert np.array_equal(np.asarray(out.data), expected)

    def test_returns_coord_values_not_indices(self):
        """The output holds coordinate values, which are not the indices."""
        # Every coord value is far past the last index, so the two differ.
        out = _row([3.0, 9.0, 1.0], [50, 60, 70]).idxmax("time")
        assert np.asarray(out.data).ravel()[0] == 60

    def test_dim_reduce_shapes(self, random_patch):
        """The default keeps the reduced dimension; squeeze drops it."""
        kept = random_patch.idxmax("time")
        assert kept.dims == random_patch.dims
        assert kept.shape[random_patch.get_axis("time")] == 1
        assert random_patch.idxmax("time", dim_reduce="squeeze").dims == ("distance",)

    def test_output_dtype_and_units(self, random_patch):
        """A time coord's unit describes its step, so it must not be copied."""
        time = random_patch.idxmax("time")
        assert time.data.dtype == random_patch.get_coord("time").dtype
        # Labelling a nanosecond magnitude "s" would scale unit maths by 1e9.
        assert time.attrs.data_units is None
        # The values are no longer whatever the patch measured.
        assert not time.attrs.data_type
        distance = random_patch.idxmax("distance")
        assert distance.attrs.data_units == random_patch.get_coord("distance").units

    @pytest.mark.parametrize(
        "name,agg", [("idxmax", np.nanargmax), ("idxmin", np.nanargmin)]
    )
    def test_partial_nan_ignored(self, random_patch, name, agg):
        """NaN samples are skipped rather than winning the comparison."""
        data = np.asarray(random_patch.data).astype(float).copy()
        data[0, :3] = np.nan
        out = getattr(random_patch.new(data=data), name)("time", dim_reduce="squeeze")
        values = random_patch.get_coord("time").values
        assert np.asarray(out.data)[0] == values[agg(data[0])]

    @pytest.mark.parametrize(
        "name,data,expected",
        [
            # A real -inf must beat a NaN for a max, and +inf for a min,
            # even though each is what the missing sample is filled with.
            ("idxmax", [np.nan, -np.inf], 20),
            ("idxmin", [np.nan, np.inf], 20),
            ("idxmax", [-np.inf, np.nan], 10),
            ("idxmin", [np.inf, np.nan], 10),
        ],
    )
    def test_nan_does_not_beat_infinity(self, name, data, expected):
        """A missing sample never outranks a genuine infinity."""
        out = getattr(_row(data, [10, 20]), name)("time")
        assert np.asarray(out.data).ravel()[0] == expected

    @pytest.mark.parametrize("name", ["idxmax", "idxmin"])
    def test_nat_in_time_like_data_is_skipped(self, name):
        """NaT is missing data too, not the largest or smallest value."""
        start = dc.to_datetime64("2020-01-01")
        data = np.array([[start + dc.to_timedelta64(5), np.datetime64("NaT", "ns")]])
        out = getattr(_row(data, [10, 20]), name)("time")
        assert np.asarray(out.data).ravel()[0] == 10

    def test_chained_calls_skip_the_null(self, dead_channel_patch):
        """The NaT one call leaves behind is skipped by the next."""
        peaks = dead_channel_patch.idxmax("time")
        assert np.isnat(np.asarray(peaks.data)).any()
        out = peaks.idxmin("distance", dim_reduce="squeeze")
        dead = dead_channel_patch.get_coord("distance").values[0]
        assert np.asarray(out.data).ravel()[0] != dead

    def test_all_nan_slice_is_null(self, dead_channel_patch):
        """A slice with no valid sample yields NaT for a time coordinate."""
        values = np.asarray(
            dead_channel_patch.idxmax("time", dim_reduce="squeeze").data
        )
        assert np.isnat(values[0])
        assert not np.isnat(values[1:]).any()

    def test_all_nan_slice_upcasts_int_coord(self, random_patch):
        """An integer coordinate widens to float so it can hold the null."""
        assert np.issubdtype(random_patch.get_coord("distance").dtype, np.integer)
        # Blank one whole time sample so reducing distance has nothing to pick.
        data = np.asarray(random_patch.data).astype(float).copy()
        data[broadcast_for_index(data.ndim, random_patch.get_axis("time"), 0)] = np.nan
        out = random_patch.new(data=data).idxmin("distance", dim_reduce="squeeze")
        values = np.asarray(out.data)
        assert np.issubdtype(values.dtype, np.floating)
        assert np.isnan(values[0])
        assert not np.isnan(values[1:]).any()

    @pytest.mark.parametrize("coord", [np.array(["a", "b"]), np.array([True, False])])
    def test_null_needs_a_coord_which_can_hold_it(self, coord):
        """A coordinate with no null value says so rather than crashing."""
        with pytest.raises(ParameterError, match="no valid sample"):
            _row([np.nan, np.nan], coord).idxmax("time")

    def test_ties_take_first(self):
        """Equal extremes resolve to the first along the dimension."""
        # The max appears at index 1 and 3; the min at index 0 and 2.
        patch = _row([0.0, 1.0, 0.0, 1.0], [10, 20, 30, 40])
        assert np.asarray(patch.idxmax("time").data).ravel()[0] == 20
        assert np.asarray(patch.idxmin("time").data).ravel()[0] == 10

    def test_integer_data(self):
        """Integer data have no missing value, so no masking is needed."""
        patch = _row([3, 9, 1, 9], [10, 20, 30, 40])
        assert np.asarray(patch.idxmax("time").data).ravel()[0] == 20
        assert np.asarray(patch.idxmin("time").data).ravel()[0] == 30

    @pytest.mark.parametrize("dim", [None, ["time"], ("time", "distance")])
    def test_non_string_dim_raises(self, random_patch, dim):
        """These reduce exactly one dimension, named by a string."""
        with pytest.raises(ParameterError, match="single dimension"):
            random_patch.idxmax(dim)

    def test_bad_dim_raises(self, random_patch):
        """A dimension the patch does not have is an error."""
        with pytest.raises(CoordError, match="not found"):
            random_patch.idxmax("not_a_dim")

    def test_reduced_dimension_raises(self, random_patch):
        """The partial coord left behind has no values to point at."""
        with pytest.raises(ParameterError, match="holds no values"):
            random_patch.idxmax("time").idxmax("time")

    def test_non_numpy_backend_warns(self, random_patch):
        """A lazy or device array is pulled to numpy, and says so."""
        dask_array = pytest.importorskip("dask.array")
        data = np.asarray(random_patch.data)
        patch = random_patch.new(data=dask_array.from_array(data, chunks=(100, 500)))
        with pytest.warns(NumpyFallbackWarning, match="idxmax"):
            out = patch.idxmax("time", dim_reduce="squeeze")
        values = random_patch.get_coord("time").values
        assert np.array_equal(np.asarray(out.data), values[data.argmax(axis=1)])
        # The fallback converts back, so the caller keeps its backend.
        assert not isinstance(out.data, np.ndarray)
