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
