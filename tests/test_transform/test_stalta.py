"""Tests for STA/LTA transform."""

from __future__ import annotations

import numpy as np
import pytest

from dascore.exceptions import ParameterError, UnitError
from dascore.transform.stalta import Stalta


class TestStaLta:
    """Tests for short-term average / long-term average ratio."""

    def test_runs_time_dimension(self, random_patch):
        """Ensure stalta runs along the time dimension."""
        out = random_patch.stalta(time=(0.01, 0.05))

        assert out.dims == random_patch.dims
        assert out.data.shape == random_patch.data.shape
        assert out.attrs.data_type == "stalta"
        assert out.attrs.data_units is None

    def test_runs_distance_dimension(self, random_patch):
        """Ensure stalta runs along the distance dimension."""
        out = random_patch.stalta(distance=(5, 25))

        assert out.dims == random_patch.dims
        assert out.data.shape == random_patch.data.shape
        assert out.attrs.data_type == "stalta"
        assert out.attrs.data_units is None

    def test_matches_expected_time_ratio(self, random_patch):
        """Ensure stalta returns rolling STA divided by rolling LTA."""
        out = random_patch.stalta(time=(0.01, 0.05))

        sta = random_patch.rolling(time=0.01).mean()
        lta = random_patch.rolling(time=0.05).mean()
        expected = sta / lta

        assert np.allclose(out.data, expected.data, equal_nan=True)

    def test_matches_expected_distance_ratio(self, random_patch):
        """Ensure stalta uses the requested dimension for rolling windows."""
        out = random_patch.stalta(distance=(5, 25))

        sta = random_patch.rolling(distance=5).mean()
        lta = random_patch.rolling(distance=25).mean()
        expected = sta / lta

        assert np.allclose(out.data, expected.data, equal_nan=True)

    def test_samples_argument(self, random_patch):
        """Ensure samples argument applies window lengths as samples."""
        out = random_patch.stalta(time=(2, 5), samples=True)

        sta = random_patch.rolling(time=2, samples=True).mean()
        lta = random_patch.rolling(time=5, samples=True).mean()
        expected = sta / lta

        assert np.allclose(out.data, expected.data, equal_nan=True)

    def test_attrs_are_set(self, random_patch):
        """Ensure output metadata are set."""
        out = random_patch.stalta(time=(0.01, 0.05))

        assert out.attrs.data_type == "stalta"
        assert out.attrs.data_units is None

    def test_missing_dimension_kwargs_raise(self, random_patch):
        """Ensure a dimension/window kwarg is required."""
        with pytest.raises(ValueError):
            random_patch.stalta()

    def test_multiple_dimension_kwargs_raise(self, random_patch):
        """Ensure only one dimension/window kwarg is accepted."""
        with pytest.raises(ValueError):
            random_patch.stalta(time=(0.01, 0.05), distance=(5, 25))

    def test_bad_window_tuple_raises(self, random_patch):
        """Ensure invalid window kwargs are rejected."""
        with pytest.raises(ValueError):
            random_patch.stalta(time=(0.01, 0.05, 0.1))

    def test_lta_must_exceed_sta(self, random_patch):
        """Ensure the long-term window must exceed the short-term window."""
        with pytest.raises(ParameterError, match="long-term window"):
            random_patch.stalta(time=(0.05, 0.01))


class TestStaltaProcessor:
    """What the class says beyond the method."""

    def test_samples_is_keyword_only(self):
        """The windows are keywords, so samples cannot be given by position."""
        with pytest.raises(TypeError, match="positional"):
            Stalta(True, time=(5, 20))

    def test_single_precision(self, random_patch):
        """The ratio of rolling means is double, as the metadata says first."""
        patch = random_patch.new(data=np.asarray(random_patch.data, np.float32))
        stalta = Stalta(time=(5, 20), samples=True)
        out, _ = stalta.get_metadata(patch.drop_data())
        assert out.dtype == stalta(patch).dtype == np.float64


class TestStaltaUnits:
    """The ratio's units are the quotient of two equal units."""

    def test_scaled_units_cancel(self, random_patch):
        """A scale in the units divides out, leaving a dimensionless ratio."""
        kwargs = dict(time=(5, 20), samples=True)
        out = random_patch.set_units("10 m/s").stalta(**kwargs)
        assert str(out.attrs.data_units) == "1"
        assert np.allclose(out.data, random_patch.stalta(**kwargs).data, equal_nan=True)

    def test_offset_units_refused(self, random_patch):
        """A temperature cannot be divided by one."""
        with pytest.raises(UnitError, match="offset units"):
            random_patch.set_units("degC").stalta(time=(5, 20), samples=True)

    def test_scaled_units_as_a_quotient(self, random_patch):
        """The scale goes into both means before dividing, as patch division does."""
        scaled = random_patch.set_units("10 m/s")
        sta, lta = (scaled.rolling(time=x, samples=True).mean() for x in (5, 20))
        out = scaled.stalta(time=(5, 20), samples=True)
        assert np.array_equal(out.data, (sta / lta).data, equal_nan=True)

    def test_single_row_rolls_with_pandas(self, random_patch):
        """A patch with one row rolls as `rolling` rolls it, through pandas."""
        row = random_patch.isel(distance=slice(0, 1))
        sta, lta = (row.rolling(time=x, samples=True).mean() for x in (5, 20))
        out = row.stalta(time=(5, 20), samples=True)
        assert np.array_equal(out.data, (sta / lta).data, equal_nan=True)
