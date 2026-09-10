"""Tests for coordinate gap utilities."""

from __future__ import annotations

import numpy as np
import pytest

from dascore.utils.gaps import get_gap_edges, is_monotonic_and_finite


class TestGetGapEdges:
    """Tests for constructing coordinate cell edges."""

    def test_numeric_without_gaps(self):
        """Numeric coordinates produce midpoint cell edges."""
        edges, gaps = get_gap_edges([0, 1, 2])
        np.testing.assert_allclose(edges, [-0.5, 0.5, 1.5, 2.5])
        assert not np.any(gaps)

    def test_numeric_with_gap(self):
        """Large numeric intervals are expanded into a gap."""
        edges, gaps = get_gap_edges([0, 1, 5, 6], gap_factor=1.5)
        np.testing.assert_allclose(edges, [-0.5, 0.5, 1.5, 4.5, 5.5, 6.5])
        np.testing.assert_array_equal(gaps, [False, True, False])

    def test_timedelta(self):
        """Timedeltas are converted to seconds before constructing edges."""
        values = np.array([0, 1, 2], dtype="timedelta64[s]")
        edges, gaps = get_gap_edges(values)
        np.testing.assert_allclose(edges, [-0.5, 0.5, 1.5, 2.5])
        assert not np.any(gaps)

    def test_datetime(self):
        """Datetime edges retain datetime dtype and support gaps."""
        values = np.array(
            ["2020-01-01", "2020-01-02", "2020-01-06"],
            dtype="datetime64[D]",
        )
        edges, gaps = get_gap_edges(values, gap_factor=1.5)
        expected = np.array(
            [
                "2019-12-31T12:00:00",
                "2020-01-01T12:00:00",
                "2020-01-03T06:00:00",
                "2020-01-04T18:00:00",
                "2020-01-07T06:00:00",
            ],
            dtype="datetime64[ns]",
        )
        assert np.issubdtype(edges.dtype, np.datetime64)
        np.testing.assert_array_equal(edges, expected)
        np.testing.assert_array_equal(gaps, [False, True])

    def test_singleton_datetime(self):
        """Singleton datetimes warn and receive a one-day default cell width."""
        with pytest.warns(UserWarning, match="Singleton coordinate"):
            edges, gaps = get_gap_edges(np.array(["2020-01-01"], dtype="datetime64[D]"))
        expected = np.array(
            ["2019-12-31T12:00:00", "2020-01-01T12:00:00"], dtype="datetime64[ns]"
        )
        np.testing.assert_array_equal(edges, expected)
        assert not len(gaps)

    def test_singleton_numeric(self):
        """Singleton numeric coordinates warn and receive a unit cell width."""
        with pytest.warns(UserWarning, match="Singleton coordinate"):
            edges, gaps = get_gap_edges([10])
        np.testing.assert_allclose(edges, [9.5, 10.5])
        assert not len(gaps)


class TestIsMonotonicAndFinite:
    """Tests for validating mesh coordinate centers."""

    @pytest.mark.parametrize("values", [[0, 1, 2], [2, 1, 0]])
    def test_valid(self, values):
        """Ascending and descending finite values are valid."""
        assert is_monotonic_and_finite(values)

    @pytest.mark.parametrize("values", [[0, 2, 1], [0, np.nan, 2]])
    def test_invalid(self, values):
        """Nonmonotonic and nonfinite values are invalid."""
        assert not is_monotonic_and_finite(values)
