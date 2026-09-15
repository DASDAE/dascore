"""Tests for composing selections on the original sample grid."""

import numpy as np
import pytest

from dascore import get_coord_manager
from dascore.io.utils import selection_windows
from dascore.utils.indexing import compose_indexers


class TestComposeIndexers:
    """Sequential selections retain their original sample positions."""

    @pytest.mark.parametrize(
        "first, second",
        [
            (slice(None, None, -1), slice(10, 10)),
            (slice(0, 0, -1), slice(None)),
            (slice(None, None, -1), slice(None, None, 2)),
            (slice(1, 8, 2), 2),
            (slice(None, None, -1), np.array([True, False] * 5)),
            (slice(2, 9, 2), np.array([False, True, True, False])),
        ],
    )
    def test_sequential_selection(self, first, second):
        """Reversed and empty ranges compose exactly like NumPy indexing."""
        data = np.arange(10) * 17 + 3
        composed = compose_indexers(len(data), first, second)
        np.testing.assert_array_equal(data[composed], data[first][second])


class TestSelectionWindows:
    """Bounding windows plus residuals reproduce original-grid selections."""

    @pytest.mark.parametrize(
        "indexer, expected_window",
        [
            (slice(None, None, -2), (1, 8)),
            (slice(0, 0), (0, 0)),
            (np.array([], dtype=int), (0, 0)),
            (np.array([6, 1, 4]), (1, 7)),
            (3, (3, 4)),
        ],
    )
    def test_window_and_residual(self, indexer, expected_window):
        """Strides, reordering, scalars, and empty selections retain exact cells."""
        data = np.arange(8) * 13 + 5
        coords = get_coord_manager(coords={"time": np.arange(8)}, dims=("time",))
        windows, residual = selection_windows(coords, {"time": indexer})
        assert windows == {"time": expected_window}
        bounded = data[slice(*windows["time"])]
        np.testing.assert_array_equal(bounded[residual], np.atleast_1d(data[indexer]))
