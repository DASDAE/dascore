"""Tests for composing selections on the original sample grid."""

import numpy as np
import pytest

from dascore.utils.indexing import compose_indexers


class TestComposeIndexers:
    """Sequential selections retain their original sample positions."""

    @pytest.mark.parametrize(
        "first, second",
        [
            (slice(None, None, -1), slice(10, 10)),
            (slice(0, 0, -1), slice(None)),
            (slice(None, None, -1), slice(None, None, 2)),
            (slice(None, None, -1), np.array([True, False] * 5)),
            (slice(2, 9, 2), np.array([False, True, True, False])),
        ],
    )
    def test_sequential_selection(self, first, second):
        """Reversed and empty ranges compose exactly like NumPy indexing."""
        data = np.arange(10) * 17 + 3
        composed = compose_indexers(len(data), first, second)
        np.testing.assert_array_equal(data[composed], data[first][second])
