"""Tests for correlate patch processing function."""


class TestCorrelateInternal:
    """Tests case of intra-patch correlation function."""

    def test_basic_correlation(self, random_patch):
        """Ensure correlation works with a random patch."""
        # dist = random_patch.get_coord("distance")[0]
        # random_patch.correlate_internal(distance=dist)
