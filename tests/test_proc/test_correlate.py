"""Tests for correlate patch processing function."""

import numpy as np
import pytest

import dascore as dc


class TestCorrelateInternal:
    """Tests case of intra-patch correlation function."""

    @pytest.fixture(scope="class")
    def corr_patch(self):
        """Create a patch of sin waves whose correlation can be easily checked."""
        patch = dc.get_example_patch(
            "sin_wav",
            sample_rate=100,
            frequency=range(10, 20),
            duration=5,
            channel_count=10,
        ).taper(time=0.5)
        # normalize energy so autocorrection is 1
        time_axis = patch.dims.index("time")
        data = patch.data
        norm = np.linalg.norm(data, axis=time_axis, keepdims=True)
        return patch.new(data=data / norm)

    def test_basic_correlation(self, corr_patch):
        """Ensure correlation works with a random patch along distance dim."""
        dist = corr_patch.get_coord("distance")[0]
        _ = corr_patch.correlate(distance=dist)
