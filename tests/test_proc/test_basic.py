"""
Tests for basic patch functions.
"""
import numpy as np


class TestAbs:
    """Test absolute values."""

    def test_no_negatives(self, random_patch):
        """simply ensure the data has no negatives."""
        # add
        data = np.array(random_patch.data)
        data[:, 0] = -2
        new = random_patch.new(data=data)
        out = new.abs()
        assert np.all(out.data >= 0)
