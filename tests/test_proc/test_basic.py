"""
Tests for basic patch functions.
"""
import numpy as np
import pytest


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


class TestNormalize:
    """Tests for normalization."""

    def test_bad_norm_raises(self, random_patch):
        """Ensure an unsupported norm raises"""
        with pytest.raises(ValueError):
            random_patch.normalize("time", norm="bob_norm")

    def test_l2(self, random_patch):
        """Ensure after operation norms are 1."""
        dims = random_patch.dims
        # test along distance axis
        dist_norm = random_patch.normalize("distance", norm="l2")
        axis = dims.index("distance")
        norm = np.linalg.norm(dist_norm.data, axis=axis)
        assert np.allclose(norm, 1)
        # tests along time axis
        time_norm = random_patch.normalize("time", norm="l2")
        axis = dims.index("time")
        norm = np.linalg.norm(time_norm.data, axis=axis)
        assert np.allclose(norm, 1)

    def test_l1(self, random_patch):
        """Ensure after operation norms are 1."""
        dims = random_patch.dims
        # test along distance axis
        dist_norm = random_patch.normalize("distance", norm="l1")
        axis = dims.index("distance")
        norm = np.abs(np.sum(dist_norm.data, axis=axis))
        assert np.allclose(norm, 1)
        # tests along time axis
        time_norm = random_patch.normalize("time", norm="l1")
        axis = dims.index("time")
        norm = np.abs(np.sum(time_norm.data, axis=axis))
        assert np.allclose(norm, 1)

    def test_max(self, random_patch):
        """Ensure after operation norms are 1."""
        dims = random_patch.dims
        # test along distance axis
        dist_norm = random_patch.normalize("distance", norm="l1")
        axis = dims.index("distance")
        norm = np.abs(np.sum(dist_norm.data, axis=axis))
        assert np.allclose(norm, 1)
        # tests along time axis
        time_norm = random_patch.normalize("time", norm="l1")
        axis = dims.index("time")
        norm = np.abs(np.sum(time_norm.data, axis=axis))
        assert np.allclose(norm, 1)
