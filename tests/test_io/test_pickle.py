"""
Tests for reading/writing pickles.
"""
import pickle

import pytest

from dascore.io.pickle.core import PickleIO


@pytest.fixture(scope="session")
def pickle_patch_path(tmp_path_factory, random_patch):
    """Pickle a patch and return the path."""
    path = tmp_path_factory.mktemp("pickle_test") / "test.pkl"
    with path.open("wb") as fi:
        pickle.dump(random_patch, fi)
    return path


class TestIsPickle:
    """Test detecting pickle format."""

    def test_detect_file(self, pickle_patch_path):
        """Simple test on output of is pickle."""
        parser = PickleIO()
        out = parser.get_format(pickle_patch_path)
        assert out
        assert out[0] == "PICKLE"
