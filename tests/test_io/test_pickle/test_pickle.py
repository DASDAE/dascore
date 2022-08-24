"""
Tests for reading/writing pickles.
"""
import pytest

import dascore
from dascore.io.pickle.core import PickleIO


@pytest.fixture(scope="session")
def pickle_patch_path(tmp_path_factory, random_patch):
    """Pickle a patch and return the path."""
    path = tmp_path_factory.mktemp("pickle_test") / "test.pkl"
    random_patch.io.write(path, "pickle")
    return path


class TestIsPickle:
    """Test detecting pickle format."""

    def test_detect_file(self, pickle_patch_path):
        """Simple test on output of is pickle."""
        parser = PickleIO()
        out = parser.get_format(pickle_patch_path)
        assert out
        assert out[0] == "PICKLE"

    def test_read_pickle(self, pickle_patch_path, random_patch):
        """Ensure a pickle file can be read."""
        out = dascore.read(pickle_patch_path)
        assert isinstance(out, dascore.MemorySpool)
        assert len(out) == 1
        assert isinstance(out[0], dascore.Patch)
        assert random_patch == out[0]
