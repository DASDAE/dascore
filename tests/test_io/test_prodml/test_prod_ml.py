"""
Generic tests for prodml support.
"""
import pytest

import dascore as dc


class TestSilixaFile:
    """
    Ensure we can read the file provided by Silixa.

    We do this since the other tests read the prodML files, even though
    the Silixa file is technical just ProdML v2.1.
    """

    @pytest.fixture(scope="class")
    def silixa_h5_patch(self, idas_h5_example_path):
        """Get the silixa file, return Patch."""
        return dc.spool(idas_h5_example_path)[0]

    def test_read_silixa(self, silixa_h5_patch):
        """Ensure we can read  Silixa file."""
        assert isinstance(silixa_h5_patch, dc.Patch)
        assert silixa_h5_patch.shape
