"""
Tests for optasense format
"""
import pytest

import dascore as dc
from dascore.utils.downloader import fetch


@pytest.fixture(scope="session")
def optasense_v2_example_path():
    """Return the path to the example terra15 file."""
    out = fetch("opta_sense_quantx_v2.h5")
    assert out.exists()
    return out


class TestV2:
    """Tests for version 2 of optasense."""

    def test_can_read(self, optasense_v2_example_path):
        """simple read test using dascore.spool."""
        opta_spool = dc.spool(optasense_v2_example_path)
        opta_patch = opta_spool[0]
        assert isinstance(opta_patch, dc.Patch)
