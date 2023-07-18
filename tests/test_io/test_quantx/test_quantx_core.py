"""
Tests for QuantXV2 format
"""
import pytest

from dascore.io.core import read
from dascore.utils.downloader import fetch
from dascore.utils.misc import register_func

PATCH_FIXTURES = []


@pytest.fixture(scope="session")
def quantx_v2_example_path():
    """Return the path to the example QuantXV2 file."""
    out = fetch("opta_sense_quantx_v2.h5")
    assert out.exists()
    return out


@pytest.fixture(scope="session")
@register_func(PATCH_FIXTURES)
def quantx_v2_das_patch(quantx_v2_example_path):
    """Read the QuantXV2 data, return contained DataArray"""
    out = read(quantx_v2_example_path, "quantx")[0]
    attr_time = out.attrs["time_max"]
    coord_time = out.coords["time"].max()
    assert attr_time == coord_time
    return out


class TestReadQuantXV2:
    """Tests for reading the QuantXV2 format."""

    def test_precision_of_time_array(self, quantx_v2_das_patch):
        """
        Ensure the time array is in ns, not native ms, in order to
        be consistent with other patches.
        """
        time = quantx_v2_das_patch.coords["time"]
        dtype = time.dtype
        assert "[ns]" in str(dtype)
