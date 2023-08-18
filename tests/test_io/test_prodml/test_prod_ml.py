"""Generic tests for prodml support."""
from __future__ import annotations

import pytest

import dascore as dc
from dascore.io.core import read
from dascore.utils.downloader import fetch


@pytest.fixture(scope="session")
def quantx_v2_example_path():
    """Return the path to the example QuantXV2 file."""
    out = fetch("opta_sense_quantx_v2.h5")
    assert out.exists()
    return out


@pytest.fixture(scope="session")
def quantx_v2_das_patch(quantx_v2_example_path):
    """Read the QuantXV2 data, return contained DataArray."""
    out = read(quantx_v2_example_path, "prodml")[0]
    attr_time = out.attrs["time_max"]
    coord_time = out.coords["time"].max()
    assert attr_time == coord_time
    return out


class TestProdMLFile:
    """
    Ensure we can read the ProdML provided by Silixa.

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

    def test_has_gauge_length(self, silixa_h5_patch):
        """Ensure gauge-length is found in patch attrs."""
        patch = silixa_h5_patch
        assert hasattr(patch.attrs, "gauge_length")


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
