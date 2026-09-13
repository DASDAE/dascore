"""Tests for the GDR file format."""

from types import SimpleNamespace

import numpy as np
import pytest

from dascore.io.gdr import GDR_V1
from dascore.io.gdr.utils_das import _get_dims
from dascore.utils.downloader import fetch


@pytest.fixture(scope="module")
def gpr_path():
    """Return the file path to a GDR file."""
    return fetch("gdr_1.h5")


class TestGDR:
    """Misc. tests not covered by common tests."""

    def test_no_snap(self, gpr_path):
        """Ensure snap or no snap produces the same coord for this file."""
        fiber_io = GDR_V1()
        patch1 = fiber_io.read(gpr_path, snap=False)[0]
        patch2 = fiber_io.read(gpr_path, snap=True)[0]
        time_1 = patch1.get_coord("time")
        time_2 = patch2.get_coord("time")
        assert len(time_1) == len(time_2)
        assert np.all(time_1.values == time_2.values)


class TestGetDims:
    """Tests for reading the dimension names the file states."""

    @staticmethod
    def _dataset(dimensions):
        """A stand-in dataset stating these DasDimensions."""
        return SimpleNamespace(attrs={"DasDimensions": dimensions})

    @pytest.mark.parametrize("locus", ["locus", b"locus", np.bytes_("locus")])
    def test_locus_is_distance(self, locus):
        """The file's locus axis is DASCore's distance, however it is stored."""
        assert _get_dims(self._dataset(["time", locus])) == ("time", "distance")

    def test_order_follows_the_file(self):
        """A distance-major file is reported distance-major."""
        assert _get_dims(self._dataset(["locus", "time"])) == ("distance", "time")

    def test_unknown_dimension_raises(self):
        """A dimension name the reader cannot map is not guessed at."""
        with pytest.raises(AssertionError, match="DasDimensions"):
            _get_dims(self._dataset(["time", "bob"]))
