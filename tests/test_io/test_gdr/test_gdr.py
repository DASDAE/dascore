"""Tests for the GDR file format."""

import numpy as np
import pytest

from dascore.io.gdr import GDR_V1
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
