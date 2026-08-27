"""
Tests for optoDAS files.
"""

import dascore as dc
from dascore.io.optodas import OptoDASV8
from dascore.utils.downloader import fetch


class TestOptoDASIssues:
    """Test case related to issues in OptoDAS parser."""

    def test_scan_distance_units_preserved(self):
        """Snapped and exact scan coordinates should retain header units."""
        path = fetch("decimated_optodas.hdf5")
        fiber_io = OptoDASV8()

        for snap in (True, False):
            payload = fiber_io.scan(path, snap=snap)[0]
            distance = payload["coords"].get_coord("distance")
            assert distance.units == dc.get_quantity("m")
