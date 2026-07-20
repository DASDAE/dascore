"""
Tests for optoDAS files.
"""

import dascore as dc
from dascore.io.optodas import OptoDASV8
from dascore.utils.downloader import fetch


class TestOptoDASIssues:
    """Test case related to issues in OptoDAS parser."""

    def test_read_decimated_patch(self):
        """Tests for reading spatially decimated patch (#419)"""
        path = fetch("decimated_optodas.hdf5")
        fiber_io = OptoDASV8()

        fmt_str, version_str = fiber_io.get_format(path)
        assert (fmt_str, version_str) == (fiber_io.name, fiber_io.version)

        spool = fiber_io.read(path)
        patch = spool[0]
        assert isinstance(patch, dc.Patch)
        assert patch.data.shape

    def test_scan_distance_units_preserved(self):
        """Snapped and exact scan coordinates should retain header units."""
        path = fetch("decimated_optodas.hdf5")
        fiber_io = OptoDASV8()

        for snap in (True, False):
            payload = fiber_io.scan(path, snap=snap)[0]
            distance = payload["coords"].get_coord("distance")
            assert distance.units == dc.get_quantity("m")
