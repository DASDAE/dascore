"""
Tests for optoDAS files.
"""

import h5py
import numpy as np
import pytest

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


class TestReadArray:
    """Tests for slicing the data dataset directly."""

    @pytest.fixture(scope="class")
    def transposed_path(self, tmp_path_factory):
        """An OptoDAS file whose header states the other dimension order."""
        source = fetch("opto_das_1.hdf5")
        path = tmp_path_factory.mktemp("optodas_transposed") / "transposed.hdf5"
        with h5py.File(source, "r") as src, h5py.File(path, "w") as dest:
            for name in src:
                src.copy(name, dest)
            names = [x.decode() for x in dest["header"]["dimensionNames"][:]]
            del dest["header"]["dimensionNames"]
            dest["header"]["dimensionNames"] = np.array(
                [x.encode() for x in names[::-1]]
            )
            data = dest["data"][:]
            del dest["data"]
            dest["data"] = data.T
        return path

    def test_dimension_order_follows_the_header(self, transposed_path):
        """The header names the stored order; a hardcoded one would transpose."""
        io = OptoDASV8()
        with h5py.File(transposed_path, "r") as h5:
            stored = h5["data"][:]
        out = io.read_array(transposed_path, {"time": (1, 4)})
        # the header now calls the first axis distance, so a time window
        # takes columns rather than rows
        assert np.array_equal(out, stored[:, 1:4])
