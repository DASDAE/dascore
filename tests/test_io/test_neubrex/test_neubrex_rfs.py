"""Tests for the Neubrex RFS (DSS/DTS) format."""

import shutil

import h5py
import pytest

import dascore as dc
from dascore.io.neubrex import NeubrexRFSV1
from dascore.utils.downloader import fetch


@pytest.fixture
def rfs_path(tmp_path):
    """A copy of an RFS file, to edit."""
    path = tmp_path / "neubrex_rfs.h5"
    shutil.copy(fetch("neubrex_dss_forge.h5"), path)
    return path


def _distance_units(path):
    """The distance units from read, scan and the spool index."""
    fiber_io = NeubrexRFSV1()
    return {
        fiber_io.read(path)[0].get_coord("distance").units,
        fiber_io.scan(path)[0].get_coord("distance").units,
        dc.spool(path)[0].get_coord("distance").units,
    }


class TestNeubrexRFSDepthUnits:
    """Tests for the units of the distance coordinate."""

    def test_stated_units(self, rfs_path):
        """The depth dataset's unit attr gives the distance units."""
        with h5py.File(rfs_path, "a") as h5:
            h5["depth"].attrs["unit"] = "ft"
        assert _distance_units(rfs_path) == {dc.get_quantity("ft")}

    def test_no_units(self, rfs_path):
        """A file stating no depth units gives a unitless distance."""
        assert _distance_units(rfs_path) == {None}

    def test_unparsable_units(self, rfs_path):
        """A unit attr which is not units leaves the distance unitless."""
        with h5py.File(rfs_path, "a") as h5:
            h5["depth"].attrs["unit"] = "not_a_unit"
        assert _distance_units(rfs_path) == {None}
