"""Tests for the Neubrex RFS (DSS/DTS) format."""

import shutil

import h5py
import pytest

import dascore as dc
from dascore.io.neubrex import NeubrexRFSV1
from dascore.io.neubrex.core import NeubrexRFSPatchAttrs
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


class TestNeubrexRFSAttrs:
    """Tests for the attrs read from the data node."""

    @pytest.mark.parametrize("name", [None, "WellBoreID", "wellboreid"])
    def test_well_bore_id(self, rfs_path, name):
        """The well bore id is read whatever the case of its name."""
        if name is not None:
            with h5py.File(rfs_path, "a") as h5:
                attrs = h5["data"].attrs
                attrs[name] = attrs.pop("WellboreID")
        attrs = NeubrexRFSV1().read(rfs_path)[0].attrs
        assert attrs.well_bore_id == "16B(78)-32"

    def test_field_name(self, rfs_path):
        """The file's field name is the one attr naming it."""
        attrs = dict(NeubrexRFSV1().read(rfs_path)[0].attrs)
        assert "filed_name" not in attrs
        assert attrs["field_name"] == "Beaver"

    def test_old_field_name_accepted(self):
        """The name the field had before is taken as it."""
        attrs = NeubrexRFSPatchAttrs(filed_name="Beaver")
        assert attrs.field_name == "Beaver"
        assert "filed_name" not in dict(attrs)
