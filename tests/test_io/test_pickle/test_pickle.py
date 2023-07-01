"""
Tests for reading/writing pickles.
"""
import pytest

import dascore as dc
from dascore.io.pickle.core import PickleIO


@pytest.fixture(scope="session")
def pickle_patch_path(tmp_path_factory, random_patch):
    """Pickle a patch and return the path."""
    path = tmp_path_factory.mktemp("pickle_test") / "test.pkl"
    random_patch.io.write(path, "pickle")
    return path


class TestIsPickle:
    """Test detecting pickle format."""

    def test_detect_file(self, pickle_patch_path):
        """Simple test on output of is pickle."""
        parser = PickleIO()
        out = parser.get_format(pickle_patch_path)
        assert out
        assert out[0] == "PICKLE"

    def test_read_pickle(self, pickle_patch_path, random_patch):
        """Ensure a pickle file can be read."""
        out = dc.read(pickle_patch_path)
        assert isinstance(out, dc.BaseSpool)
        assert len(out) == 1
        assert isinstance(out[0], dc.Patch)
        assert random_patch == out[0]


class TestScan:
    """Tests for scanning pickle files/"""

    comp_attrs = (
        "data_type",
        "data_units",
        "d_time",
        "time_min",
        "time_max",
        "distance_min",
        "distance_max",
        "d_distance",
        "tag",
        "network",
    )

    def test_scan_attrs_eq_read_attrs(self, pickle_patch_path):
        """Ensure read/scan produce the same attrs."""
        scan_list = dc.scan(pickle_patch_path)
        patch_attrs_list = [x.attrs for x in dc.read(pickle_patch_path)]

        for scan_attrs, patch_attrs in zip(scan_list, patch_attrs_list):
            for attr in self.comp_attrs:
                scan_attr = getattr(scan_attrs, attr)
                patch_attr = getattr(patch_attrs, attr)
                assert scan_attr == patch_attr
