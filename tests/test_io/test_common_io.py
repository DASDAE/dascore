"""
Common tests for IO operations.

The objective of this test file is to reduce the amount of repetition
in testing common operations with various formats. Format specific
tests should go in their respective test modules. Tests for *how* specific
IO functions (i.e., not that they work on various files) should go in
test_io_core.py
"""
import pytest

import dascore as dc
from dascore.io.dasdae import DASDAEV1
from dascore.io.prodml import ProdMLV2_0, ProdMLV2_1
from dascore.io.quantx import QuantXV2
from dascore.io.tdms import TDMSFormatterV4713
from dascore.io.terra15 import (
    Terra15FormatterV4,
    Terra15FormatterV5,
    Terra15FormatterV6,
)
from dascore.utils.downloader import fetch, get_registry_df
from dascore.utils.misc import iterate

# --- Fixtures

# These fixtures are for attaching FiberIO objects to a suite of common
# read, scan, and get_format tests that all FiberIO instances which implement
# more than just a write method should pass.
# The format is {FiberIO class: [path_or_fetch_name]} where fetch_name
# is the name passed to dascore.utils.downloader.fetch to load the file.
# See the docs on adding a new IO format, in the contributing section,
# for more details.
COMMON_IO_READ_TESTS = {
    DASDAEV1(): ("example_dasdae_event_1.h5",),
    ProdMLV2_0(): ("prodml_2.0.h5",),
    ProdMLV2_1(): (
        "prodml_2.1.h5",
        "iDAS005_hdf5_example.626.h5",
    ),
    TDMSFormatterV4713(): ("sample_tdms_file_v4713.tdms",),
    Terra15FormatterV4(): (
        "terra15_das_1_trimmed.hdf5",
        "terra15_das_unfinished.hdf5",
    ),
    Terra15FormatterV5(): ("terra15_v5_test_file.hdf5",),
    Terra15FormatterV6(): ("terra15_v6_test_file.hdf5",),
    QuantXV2(): ("opta_sense_quantx_v2.h5",),
}


def _get_flat_io_test():
    """Flatten list to [(fiberio, path)] so it can be parametrized."""
    flat_io = []
    for io, fetch_name_list in COMMON_IO_READ_TESTS.items():
        for fetch_name in iterate(fetch_name_list):
            flat_io.append([io, fetch_name])
    return flat_io


@pytest.fixture(scope="session", params=_get_flat_io_test())
def io_path_tuple(request):
    """
    A fixture which returns io instance, path_to_file.
    This is used for common testing.
    """
    io, fetch_name = request.param
    return io, fetch(fetch_name)


@pytest.fixture(scope="session")
def read_spool(io_path_tuple):
    """Read path without any parameters."""
    io, path = io_path_tuple
    return io.read(path)


@pytest.fixture(scope="class", params=get_registry_df()["name"])
def data_file_path(request):
    """A fixture of all data files. Will download if needed."""
    return fetch(request.param)


# --- Tests


class TestGetFormat:
    """Test suite for getting the version/format of files."""

    def test_expected_version(self, io_path_tuple):
        """Each io should get its own version/name for its test files."""
        io, path = io_path_tuple
        expected = (io.name, io.version)
        out = io.get_format(path)
        assert out == expected

    def test_each_file_has_version(self, data_file_path):
        """We should be able to get each file's format/version."""
        out = dc.get_format(data_file_path)
        assert out
        assert len(out) == 2
        format_str, version_str = out
        assert isinstance(format_str, str)
        assert isinstance(version_str, str)


class TestRead:
    """Test suite for reading formats."""

    def test_scan_read_same_attrs(self, io_path_tuple):
        """Ensure"""


class TestScan:
    """Tests for generic scanning"""


class TestIntegration:
    """Test suite for generic scanning."""

    def test_scan_attrs_match_patch_attrs(self, data_file_path):
        """We need to make sure scan and patch attrs are identical."""
        # Since dasdae format stores attrs and coords, we need to
        # skip events created before coords/attrs were more closely
        # aligned.
        if data_file_path.name == "example_dasdae_event_1.h5":
            return
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
        scan_attrs_list = dc.scan(data_file_path)
        patch_attrs_list = [x.attrs for x in dc.read(data_file_path)]
        assert len(scan_attrs_list) == len(patch_attrs_list)
        for pat_attrs1, scan_attrs2 in zip(patch_attrs_list, scan_attrs_list):
            for attr_name in comp_attrs:
                patch_attr = getattr(pat_attrs1, attr_name)
                scan_attr = getattr(scan_attrs2, attr_name)
                assert scan_attr == patch_attr
