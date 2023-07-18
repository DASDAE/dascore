"""
Common tests for IO operations.

The objective of this test file is to reduce the amount of repetition
in testing common operations with various formats. Format specific
tests should go in their respective test modules. Tests for *how* specific
IO functions (i.e., not that they work on various files) should go in
test_io_core.py
"""
from functools import cache
from operator import eq, ge, le

import pandas as pd
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
from dascore.utils.misc import all_close, iterate

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


@cache
def _cached_read(path, io=None):
    """
    A function to read data without any params and cache.
    This ensures each files is read at most twice.
    """
    if io is None:
        return dc.read(path)
    return io.read(path)


def _get_flat_io_test():
    """Flatten list to [(fiberio, path)] so it can be parametrized."""
    flat_io = []
    for io, fetch_name_list in COMMON_IO_READ_TESTS.items():
        for fetch_name in iterate(fetch_name_list):
            flat_io.append([io, fetch_name])
    return flat_io


@pytest.fixture(scope="session", params=list(COMMON_IO_READ_TESTS))
def io_instance(request):
    """Fixture for returning fiber io instances."""
    return request.param


@pytest.fixture(scope="session", params=_get_flat_io_test())
def io_path_tuple(request):
    """
    A fixture which returns io instance, path_to_file.
    This is used for common testing.
    """
    io, fetch_name = request.param
    return io, fetch(fetch_name)


@pytest.fixture(scope="session", params=get_registry_df()["name"])
def data_file_path(request):
    """A fixture of all data files. Will download if needed."""
    return fetch(request.param)


@pytest.fixture(scope="session")
def read_spool(data_file_path):
    """Read each file into a spool."""
    return dc.read(data_file_path)


# --- Helper functions


def _assert_coords_attrs_match(patch):
    """Ensure both the coordinates and attributes match on patch."""
    attrs = patch.attrs
    coords = patch.coords
    assert attrs.dim_tuple == coords.dims
    for dim in attrs.dim_tuple:
        coord = patch.get_coord(dim)
        assert coord.min() == getattr(attrs, f"{dim}_min")
        assert coord.max() == getattr(attrs, f"{dim}_max")
        assert coord.step == getattr(attrs, f"{dim}_step")


def _assert_op_or_close(val1, val2, op):
    """assert op(val1, val2) or isclose(val1, val2)."""
    meets_eq = op(val1, val2)
    both_null = pd.isnull(val1) and pd.isnull(val2)
    all_close_vals = all_close(val1, val2)
    if meets_eq or both_null or all_close_vals:
        return
    msg = f"{val1} and {val2} do not pass {op} or all close."
    assert False, msg


# --- Tests

DIM_RELATED_ATTRS = ("{dim}_min", "{dim}_max", "d_{dim}")


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

    def test_random_textfile_isnt_format(self, io_instance, dummy_text_file):
        """Ensure a dummy text file the format (it isn't any fiber format)."""
        assert not io_instance.get_format(dummy_text_file)

    def test_random_h5_isnt_format(self, io_instance, generic_hdf5):
        """Ensure a dummy h5 file the format (it isn't any fiber format)."""
        assert not io_instance.get_format(generic_hdf5)


class TestRead:
    """Test suite for reading formats."""

    def test_read_returns_spools(self, io_path_tuple):
        """Each read function must return a spool."""
        io, path = io_path_tuple
        out = _cached_read(path, io=io)
        assert isinstance(out, dc.BaseSpool)
        assert all([isinstance(x, dc.Patch) for x in out])

    def test_coord_attrs_match(self, read_spool):
        """The attributes and coordinate should match in min/max/step."""
        for patch in read_spool:
            _assert_coords_attrs_match(patch)

    def test_slice_single_dim_both_ends(self, io_path_tuple):
        """
        Ensure each dimension can be passed as an argument to `read` and
        a patch containing the requested data is returned.
        """
        io, path = io_path_tuple
        attrs = dc.scan(path)
        # skip files that have more than one patch for now
        # TODO just write better test logic to handle this case.
        if len(attrs) > 1:
            pytest.skip("Havent implemented test for multipatch files.")
        attrs_init = attrs[0]
        for dim in attrs_init.dim_tuple:
            start = getattr(attrs_init, f"{dim}_min")
            stop = getattr(attrs_init, f"{dim}_max")
            duration = stop - start
            # first test double ended query
            trim_tuple = (start + duration / 10, start + 2 * duration // 10)
            spool = io.read(path, **{dim: trim_tuple})
            assert len(spool) == 1
            patch = spool[0]
            _assert_coords_attrs_match(patch)
            coord = patch.get_coord(dim)
            _assert_op_or_close(coord.min(), trim_tuple[0], ge)
            _assert_op_or_close(coord.max(), trim_tuple[1], le)
            # then single-ended query on start side
            trim_tuple = (start + duration / 10, ...)
            spool = io.read(path, **{dim: trim_tuple})
            assert len(spool) == 1
            patch = spool[0]
            attrs = patch.attrs
            _assert_coords_attrs_match(patch)
            coord = patch.get_coord(dim)
            _assert_op_or_close(coord.min(), trim_tuple[0], ge)
            _assert_op_or_close(coord.max(), getattr(attrs, f"{dim}_max"), eq)
            # then single-ended query on end side
            trim_tuple = (None, start + duration / 10)
            spool = io.read(path, **{dim: trim_tuple})
            assert len(spool) == 1
            patch = spool[0]
            attrs = patch.attrs
            _assert_coords_attrs_match(patch)
            coord = patch.get_coord(dim)
            _assert_op_or_close(coord.min(), getattr(attrs, f"{dim}_min"), eq)
            _assert_op_or_close(coord.max(), trim_tuple[1], le)


class TestScan:
    """Tests for generic scanning"""

    def test_scan_basics(self, data_file_path):
        """Ensure each file can be scanned."""
        attrs_list = dc.scan(data_file_path)
        assert len(attrs_list)

        for attrs in attrs_list:
            assert isinstance(attrs, dc.PatchAttrs)
            assert str(attrs.path) == str(data_file_path)

    def test_scan_has_version_and_format(self, io_path_tuple):
        """Scan output should contain version and format."""
        io, path = io_path_tuple
        attr_list = io.scan(path)
        for attrs in attr_list:
            assert attrs.file_format == io.name
            assert attrs.file_version == io.version


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
            "tag",
            "network",
        )
        scan_attrs_list = dc.scan(data_file_path)
        patch_attrs_list = [x.attrs for x in _cached_read(data_file_path)]
        assert len(scan_attrs_list) == len(patch_attrs_list)
        for pat_attrs1, scan_attrs2 in zip(patch_attrs_list, scan_attrs_list):
            assert pat_attrs1.dims == scan_attrs2.dims
            # first compare dimensions are related attributes
            for dim in pat_attrs1.dim_tuple:
                assert getattr(pat_attrs1, f"{dim}_min") == getattr(
                    pat_attrs1, f"{dim}_min"
                )
                for dim_attr in DIM_RELATED_ATTRS:
                    attr_name = dim_attr.format(dim=dim)
                    attr1 = getattr(pat_attrs1, attr_name)
                    attr2 = getattr(pat_attrs1, attr_name)
                    assert attr1 == attr2
            # then other expected attributes.
            for attr_name in comp_attrs:
                patch_attr = getattr(pat_attrs1, attr_name)
                scan_attr = getattr(scan_attrs2, attr_name)
                assert scan_attr == patch_attr
