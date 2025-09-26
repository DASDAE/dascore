"""
Common tests for IO operations.

The objective of this test file is to reduce the amount of repetition
in testing common operations with various formats. Format specific
tests should go in their respective test modules. Tests for *how* specific
IO functions (i.e., not that they work on various files) should go in
test_io_core.py
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
from functools import cache
from io import BytesIO, UnsupportedOperation
from operator import eq, ge, le
from pathlib import Path
from urllib import error as urllib_error

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.exceptions import MissingOptionalDependencyError
from dascore.io import BinaryReader
from dascore.io.ap_sensing import APSensingV10
from dascore.io.dasdae import DASDAEV1
from dascore.io.dashdf5 import DASHDF5
from dascore.io.febus import Febus2
from dascore.io.gdr import GDR_V1
from dascore.io.h5simple import H5Simple
from dascore.io.neubrex import NeubrexDASV1, NeubrexRFSV1
from dascore.io.optodas import OptoDASV8
from dascore.io.pickle import PickleIO
from dascore.io.prodml import ProdMLV2_0, ProdMLV2_1
from dascore.io.segy import SegyV1_0
from dascore.io.sentek import SentekV5
from dascore.io.silixah5 import SilixaH5V1
from dascore.io.sintela_binary import SintelaBinaryV3
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
    SintelaBinaryV3(): ("sintela_binary_v3_test_1.raw",),
    GDR_V1(): ("gdr_1.h5",),
    NeubrexDASV1(): ("neubrex_das_1.h5",),
    NeubrexRFSV1(): ("neubrex_dss_forge.h5", "neubrex_dts_forge.h5"),
    SilixaH5V1(): ("silixa_h5_1.hdf5",),
    ProdMLV2_0(): ("prodml_2.0.h5", "opta_sense_quantx_v2.h5"),
    ProdMLV2_1(): (
        "prodml_2.1.h5",
        "iDAS005_hdf5_example.626.h5",
    ),
    H5Simple(): ("h5_simple_2.h5", "h5_simple_1.h5"),
    APSensingV10(): ("ap_sensing_1.hdf5",),
    Febus2(): ("febus_1.h5",),
    OptoDASV8(): ("opto_das_1.hdf5",),
    DASDAEV1(): ("example_dasdae_event_1.h5",),
    TDMSFormatterV4713(): ("sample_tdms_file_v4713.tdms",),
    Terra15FormatterV4(): (
        "terra15_das_1_trimmed.hdf5",
        "terra15_das_unfinished.hdf5",
    ),
    Terra15FormatterV5(): ("terra15_v5_test_file.hdf5",),
    Terra15FormatterV6(): ("terra15_v6_test_file.hdf5",),
    SegyV1_0(): ("conoco_segy_1.sgy",),
    DASHDF5(): ("PoroTomo_iDAS_1.h5",),
    SentekV5(): ("DASDMSShot00_20230328155653619.das",),
}

# This tuple is for fiber io which support a write method and can write
# generic patches. If the patch has to be in some special form, for example
# only flat patches can be written to WAV, don't put it here.
COMMON_IO_WRITE_TESTS = (
    PickleIO(),
    DASDAEV1(),
)

# Specifies data registry entries which should not be tested.
SKIP_DATA_FILES = {"whale_1.hdf5", "brady_hs_DAS_DTS_coords.csv"}


@contextmanager
def skip_missing():
    """Skip if missing dependencies found."""
    try:
        yield
    except MissingOptionalDependencyError as exc:
        pytest.skip(f"Missing optional dependency required to read file: {exc}")
    except TimeoutError as exc:
        pytest.skip(f"Unable to fetch data due to timeout: {exc}")


@contextmanager
def skip_timeout():
    """Skip if downloading file times out."""
    try:
        yield
    except (TimeoutError, urllib_error.URLError) as exc:
        pytest.skip(f"Unable to fetch data due to timeout: {exc}")


@cache
def _cached_read(path, io=None):
    """
    A function to read data without any params and cache.
    This ensures each files is read at most twice.
    """
    if io is None:
        read = dc.read
    else:
        read = io.read
    with skip_missing():
        out = read(path)
    return out


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
    with skip_timeout():
        return io, fetch(fetch_name)


@pytest.fixture(scope="session", params=get_registry_df()["name"])
def data_file_path(request):
    """A fixture of all data files. Will download if needed."""
    param = request.param
    # Some files should be skipped if not DAS or too big.
    if str(param) in SKIP_DATA_FILES:
        pytest.skip(f"Skipping {param}")
    with skip_timeout():
        return fetch(request.param)


@pytest.fixture(scope="session")
def read_spool(data_file_path):
    """Read each file into a spool."""
    with skip_missing():
        out = dc.read(data_file_path)
    return out


@pytest.fixture(scope="session")
def scanned_attrs(data_file_path):
    """Read each file into a spool."""
    with skip_missing():
        out = dc.scan(data_file_path)
    return out


@pytest.fixture(scope="session", params=COMMON_IO_WRITE_TESTS)
def fiber_io_writer(request):
    """Meta fixture for IO that implement write."""
    return request.param


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
    """Assert op(val1, val2) or isclose(val1, val2)."""
    meets_eq = op(val1, val2)
    both_null = pd.isnull(val1) and pd.isnull(val2)
    all_close_vals = all_close(val1, val2)
    if meets_eq or both_null or all_close_vals:
        return
    msg = f"{val1} and {val2} do not pass {op} or all close."
    raise AssertionError(msg)


# --- Tests

DIM_RELATED_ATTRS = ("{dim}_min", "{dim}_max", "{dim}_step")


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

    def test_all_other_files_arent_format(self, io_instance):
        """All other data files should not show up as this format."""
        for other_io, data_files in COMMON_IO_READ_TESTS.items():
            if isinstance(other_io, type(io_instance)):
                continue
            for key in data_files:
                with skip_timeout():
                    path = fetch(key)
                out = io_instance.get_format(path)
                if out:
                    format_name, version = out
                    assert version != io_instance.version


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

    def test_time_coords_are_datetimes(self, read_spool):
        """Ensure the time coordinates have the type of datetime."""
        for patch in read_spool:
            with suppress(KeyError):
                time = patch.get_coord("time")
                assert "datetime64" in str(np.dtype(time.dtype))

    def test_read_stream(self, io_path_tuple):
        """If the format supports reading from a stream, test it out."""
        io, path = io_path_tuple
        req_type = getattr(io.read, "_required_type", None)
        if req_type is not BinaryReader:
            pytest.skip(f"{io} doesn't support BinaryReader streams.")

        spool1 = _cached_read(path)
        # write file contents to bytes io and ensure it can be read.
        bio = BytesIO()
        bio.write(Path(path).read_bytes())
        bio.seek(0)
        try:
            spool2 = io.read(bio)
        except (AttributeError, OSError, UnsupportedOperation) as e:
            # Skip if the format doesn't support BytesIO (e.g., missing
            # 'name' attribute, fileno() not supported, or other BytesIO
            # incompatibilities)
            pytest.skip(f"{io} doesn't support BytesIO streams: {e}")
        for patch1, patch2 in zip(spool1, spool2):
            assert patch1.equals(patch2)

    def test_slice_single_dim_both_ends(self, io_path_tuple):
        """
        Ensure each dimension can be passed as an argument to `read` and
        a patch containing the requested data is returned.
        """
        io, path = io_path_tuple
        with skip_missing():
            attrs_from_file = dc.scan(path)
        assert len(attrs_from_file)
        # skip files that have more than one patch for now
        # TODO just write better test logic to handle this case.
        if len(attrs_from_file) > 1:
            pytest.skip("Haven't implemented test for multipatch files.")
        attrs_init = attrs_from_file[0]
        for dim in attrs_init.dim_tuple:
            start = getattr(attrs_init, f"{dim}_min")
            stop = getattr(attrs_init, f"{dim}_max")
            duration = stop - start
            # first test double ended query
            trim_tuple = (start + duration / 10, start + 2 * duration / 10)
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
    """Tests for generic scanning."""

    def test_scan_basics(self, data_file_path):
        """Ensure each file can be scanned."""
        with skip_missing():
            attrs_list = dc.scan(data_file_path)
        assert len(attrs_list)

        for attrs in attrs_list:
            assert isinstance(attrs, dc.PatchAttrs)
            assert str(attrs.path) == str(data_file_path)

    def test_scan_has_version_and_format(self, io_path_tuple):
        """Scan output should contain version and format."""
        io, path = io_path_tuple
        with skip_missing():
            attr_list = io.scan(path)
        for attrs in attr_list:
            assert attrs.file_format == io.name
            assert attrs.file_version == io.version

    def test_time_coord_is_time(self, scanned_attrs):
        """Ensure scanned attrs have correct dtype for time."""
        for patch_attr in scanned_attrs:
            with suppress(KeyError):
                time = patch_attr.coords["time"]
                assert "datetime64" in str(np.dtype(time.dtype))

    def test_dist_coord_is_float_or_int(self, scanned_attrs):
        """Distance can be either float or int, but must be numeric."""
        for patch_attr in scanned_attrs:
            with suppress(KeyError):
                distance = patch_attr.coords["distance"]
                dtype = np.dtype(distance.dtype)
                assert np.issubdtype(dtype, np.number)

    def test_no_bytes(self, scanned_attrs):
        """Sometimes bytes are returned from scanning, we need str."""
        for patch_attr in scanned_attrs:
            model = patch_attr.model_dump()
            for key, value in model.items():
                assert not isinstance(value, bytes | np.bytes_)


class TestWrite:
    """Tests for writing data to disk."""

    @pytest.fixture(scope="session")
    def written_fiber_path(self, random_patch, fiber_io_writer, tmp_path_factory):
        """Write patch to disk, return path."""
        tmp_path = Path(tmp_path_factory.mktemp("generic_write_test"))
        pre_ext = list(iterate(fiber_io_writer.preferred_extensions))
        ext = "" if not len(pre_ext) else pre_ext[0]
        path = tmp_path / f"{fiber_io_writer.name}.{ext}"
        fiber_io_writer.write(random_patch, path)
        return path

    def test_write_random_patch(self, written_fiber_path):
        """Ensure the random patch can be written."""
        assert written_fiber_path.exists()

    def test_roundtrip(self, random_patch, written_fiber_path, fiber_io_writer):
        """If the writer can read, ensure round-tripping patch is equal."""
        if not fiber_io_writer.implements_read:
            pytest.skip("FiberIO doesn't implement read")
        new = fiber_io_writer.read(written_fiber_path)[0]
        assert new == random_patch


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
        with skip_missing():
            scan_attrs_list = dc.scan(data_file_path)
        patch_attrs_list = [x.attrs for x in _cached_read(data_file_path)]
        assert len(scan_attrs_list) == len(patch_attrs_list)
        for pat_attrs1, scan_attrs2 in zip(patch_attrs_list, scan_attrs_list):
            assert pat_attrs1.dims == scan_attrs2.dims
            # first compare dimensions are related attributes
            for dim in pat_attrs1.dim_tuple:
                assert getattr(pat_attrs1, f"{dim}_min") == getattr(
                    scan_attrs2, f"{dim}_min"
                )
                for dim_attr in DIM_RELATED_ATTRS:
                    attr_name = dim_attr.format(dim=dim)
                    attr1 = getattr(pat_attrs1, attr_name)
                    attr2 = getattr(scan_attrs2, attr_name)
                    # Use close comparison for floating point values
                    if isinstance(attr1, float | np.floating) and isinstance(
                        attr2, float | np.floating
                    ):
                        np.testing.assert_allclose(attr1, attr2, rtol=1e-12)
                    else:
                        assert attr1 == attr2
            # then other expected attributes.
            for attr_name in comp_attrs:
                patch_attr = getattr(pat_attrs1, attr_name)
                scan_attr = getattr(scan_attrs2, attr_name)
                assert scan_attr == patch_attr
