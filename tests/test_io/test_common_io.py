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
from dascore.exceptions import CoordError, MissingOptionalDependencyError
from dascore.io import BinaryReader
from dascore.io.ap_sensing import APSensingV10
from dascore.io.dasdae import DASDAEV1
from dascore.io.dashdf5 import DASHDF5
from dascore.io.dasvader import DASVaderV1
from dascore.io.febus import Febus1, Febus2
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
    APSensingV10(): ("ap_sensing_1.hdf5",),
    DASDAEV1(): ("example_dasdae_event_1.h5",),
    DASHDF5(): ("PoroTomo_iDAS_1.h5",),
    DASVaderV1(): ("das_vader_1.jld2",),
    Febus1(): ("valencia_febus_example.h5",),
    Febus2(): ("febus_1.h5", "febus_2.h5"),
    GDR_V1(): ("gdr_1.h5",),
    H5Simple(): ("h5_simple_2.h5", "h5_simple_1.h5"),
    NeubrexDASV1(): ("neubrex_das_1.h5",),
    NeubrexRFSV1(): ("neubrex_dss_forge.h5", "neubrex_dts_forge.h5"),
    OptoDASV8(): ("opto_das_1.hdf5",),
    ProdMLV2_0(): ("prodml_2.0.h5", "opta_sense_quantx_v2.h5"),
    ProdMLV2_1(): (
        "prodml_2.1.h5",
        "iDAS005_hdf5_example.626.h5",
    ),
    TDMSFormatterV4713(): ("sample_tdms_file_v4713.tdms",),
    SegyV1_0(): ("conoco_segy_1.sgy",),
    SentekV5(): ("DASDMSShot00_20230328155653619.das",),
    SilixaH5V1(): ("silixa_h5_1.hdf5",),
    SintelaBinaryV3(): ("sintela_binary_v3_test_1.raw",),
    Terra15FormatterV4(): (
        "terra15_das_1_trimmed.hdf5",
        "terra15_das_unfinished.hdf5",
    ),
    Terra15FormatterV5(): ("terra15_v5_test_file.hdf5",),
    Terra15FormatterV6(): ("terra15_v6_test_file.hdf5",),
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


def _scan_summary(scan_result):
    """Normalize scan output to the patch summary view."""
    return scan_result.summary if isinstance(scan_result, dc.Patch) else scan_result


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
def scanned_summaries(data_file_path):
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
    """Ensure patch summary and coordinates match."""
    summary = patch.summary
    coords = patch.coords
    assert summary.dims == coords.dims
    for dim in summary.dims:
        coord = patch.get_coord(dim)
        summary_coord = summary.get_coord_summary(dim)
        assert coord.min() == summary_coord.min
        assert coord.max() == summary_coord.max
        assert coord.step == summary_coord.step


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
            for key in iterate(data_files):
                with skip_timeout():
                    path = fetch(key)
                out = io_instance.get_format(path)
                if out:
                    _format_name, version = out
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
            summaries_from_file = [_scan_summary(x) for x in dc.scan(path)]
        assert len(summaries_from_file)
        summary_init = summaries_from_file[0]
        for dim in summary_init.dim_tuple:
            summary_coord = summary_init.get_coord_summary(dim)
            start = summary_coord.min
            stop = summary_coord.max
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
            summary = patch.summary
            _assert_coords_attrs_match(patch)
            coord = patch.get_coord(dim)
            _assert_op_or_close(coord.min(), trim_tuple[0], ge)
            _assert_op_or_close(coord.max(), summary.get_coord_summary(dim).max, eq)
            # then single-ended query on end side
            trim_tuple = (None, start + duration / 10)
            spool = io.read(path, **{dim: trim_tuple})
            assert len(spool) == 1
            patch = spool[0]
            summary = patch.summary
            _assert_coords_attrs_match(patch)
            coord = patch.get_coord(dim)
            _assert_op_or_close(coord.min(), summary.get_coord_summary(dim).min, eq)
            _assert_op_or_close(coord.max(), trim_tuple[1], le)

    def test_slice_out_all_patches_time(self, io_path_tuple):
        """Ensure slicing outside of file time range returns an empty spool."""
        io, path = io_path_tuple
        with skip_missing():
            scan_patches = dc.scan(path)
        dims = {y for x in scan_patches for y in x.dims}
        if "time" not in dims:
            pytest.skip("Test requires patch with time and distance dimensions.")
        # First test on selecting outside time range.
        end_time = np.max([x.get_coord_summary("time").max for x in scan_patches])
        one_second = np.timedelta64(1, "s")
        spool = io.read(path, time=(end_time + one_second, ...))
        assert len(spool) == 0

    def test_slice_out_all_patches_distance(self, io_path_tuple):
        """Ensure slicing outside file distance range returns an empty spool."""
        io, path = io_path_tuple
        with skip_missing():
            scan_patches = dc.scan(path)
        dims = {y for x in scan_patches for y in x.dims}
        if "distance" not in dims:
            pytest.skip("Test requires patch with time and distance dimensions.")
        # The outside distance range.
        max_dist = np.max([x.get_coord_summary("distance").max for x in scan_patches])
        spool = io.read(path, distance=(max_dist + 1, ...))
        assert len(spool) == 0


class TestScan:
    """Tests for generic scanning."""

    def test_scan_basics(self, data_file_path):
        """Ensure each file can be scanned."""
        with skip_missing():
            summary_list = dc.scan(data_file_path)
        assert len(summary_list)

        for summary in summary_list:
            assert isinstance(summary, dc.PatchSummary)
            assert str(summary.path) == str(data_file_path)

    def test_raw_scan_excludes_source_metadata(self, io_path_tuple):
        """Direct FiberIO scans should not attach source metadata."""
        io, path = io_path_tuple
        with skip_missing():
            summary_list = io.scan(path)
        for summary in summary_list:
            assert not summary.path
            assert not summary.file_format
            assert not summary.file_version
            attr_dump = summary.attrs.model_dump()
            assert "path" not in attr_dump
            assert "file_format" not in attr_dump
            assert "file_version" not in attr_dump

    def test_public_scan_has_version_and_format(self, io_path_tuple):
        """Public scan output should contain source metadata."""
        io, path = io_path_tuple
        with skip_missing():
            summary_list = dc.scan(path)
        for summary in summary_list:
            assert str(summary.path) == str(path)
            assert summary.file_format == io.name
            assert summary.file_version == io.version

    def test_time_coord_is_time(self, scanned_summaries):
        """Ensure scanned summaries have correct dtype for time."""
        for summary in scanned_summaries:
            with suppress(KeyError):
                time = summary.get_coord_summary("time")
                assert "datetime64" in str(np.dtype(time.dtype))

    def test_dist_coord_is_float_or_int(self, scanned_summaries):
        """Distance can be either float or int, but must be numeric."""
        for summary in scanned_summaries:
            with suppress(KeyError, CoordError):
                distance = summary.get_coord_summary("distance")
                dtype = np.dtype(distance.dtype)
                assert np.issubdtype(dtype, np.number)

    def test_coord_dtype_non_empty(self, scanned_summaries):
        """Each coordinate summary should have a non-empty dtype string."""
        for summary in scanned_summaries:
            for coord_name, coord in summary.coords.items():
                assert coord.dtype, f"coord '{coord_name}' has empty dtype"

    def test_patch_dims_non_empty(self, scanned_summaries):
        """Scan results should carry at least one dimension."""
        for summary in scanned_summaries:
            assert summary.dims, "PatchSummary has empty dims"

    def test_patch_dtype_non_empty(self, scanned_summaries):
        """Scan results should carry a non-empty data dtype."""
        for summary in scanned_summaries:
            assert summary.dtype, "PatchSummary has empty dtype"

    def test_coord_min_max_ordered(self, scanned_summaries):
        """Coord min should be <= max for all coordinates."""
        for summary in scanned_summaries:
            for coord_name, coord in summary.coords.items():
                with suppress(TypeError):  # incomparable types (e.g. NaT/NaN)
                    assert (
                        coord.min <= coord.max
                    ), f"{coord_name}: min ({coord.min}) > max ({coord.max})"

    def test_no_bytes(self, scanned_summaries):
        """Sometimes bytes are returned from scanning, we need str."""
        for summary in scanned_summaries:
            model = _scan_summary(summary).model_dump()
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

    def test_scan_summary_matches_patch_summary(self, data_file_path):
        """We need to make sure scan and patch summaries are identical."""
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
            scan_summary_list = [_scan_summary(x) for x in dc.scan(data_file_path)]
        patch_summary_list = [x.summary for x in _cached_read(data_file_path)]
        assert len(scan_summary_list) == len(patch_summary_list)
        for patch_summary, scan_summary in zip(patch_summary_list, scan_summary_list):
            assert patch_summary.dims == scan_summary.dims
            # first compare dimensions are related attributes
            for dim in patch_summary.dim_tuple:
                patch_coord = patch_summary.get_coord_summary(dim)
                scan_coord = scan_summary.get_coord_summary(dim)
                assert patch_coord.min == scan_coord.min
                for attr_name in ("min", "max", "step"):
                    attr1 = getattr(patch_coord, attr_name)
                    attr2 = getattr(scan_coord, attr_name)
                    # Use close comparison for floating point values
                    if isinstance(attr1, float | np.floating) and isinstance(
                        attr2, float | np.floating
                    ):
                        np.testing.assert_allclose(attr1, attr2, rtol=1e-12)
                    else:
                        assert attr1 == attr2
            # then other expected attributes.
            for attr_name in comp_attrs:
                patch_value = getattr(patch_summary.attrs, attr_name)
                scan_value = getattr(scan_summary.attrs, attr_name)
                if scan_value in ("", None):
                    continue
                assert scan_value == patch_value
