"""Test for basic IO and related functions."""
from __future__ import annotations

import copy
import io
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.constants import SpoolType
from dascore.exceptions import InvalidFiberIO, UnknownFiberFormat
from dascore.io.core import FiberIO, PatchFileSummary
from dascore.io.dasdae.core import DASDAEV1
from dascore.utils.io import BinaryReader, BinaryWriter
from dascore.utils.time import to_datetime64

tvar = TypeVar("tvar", int, float, str, Path)


class _FiberFormatTestV1(FiberIO):
    """A test format v1."""

    name = "_TestFormatter"
    version = "1"


class _FiberFormatTestV2(FiberIO):
    """A test format v2."""

    name = "_TestFormatter"
    version = "2"


class _FiberImplementer(FiberIO):
    """A fiber io which implements all the methods (poorly)."""

    name = "_Implementer"
    version = "2"

    def read(self, resource, **kwargs):
        """Dummy read."""

    def write(self, spool: SpoolType, resource):
        """Dummy write."""

    def scan(self, resource: BinaryReader):
        """Dummy scan."""

    def get_format(self, resource):
        """Dummy get_format."""


class _FiberCaster(FiberIO):
    """A test class for casting inputs to certain types."""

    name = "_TestFormatter"
    version = "2"

    def read(self, resource: BinaryReader, **kwargs) -> SpoolType:
        """Just ensure read was cast to correct type."""
        assert isinstance(resource, io.BufferedReader)

    def write(self, spool: SpoolType, resource: BinaryWriter):
        """Ditto for write."""
        assert isinstance(resource, io.BufferedWriter)

    def get_format(self, resource: Path) -> tuple[str, str] | bool:
        """And get format."""
        assert isinstance(resource, Path)
        return False

    def scan(self, not_path: BinaryReader):
        """Ensure an off-name still works for type casting."""
        assert isinstance(not_path, io.BufferedReader)


class _FiberUnsupportedTypeHints(FiberIO):
    """A fiber io which implements typehints which have not casting meaning."""

    name = "_TypeHinterNotRight"
    version = "2"

    def read(self, resource: tvar, **kwargs):
        """Dummy read."""
        with open(resource) as fi:
            return fi.read()


class TestPatchFileSummary:
    """Tests for getting patch file information."""

    def test_d_translates(self):
        """Ensure d_{whatever} translates to step_{whatever}."""
        out = PatchFileSummary(d_time=10)
        assert out.time_step == dc.to_timedelta64(10)

    def test_dim_typle(self):
        """Ensure patch file summaries can be converted to tuples."""
        out = PatchFileSummary(d_time=10, dims="time,distance")
        assert out.dim_tuple == ("time", "distance")

    def test_flat_dump(self):
        """Simple test to show summary can be flat dumped."""
        # flat dump is just here for compatibility with dc.PatchAttrs
        out = PatchFileSummary(d_time=10, dims="time,distance")
        assert isinstance(out.flat_dump(), dict)


class TestFormatManager:
    """Tests for the format manager."""

    @pytest.fixture(scope="class")
    def format_manager(self):
        """Deep copy manager to avoid changing state used by other objects."""
        manager = copy.deepcopy(FiberIO.manager)
        return manager

    def test_specific_format_and_version(self, format_manager):
        """Specifying a known format and version should return exactly one formatter."""
        out = list(format_manager.yield_fiberio("DASDAE", "1"))
        assert len(out) == 1
        assert isinstance(out[0], DASDAEV1)

    def test_get_all_formatters(self, format_manager):
        """Ensure getting all formatters through yield_fiberio works."""
        formatters = list(format_manager.yield_fiberio())
        assert len(formatters) >= len(format_manager._eps)

    def test_extension_priority(self, format_manager):
        """Ensure the extension priority is honored."""
        ext = "h5"
        ext_formatters = list(format_manager.yield_fiberio(extension=ext))
        all_formatters = list(format_manager.yield_fiberio())
        in_formatter = [ext in x.preferred_extensions for x in ext_formatters]
        format_array = np.array(in_formatter).astype(bool)
        # ensure all the start of the arrays are True.
        assert np.argmin(format_array) == np.sum(format_array)
        # ensure all formats are represented.
        assert len(format_array) == len(all_formatters)
        # ensure V2 of the Test formatter appears first
        v2_arg = np.argmax([isinstance(x, _FiberImplementer) for x in ext_formatters])
        v1_arg = np.argmax([isinstance(x, _FiberFormatTestV1) for x in ext_formatters])
        assert v2_arg < v1_arg

    def test_format_raises_unknown_format(self, format_manager):
        """Ensure we raise for unknown formats."""
        with pytest.raises(UnknownFiberFormat, match="format"):
            list(format_manager.yield_fiberio(format="bob_2"))

    def test_format_raises_just_version(self, format_manager):
        """Providing only a version should also raise."""
        with pytest.raises(UnknownFiberFormat, match="version"):
            list(format_manager.yield_fiberio(version="1"))

    def test_format_bad_version(self, format_manager):
        """Ensure providing a bad version but valid format raises."""
        with pytest.raises(UnknownFiberFormat, match="known versions"):
            iterator = format_manager.yield_fiberio(format="DASDAE", version="-1")
            list(iterator)

    def test_format_format_no_version(self, format_manager):
        """Ensure providing a bad version but valid format raises."""
        with pytest.raises(UnknownFiberFormat, match="known versions"):
            iterator = format_manager.yield_fiberio(format="DASDAE", version="-1")
            list(iterator)

    def test_format_multiple_versions(self, format_manager):
        """Ensure multiple versions are returned when only format is specified."""
        file_format = _FiberFormatTestV1.name
        out = list(format_manager.yield_fiberio(format=file_format))
        assert len(out) == 2


class TestFormatter:
    """Tests for adding file supports through Formatter."""

    # the methods a formatter can implement.

    class FormatterWithName(FiberIO):
        """A formatter with a file name."""

        name = "_test_format"

    def test_empty_formatter_raises(self):
        """An empty formatter can't exist; it at least needs a name."""
        with pytest.raises(InvalidFiberIO):

            class empty_formatter(FiberIO):
                """formatter with no name."""

    def test_empty_formatter_undefined_methods(self, random_patch):
        """
        Ensure a Not Implemented error is raised for un-implemented methods
        of FormatterWithName.
        """
        instance = self.FormatterWithName()
        with pytest.raises(NotImplementedError):
            instance.read("empty_path")
        with pytest.raises(NotImplementedError):
            instance.write(random_patch, "empty_path")
        with pytest.raises(NotImplementedError):
            instance.get_format("empty_path")
        with pytest.raises(NotImplementedError):
            instance.scan("bad_path")

    def test_doesnt_implements(self):
        """Tests for implements_x methods."""
        # this test fiber io don't implement anything
        fio = _FiberFormatTestV1()
        assert not fio.implements_scan
        assert not fio.implements_get_format
        assert not fio.implements_read
        assert not fio.implements_write

    def test_implements(self):
        """Tests for implements_x methods."""
        # this test fiber implements all the things
        fio = _FiberImplementer()
        assert fio.implements_scan
        assert fio.implements_get_format
        assert fio.implements_read
        assert fio.implements_write


class TestGetFormat:
    """Tests to ensure formats can be retrieved."""

    def test_not_known(self, dummy_text_file):
        """Ensure a non-path/str object raises."""
        with pytest.raises(UnknownFiberFormat):
            dc.get_format(dummy_text_file)

    def test_missing_file(self):
        """Ensure a missing file raises."""
        with pytest.raises(FileNotFoundError):
            dc.get_format("bad/file")


class TestScan:
    """Tests for scanning fiber files."""

    @pytest.fixture(scope="class")
    def nested_directory_with_patches(self, tmpdir_factory, random_patch):
        """Return a nested directory with patch files interlaced."""
        out = Path(tmpdir_factory.mktemp("nested_random_patch"))
        path_1 = out / "patch_1.h5"
        path_2 = out / "subdir" / "patch_2.h5"
        path_3 = out / "subdir" / "suber_dir" / "patch_3.h5"
        random_patch.io.write(path_1, "dasdae")
        random_patch.io.write(path_2, "dasdae")
        random_patch.io.write(path_3, "dasdae")
        return out

    def test_scan_no_good_files(self, tmp_path):
        """Scan with no fiber files should return []."""
        dummy_file = tmp_path / "data.txt"
        dummy_file.touch()
        out = dc.scan(dummy_file)
        assert not len(out)
        assert out == []

    def test_scan_directory(self, tmp_path):
        """Trying to scan an empty directory should return empty list."""
        out = dc.scan(tmp_path)
        assert len(out) == 0

    def test_scan_bad_files(self, tmp_path):
        """Trying to scan a directory should raise a nice error."""
        new = tmp_path / "myfile.txt"
        with pytest.raises(FileNotFoundError):
            _ = dc.scan(new)

    def test_scan_patch(self, random_patch):
        """Scan should also work on a patch."""
        out = dc.scan_to_df(random_patch)
        attrs = random_patch.attrs
        assert len(out) == 1
        ser = out.iloc[0]
        assert to_datetime64(ser["time_min"]) == to_datetime64(attrs["time_min"])
        assert to_datetime64(ser["time_max"]) == to_datetime64(attrs["time_max"])

    def test_scan_nested_directory(self, nested_directory_with_patches):
        """Ensure scan picks up files in nested directories."""
        out = dc.scan(nested_directory_with_patches)
        assert len(out) == 3

    def test_can_raise(self):
        """
        Scan, when called from a FiberIO, should be able to raise if
        type coercion fails.
        """
        fio = _FiberImplementer()
        bad_input = _FiberFormatTestV1()
        with pytest.raises(NotImplementedError):
            fio.scan(bad_input)


class TestCastType:
    """Test suite to ensure types are intelligently cast to type hints."""

    def test_read(self, dummy_text_file):
        """Ensure write casts type."""
        io = _FiberCaster()
        # this passes if it doesnt raise.
        io.read(dummy_text_file)

    def test_write(self, tmp_path, random_spool):
        """Ensure write casts type."""
        path = tmp_path / "write_fiber_cast.txt"
        io = _FiberCaster()
        # this passes if it doesnt raise.
        io.write(random_spool, path)

    def test_non_standard_name(self, dummy_text_file):
        """Ensure non-standard names still work."""
        io = _FiberCaster()
        io.scan(dummy_text_file)

    def test_unsupported_typehints(self, dummy_text_file):
        """Ensure FiberIO with non-"special" type hints still works."""
        fiberio = _FiberUnsupportedTypeHints()
        out = fiberio.read(dummy_text_file)
        assert out == Path(dummy_text_file).read_text()

    def test_unsupported_type(self, dummy_text_file):
        """Ensure FiberIO from above works with dascore.read."""
        name = _FiberUnsupportedTypeHints.name
        version = _FiberUnsupportedTypeHints.version
        out = dc.read(dummy_text_file, name, version)
        assert out == Path(dummy_text_file).read_text()


class TestGetSupportedIOTable:
    """A test for creating the supported io table."""

    def test_get_supported_io_table(self):
        """Test the get_supported_io_table function."""
        # call the function to get the result
        result_df = FiberIO.get_supported_io_table()

        # assert that the result is a DataFrame
        assert isinstance(result_df, pd.DataFrame)

        # assert that the length of the DataFrame is not 0
        assert len(result_df) > 0
