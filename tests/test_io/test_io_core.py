"""
Test for basic IO and related functions.
"""
import copy
from pathlib import Path
from typing import Union

import numpy as np
import pytest

import dascore as dc
from dascore.constants import SpoolType
from dascore.exceptions import InvalidFiberFile, InvalidFiberIO, UnknownFiberFormat
from dascore.io.core import FiberIO
from dascore.io.dasdae.core import DASDAEV1
from dascore.utils.io import BinaryReader, BinaryWriter
from dascore.utils.time import to_datetime64


class _FiberFormatTestV1(FiberIO):
    """A test format v1"""

    name = "_TestFormatter"
    version = "1"


class _FiberFormatTestV2(FiberIO):
    """A test format v2"""

    name = "_TestFormatter"
    version = "2"


class _FiberCaster(FiberIO):
    """A test class for casting inputs to certain types."""

    name = "_TestFormatter"
    version = "2"

    def read(self, path: BinaryReader, **kwargs) -> SpoolType:
        """Just ensure read was cast to correct type"""
        assert isinstance(path, BinaryReader)

    def write(self, spool: SpoolType, path: BinaryWriter):
        """ditto for write"""
        assert isinstance(path, BinaryWriter)

    def get_format(self, path: Path) -> Union[tuple[str, str], bool]:
        """and get format"""
        assert isinstance(path, Path)
        return False

    def scan(self, not_path: BinaryReader):
        """Ensure an off-name still works for type casting."""
        assert isinstance(not_path, BinaryWriter)


class TestFormatManager:
    """tests for the format manager."""

    @pytest.fixture(scope="class")
    def format_manager(self):
        """Deep copy manager to avoid changing state used by other objects."""
        manager = copy.deepcopy(FiberIO.manager)
        return manager

    def test_specific_format_and_version(self, format_manager):
        """
        Specifying a known format and version should return exactly one formatter.
        """
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
        v2_arg = np.argmax([isinstance(x, _FiberFormatTestV2) for x in ext_formatters])
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
        """Ensure providing a bad version but valid format raises"""
        with pytest.raises(UnknownFiberFormat, match="known versions"):
            iterator = format_manager.yield_fiberio(format="DASDAE", version="-1")
            list(iterator)

    def test_format_format_no_version(self, format_manager):
        """Ensure providing a bad version but valid format raises"""
        with pytest.raises(UnknownFiberFormat, match="known versions"):
            iterator = format_manager.yield_fiberio(format="DASDAE", version="-1")
            list(iterator)

    def test_format_multiple_versions(self, format_manager):
        """Ensure multiple versions are returned when only format is specified."""
        file_format = _FiberFormatTestV1.name
        out = list(format_manager.yield_fiberio(format=file_format))
        assert len(out) == 2


class TestFormatter:
    """Tests for adding file supports through Formatter"""

    # the methods a formatter can implement.

    class FormatterWithName(FiberIO):
        """A formatter with a file name."""

        name = "_test_format"

    def test_empty_formatter_raises(self):
        """An empty formatter can't exist; it at least needs a name."""

        with pytest.raises(InvalidFiberIO):

            class empty_formatter(FiberIO):
                """formatter with no name"""

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


class TestGetFormat:
    """Tests to ensure formats can be retrieved."""

    def test_not_known(self, dummy_text_file):
        """Ensure a non-path/str object raises."""
        with pytest.raises(UnknownFiberFormat):
            dc.get_format(dummy_text_file)

    def test_missing_file(self):
        """Ensure a missing file raises"""
        with pytest.raises(FileNotFoundError):
            dc.get_format("bad/file")


class TestScan:
    """Tests for scanning fiber files."""

    def test_scan_with_ignore(self, tmp_path):
        """ignore option should make scan return []"""
        # no ignore should still raise
        dummy_file = tmp_path / "data.txt"
        dummy_file.touch()
        with pytest.raises(UnknownFiberFormat):
            _ = dc.scan(dummy_file)
        out = dc.scan(dummy_file, ignore=True)
        assert not len(out)
        assert out == []

    def test_scan_directory(self, tmp_path):
        """Trying to scan an empty directory should return empty list."""
        out = dc.scan(tmp_path)
        assert len(out) == 0

    def test_scan_bad_files(self, tmp_path):
        """Trying to scan a directory should raise a nice error"""
        new = tmp_path / "myfile.txt"
        with pytest.raises(InvalidFiberFile):
            _ = dc.scan(new)

    def test_scan_patch(self, random_patch):
        """Scan should also work on a patch"""
        out = dc.scan_to_df(random_patch)
        attrs = random_patch.attrs
        assert len(out) == 1
        ser = out.iloc[0]
        assert to_datetime64(ser["time_min"]) == to_datetime64(attrs["time_min"])
        assert to_datetime64(ser["time_max"]) == to_datetime64(attrs["time_max"])

    def test_implements_scan(self):
        """Test for checking is subclass implements_scan"""
        assert not _FiberFormatTestV2().implements_scan
        assert not _FiberFormatTestV1().implements_scan
        dasdae = FiberIO.manager.get_fiberio("DASDAE")
        dasdae.implements_scan
        assert dasdae.implements_scan

    def test_implements_get_format(self):
        """Test for checking is subclass implements_get_format"""
        assert not _FiberFormatTestV2().implements_get_format
        assert not _FiberFormatTestV1().implements_get_format
        dasdae = FiberIO.manager.get_fiberio("DASDAE")
        assert dasdae.implements_get_format


class TestCastType:
    """Test suite to ensure types are intelligently cast to type hints."""

    def test_read(self, dummy_text_file):
        """ensure write casts type."""
        io = _FiberCaster()
        # this passes if it doesnt raise.
        io.read(dummy_text_file)

    def test_write(self, tmp_path, random_spool):
        """Ensure write casts type."""
        path = tmp_path / "write_fiber_cast.txt"
        io = _FiberCaster()
        # this passes if it doesnt raise.
        io.write(random_spool, path)  # noqa

    def test_non_standard_name(self, dummy_text_file):
        """Ensure non-standard names still work."""
        io = _FiberCaster()
        io.scan(dummy_text_file)
