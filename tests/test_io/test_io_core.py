"""
Test for basic IO and related functions.
"""
import copy

import numpy as np
import pytest

import dascore
from dascore.exceptions import (
    InvalidFiberFile,
    InvalidFileFormatter,
    UnknownFiberFormat,
)
from dascore.io.core import FiberIO
from dascore.io.dasdae.core import DASDAEV1


class FiberFormatTestV1(FiberIO):
    """A test format v1"""

    name = "_TestFormatter"
    version = "1"


class FiberFormatTestV2(FiberIO):
    """A test format v2"""

    name = "_TestFormatter"
    version = "2"


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
        v2_arg = np.argmax([isinstance(x, FiberFormatTestV2) for x in ext_formatters])
        v1_arg = np.argmax([isinstance(x, FiberFormatTestV1) for x in ext_formatters])
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
        file_format = FiberFormatTestV1.name
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

        with pytest.raises(InvalidFileFormatter):

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

    def test_terra_15(self, terra15_das_example_path):
        """Ensure terra15 v2 can be read"""
        out = dascore.get_format(terra15_das_example_path)
        vtuple = (out[0].upper(), out[1])
        assert vtuple == ("TERRA15", "4")

    def test_not_known(self, dummy_text_file):
        """Ensure a non-path/str object raises."""
        with pytest.raises(UnknownFiberFormat):
            dascore.get_format(dummy_text_file)


class TestRead:
    """Basic tests for reading files."""

    def test_read_terra15(self, terra15_das_example_path, terra15_das_patch):
        """Ensure terra15 can be read."""
        out = dascore.read(terra15_das_example_path)
        assert isinstance(out, dascore.MemorySpool)
        assert len(out) == 1
        assert out[0].equals(terra15_das_patch)


class TestScan:
    """Tests for scanning fiber files."""

    def test_scan_terra15(self, terra15_das_example_path):
        """Ensure terra15 format can be automatically determined."""
        out = dascore.scan(terra15_das_example_path)
        assert isinstance(out, list)
        assert len(out)

    def test_scan_with_ignore(self, tmp_path):
        """ignore option should make scan return []"""
        # no ignore should still raise
        dummy_file = tmp_path / "data.txt"
        dummy_file.touch()
        with pytest.raises(UnknownFiberFormat):
            _ = dascore.scan(dummy_file)
        out = dascore.scan(dummy_file, ignore=True)
        assert not len(out)
        assert out == []

    def test_scan_directory(self, tmp_path):
        """Trying to scan a directory should raise a nice error"""
        with pytest.raises(InvalidFiberFile, match="a directory"):
            _ = dascore.scan(tmp_path)

    def test_scan_patch(self, random_patch):
        """Scan should also work on a patch"""
        out = dascore.scan_to_df(random_patch)
        attrs = random_patch.attrs
        assert len(out) == 1
        ser = out.iloc[0]
        assert ser["time_min"] == attrs["time_min"]
        assert ser["time_max"] == attrs["time_max"]

    def test_implements_scan(self):
        """Test for checking is subclass implements_scan"""
        assert not FiberFormatTestV2().implements_scan
        assert not FiberFormatTestV1().implements_scan
        dasdae = FiberIO.manager.get_fiberio("DASDAE")
        assert dasdae.implements_scan

    def test_implements_get_format(self):
        """Test for checking is subclass implements_get_format"""
        assert not FiberFormatTestV2().implements_get_format
        assert not FiberFormatTestV1().implements_get_format
        dasdae = FiberIO.manager.get_fiberio("DASDAE")
        assert dasdae.implements_get_format
