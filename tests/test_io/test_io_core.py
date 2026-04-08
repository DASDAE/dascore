"""Test for basic IO and related functions."""

from __future__ import annotations

import copy
import io
from pathlib import Path
from typing import TypeVar

import h5py
import numpy as np
import pandas as pd
import pytest
import rich.progress as prog
from upath import UPath

import dascore as dc
from dascore.config import set_config
from dascore.constants import SpoolType
from dascore.exceptions import (
    InvalidFiberIOError,
    MissingOptionalDependencyError,
    RemoteCacheError,
    UnknownFiberFormatError,
)
from dascore.io.core import (
    FiberIO,
    PatchFileSummary,
    _get_reloadable_source_path,
    _make_scan_payload,
    _scan_result_to_summary,
)
from dascore.io.dasdae.core import DASDAEV1
from dascore.utils.io import BinaryReader, BinaryWriter, IOResourceManager
from dascore.utils.misc import suppress_warnings
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
    """A fiber io which implements typehints which have no casting meaning."""

    name = "_TypeHinterNotRight"
    version = "2"

    def read(self, resource: tvar, **kwargs):
        """Dummy read."""
        with open(resource) as fi:
            return fi.read()


class _FiberDirectory(FiberIO):
    """A FiberIO which accepts a directory."""

    name = "_directory_test_io"
    version = "0.1"
    input_type = "directory"

    def get_format(self, resource) -> tuple[str, str] | bool:
        """Only accept directories which have specific naming."""
        path = Path(resource)
        name = path.name
        if self.name in name:
            return self.name, self.version
        return False


class _ReadOnlySummaryFormatter(FiberIO):
    """A formatter that relies on FiberIO.scan falling back to read()."""

    name = "_read_only_summary_formatter"
    version = "1"

    def read(self, resource: Path, **kwargs) -> SpoolType:
        """Return a simple spool for default scan conversion."""
        return dc.spool([dc.get_example_patch().update_attrs(tag="fallback")])

    def get_format(self, resource: Path) -> tuple[str, str] | bool:
        """Only accept the explicit fallback-scan test resource."""
        path = Path(resource)
        if path.suffix == ".h5" and path.name == "fallback_scan.h5":
            return self.name, self.version
        return False


class _MissingOptionalFormatter(FiberIO):
    """A formatter whose scan path requires an unavailable optional dependency."""

    name = "_missing_optional_formatter"
    version = "1"

    def scan(self, resource: Path, **kwargs):
        """Raise a stable missing-optional error for scan coverage tests."""
        msg = (
            "not_optional_pkg is not installed but is required for the requested "
            "functionality."
        )
        raise MissingOptionalDependencyError(msg)

    def get_format(self, resource: Path) -> tuple[str, str] | bool:
        """Only accept the explicit missing-optional test resource."""
        path = Path(resource)
        if path.suffix == ".opt" and path.name == "missing_optional.opt":
            return self.name, self.version
        return False


class TestPatchFileSummary:
    """Tests for getting patch file information."""

    def test_d_translates(self):
        """Ensure d_{whatever} translates to step_{whatever}."""
        out = PatchFileSummary(d_time=10)
        assert out.time_step == dc.to_timedelta64(10)

    def test_dim_tuple(self):
        """Ensure patch file summaries can be converted to tuples."""
        out = PatchFileSummary(d_time=10, dims="time,distance")
        assert out.dim_tuple == ("time", "distance")

    def test_flat_dump(self):
        """Simple test to show summary can be flat dumped."""
        # flat dump is just here for compatibility with dc.PatchAttrs
        out = PatchFileSummary(d_time=10, dims="time,distance")
        assert isinstance(out.flat_dump(), dict)


class TestScanResultToSummary:
    """Tests for converting scan metadata into summaries."""

    def test_scan_payload_dict_input_builds_summary(self):
        """Structured scan payloads should normalize to PatchSummary."""
        patch = dc.get_example_patch()
        payload = {
            "attrs": patch.attrs,
            "coords": patch.coords,
            "dims": patch.dims,
            "shape": patch.shape,
            "dtype": str(patch.data.dtype),
            "source_patch_id": "node-1",
        }
        out = _scan_result_to_summary(payload, source_path="some_path")
        assert isinstance(out, dc.PatchSummary)
        assert (
            out.get_coord_summary("time").fingerprint
            == patch.get_coord("time").fingerprint()
        )
        assert out.source_patch_id == "node-1"
        assert str(out.source_path) == "some_path"

    def test_scan_payload_missing_dtype_raises(self):
        """Structured scan payloads should require dtype metadata."""
        patch = dc.get_example_patch()
        payload = {
            "attrs": patch.attrs,
            "coords": patch.coords,
            "dims": patch.dims,
            "shape": patch.shape,
        }
        msg = r"requires a mapping with `coords`, `attrs`, and `dtype`"
        with pytest.raises(TypeError, match=msg):
            _scan_result_to_summary(payload, source_path="some_path")

    def test_make_scan_payload_uses_dtype_key(self):
        """The helper should emit the normalized dtype field."""
        patch = dc.get_example_patch()
        out = _make_scan_payload(
            attrs=patch.attrs,
            coords=patch.coords,
            dims=patch.dims,
            shape=patch.shape,
            dtype=str(patch.data.dtype),
        )
        assert out["dtype"] == str(patch.data.dtype)

    def test_invalid_dict_input_raises(self):
        """Untyped dict payloads should still be rejected."""
        msg = r"requires a mapping with `coords`, `attrs`, and `dtype`"
        with pytest.raises(TypeError, match=msg):
            _scan_result_to_summary({"tag": "x"})

    def test_invalid_non_mapping_input_raises(self):
        """Unsupported scan outputs should mention allowed input shapes."""
        msg = "only accepts PatchSummary or structured scan payload mappings"
        with pytest.raises(TypeError, match=msg):
            _scan_result_to_summary("bad scan output")

    def test_patch_attrs_input_raises(self):
        """PatchAttrs scan outputs should fail with a migration hint."""
        patch = dc.get_example_patch()
        msg = (
            "DASCore no longer accepts PatchAttrs from FiberIO.scan\\(\\).*"
            "docs/contributing/new_format.qmd"
        )
        with pytest.raises(ValueError, match=msg):
            _scan_result_to_summary(patch.attrs)

    def test_summary_source_patch_id_sets_private_attr(self):
        """Summary source ids should be copied onto private attrs."""
        summary = dc.PatchSummary(
            attrs=dc.PatchAttrs(tag="x"),
            source_patch_id="node-1",
        )
        assert summary.source_patch_id == "node-1"
        assert summary.attrs["_source_patch_id"] == "node-1"

    def test_private_attr_source_patch_id_sets_summary(self):
        """Private attr source ids should populate the summary field."""
        summary = dc.PatchSummary(
            attrs=dc.PatchAttrs(tag="x", _source_patch_id="node-2"),
        )
        assert summary.source_patch_id == "node-2"
        assert summary.attrs["_source_patch_id"] == "node-2"

    def test_summary_source_patch_id_wins_on_conflict(self):
        """Conflicting ids should resolve in favor of the summary field."""
        summary = dc.PatchSummary(
            attrs=dc.PatchAttrs(tag="x", _source_patch_id="attrs-id"),
            source_patch_id="summary-id",
        )
        assert summary.source_patch_id == "summary-id"
        assert summary.attrs["_source_patch_id"] == "summary-id"


class TestFormatManager:
    """Tests for the format manager."""

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
        v2_arg = np.argmax([isinstance(x, _FiberImplementer) for x in ext_formatters])
        v1_arg = np.argmax([isinstance(x, _FiberFormatTestV1) for x in ext_formatters])
        assert v2_arg < v1_arg

    def test_format_raises_unknown_format(self, format_manager):
        """Ensure we raise for unknown formats."""
        with pytest.raises(UnknownFiberFormatError, match="format"):
            list(format_manager.yield_fiberio(format="bob_2"))

    def test_format_raises_just_version(self, format_manager):
        """Providing only a version should also raise."""
        with pytest.raises(UnknownFiberFormatError, match="version"):
            list(format_manager.yield_fiberio(version="1"))

    def test_format_bad_version(self, format_manager):
        """Ensure providing a bad version but valid format raises."""
        with pytest.raises(UnknownFiberFormatError, match="known versions"):
            iterator = format_manager.yield_fiberio(format="DASDAE", version="-1")
            list(iterator)

    def test_format_format_no_version(self, format_manager):
        """Ensure providing a bad version but valid format raises."""
        with pytest.raises(UnknownFiberFormatError, match="known versions"):
            iterator = format_manager.yield_fiberio(format="DASDAE", version="-1")
            list(iterator)

    def test_format_multiple_versions(self, format_manager):
        """Ensure multiple versions are returned when only format is specified."""
        file_format = _FiberFormatTestV1.name
        out = list(format_manager.yield_fiberio(format=file_format))
        assert len(out) == 2

    def test_unique_values_extensions(self, format_manager):
        """Ensure unique FiberIO are returned for an extension."""
        out = list(format_manager.yield_fiberio(extension="h5"))
        name_ver = [(x.name, x.version) for x in out]
        assert len(name_ver) == len(set(name_ver))

    def test_unique_values_no_extensions(self, format_manager):
        """Ensure unique FiberIO are returned when nothing specified."""
        out = list(format_manager.yield_fiberio())
        name_ver = [(x.name, x.version) for x in out]
        assert len(name_ver) == len(set(name_ver))

    def test_known_formats_empty_entry_points(self, format_manager):
        """Known formats should tolerate an empty/non-string entry-point index."""
        format_manager.__dict__.pop("_eps", None)
        format_manager.__dict__.pop("known_formats", None)
        format_manager._eps = pd.Series(dtype=object)
        assert isinstance(format_manager.known_formats, set)

    def test_load_plugins_empty_entry_points(self, format_manager):
        """Loading plugins should no-op when no entry points are present."""
        format_manager.__dict__.pop("_eps", None)
        format_manager.__dict__.pop("known_formats", None)
        format_manager._eps = pd.Series(dtype=object)
        format_manager.load_plugins()


class TestFormatter:
    """Tests for adding file supports through Formatter."""

    # the methods a formatter can implement.

    class FormatterWithName(FiberIO):
        """A formatter with a file name."""

        name = "_test_format"

    def test_empty_formatter_raises(self):
        """An empty formatter can't exist; it at least needs a name."""
        with pytest.raises(InvalidFiberIOError):

            class EmptyFormatter(FiberIO):
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

    @pytest.fixture(scope="class")
    def empty_h5_path(self, tmpdir_factory):
        """Create an empty HDF5 file."""
        path = tmpdir_factory.mktemp("empty") / "empty.h5"
        with h5py.File(path, "w"):
            pass
        return path

    def test_not_known(self, dummy_text_file):
        """Ensure a non-path/str object raises."""
        with pytest.raises(UnknownFiberFormatError):
            dc.get_format(dummy_text_file)

    def test_missing_file(self):
        """Ensure a missing file raises."""
        with pytest.raises(FileNotFoundError):
            dc.get_format("bad/file")

    def test_fiberio_directory(self, tmp_path_factory):
        """Ensure a directory can be recognized as a FiberIO."""
        fiber_io = _FiberDirectory()
        path = tmp_path_factory.mktemp(fiber_io.name)
        assert fiber_io.get_format(path)
        (name, version) = dc.get_format(path)
        assert fiber_io.name == name
        assert fiber_io.version == version

    def test_manager_get_format_invokes_fiberio_get_format(self, monkeypatch, tmp_path):
        """Manager format detection should execute FiberIO get_format loop bodies."""
        path = tmp_path / "format_loop.h5"
        path.write_text("placeholder")
        fiber_io = _ReadOnlySummaryFormatter()
        seen = {}

        def _yield_fiberio(*_args, **_kwargs):
            yield fiber_io

        def _get_format(resource, **_kwargs):
            seen["resource"] = resource
            return (fiber_io.name, fiber_io.version)

        monkeypatch.setattr(FiberIO.manager, "yield_fiberio", _yield_fiberio)
        monkeypatch.setattr(fiber_io, "get_format", _get_format)
        fiber_io.get_format._required_type = Path

        assert FiberIO.manager._get_format(path=path) == (
            fiber_io.name,
            fiber_io.version,
        )
        assert seen["resource"] == path


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
        summary = random_patch.summary
        assert len(out) == 1
        ser = out.iloc[0]
        time_summary = summary.get_coord_summary("time")
        assert to_datetime64(ser["time_min"]) == to_datetime64(time_summary.min)
        assert to_datetime64(ser["time_max"]) == to_datetime64(time_summary.max)

    def test_scan_patch_returns_summary(self, random_patch):
        """Direct patch scan should normalize to a PatchSummary."""
        out = dc.scan(random_patch)
        assert len(out) == 1
        scanned = out[0]
        assert isinstance(scanned, dc.PatchSummary)
        assert scanned.dtype == str(random_patch.dtype)
        assert not scanned.source_patch_id

    def test_scan_multi_patch_includes_source_patch_id(self, tmp_path):
        """Multi-patch scan rows should include a stable source patch id."""
        path = tmp_path / "multi_patch.h5"
        spool = dc.examples.get_example_spool("random_das", length=2)
        dc.write(spool, path, "DASDAE", file_version="1")
        out = dc.scan_to_df(path)
        assert "source_patch_id" in out.columns
        assert out["source_patch_id"].astype(bool).all()

    def test_scan_nested_directory(self, nested_directory_with_patches):
        """Ensure scan picks up files in nested directories."""
        out = dc.scan(nested_directory_with_patches)
        assert len(out) == 3

    def test_scan_single_file(self, terra15_v6_path):
        """Ensure scan works on a single file."""
        out = dc.scan(terra15_v6_path)
        assert len(out) == 1

    def test_scan_missing_optional_dependency_raises(self, tmp_path):
        """Scan should raise if optional deps are missing and nothing else loads."""
        path = tmp_path / "missing_optional.opt"
        path.write_text("placeholder")

        msg = "found files that can be read if additional packages"
        with pytest.raises(MissingOptionalDependencyError, match=msg):
            dc.scan(path)

    def test_scan_missing_optional_dependency_warns_with_other_outputs(self, tmp_path):
        """Scan should warn if optional deps are missing but other files load."""
        missing_path = tmp_path / "missing_optional.opt"
        missing_path.write_text("placeholder")
        readable_path = tmp_path / "fallback_scan.h5"
        readable_path.write_text("placeholder")

        msg = "found files that can be read if additional packages"
        with pytest.warns(UserWarning, match=msg):
            out = dc.scan([missing_path, readable_path])

        assert len(out) == 1
        assert out[0].source_format == _ReadOnlySummaryFormatter.name

    def test_local_upath_file_interfaces(self, terra15_v6_path):
        """Ensure core file IO accepts local UPath inputs."""
        path = UPath(terra15_v6_path)
        file_format, file_version = dc.get_format(path)
        assert file_format
        assert file_version
        assert len(dc.scan(path)) == 1
        assert len(dc.read(path)) == 1

    def test_updated_after_warns_when_remote_mtime_missing(self, monkeypatch):
        """Timestamp filtering should warn and continue on unsupported backends."""
        fiber_io = _FiberFormatTestV1()
        resource = UPath("memory://dascore/fiberio/mtime.txt")
        resource.write_text("x")
        path_type = type(resource)
        original_stat = path_type.stat

        def _stat(self, *args, **kwargs):
            raise OSError("no mtime")

        monkeypatch.setattr(path_type, "stat", _stat)
        with pytest.warns(UserWarning, match="does not expose reliable mtime"):
            assert fiber_io._updated_after(resource, 1) is True
        monkeypatch.setattr(path_type, "stat", original_stat)

    def test_local_stat_failure_returns_false(self, monkeypatch, tmp_path):
        """Local stat failures should conservatively skip timestamp-matched scans.

        Remote backends sometimes cannot provide mtimes at all, so DASCore warns
        and continues scanning in that case. For local files, a failed ``stat``
        usually means the path disappeared or became unreadable, so
        ``_updated_after`` should return ``False`` instead of treating that as an
        implicit update.
        """
        fiber_io = _FiberFormatTestV1()
        path = tmp_path / "mtime.txt"
        path.write_text("x")

        def _stat(_self, *args, **kwargs):
            raise OSError("no stat")

        monkeypatch.setattr(Path, "stat", _stat)
        with suppress_warnings(action="always", record=True) as record:
            assert fiber_io._updated_after(path, 1) is False
        assert not record


class TestReloadableSourcePath:
    """Tests for reloading source path extraction."""

    def test_io_resource_manager_source(self, tmp_path):
        """IOResourceManager candidates should resolve to their source path."""
        path = tmp_path / "example.txt"
        path.write_text("x")
        manager = IOResourceManager(path)
        out = _get_reloadable_source_path(manager)
        assert isinstance(out, UPath)
        assert out == UPath(path)

    def test_can_raise(self):
        """
        Scan, when called from a FiberIO, should be able to raise if
        type coercion fails.
        """
        fio = _FiberImplementer()
        bad_input = _FiberFormatTestV1()
        with pytest.raises(NotImplementedError):
            fio.scan(bad_input)

    def test_bad_checksum(self, monkeypatch, terra15_v6_path):
        """Test for when format is identified but can't read part of file #346"""
        # Monkey patch scan to raise OSError. This simulates observed behavior.
        fname, ver = FiberIO.manager._get_format(path=terra15_v6_path)
        fiber_io = FiberIO.manager.get_fiberio(format=fname, version=ver)

        def raise_os_error(*args, **kwargs):
            raise OSError("Simulated OS issue")

        monkeypatch.setattr(fiber_io, "scan", raise_os_error)

        # Ensure scanning doesn't raise and warns
        msg = "Failed to scan"
        with pytest.warns(UserWarning, match=msg):
            scan = dc.scan(terra15_v6_path)
        assert not len(scan)

    def test_remote_cache_error_is_not_swallowed(self, monkeypatch, terra15_v6_path):
        """Remote cache policy errors during scan should propagate to callers."""
        fname, ver = FiberIO.manager._get_format(path=terra15_v6_path)
        fiber_io = FiberIO.manager.get_fiberio(format=fname, version=ver)

        def raise_remote_cache_error(*args, **kwargs):
            raise RemoteCacheError("metadata cache blocked")

        monkeypatch.setattr(fiber_io, "scan", raise_remote_cache_error)

        with pytest.raises(RemoteCacheError, match="metadata cache blocked"):
            dc.scan(terra15_v6_path)

    def test_scan_legacy_patch_attrs_raises(self, monkeypatch, terra15_v6_path):
        """FiberIO returning PatchAttrs should now fail loudly."""
        fname, ver = FiberIO.manager._get_format(path=terra15_v6_path)
        fiber_io = FiberIO.manager.get_fiberio(format=fname, version=ver)

        def return_patch_attrs(*args, **kwargs):
            return [dc.PatchAttrs(tag="legacy")]

        monkeypatch.setattr(fiber_io, "scan", return_patch_attrs)

        with pytest.raises(ValueError, match=r"PatchAttrs from FiberIO\.scan"):
            dc.scan(terra15_v6_path)

    def test_default_fiberio_scan_uses_reloadable_source_path(self, tmp_path):
        """Default FiberIO.scan should return structured scan payloads."""
        path = tmp_path / "fallback_scan.h5"
        path.write_text("placeholder")
        fio = _ReadOnlySummaryFormatter()

        out = fio.scan(path)

        assert len(out) == 1
        assert isinstance(out[0], dict)
        assert "source_path" not in out[0]
        assert "source_format" not in out[0]
        assert "source_version" not in out[0]
        assert not out[0]["source_patch_id"]

    def test_dc_scan_adds_source_metadata_to_raw_fiberio_scan(self, tmp_path):
        """dc.scan should add path/format/version on top of raw formatter scan."""
        path = tmp_path / "fallback_scan.h5"
        path.write_text("placeholder")

        raw = _ReadOnlySummaryFormatter().scan(path)
        assert len(raw) == 1
        assert "source_path" not in raw[0]
        assert "source_format" not in raw[0]
        assert "source_version" not in raw[0]

        out = dc.scan(path)
        assert len(out) == 1
        assert isinstance(out[0], dc.PatchSummary)
        assert str(out[0].source_path) == str(path)
        assert out[0].source_format == _ReadOnlySummaryFormatter.name
        assert out[0].source_version == _ReadOnlySummaryFormatter.version
        assert "path" not in out[0].attrs.model_dump()
        assert "file_format" not in out[0].attrs.model_dump()
        assert "file_version" not in out[0].attrs.model_dump()

    def test_default_fiberio_scan_multi_patch_does_not_set_source_patch_id(
        self, tmp_path
    ):
        """Default scan should not invent source ids for multi-patch readers."""
        path = tmp_path / "fallback_scan.h5"
        path.write_text("placeholder")
        fio = _ReadOnlySummaryFormatter()

        def read_two_patches(resource: Path, **kwargs) -> SpoolType:
            patches = [
                dc.get_example_patch().update_attrs(tag="first"),
                dc.get_example_patch().update_attrs(tag="second"),
            ]
            return dc.spool(patches)

        fio.read = read_two_patches  # type: ignore[method-assign]
        out = fio.scan(path)

        assert len(out) == 2
        assert not any(summary["source_patch_id"] for summary in out)

    def test_keyboard_interrupt(self, monkeypatch):
        """Ensure a keyboard interrupt works when progress bar is going"""

        class Progress(prog.Progress):
            """A dummy class for progress that just raises interrupt."""

            def track(self, *args, **kwargs):
                """Track progress."""
                raise KeyboardInterrupt("test interrupt")

        # Switch off debug to force progress bar, then make contents to scan.
        contents = list(dc.examples.get_example_spool(length=22))

        with set_config(debug=False):
            with pytest.raises(KeyboardInterrupt, match="test interrupt"):
                dc.scan(contents, progress=Progress())


class TestScanToDF:
    """Tests for scanning to dataframes."""

    def test_input_dataframe(self, random_spool):
        """Ensure a dataframe returns a dataframe."""
        df = random_spool.get_contents()
        out = dc.scan_to_df(df)
        assert out is df

    def test_spool_dataframe(self, random_directory_spool):
        """Ensure scan_to_df just gets the dataframe from the spool."""
        expected = random_directory_spool.get_contents()
        out = dc.scan_to_df(random_directory_spool)
        assert out.equals(expected)


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
