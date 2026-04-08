"""Deterministic remote-path tests using an in-memory filesystem."""

from __future__ import annotations

from pathlib import Path

import pytest
from upath import UPath

import dascore as dc
from dascore.config import set_config
from dascore.exceptions import InvalidSpoolError
from dascore.utils.downloader import fetch
from dascore.utils.remote_io import clear_remote_file_cache, get_remote_cache_path


def _copy_file_to_memory(source: Path, dest: UPath) -> UPath:
    """Copy a local file into the in-memory remote filesystem."""
    with source.open("rb") as src, dest.open("wb") as dst:
        dst.write(src.read())
    return dest


@pytest.fixture()
def memory_prodml_path():
    """Return a remote memory-backed ProdML file path."""
    source = Path(fetch("prodml_2.1.h5"))
    dest = UPath("memory://dascore/remote/prodml_2.1.h5")
    return _copy_file_to_memory(source, dest)


@pytest.fixture()
def memory_fetch_copy():
    """Copy one fetched registry file into the in-memory filesystem."""

    def _copy(fetch_name: str, namespace: str) -> tuple[Path, UPath]:
        source = Path(fetch(fetch_name))
        dest = UPath(f"memory://dascore/{namespace}/{source.name}")
        return source, _copy_file_to_memory(source, dest)

    return _copy


@pytest.fixture(autouse=True)
def isolated_remote_cache(tmp_path):
    """Use one isolated remote cache per test to avoid cross-test cleanup cost."""
    with set_config(remote_cache_dir=tmp_path / "remote_cache"):
        clear_remote_file_cache()
        yield
        clear_remote_file_cache()


@pytest.fixture()
def memory_dasdae_path(tmp_path):
    """Return a remote memory-backed DASDAE file path."""
    source = tmp_path / "example_dasdae_event_1.h5"
    dc.write(dc.get_example_patch(), source, "DASDAE")
    dest = UPath("memory://dascore/remote/example_dasdae_event_1.h5")
    return _copy_file_to_memory(source, dest)


class TestMemoryRemoteRead:
    """Tests for reading remote files from an in-memory backend."""

    def test_read_prodml(self, memory_prodml_path):
        """ProdML files should be readable through a remote UPath."""
        spool = dc.read(memory_prodml_path)
        assert len(spool) > 0
        assert spool[0].dims

    def test_read_dasdae(self, memory_dasdae_path):
        """DASDAE files should be readable through a remote UPath."""
        spool = dc.read(memory_dasdae_path)
        assert len(spool) == 1
        assert spool[0].dims == ("distance", "time")


class TestMemoryRemoteWrite:
    """Tests for writing remote files to an in-memory backend."""

    def test_write_pickle_round_trip(self):
        """Pickle writes should support remote UPath destinations."""
        path = UPath("memory://dascore/remote/write_patch.pkl")
        patch = dc.get_example_patch()
        dc.write(patch, path, "pickle")
        assert path.exists()
        spool = dc.read(path)
        assert len(spool) == 1
        assert spool[0].dims == patch.dims

    def test_write_dasdae_round_trip(self):
        """DASDAE writes should support remote UPath destinations."""
        path = UPath("memory://dascore/remote/write_patch.h5")
        patch = dc.get_example_patch()
        dc.write(patch, path, "DASDAE")
        assert path.exists()
        spool = dc.read(path)
        assert len(spool) == 1
        assert spool[0].dims == patch.dims


class TestMemoryRemoteScan:
    """Tests for scanning remote files from an in-memory backend."""

    def test_scan_prodml(self, memory_prodml_path):
        """ProdML files should be scannable through a remote UPath."""
        attrs = dc.scan(memory_prodml_path)
        assert len(attrs) > 0
        assert attrs[0].source_format == "PRODML"
        assert isinstance(attrs[0].source_path, UPath)
        assert attrs[0].source_path == memory_prodml_path
        assert attrs[0].get_coord_summary("time").min is not None

    def test_scan_dasdae(self, memory_dasdae_path):
        """DASDAE files should be scannable through a remote UPath."""
        attrs = dc.scan(memory_dasdae_path)
        assert len(attrs) == 1
        assert attrs[0].source_format == "DASDAE"
        assert isinstance(attrs[0].source_path, UPath)
        assert attrs[0].source_path == memory_dasdae_path

    def test_scan_remote_directory_string_path(self, memory_dasdae_path):
        """Remote directory strings should recurse via UPath-aware scanning."""
        root = str(memory_dasdae_path.parent)
        attrs = dc.scan(root)
        paths = {str(x.source_path) for x in attrs}
        assert str(memory_dasdae_path) in paths

    def test_scan_remote_directory_timestamp_warns_once_without_mtime(
        self, monkeypatch, tmp_path
    ):
        """Remote directory scans should warn once when mtime is unavailable."""
        source = tmp_path / "example_dasdae_event_1.h5"
        dc.write(dc.get_example_patch(), source, "DASDAE")
        path = _copy_file_to_memory(
            source, UPath("memory://dascore/remote_timestamp/example_dasdae_event_1.h5")
        )
        root = path.parent
        upath_type = type(path)
        original_stat = upath_type.stat

        def _stat(self, *args, **kwargs):
            if self.name == path.name:
                raise OSError("mtime unavailable")
            return original_stat(self, *args, **kwargs)

        monkeypatch.setattr(upath_type, "stat", _stat)

        with pytest.warns(
            UserWarning, match="does not expose reliable mtime"
        ) as record:
            attrs = dc.scan(root, timestamp=0)

        assert len(attrs) == 1
        assert attrs[0].source_path == path
        assert len(record) >= 1

    def test_scan_remote_string_path_promotes_to_upath(self, memory_dasdae_path):
        """Remote URL strings should be preserved as UPath on summaries."""
        attrs = dc.scan(str(memory_dasdae_path))
        assert len(attrs) == 1
        assert isinstance(attrs[0].source_path, UPath)
        assert attrs[0].source_path == memory_dasdae_path


class TestMemoryRemoteSpool:
    """Tests for spool behavior on remote memory-backed paths."""

    def test_spool_file_path(self, memory_dasdae_path):
        """A remote file path should still produce a file-backed spool."""
        spool = dc.spool(memory_dasdae_path)
        assert len(spool) == 1
        assert spool[0].dims == ("distance", "time")

    def test_spool_file_string_path(self, memory_dasdae_path):
        """A remote file URL string should route through UPath handling."""
        spool = dc.spool(str(memory_dasdae_path))
        assert len(spool) == 1
        assert spool[0].dims == ("distance", "time")

    def test_spool_directory_rejected(self):
        """Remote directories should remain unsupported for directory spools."""
        root = UPath("memory://dascore/remote_dir")
        (root / "file.txt").write_text("x")
        with pytest.raises(InvalidSpoolError, match="local filesystem"):
            dc.spool(root)


class TestMemoryRemoteMetadataAccess:
    """Tests for remote-first metadata access on in-memory filesystems."""

    @pytest.mark.parametrize(
        ("fetch_name", "expected"),
        [
            ("h5_simple_2.h5", ("H5Simple", "1")),
            ("sample_tdms_file_v4713.tdms", ("TDMS", "4713")),
            ("DASDMSShot00_20230328155653619.das", ("sentek", "5")),
            ("sintela_binary_v3_test_1.raw", ("Sintela_Binary", "3")),
            ("sintela_protobuf_1.pb", ("Sintela_Protobuf", "1")),
        ],
    )
    def test_get_format_avoids_local_cache(
        self, fetch_name, expected, memory_fetch_copy
    ):
        """Remote-first get_format paths should not materialize local cache files."""
        source, path = memory_fetch_copy(fetch_name, "meta")
        assert dc.get_format(path) == expected
        assert not list(get_remote_cache_path().rglob(source.name))

    @pytest.mark.parametrize(
        ("fetch_name", "expected"),
        [
            ("h5_simple_2.h5", ("H5Simple", "1")),
            ("sample_tdms_file_v4713.tdms", ("TDMS", "4713")),
            ("DASDMSShot00_20230328155653619.das", ("sentek", "5")),
            ("sintela_binary_v3_test_1.raw", ("Sintela_Binary", "3")),
            ("sintela_protobuf_1.pb", ("Sintela_Protobuf", "1")),
        ],
    )
    def test_scan_avoids_local_cache(self, fetch_name, expected, memory_fetch_copy):
        """Remote-first scans should not materialize local cache files."""
        source, path = memory_fetch_copy(fetch_name, "scan")
        attrs = dc.scan(path)
        assert len(attrs) > 0
        assert attrs[0].source_format == expected[0]
        assert attrs[0].source_version == expected[1]
        assert not list(get_remote_cache_path().rglob(source.name))

    def test_dasdae_get_format_avoids_local_cache(self, memory_dasdae_path):
        """DASDAE format detection should stay remote-first."""
        assert dc.get_format(memory_dasdae_path) == ("DASDAE", "1")
        assert not list(get_remote_cache_path().rglob(memory_dasdae_path.name))

    def test_dasdae_scan_avoids_local_cache(self, memory_dasdae_path):
        """DASDAE scan should not materialize a local cache file."""
        attrs = dc.scan(memory_dasdae_path)
        assert len(attrs) == 1
        assert attrs[0].source_format == "DASDAE"
        assert not list(get_remote_cache_path().rglob(memory_dasdae_path.name))
