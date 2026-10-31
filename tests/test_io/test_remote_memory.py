"""Deterministic remote-path tests using an in-memory filesystem."""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest
from upath import UPath

import dascore as dc
from dascore.config import config_context
from dascore.exceptions import (
    InvalidSpoolError,
    RemoteCacheError,
    UnknownFiberFormatError,
)
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
    with config_context(remote_cache_dir=tmp_path / "remote_cache"):
        clear_remote_file_cache()
        yield
        clear_remote_file_cache()


def _write_netcdf_v0(path: Path) -> Path:
    """Write a NetCDF file with no CF version, claimed only by a fallback."""
    pytest.importorskip("h5netcdf")
    pytest.importorskip("xarray")
    dc.write(dc.get_example_patch("random_das"), path, "NETCDF_CF")
    # H5Simple also claims a root "data" beside "time".
    with h5py.File(path, "r+") as handle:
        handle.move("data", "strain")
        del handle.attrs["Conventions"]
    return path


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

    def test_spool_directory_with_metadata_xml_rejected(self):
        """A remote folder a directory reader cannot probe gives the same error."""
        root = UPath("memory://dascore/remote_xml_dir")
        (root / "metadata.xml").write_text("<a/>")
        with pytest.raises(InvalidSpoolError, match="local filesystem"):
            dc.spool(root)

    def test_spool_unknown_file(self):
        """A remote file no reader claims is refused as a local one is."""
        path = UPath("memory://dascore/remote_unknown/file.txt")
        path.write_text("x")
        with (
            config_context(allow_remote_cache_for_metadata=True),
            pytest.raises(UnknownFiberFormatError),
        ):
            dc.spool(path)


class TestMemoryRemoteMetadataAccess:
    """Tests for remote-first metadata access on in-memory filesystems."""

    @pytest.mark.parametrize(
        ("fetch_name", "expected"),
        [
            # One HDF5 reader and one which walks a plain binary file: what
            # this is about is the cache, not the readers.
            ("h5_simple_2.h5", ("H5Simple", "1")),
            ("sample_tdms_file_v4713.tdms", ("TDMS", "4713")),
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
        assert dc.get_format(memory_dasdae_path) == ("DASDAE", "2")
        assert not list(get_remote_cache_path().rglob(memory_dasdae_path.name))

    def test_dasdae_scan_avoids_local_cache(self, memory_dasdae_path):
        """DASDAE scan should not materialize a local cache file."""
        attrs = dc.scan(memory_dasdae_path)
        assert len(attrs) == 1
        assert attrs[0].source_format == "DASDAE"
        assert not list(get_remote_cache_path().rglob(memory_dasdae_path.name))

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("v0.nc", ("NETCDF_CF", "0")), ("h5_simple_2", ("H5Simple", "1"))],
    )
    def test_detection_needs_no_download(self, tmp_path, name, expected):
        """
        Files claimed only after the MiniSEED and SEG-Y readers are asked are
        detected and scanned without a download.
        """
        if name == "v0.nc":
            source = _write_netcdf_v0(tmp_path / name)
        else:
            source = Path(fetch("h5_simple_2.h5"))
        path = _copy_file_to_memory(source, UPath(f"memory://dascore/late/{name}"))
        assert dc.get_format(path) == expected
        attrs = dc.scan(path)
        assert {(x.source_format, x.source_version) for x in attrs} == {expected}
        assert not list(get_remote_cache_path().rglob("*"))

    def test_mseed_streams(self, memory_fetch_copy):
        """MiniSEED is detected, scanned and read without a download."""
        source, path = memory_fetch_copy("etna_9n_3chan_10s.mseed", "mseed")
        assert dc.get_format(path) == dc.get_format(source)
        scan = dc.scan(path)
        assert [x.coords for x in scan] == [x.coords for x in dc.scan(source)]
        assert list(dc.read(path)) == list(dc.read(source))
        assert not list(get_remote_cache_path().rglob("*"))

    def test_segy_scan_needs_metadata_cache(self, memory_fetch_copy):
        """
        SEG-Y is detected from its headers, but scanning it needs a local
        file, so it waits for the metadata cache opt-in.
        """
        source, path = memory_fetch_copy("small_channel_patch.sgy", "segy")
        assert dc.get_format(path) == dc.get_format(source)
        assert not list(get_remote_cache_path().rglob("*"))
        with pytest.raises(RemoteCacheError, match="allow_remote_cache_for"):
            dc.scan(path)
        with config_context(
            allow_remote_cache_for_metadata=True, warn_on_remote_cache=False
        ):
            assert len(dc.scan(path)) == 1
