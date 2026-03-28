"""Deterministic remote-path tests using an in-memory filesystem."""

from __future__ import annotations

from pathlib import Path

import pytest
from upath import UPath

import dascore as dc
from dascore.exceptions import InvalidSpoolError
from dascore.utils.downloader import fetch


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


class TestMemoryRemoteScan:
    """Tests for scanning remote files from an in-memory backend."""

    def test_scan_prodml(self, memory_prodml_path):
        """ProdML files should be scannable through a remote UPath."""
        attrs = dc.scan(memory_prodml_path)
        assert len(attrs) > 0
        assert attrs[0].file_format == "PRODML"
        assert attrs[0].get_coord_summary("time").min is not None

    def test_scan_dasdae(self, memory_dasdae_path):
        """DASDAE files should be scannable through a remote UPath."""
        attrs = dc.scan(memory_dasdae_path)
        assert len(attrs) == 1
        assert attrs[0].file_format == "DASDAE"


class TestMemoryRemoteSpool:
    """Tests for spool behavior on remote memory-backed paths."""

    def test_spool_file_path(self, memory_dasdae_path):
        """A remote file path should still produce a file-backed spool."""
        spool = dc.spool(memory_dasdae_path)
        assert len(spool) == 1
        assert spool[0].dims == ("distance", "time")

    def test_spool_directory_rejected(self):
        """Remote directories should remain unsupported for directory spools."""
        root = UPath("memory://dascore/remote_dir")
        (root / "file.txt").write_text("x")
        with pytest.raises(InvalidSpoolError, match="local filesystem"):
            dc.spool(root)
