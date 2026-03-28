"""Common remote IO tests for localhost HTTP-backed paths."""

from __future__ import annotations

import pytest

import dascore as dc
from tests.test_io._common_io_test_utils import (
    get_flat_io_test,
    skip_missing,
    skip_timeout,
)
from tests.test_io.test_common_io import COMMON_IO_READ_TESTS

pytestmark = pytest.mark.network


@pytest.fixture(scope="session", params=get_flat_io_test(COMMON_IO_READ_TESTS))
def remote_io_path_tuple(request, to_http_path):
    """Return an IO instance with the matching HTTP-backed path."""
    io, fetch_name = request.param
    with skip_timeout():
        local_path = dc.utils.downloader.fetch(fetch_name)
    return io, to_http_path(local_path)


class TestRemoteGetFormat:
    """Test remote format detection against the local IO support matrix."""

    def test_expected_version(self, remote_io_path_tuple):
        """Each IO should identify its own remote test fixtures."""
        io, path = remote_io_path_tuple
        with skip_missing():
            out = dc.get_format(path)
        assert out == (io.name, io.version)


class TestRemoteRead:
    """Test remote reads against the local IO support matrix."""

    def test_read_returns_spools(self, remote_io_path_tuple):
        """Each remotely supported file should read into a spool."""
        _io, path = remote_io_path_tuple
        with skip_missing():
            out = dc.read(path)
        assert isinstance(out, dc.BaseSpool)
        assert len(out) > 0
        assert all(isinstance(x, dc.Patch) for x in out)


class TestRemoteScan:
    """Test remote scans against the local IO support matrix."""

    def test_scan_has_source_metadata(self, remote_io_path_tuple):
        """Public scans of remote files should retain source metadata."""
        io, path = remote_io_path_tuple
        with skip_missing():
            summary_list = dc.scan(path)
        assert len(summary_list) > 0
        for summary in summary_list:
            assert str(summary.path) == str(path)
            assert summary.file_format == io.name
            assert summary.file_version == io.version
