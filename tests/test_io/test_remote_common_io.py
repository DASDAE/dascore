"""Common remote IO tests for localhost HTTP-backed paths."""

from __future__ import annotations

import pytest

import dascore as dc
from dascore.config import set_config
from dascore.utils.misc import suppress_warnings
from tests.test_io._common_io_test_utils import (
    get_flat_io_test,
    get_representative_io_test,
    skip_missing,
    skip_timeout,
)
from tests.test_io.test_common_io import COMMON_IO_READ_TESTS

pytestmark = [pytest.mark.network, pytest.mark.timeout(30)]

REMOTE_GET_FORMAT_CASES = get_flat_io_test(COMMON_IO_READ_TESTS)
REMOTE_REPRESENTATIVE_CASES = get_representative_io_test(COMMON_IO_READ_TESTS)


@pytest.fixture(autouse=True)
def suppress_expected_remote_cache_warnings():
    """Keep expected remote-cache download warnings out of test output."""
    with suppress_warnings(UserWarning):
        yield


@pytest.fixture(scope="module", autouse=True)
def isolated_remote_cache(tmp_path_factory):
    """Keep the common remote matrix in its own cache root."""
    with set_config(
        remote_cache_dir=tmp_path_factory.mktemp("remote_common_cache"),
        allow_remote_cache_for_metadata=True,
    ):
        yield


def _get_remote_case(fetch_name: str, to_http_range_path):
    """Return a range-capable HTTP path for one fetched local test file."""
    with skip_timeout():
        local_path = dc.utils.downloader.fetch(fetch_name)
    return to_http_range_path(local_path)


@pytest.fixture(
    scope="session",
    params=REMOTE_GET_FORMAT_CASES,
    ids=lambda case: f"{case[0].name}-{case[0].version}-{case[1]}",
)
def remote_get_format_case(request, to_http_range_path):
    """Return one remote get-format case per IO/file pairing."""
    io, fetch_name = request.param
    return io, _get_remote_case(fetch_name, to_http_range_path)


@pytest.fixture(scope="session", params=REMOTE_REPRESENTATIVE_CASES)
def remote_read_case(request, to_http_range_path):
    """Return one representative remote read case per FiberIO entry."""
    io, fetch_name = request.param
    return io, _get_remote_case(fetch_name, to_http_range_path)


@pytest.fixture(scope="session", params=REMOTE_REPRESENTATIVE_CASES)
def remote_scan_case(request, to_http_range_path):
    """Return one representative remote scan case per FiberIO entry."""
    io, fetch_name = request.param
    return io, _get_remote_case(fetch_name, to_http_range_path)


class TestRemoteGetFormat:
    """Test remote format detection against the local IO support matrix."""

    def test_expected_version(self, remote_get_format_case):
        """Each IO should identify its own remote test fixture."""
        io, path = remote_get_format_case
        with skip_missing():
            out = dc.get_format(path)
        assert out == (io.name, io.version)


class TestRemoteRead:
    """Test remote reads against the local IO support matrix."""

    def test_read_returns_spools(self, remote_read_case):
        """Each remotely supported file should read into a spool."""
        _io, path = remote_read_case
        with skip_missing():
            out = dc.read(path)
        assert isinstance(out, dc.BaseSpool)
        assert len(out) > 0
        assert all(isinstance(x, dc.Patch) for x in out)


class TestRemoteScan:
    """Test remote scans against the local IO support matrix."""

    def test_scan_has_source_metadata(self, remote_scan_case):
        """Public scans of remote files should retain source metadata."""
        io, path = remote_scan_case
        with skip_missing():
            summary_list = dc.scan(path)
        assert len(summary_list) > 0
        for summary in summary_list:
            assert str(summary.source_path) == str(path)
            assert summary.source_format == io.name
            assert summary.source_version == io.version
