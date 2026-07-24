"""Common remote IO tests for localhost HTTP-backed paths."""

from __future__ import annotations

import sys

import pytest

import dascore as dc
from dascore.config import get_config, set_config
from dascore.utils.misc import suppress_warnings
from tests.test_io._common_io_test_utils import (
    get_flat_io_test,
    get_representative_io_test,
    skip_missing,
    skip_on_timeout,
    skip_timeout,
)
from tests.test_io.test_common_io import COMMON_IO_READ_TESTS

# The localhost HTTP + fsspec/aiohttp streaming path can intermittently deadlock
# on Windows (the async read stalls while h5py probes remote HDF5 metadata),
# which pytest-timeout then aborts. This is a known Windows flakiness in that
# fallback path, not a DASCore logic issue; see the win32 skip in
# test_remote_http.py. Skip the localhost-HTTP matrix on Windows to keep CI
# deterministic while still exercising it fully on Linux and macOS.
pytestmark = [
    pytest.mark.network,
    pytest.mark.timeout(30),
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="Flaky localhost-HTTP fsspec/aiohttp streaming on Windows.",
    ),
]

# Sintela protobuf walks its MTLV envelope with three small sequential reads
# per record (magic, header, payload), so a modest file issues hundreds of
# reads. That is fine locally and over memory://, but each read becomes a
# request on the localhost-HTTP range-streaming path, which blows the timeouts
# below. Remote coverage for this format stays at the memory:// level.
REMOTE_COMMON_IO_READ_TESTS = {
    io: fetch_names
    for io, fetch_names in COMMON_IO_READ_TESTS.items()
    if io.name != "Sintela_Protobuf"
}
REMOTE_GET_FORMAT_CASES = get_flat_io_test(REMOTE_COMMON_IO_READ_TESTS)
REMOTE_REPRESENTATIVE_CASES = get_representative_io_test(REMOTE_COMMON_IO_READ_TESTS)

# The localhost HTTP/fsspec/h5py streaming path can intermittently stall while
# probing remote HDF5 metadata (see the TODO in test_remote_http.py). Bound each
# remote operation below the 30s pytest-timeout so a stall skips with a useful
# message instead of aborting the whole job as a hard timeout failure.
REMOTE_OP_TIMEOUT = 15


@pytest.fixture(autouse=True)
def suppress_expected_remote_cache_warnings():
    """Keep expected remote-cache download warnings out of test output."""
    with suppress_warnings(UserWarning):
        yield


@pytest.fixture(scope="module", autouse=True)
def isolated_remote_cache(tmp_path_factory):
    """Keep the common remote matrix in its own cache root.

    Uses the permanent config base (not a scoped ``config_context``) because a
    module-scoped fixture spans many tests; the scoped override belongs to a
    single call block, not a fixture that stays open across the module.
    """
    previous = get_config()
    set_config(
        remote_cache_dir=tmp_path_factory.mktemp("remote_common_cache"),
        allow_remote_cache_for_metadata=True,
    )
    try:
        yield
    finally:
        set_config(previous)


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
        with skip_missing(), skip_on_timeout(REMOTE_OP_TIMEOUT, "remote get_format"):
            out = dc.get_format(path)
        assert out == (io.name, io.version)


class TestRemoteRead:
    """Test remote reads against the local IO support matrix."""

    def test_read_returns_spools(self, remote_read_case):
        """Each remotely supported file should read into a spool."""
        _io, path = remote_read_case
        with skip_missing(), skip_on_timeout(REMOTE_OP_TIMEOUT, "remote read"):
            out = dc.read(path)
        assert isinstance(out, dc.BaseSpool)
        assert len(out) > 0
        assert all(isinstance(x, dc.Patch) for x in out)


class TestRemoteScan:
    """Test remote scans against the local IO support matrix."""

    def test_scan_has_source_metadata(self, remote_scan_case):
        """Public scans of remote files should retain source metadata."""
        io, path = remote_scan_case
        with skip_missing(), skip_on_timeout(REMOTE_OP_TIMEOUT, "remote scan"):
            summary_list = dc.scan(path)
        assert len(summary_list) > 0
        for summary in summary_list:
            assert str(summary.source_path) == str(path)
            assert summary.source_format == io.name
            assert summary.source_version == io.version
