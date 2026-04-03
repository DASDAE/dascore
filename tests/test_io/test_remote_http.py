"""Remote HTTP path tests for universal_pathlib support."""

from __future__ import annotations

from urllib.request import Request, urlopen

import pytest
from upath import UPath

import dascore as dc
from dascore.config import set_config
from dascore.exceptions import InvalidSpoolError, RemoteCacheError
from dascore.utils.misc import suppress_warnings
from dascore.utils.remote_io import clear_remote_file_cache, get_remote_cache_path

pytestmark = pytest.mark.network


@pytest.fixture(autouse=True)
def suppress_expected_remote_cache_warnings(request):
    """Keep expected remote-cache download warnings out of test output."""
    # This test is supposed to raise warning.
    if (
        request.node.name
        == "test_http_hdf5_fallback_warns_once_and_reuses_cached_local_copy"
    ):
        yield
        return
    msg = "Downloading remote file .* to local cache at .*"
    with suppress_warnings(message=msg):
        yield


@pytest.fixture(autouse=True)
def isolated_remote_cache(tmp_path):
    """Use one isolated remote cache per test to avoid suite-order slowdowns."""
    with set_config(remote_cache_dir=tmp_path / "remote_cache"):
        clear_remote_file_cache()
        yield
        clear_remote_file_cache()


class TestHTTPRead:
    """Tests for reading DAS files over a local HTTP backend."""

    def test_read_dasdae_from_http(self, http_regression_das_path):
        """Ensure a top-level DASDAE file can be read from the HTTP tree."""
        path = http_regression_das_path / "example_dasdae_event_1.h5"
        spool = dc.read(path)
        assert len(spool) > 0
        assert spool[0].dims == ("distance", "time")

    def test_read_nested_dasdae_from_http(self, http_regression_das_path):
        """Ensure a nested DASDAE file can be read from the HTTP tree."""
        path = http_regression_das_path / "nested" / "example_dasdae_event_2.h5"
        spool = dc.read(path)
        assert len(spool) > 0
        assert spool[0].dims == ("distance", "time")


class TestHTTPScan:
    """Tests for scanning DAS files from a local HTTP backend."""

    def test_scan_top_level_dasdae_from_http(self, http_regression_das_path):
        """Ensure a top-level DASDAE file can be scanned from the HTTP tree."""
        path = http_regression_das_path / "example_dasdae_event_1.h5"
        attrs = dc.scan(path)
        assert len(attrs) > 0
        assert attrs[0].source_format == "DASDAE"
        assert isinstance(attrs[0].source_path, UPath)
        assert attrs[0].source_path == path

    def test_scan_nested_dasdae_from_http(self, http_regression_das_path):
        """Ensure a nested DASDAE file can be scanned from the HTTP tree."""
        path = http_regression_das_path / "nested" / "example_dasdae_event_2.h5"
        attrs = dc.scan(path)
        assert len(attrs) > 0
        assert attrs[0].source_format == "DASDAE"
        assert isinstance(attrs[0].source_path, UPath)
        assert attrs[0].source_path == path

    def test_scan_http_string_path_promotes_to_upath(self, http_regression_das_path):
        """HTTP URL strings should be preserved as UPath on summaries."""
        path = http_regression_das_path / "example_dasdae_event_1.h5"
        attrs = dc.scan(str(path))
        assert len(attrs) > 0
        assert isinstance(attrs[0].source_path, UPath)
        assert attrs[0].source_path == path

    def test_scan_http_directory_traverses_nested_files(self, http_regression_das_path):
        """Ensure directory scans recurse through HTTP directory listings."""
        attrs = dc.scan(http_regression_das_path)
        paths = {str(x.source_path) for x in attrs}
        assert str(http_regression_das_path / "example_dasdae_event_1.h5") in paths
        assert (
            str(http_regression_das_path / "nested" / "example_dasdae_event_2.h5")
            in paths
        )


class TestHTTPFormatAndSpool:
    """Tests for format detection and spool behavior over HTTP."""

    def test_get_format_from_http(self, http_regression_das_path):
        """Ensure format detection works for HTTP-served DASDAE files."""
        path = http_regression_das_path / "example_dasdae_event_1.h5"
        out = dc.get_format(path)
        assert out == ("DASDAE", "1")

    def test_http_hdf5_get_format_requires_metadata_cache_opt_in(
        self, http_regression_das_path, ensure_http_regression_file
    ):
        """Plain HTTP HDF5 metadata access should fail unless opted in."""
        ensure_http_regression_file("prodml_2.1.h5")
        path = http_regression_das_path / "prodml_2.1.h5"
        with pytest.raises(RemoteCacheError, match="allow_remote_cache_for_metadata"):
            dc.get_format(path)

    # TODO(#645): Remove this temporary timeout once the intermittent
    # full-suite hang on the HTTP HDF5 fallback path is fully understood.
    @pytest.mark.timeout(30)
    def test_http_hdf5_fallback_warns_once_and_reuses_cached_local_copy(
        self, http_regression_das_path, ensure_http_regression_file
    ):
        """First local cache materialization should warn, then reuse the artifact."""
        fname = "prodml_2.1.h5"
        ensure_http_regression_file(fname)
        path = http_regression_das_path / fname
        with set_config(
            allow_remote_cache_for_metadata=True, warn_on_remote_cache=True
        ):
            with pytest.warns(UserWarning, match="Downloading remote file"):
                dc.get_format(path)
        cached_files = list(get_remote_cache_path().rglob(fname))
        assert len(cached_files) <= 1
        assert len(dc.read(path))

        cached_files_2 = list(get_remote_cache_path().rglob(fname))
        assert len(cached_files_2) == 1
        assert cached_files_2[0].exists()
        assert dc.read(path)
        cached_files_3 = list(get_remote_cache_path().rglob(fname))
        assert cached_files_3 == cached_files_2

    def test_http_range_server_supports_partial_reads(
        self, http_range_das_path, ensure_http_fetch_file
    ):
        """The ranged HTTP fixture should respond with partial content."""
        ensure_http_fetch_file("prodml_2.1.h5")
        url = str(http_range_das_path / "prodml_2.1.h5")
        request = Request(url, headers={"Range": "bytes=0-15"})
        with urlopen(request) as response:
            data = response.read()
            assert response.status == 206
            assert response.headers["Accept-Ranges"] == "bytes"
            assert response.headers["Content-Range"].startswith("bytes 0-15/")
            assert len(data) == 16

    def test_http_range_hdf5_read_succeeds(
        self, http_range_das_path, ensure_http_fetch_file
    ):
        """Range-capable HTTP servers should support DASCore HDF5 reads."""
        ensure_http_fetch_file("prodml_2.1.h5")
        path = http_range_das_path / "prodml_2.1.h5"
        assert dc.get_format(path) == ("PRODML", "2.1")
        assert dc.read(path)
        assert not list(get_remote_cache_path().rglob("prodml_2.1.h5"))

    def test_spool_file_path(self, http_regression_das_path):
        """A remote HTTP file should still produce a file-backed spool."""
        path = http_regression_das_path / "nested" / "example_dasdae_event_2.h5"
        spool = dc.spool(path)
        assert len(spool) == 1
        assert spool[0].dims == ("distance", "time")

    def test_spool_directory_rejected(self, http_regression_das_path):
        """Remote HTTP directories should remain unsupported for spooling."""
        with pytest.raises(InvalidSpoolError, match="local filesystem"):
            dc.spool(http_regression_das_path)
