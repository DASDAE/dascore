"""Remote HTTP path tests for universal_pathlib support."""

from __future__ import annotations

from urllib.request import Request, urlopen

import pytest

import dascore as dc
from dascore.exceptions import InvalidSpoolError
from dascore.utils.io import clear_remote_file_cache, get_remote_cache_path

pytestmark = pytest.mark.network


@pytest.fixture(autouse=True)
def clear_remote_cache():
    """Ensure remote cache state doesn't leak across HTTP tests."""
    clear_remote_file_cache()
    yield
    clear_remote_file_cache()


class TestHTTPRead:
    """Tests for reading DAS files over a local HTTP backend."""

    def test_read_dasdae_from_http(self, http_das_path):
        """Ensure a top-level DASDAE file can be read from the HTTP tree."""
        path = http_das_path / "example_dasdae_event_1.h5"
        spool = dc.read(path)
        assert len(spool) > 0
        assert spool[0].dims == ("distance", "time")

    def test_read_nested_dasdae_from_http(self, http_das_path):
        """Ensure a nested DASDAE file can be read from the HTTP tree."""
        path = http_das_path / "nested" / "example_dasdae_event_2.h5"
        spool = dc.read(path)
        assert len(spool) > 0
        assert spool[0].dims == ("distance", "time")


class TestHTTPScan:
    """Tests for scanning DAS files from a local HTTP backend."""

    def test_scan_top_level_dasdae_from_http(self, http_das_path):
        """Ensure a top-level DASDAE file can be scanned from the HTTP tree."""
        path = http_das_path / "example_dasdae_event_1.h5"
        attrs = dc.scan(path)
        assert len(attrs) > 0
        assert attrs[0].file_format == "DASDAE"

    def test_scan_nested_dasdae_from_http(self, http_das_path):
        """Ensure a nested DASDAE file can be scanned from the HTTP tree."""
        path = http_das_path / "nested" / "example_dasdae_event_2.h5"
        attrs = dc.scan(path)
        assert len(attrs) > 0
        assert attrs[0].file_format == "DASDAE"

    def test_scan_http_directory_traverses_nested_files(self, http_das_path):
        """Ensure directory scans recurse through HTTP directory listings."""
        attrs = dc.scan(http_das_path)
        paths = {str(x.path) for x in attrs}
        assert str(http_das_path / "example_dasdae_event_1.h5") in paths
        assert str(http_das_path / "nested" / "example_dasdae_event_2.h5") in paths


class TestHTTPFormatAndSpool:
    """Tests for format detection and spool behavior over HTTP."""

    def test_get_format_from_http(self, http_das_path):
        """Ensure format detection works for HTTP-served DASDAE files."""
        path = http_das_path / "example_dasdae_event_1.h5"
        out = dc.get_format(path)
        assert out == ("DASDAE", "1")

    def test_http_hdf5_fallback_reuses_cached_local_copy(self, http_das_path):
        """HDF5 files should fall back to one reusable cached local artifact."""
        path = http_das_path / "prodml_2.1.h5"
        assert dc.get_format(path) == ("PRODML", "2.1")
        cached_files = list(get_remote_cache_path().rglob("prodml_2.1.h5"))
        assert len(cached_files) <= 1
        assert dc.read(path)
        cached_files_2 = list(get_remote_cache_path().rglob("prodml_2.1.h5"))
        assert len(cached_files_2) == 1
        assert cached_files_2[0].exists()
        assert dc.read(path)
        cached_files_3 = list(get_remote_cache_path().rglob("prodml_2.1.h5"))
        assert cached_files_3 == cached_files_2

    def test_http_range_server_supports_partial_reads(self, http_range_das_path):
        """The ranged HTTP fixture should respond with partial content."""
        url = str(http_range_das_path / "prodml_2.1.h5")
        request = Request(url, headers={"Range": "bytes=0-15"})
        with urlopen(request) as response:
            data = response.read()
            assert response.status == 206
            assert response.headers["Accept-Ranges"] == "bytes"
            assert response.headers["Content-Range"].startswith("bytes 0-15/")
            assert len(data) == 16

    def test_http_range_hdf5_read_succeeds(self, http_range_das_path):
        """Range-capable HTTP servers should support DASCore HDF5 reads."""
        path = http_range_das_path / "prodml_2.1.h5"
        assert dc.get_format(path) == ("PRODML", "2.1")
        assert dc.read(path)
        assert not list(get_remote_cache_path().rglob("prodml_2.1.h5"))

    def test_spool_file_path(self, http_das_path):
        """A remote HTTP file should still produce a file-backed spool."""
        path = http_das_path / "nested" / "example_dasdae_event_2.h5"
        spool = dc.spool(path)
        assert len(spool) == 1
        assert spool[0].dims == ("distance", "time")

    def test_spool_directory_rejected(self, http_das_path):
        """Remote HTTP directories should remain unsupported for spooling."""
        with pytest.raises(InvalidSpoolError, match="local filesystem"):
            dc.spool(http_das_path)
