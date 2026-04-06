"""Fixtures for IO tests, including local HTTP-backed remote paths.

These remote-style tests share the on-disk DASCore remote cache under ``/tmp``.
Running remote suites in separate processes is fine when each process uses an
isolated cache directory. The current cache behavior should not be treated as
safe for concurrent threaded access within a single process.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from collections.abc import Callable
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from dascore.compat import UPath
from dascore.utils.downloader import fetch
from tests.test_io._common_io_test_utils import fail_on_timeout


class _SilentSimpleHTTPRequestHandler(SimpleHTTPRequestHandler):
    """A simple HTTP handler that suppresses request noise in tests."""

    def log_message(self, format, *args):
        """Silence request logs during localhost-backed tests."""
        return None

    def copyfile(self, source, outputfile):
        """Silence BrokenPipeError/ConnectionResetError from test clients."""
        try:
            return super().copyfile(source, outputfile)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return None


class _RegressionHTTPRequestHandler(_SilentSimpleHTTPRequestHandler):
    """A stricter localhost handler for flaky plain-HTTP regression tests."""

    protocol_version = "HTTP/1.0"

    def end_headers(self):
        """Disable keep-alive so one test request cannot leak into the next."""
        self.send_header("Connection", "close")
        super().end_headers()


def _link_or_copy(source: Path, dest: Path) -> None:
    """Populate one served file path using the cheapest available local copy."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        dest.hardlink_to(source)
        return
    except OSError:
        pass
    try:
        dest.symlink_to(source)
        return
    except OSError:
        pass
    shutil.copy2(source, dest)


def _prime_http_test_tree(
    ensure_file: Callable[[str, str | Path | None], Path],
) -> None:
    """Populate the fixed files used by the dedicated HTTP regression tests."""
    ensure_file("example_dasdae_event_1.h5")
    ensure_file(
        "example_dasdae_event_1.h5",
        Path("nested") / "example_dasdae_event_2.h5",
    )


def _coerce_http_relative_path(path_or_name: str | Path) -> Path:
    """Map fixture inputs onto paths within the served HTTP tree."""
    path = Path(path_or_name)
    if path.is_absolute():
        return Path(path.name)
    return path if len(path.parts) > 1 else Path(path.name)


def _wait_for_http_path(path: UPath, timeout: float = 5.0) -> None:
    """Probe one served HTTP path until it is reachable."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(str(path), timeout=5):
                return
        except (URLError, OSError):
            time.sleep(0.1)
    pytest.fail(f"HTTP fixture path did not become ready in time: {path}")


@pytest.fixture(scope="session")
def http_test_data_root(tmp_path_factory):
    """Return a lazy local directory tree served over HTTP."""
    root = tmp_path_factory.mktemp("http_test_data") / "served_root" / "das"
    (root / "nested").mkdir(parents=True, exist_ok=True)
    return root.parent


@pytest.fixture(scope="session")
def ensure_http_fetch_file(http_test_data_root):
    """Materialize one fetched file into the served HTTP tree on demand."""
    served_root = Path(http_test_data_root) / "das"

    def _ensure(fetch_name: str, relative_path: str | Path | None = None) -> Path:
        source = Path(fetch(fetch_name))
        dest = served_root / Path(relative_path or source.name)
        _link_or_copy(source, dest)
        return dest

    # Pre-materialize the fixed files used directly by the HTTP regression suite.
    _prime_http_test_tree(_ensure)
    return _ensure


@pytest.fixture(scope="session")
def http_das_path(http_test_data_root, ensure_http_fetch_file):
    """Return a UPath pointing at a localhost HTTP view of DAS test data."""
    handler = partial(
        _SilentSimpleHTTPRequestHandler,
        directory=str(http_test_data_root),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    probe_path = "example_dasdae_event_1.h5"
    try:
        host, port = server.server_address
        probe_url = f"http://{host}:{port}/das/{probe_path}"
        with fail_on_timeout(10, "http_das_path readiness probe"):
            for _ in range(50):
                try:
                    with urlopen(probe_url, timeout=5):
                        break
                except (URLError, OSError):
                    time.sleep(0.1)
            else:
                pytest.fail("HTTP test server did not become ready in time.")
        yield UPath(f"http://{host}:{port}/das")
    finally:
        with fail_on_timeout(10, "http_das_path teardown"):
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        if thread.is_alive():
            pytest.fail("HTTP test server thread did not exit cleanly.")


@pytest.fixture(scope="session")
def http_regression_data_root(tmp_path_factory):
    """Return an isolated local directory tree for remote HTTP regression tests."""
    root = tmp_path_factory.mktemp("http_regression_data") / "served_root" / "das"
    (root / "nested").mkdir(parents=True, exist_ok=True)
    return root.parent


@pytest.fixture(scope="session")
def ensure_http_regression_file(http_regression_data_root):
    """Materialize only the fixed regression files into the isolated HTTP tree."""
    served_root = Path(http_regression_data_root) / "das"

    def _ensure(fetch_name: str, relative_path: str | Path | None = None) -> Path:
        source = Path(fetch(fetch_name))
        dest = served_root / Path(relative_path or source.name)
        _link_or_copy(source, dest)
        return dest

    _prime_http_test_tree(_ensure)
    return _ensure


@pytest.fixture(scope="session")
def http_regression_das_path(http_regression_data_root, ensure_http_regression_file):
    """Return an isolated HTTP tree containing only the regression fixtures."""
    handler = partial(
        _RegressionHTTPRequestHandler,
        directory=str(http_regression_data_root),
    )
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    probe_path = "example_dasdae_event_1.h5"
    try:
        host, port = server.server_address
        probe_url = f"http://{host}:{port}/das/{probe_path}"
        with fail_on_timeout(10, "http_regression_das_path readiness probe"):
            for _ in range(50):
                try:
                    with urlopen(probe_url, timeout=5):
                        break
                except (URLError, OSError):
                    time.sleep(0.1)
            else:
                pytest.fail("HTTP test server did not become ready in time.")
        yield UPath(f"http://{host}:{port}/das")
    finally:
        with fail_on_timeout(10, "http_regression_das_path teardown"):
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        if thread.is_alive():
            pytest.fail("HTTP regression server thread did not exit cleanly.")


@pytest.fixture(scope="session")
def http_range_das_path(http_test_data_root, ensure_http_fetch_file):
    """Return a UPath pointing at a localhost HTTP server with range support."""
    uvicorn = pytest.importorskip("uvicorn")
    starlette_cls = pytest.importorskip("starlette.applications").Starlette
    responses = pytest.importorskip("starlette.responses")
    file_response_cls = responses.FileResponse
    response_cls = responses.Response
    route_cls = pytest.importorskip("starlette.routing").Route
    served_root = Path(http_test_data_root)

    async def _serve_file(request):
        rel_path = Path(request.path_params["path"])
        file_path = served_root / rel_path
        root_path = os.path.abspath(served_root)
        candidate_path = os.path.abspath(file_path)
        if os.path.commonpath([root_path, candidate_path]) != root_path:
            return response_cls(status_code=404)
        if not file_path.exists() or not file_path.is_file():
            return response_cls(status_code=404)
        return file_response_cls(file_path)

    app = starlette_cls(routes=[route_cls("/{path:path}", _serve_file)])
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        log_level="warning",
        ws="none",
        # Avoid indefinite teardown hangs if a client leaves a keep-alive
        # connection open when the fixture shuts the server down.
        timeout_graceful_shutdown=1,
    )
    server = uvicorn.Server(config)
    sock = config.bind_socket()
    host, port = sock.getsockname()[:2]

    def _run():
        server.run(sockets=[sock])

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    try:
        probe_url = f"http://{host}:{port}/das/example_dasdae_event_1.h5"
        with fail_on_timeout(10, "http_range_das_path readiness probe"):
            for _ in range(50):
                try:
                    with urlopen(probe_url, timeout=5):
                        break
                except (URLError, OSError):
                    time.sleep(0.1)
            else:
                pytest.fail(
                    "Range-capable HTTP test server did not become ready in time."
                )
        yield UPath(f"http://{host}:{port}/das")
    finally:
        with fail_on_timeout(10, "http_range_das_path teardown"):
            server.should_exit = True
            thread.join(timeout=5)
        if thread.is_alive():
            sock.close()
            thread.join(timeout=1)
            pytest.fail("Range-capable HTTP server thread did not exit cleanly.")


@pytest.fixture(scope="session")
def to_http_path(
    ensure_http_fetch_file, http_das_path
) -> Callable[[str | Path], UPath]:
    """Convert a local fixture path or fetch name to its HTTP-backed UPath."""

    def _convert(path_or_name: str | Path) -> UPath:
        path = Path(path_or_name)
        relative_path = _coerce_http_relative_path(path_or_name)
        ensure_http_fetch_file(path.name, relative_path)
        out = http_das_path / relative_path
        _wait_for_http_path(out)
        return out

    return _convert


@pytest.fixture(scope="session")
def to_http_range_path(
    ensure_http_fetch_file, http_range_das_path
) -> Callable[[str | Path], UPath]:
    """Convert a local fixture path or fetch name to its range-capable HTTP UPath."""

    def _convert(path_or_name: str | Path) -> UPath:
        relative_path = _coerce_http_relative_path(path_or_name)
        ensure_http_fetch_file(Path(path_or_name).name, relative_path)
        out = http_range_das_path / relative_path
        _wait_for_http_path(out)
        return out

    return _convert
