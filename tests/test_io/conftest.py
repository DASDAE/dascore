"""Fixtures for IO tests, including local HTTP-backed remote paths."""

from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Callable
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import pytest

from dascore.compat import UPath
from dascore.utils.downloader import fetch
from dascore.utils.misc import iterate


class _QuietSimpleHTTPRequestHandler(SimpleHTTPRequestHandler):
    """A simple HTTP handler that ignores expected client disconnects."""

    def copyfile(self, source, outputfile):
        """Silence BrokenPipeError/ConnectionResetError from test clients."""
        try:
            return super().copyfile(source, outputfile)
        except (BrokenPipeError, ConnectionResetError):
            return None


def _get_common_io_fetch_names() -> tuple[str, ...]:
    """Return the fetch names used by the common readable IO matrix."""
    from tests.test_io.test_common_io import COMMON_IO_READ_TESTS

    seen: set[str] = set()
    ordered: list[str] = []
    for fetch_names in COMMON_IO_READ_TESTS.values():
        for fetch_name in iterate(fetch_names):
            if fetch_name not in seen:
                seen.add(fetch_name)
                ordered.append(fetch_name)
    return tuple(ordered)


@pytest.fixture(scope="session")
def http_test_data_root(tmp_path_factory):
    """Return a curated local directory tree served over HTTP."""
    root = tmp_path_factory.mktemp("http_test_data") / "served_root" / "das"
    nested = root / "nested"
    nested.mkdir(parents=True, exist_ok=True)
    for fetch_name in _get_common_io_fetch_names():
        shutil.copy2(fetch(fetch_name), root / fetch_name)
    shutil.copy2(
        root / "example_dasdae_event_1.h5",
        nested / "example_dasdae_event_2.h5",
    )
    return root.parent


@pytest.fixture(scope="session")
def http_das_path(http_test_data_root):
    """Return a UPath pointing at a localhost HTTP view of DAS test data."""
    handler = partial(
        _QuietSimpleHTTPRequestHandler,
        directory=str(http_test_data_root),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield UPath(f"http://{host}:{port}/das")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="session")
def http_range_das_path(http_test_data_root):
    """Return a UPath pointing at a localhost HTTP server with range support."""
    uvicorn = pytest.importorskip("uvicorn")
    starlette_cls = pytest.importorskip("starlette.applications").Starlette
    responses = pytest.importorskip("starlette.responses")
    file_response_cls = responses.FileResponse
    response_cls = responses.Response
    route_cls = pytest.importorskip("starlette.routing").Route
    served_root = Path(http_test_data_root).resolve()

    async def _serve_file(request):
        rel_path = Path(request.path_params["path"])
        file_path = (served_root / rel_path).resolve()
        if served_root not in file_path.parents and file_path != served_root:
            return response_cls(status_code=404)
        if not file_path.exists() or not file_path.is_file():
            return response_cls(status_code=404)
        return file_response_cls(file_path)

    app = starlette_cls(routes=[route_cls("/{path:path}", _serve_file)])
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    sock = config.bind_socket()
    host, port = sock.getsockname()[:2]

    def _run():
        server.run(sockets=[sock])

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    probe_url = f"http://{host}:{port}/das/example_dasdae_event_1.h5"
    for _ in range(50):
        try:
            with urlopen(probe_url):
                break
        except Exception:
            time.sleep(0.1)
    try:
        yield UPath(f"http://{host}:{port}/das")
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture(scope="session")
def to_http_path(http_das_path) -> Callable[[str | Path], UPath]:
    """Convert a local fixture path or fetch name to its HTTP-backed UPath."""

    def _convert(path_or_name: str | Path) -> UPath:
        name = Path(path_or_name).name
        return http_das_path / name

    return _convert
