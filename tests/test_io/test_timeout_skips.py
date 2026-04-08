"""Tests for timeout handling in IO network fixtures."""

from __future__ import annotations

from itertools import chain, repeat
from urllib.error import URLError

import pytest
from upath import UPath

from tests.test_io import conftest as io_conftest
from tests.test_io._common_io_test_utils import skip_on_timeout


class TestTimeoutSkipHelpers:
    """Ensure flaky timeout paths skip rather than fail."""

    def test_skip_on_timeout_skips(self):
        """Timeouts in fixture lifecycle helpers should skip the test."""
        with pytest.raises(pytest.skip.Exception, match="fixture setup timed out"):
            with skip_on_timeout(1, "fixture setup"):
                raise TimeoutError("fixture setup timed out")

    def test_wait_for_http_path_raises_timeout(self, monkeypatch):
        """The low-level probe helper should only report a timeout condition."""
        values = chain([0.0, 6.0], repeat(6.0))

        def _fake_monotonic():
            return next(values)

        def _fake_urlopen(*args, **kwargs):
            raise URLError(TimeoutError("timed out"))

        monkeypatch.setattr(io_conftest.time, "monotonic", _fake_monotonic)
        monkeypatch.setattr(io_conftest.time, "sleep", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(io_conftest, "urlopen", _fake_urlopen)

        with pytest.raises(
            TimeoutError, match="HTTP fixture path did not become ready"
        ):
            io_conftest._wait_for_http_path(UPath("http://example.com/das/test.h5"))
