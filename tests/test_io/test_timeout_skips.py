"""Tests for timeout handling in IO network fixtures."""

from __future__ import annotations

import signal as signal_mod
from itertools import chain, repeat
from urllib.error import URLError

import pytest
from upath import UPath

from tests.test_io import _common_io_test_utils as io_test_utils
from tests.test_io import conftest as io_conftest
from tests.test_io._common_io_test_utils import skip_on_timeout


class TestTimeoutSkipHelpers:
    """Ensure flaky timeout paths skip rather than fail."""

    def test_skip_on_timeout_skips(self):
        """Timeouts in fixture lifecycle helpers should skip the test."""
        required_attrs = ("SIGALRM", "ITIMER_REAL", "setitimer")
        if not all(hasattr(signal_mod, attr) for attr in required_attrs):
            pytest.skip("skip_on_timeout skip behavior requires SIGALRM support")
        with pytest.raises(pytest.skip.Exception, match="fixture setup timed out"):
            with skip_on_timeout(1, "fixture setup"):
                raise TimeoutError("fixture setup timed out")

    def test_skip_on_timeout_skips_without_signal_support(self, monkeypatch):
        """Platforms without SIGALRM support should still skip on timeout."""
        monkeypatch.setattr(io_test_utils.threading, "main_thread", lambda: object())

        with pytest.raises(pytest.skip.Exception, match="fixture setup timed out"):
            with skip_on_timeout(1, "fixture setup"):
                raise TimeoutError("fixture setup timed out")

    def test_earlier_outer_alarm_unchanged(self, monkeypatch):
        """An earlier outer timeout must remain authoritative."""
        timer_calls = []
        handler_calls = []
        monkeypatch.setattr(io_test_utils.signal_mod, "getsignal", lambda *_: object())
        monkeypatch.setattr(
            io_test_utils.signal_mod, "getitimer", lambda *_: (2.0, 0.0)
        )
        monkeypatch.setattr(
            io_test_utils.signal_mod,
            "setitimer",
            lambda *args: timer_calls.append(args),
        )
        monkeypatch.setattr(
            io_test_utils.signal_mod,
            "signal",
            lambda *args: handler_calls.append(args),
        )

        with pytest.raises(pytest.skip.Exception, match="operation timed out"):
            with skip_on_timeout(5, "inner timeout"):
                raise TimeoutError("operation timed out")

        assert timer_calls == []
        assert handler_calls == []

    def test_later_outer_alarm_restored(self, monkeypatch):
        """A later outer timeout is restored with elapsed time deducted."""
        previous_handler = object()
        timer_calls = []
        handler_calls = []
        monotonic = iter((100.0, 102.0))
        monkeypatch.setattr(
            io_test_utils.signal_mod, "getsignal", lambda *_: previous_handler
        )
        monkeypatch.setattr(
            io_test_utils.signal_mod, "getitimer", lambda *_: (10.0, 0.5)
        )
        monkeypatch.setattr(
            io_test_utils.signal_mod,
            "setitimer",
            lambda *args: timer_calls.append(args),
        )
        monkeypatch.setattr(
            io_test_utils.signal_mod,
            "signal",
            lambda *args: handler_calls.append(args),
        )
        monkeypatch.setattr(io_test_utils.time, "monotonic", lambda: next(monotonic))

        with skip_on_timeout(5, "inner timeout"):
            pass

        timer = io_test_utils.signal_mod.ITIMER_REAL
        assert timer_calls == [(timer, 5), (timer, 0), (timer, 8.0, 0.5)]
        assert handler_calls[-1] == (signal_mod.SIGALRM, previous_handler)

    def test_wait_for_http_path_raises_timeout(self, monkeypatch):
        """The low-level probe helper should only report a timeout condition."""
        values = chain([0.0, 0.1, 6.0], repeat(6.0))

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
