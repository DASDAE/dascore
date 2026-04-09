"""Shared helpers for common IO test matrices."""

from __future__ import annotations

import signal as signal_mod
import socket
import threading
from contextlib import contextmanager
from urllib import error as urllib_error

import pytest

import dascore as dc
from dascore.exceptions import MissingOptionalDependencyError
from dascore.utils.misc import iterate


@contextmanager
def skip_missing():
    """Skip if missing dependencies found."""
    try:
        yield
    except MissingOptionalDependencyError as exc:
        pytest.skip(f"Missing optional dependency required to read file: {exc}")
    except TimeoutError as exc:
        pytest.skip(f"Unable to fetch data due to timeout: {exc}")


@contextmanager
def skip_timeout():
    """Skip if downloading file times out."""
    try:
        yield
    except (TimeoutError, urllib_error.URLError) as exc:
        if not _is_timeout_error(exc):
            raise
        pytest.skip(f"Unable to fetch data due to timeout: {exc}")


def _is_timeout_error(exc: BaseException) -> bool:
    """Return True if the exception chain indicates a timeout."""
    if isinstance(exc, TimeoutError | socket.timeout):
        return True
    if isinstance(exc, pytest.fail.Exception):
        return "Timeout" in str(exc)
    if isinstance(exc, urllib_error.URLError):
        reason = exc.reason
        return isinstance(reason, TimeoutError | socket.timeout)
    return False


def get_flat_io_test(common_io_read_tests: dict) -> list[list[dc.FiberIO | str]]:
    """Flatten the common IO matrix for parametrized tests."""
    flat_io = []
    for io, fetch_name_list in common_io_read_tests.items():
        for fetch_name in iterate(fetch_name_list):
            flat_io.append([io, fetch_name])
    return flat_io


def get_representative_io_test(
    common_io_read_tests: dict,
) -> list[list[dc.FiberIO | str]]:
    """Return one deterministic representative file for each FiberIO entry."""
    out = []
    for io, fetch_name_list in common_io_read_tests.items():
        out.append([io, next(iter(iterate(fetch_name_list)))])
    return out


@contextmanager
def skip_on_timeout(seconds: float, label: str):
    """Skip flaky network-bound fixture lifecycle work when it exceeds a time budget."""
    if (
        threading.current_thread() is not threading.main_thread()
        or not hasattr(signal_mod, "SIGALRM")
        or not hasattr(signal_mod, "ITIMER_REAL")
        or not hasattr(signal_mod, "setitimer")
    ):
        try:
            yield
        # Broad catch is intentional so _is_timeout_error can normalize
        # pytest-timeout/framework-specific timeout exceptions and re-raise the rest.
        except BaseException as exc:
            if not _is_timeout_error(exc):
                raise
            pytest.skip(str(exc))
        return

    previous_handler = signal_mod.getsignal(signal_mod.SIGALRM)

    def _handle_timeout(_signum, _frame):
        raise TimeoutError(f"{label} exceeded {seconds} seconds")

    try:
        signal_mod.signal(signal_mod.SIGALRM, _handle_timeout)
        signal_mod.setitimer(signal_mod.ITIMER_REAL, seconds)
        yield
    # Broad catch is intentional so _is_timeout_error can normalize
    # pytest-timeout/framework-specific timeout exceptions and re-raise the rest.
    except BaseException as exc:
        if not _is_timeout_error(exc):
            raise
        pytest.skip(str(exc))
    finally:
        signal_mod.setitimer(signal_mod.ITIMER_REAL, 0)
        signal_mod.signal(signal_mod.SIGALRM, previous_handler)
