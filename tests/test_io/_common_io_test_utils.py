"""Shared helpers for common IO test matrices."""

from __future__ import annotations

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
        pytest.skip(f"Unable to fetch data due to timeout: {exc}")


def get_flat_io_test(common_io_read_tests: dict) -> list[list[dc.FiberIO | str]]:
    """Flatten the common IO matrix for parametrized tests."""
    flat_io = []
    for io, fetch_name_list in common_io_read_tests.items():
        for fetch_name in iterate(fetch_name_list):
            flat_io.append([io, fetch_name])
    return flat_io
