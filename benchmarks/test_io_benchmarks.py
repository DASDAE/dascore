"""Benchmarks for generic IO operations using pytest-codspeed."""

from __future__ import annotations

from contextlib import suppress
from functools import cache

import pytest

import dascore as dc
from dascore.exceptions import MissingOptionalDependencyError
from dascore.utils.downloader import fetch, get_registry_df


@cache
def get_test_file_paths():
    """Get a dict of name: path for all files in data registry."""
    df = get_registry_df().loc[lambda x: ~x["name"].str.endswith(".csv")]
    out = {row["name"]: fetch(row["name"]) for _, row in df.iterrows()}
    return out


@pytest.fixture(scope="session")
def test_file_paths():
    """Get paths of test files."""
    return get_test_file_paths()


class TestIOBenchmarks:
    """Benchmarks for IO operations."""

    @pytest.mark.benchmark
    def test_scan(self, test_file_paths):
        """Time for basic scanning of all datafiles."""
        for path in test_file_paths.values():
            with suppress(MissingOptionalDependencyError):
                dc.scan(path)

    @pytest.mark.benchmark
    def test_scan_df(self, test_file_paths):
        """Time for basic scanning of all datafiles to DataFrame."""
        for path in test_file_paths.values():
            with suppress(MissingOptionalDependencyError):
                dc.scan_to_df(path)

    @pytest.mark.benchmark
    def test_get_format(self, test_file_paths):
        """Time for format detection of all datafiles."""
        for path in test_file_paths.values():
            with suppress(MissingOptionalDependencyError):
                dc.get_format(path)

    @pytest.mark.benchmark
    def test_read(self, test_file_paths):
        """Time for basic reading of all datafiles."""
        for path in test_file_paths.values():
            with suppress(MissingOptionalDependencyError):
                dc.read(path)[0]

    @pytest.mark.benchmark
    def test_spool(self, test_file_paths):
        """Time for creating spools from all datafiles."""
        for path in test_file_paths.values():
            with suppress(MissingOptionalDependencyError):
                dc.spool(path)[0]
