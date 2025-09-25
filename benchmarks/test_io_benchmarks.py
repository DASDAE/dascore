"""Benchmarks for generic IO operations using pytest-codspeed."""

from __future__ import annotations

from functools import cache

import pytest

import dascore as dc
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
    def test_scan_performance(self, test_file_paths):
        """Time for basic scanning of all datafiles."""
        for path in test_file_paths.values():
            dc.scan(path)

    @pytest.mark.benchmark
    def test_scan_df_performance(self, test_file_paths):
        """Time for basic scanning of all datafiles to DataFrame."""
        for path in test_file_paths.values():
            dc.scan_to_df(path)

    @pytest.mark.benchmark
    def test_get_format_performance(self, test_file_paths):
        """Time for format detection of all datafiles."""
        for path in test_file_paths.values():
            dc.get_format(path)

    @pytest.mark.benchmark
    def test_read_performance(self, test_file_paths):
        """Time for basic reading of all datafiles."""
        for path in test_file_paths.values():
            dc.read(path)[0]

    @pytest.mark.benchmark
    def test_spool_performance(self, test_file_paths):
        """Time for creating spools from all datafiles."""
        for path in test_file_paths.values():
            dc.spool(path)[0]
