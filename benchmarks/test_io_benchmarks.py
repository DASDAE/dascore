"""Benchmarks for generic IO operations using pytest-codspeed."""

from __future__ import annotations

from contextlib import suppress
from functools import cache

import pytest

import dascore as dc
from dascore.config import get_config, set_config
from dascore.exceptions import DependencyError
from dascore.utils.downloader import fetch, get_registry_df


@cache
def get_test_file_paths():
    """Get a dict of name: path for all files in data registry."""
    df = get_registry_df().loc[lambda x: ~x["name"].str.endswith(".csv")]
    out = {row["name"]: fetch(row["name"]) for _, row in df.iterrows()}
    return out


# Benchmarked one file at a time so a regression in a single reader is visible
# rather than averaged away by the whole-registry benchmarks below. Chosen to
# cover distinct reader strategies: record-framed protobuf, HDF5, SEG-Y, and a
# memory-mapped binary.
SINGLE_FILE_BENCHMARKS = (
    "sintela_protobuf_1.pb",
    "terra15_v6_test_file.hdf5",
    "conoco_segy_1.sgy",
    "sample_tdms_file_v4713.tdms",
)


@pytest.fixture(scope="session")
def test_file_paths():
    """Get paths of test files."""
    return get_test_file_paths()


@pytest.fixture(scope="session", params=SINGLE_FILE_BENCHMARKS)
def single_file_path(request):
    """Path to one registry file, parametrized for per-format benchmarks."""
    return get_test_file_paths()[request.param]


@pytest.fixture(scope="module", autouse=True)
def allow_legacy_dasdae_coord_unpickle():
    """Benchmarks include trusted historical DASDAE fixtures from the registry.

    Uses the permanent config base (not a scoped ``config_context``) because a
    module-scoped fixture spans many benchmarks.
    """
    previous = get_config()
    set_config(allow_dasdae_format_unpickle=True)
    try:
        yield
    finally:
        set_config(previous)


class TestIOBenchmarks:
    """Benchmarks for IO operations."""

    @pytest.mark.benchmark
    def test_scan(self, test_file_paths):
        """Time for basic scanning of all datafiles."""
        for path in test_file_paths.values():
            with suppress(DependencyError):
                dc.scan(path)

    @pytest.mark.benchmark
    def test_scan_df(self, test_file_paths):
        """Time for basic scanning of all datafiles to DataFrame."""
        for path in test_file_paths.values():
            with suppress(DependencyError):
                dc.scan_to_df(path)

    @pytest.mark.benchmark
    def test_get_format(self, test_file_paths):
        """Time for format detection of all datafiles."""
        for path in test_file_paths.values():
            with suppress(DependencyError):
                dc.get_format(path)

    @pytest.mark.benchmark
    def test_read(self, test_file_paths):
        """Time for basic reading of all datafiles."""
        for path in test_file_paths.values():
            with suppress(DependencyError):
                dc.read(path)[0]

    @pytest.mark.benchmark
    def test_scan_single_file(self, single_file_path):
        """
        Time one file's scan, to pair with the read below.

        A scan reads headers and a read reads samples, so the two should be far
        apart for any file whose samples dominate it. Comparing the pair is
        what makes a reader that quietly started reading everything obvious.
        """
        with suppress(DependencyError):
            dc.scan(single_file_path)

    @pytest.mark.benchmark
    def test_read_single_file(self, single_file_path):
        """Time one file's read, as the counterpart to the scan above."""
        with suppress(DependencyError):
            dc.read(single_file_path)[0]

    @pytest.mark.benchmark
    def test_spool(self, test_file_paths):
        """Time for creating spools from all datafiles."""
        for path in test_file_paths.values():
            with suppress(DependencyError):
                dc.spool(path)
