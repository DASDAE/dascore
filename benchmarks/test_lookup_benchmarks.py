"""
Benchmarks for the small lookups DASCore performs on its hot paths.

These run entirely in memory. The end-to-end benchmarks cannot resolve
changes at this scale, because one file read costs far more than the
lookups it makes along the way, so a regression here hides inside their
noise.

Each benchmark repeats its lookup enough times to do a few milliseconds
of work, so the measurement is not dominated by test setup. The repeat
counts differ because the operations differ in cost by two orders of
magnitude.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from upath import UPath

import dascore as dc
from dascore.config import get_config, set_config
from dascore.io.core import FiberIO
from dascore.utils.io import IOResourceManager
from dascore.utils.remote_io import ensure_local_file


class TestFiberIOLookupBenchmarks:
    """Format resolution once every plugin has been loaded."""

    @pytest.fixture(scope="class")
    def manager(self):
        """Return the FiberIO manager with all plugins loaded."""
        manager = FiberIO.manager
        manager.load_plugins()
        return manager

    @pytest.mark.benchmark
    def test_load_plugins_already_loaded(self, manager):
        """Time the repeat no-op call every format lookup makes."""
        for _ in range(100_000):
            manager.load_plugins()

    @pytest.mark.benchmark
    def test_yield_fiberio_by_extension(self, manager):
        """Time the extension path used when the format is unknown."""
        for _ in range(1_000):
            list(manager.yield_fiberio(extension="h5"))

    @pytest.mark.benchmark
    def test_yield_fiberio_by_format(self, manager):
        """Time resolution when the format and version are known."""
        for _ in range(20_000):
            list(manager.yield_fiberio("DASDAE", "1"))


class TestRemoteCacheLookupBenchmarks:
    """Resolving a remote file which is already in the local cache."""

    @pytest.fixture(scope="class")
    def cached_remote_file(self, tmp_path_factory):
        """Return a remote path already materialized into a temp cache.

        Uses the permanent config tier because the fixture is class
        scoped; see the note in the parallelization recipe.
        """
        previous = get_config()
        set_config(
            remote_cache_dir=tmp_path_factory.mktemp("remote_cache_benchmark"),
            warn_on_remote_cache=False,
        )
        resource = UPath("memory://dascore/benchmark/cached_file.bin")
        with resource.open("wb") as fi:
            fi.write(b"dascore" * 1024)
        ensure_local_file(resource)
        yield resource
        set_config(previous)

    @pytest.mark.benchmark
    def test_ensure_local_file_already_cached(self, cached_remote_file):
        """Time resolving a remote file which is already downloaded."""
        for _ in range(1_000):
            ensure_local_file(cached_remote_file)


class TestIOResourceBenchmarks:
    """Handle lookups inside one IO operation."""

    @pytest.fixture(scope="class")
    def local_file(self, tmp_path_factory):
        """Return a small local file."""
        path = tmp_path_factory.mktemp("io_manager_benchmark") / "sample.bin"
        path.write_bytes(b"dascore" * 128)
        return path

    @pytest.mark.benchmark
    def test_repeat_get_resource(self, local_file):
        """Time repeat handle lookups on one manager."""
        with IOResourceManager(local_file) as manager:
            for _ in range(50_000):
                manager.get_resource(Path)

    @pytest.mark.benchmark
    def test_ensure_local_file_for_local_path(self, local_file):
        """Time the local fast path, which must not touch the remote cache."""
        for _ in range(10_000):
            ensure_local_file(local_file)


class TestSpoolAccessBenchmarks:
    """Repeat metadata access on an in-memory spool."""

    @pytest.fixture(scope="class")
    def memory_spool(self):
        """Return a small in-memory spool."""
        return dc.get_example_spool("random_das")

    @pytest.mark.benchmark
    def test_repeat_patch_access(self, memory_spool):
        """Time indexing the same spool many times."""
        for _ in range(20_000):
            memory_spool[0]

    @pytest.mark.benchmark
    def test_repeat_len(self, memory_spool):
        """Time the length of the same spool many times."""
        for _ in range(50_000):
            len(memory_spool)
