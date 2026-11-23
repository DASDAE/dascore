"""Benchmarks for generic IO operations using pytest-codspeed."""

from __future__ import annotations

from contextlib import suppress
from functools import cache
from pathlib import Path

import numpy as np
import pytest

import dascore as dc
from dascore.config import get_config, set_config
from dascore.core.source import ArraySource
from dascore.exceptions import DependencyError
from dascore.io import FiberIO
from dascore.io.utils import windows_to_slices
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
    pytest.param("terra15_v6_test_file.hdf5", marks=pytest.mark.smoke),
    pytest.param("conoco_segy_1.sgy", marks=pytest.mark.smoke),
    "sample_tdms_file_v4713.tdms",
)


# Patches in the synthetic resource the keyed-read benchmarks pick from.
MANY_PATCHES = 2_000
# The patches each keyed-read benchmark reads, spread over the resource.
PICKED = tuple(str(x) for x in np.linspace(0, MANY_PATCHES - 1, 5, dtype=int))


@cache
def _many_patch_metadata(count: int) -> tuple[dc.PatchMeta, ...]:
    """Build the metadata once, so the reader's own parse is not timed."""
    base = dc.get_example_patch("random_das", shape=(3, 4))
    meta = dc.PatchMeta(
        attrs=base.attrs,
        coords=base.coords,
        dims=base.dims,
        dtype="float64",
        source=ArraySource(),
    )
    return (meta,) * count


class _ManyPatchBenchmarkIO(FiberIO):
    """
    A reader of many small keyless patches; the file's text gives the count.

    Beyond finding the keys, reading a few should not cost more as the count
    grows; any growth is framework work on unread patches.
    """

    name = "_many_patch_benchmark_io"
    version = "1"

    def get_version(self, resource: Path, **kwargs) -> str | None:
        """Claim only this benchmark's files."""
        return self.version if Path(resource).suffix == ".manybench" else None

    def get_metadata(self, resource: Path, *, snap=True, **kwargs):
        """Describe every patch in the resource."""
        return list(_many_patch_metadata(int(Path(resource).read_text())))

    def read_array(self, resource: Path, windows=(), key=""):
        """Return a tiny array; the samples are not what is measured."""
        shape = (3, 4)
        return np.zeros(shape)[windows_to_slices(windows, shape)]


@pytest.fixture(scope="module")
def many_patch_path(tmp_path_factory):
    """One resource holding many patches."""
    path = tmp_path_factory.mktemp("many_patch") / "patches.manybench"
    path.write_text(str(MANY_PATCHES))
    _many_patch_metadata(MANY_PATCHES)  # built here, not in a timed run
    return path


@pytest.fixture(scope="module")
def many_patch_spool(many_patch_path):
    """An indexed spool over the many-patch resource, and the rows to read."""
    spool = dc.spool(many_patch_path.parent).update()
    keys = spool.get_contents()["source_patch_key"].astype(str).tolist()
    return spool, [keys.index(key) for key in PICKED]


@pytest.fixture(scope="session")
def test_file_paths():
    """Get paths of test files."""
    return get_test_file_paths()


@pytest.fixture(scope="session", params=SINGLE_FILE_BENCHMARKS)
def single_file_path(request):
    """Path to one registry file, parametrized for per-format benchmarks."""
    # Fetched by name rather than through get_test_file_paths, so running
    # only these benchmarks does not pull the whole registry, and an
    # unrelated file failing to fetch cannot take them down with it.
    return fetch(request.param)


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
    @pytest.mark.smoke
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


class TestKeyedReadBenchmarks:
    """
    Reads of a few patches from a resource holding many.

    Beyond finding the keys, their cost should not grow with the patches
    left unread; see `MANY_PATCHES`.
    """

    @pytest.mark.benchmark
    def test_read_patches_by_key(self, many_patch_path):
        """Read a few patches, one key per read."""
        for key in PICKED:
            dc.read(many_patch_path, source_patch_key=key)

    @pytest.mark.benchmark
    def test_read_several_keys(self, many_patch_path):
        """Read a few patches named together in one read."""
        dc.read(many_patch_path, source_patch_key=PICKED)

    @pytest.mark.benchmark
    def test_spool_patches_by_key(self, many_patch_spool):
        """Resolve a few rows of an indexed spool, as a viewer tile does."""
        spool, rows = many_patch_spool
        for row in rows:
            spool[row]
