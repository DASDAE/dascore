"""Benchmarks for spool operations using pytest-codspeed."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.utils.time import to_timedelta64


@pytest.fixture
def spool_no_gap():
    """Get test spool with no gaps."""
    return dc.get_example_spool("random_das", length=10)


def _make_contiguous_patches(count, shape=(50, 100), time_step=0.004):
    """Make small patches contiguous in time."""
    base = dc.get_example_patch(
        "random_das",
        time_min="2023-01-01",
        shape=shape,
        time_step=time_step,
        distance_step=1.0,
    ).update_attrs(history=[])
    duration = to_timedelta64(shape[1] * time_step)
    start = np.datetime64("2023-01-01")
    return [
        base.update_coords(time_min=start + i * duration).update_attrs(history=[])
        for i in range(count)
    ]


@pytest.fixture(scope="module")
def many_file_directory(tmp_path_factory):
    """Create a directory of many small contiguous DASDAE files."""
    path = tmp_path_factory.mktemp("many_file_benchmark")
    for i, patch in enumerate(_make_contiguous_patches(25)):
        patch.io.write(path / f"file_{i:03d}.h5", "dasdae")
    return path


@pytest.fixture
def gapped_spool_no_overlap():
    """Get test spool with no overlap."""
    return dc.get_example_spool("random_das", length=10, time_gap=10)


@pytest.fixture
def diverse_spool():
    """Get diverse test spool."""
    return dc.get_example_spool("diverse_das")


def _chunk_and_check(spool, length: int | None = 1, time=None):
    """Helper function to merge and check the spool."""
    new = spool.chunk(time=time)
    if length is not None:
        assert len(new) == length
    # Check contents
    for patch in new:
        assert isinstance(patch, dc.Patch)


class TestChunkBenchmarks:
    """Benchmarks for spool chunking operations."""

    @pytest.mark.benchmark
    def test_contiguous_merge(self, spool_no_gap):
        """Time merging contiguous patches from in-memory spool."""
        _chunk_and_check(spool_no_gap)

    @pytest.mark.benchmark
    def test_no_overlap_merge(self, gapped_spool_no_overlap):
        """Timing for trying to chunk patches that have no overlap."""
        chunked = gapped_spool_no_overlap.chunk(time=None)
        # In this case the spool should not be merged.
        assert len(chunked) == len(gapped_spool_no_overlap)

    @pytest.mark.benchmark
    def test_diverse_merge(self, diverse_spool):
        """Time trying to merge the diverse spool."""
        _chunk_and_check(diverse_spool, length=None)

    @pytest.mark.benchmark
    def test_1second_chunk(self, spool_no_gap):
        """Time chunking for one second along no gap spool."""
        _chunk_and_check(spool_no_gap, time=1, length=None)

    @pytest.mark.benchmark
    def test_half_second_chunk(self, spool_no_gap):
        """Time chunking for 0.5 along no gap spool."""
        _chunk_and_check(spool_no_gap, time=0.5, length=None)


class TestSelectionBenchmarks:
    """Benchmarks for spool selection operations."""

    @pytest.fixture
    def spool_no_gap_df(self, spool_no_gap):
        """Get contents dataframe for no gap spool."""
        return spool_no_gap.get_contents()

    @pytest.mark.benchmark
    def test_select_full_range(self, spool_no_gap, spool_no_gap_df):
        """Timing selecting the full time range."""
        df, spool = spool_no_gap_df, spool_no_gap
        t1, t2 = df["time_min"].min(), df["time_max"].max()
        spool.select(time=(t1, t2))
        spool.select(time=(None, t2))
        spool.select(time=(t1, None))

    @pytest.mark.benchmark
    def test_select_half_range(self, spool_no_gap, spool_no_gap_df):
        """Time selecting and trimming."""
        df, spool = spool_no_gap_df, spool_no_gap
        t1, t2 = df["time_min"].min(), df["time_max"].max()
        duration = (t2 - t1) / 2
        spool.select(time=(t1, t2 - duration))
        spool.select(time=(t1 + duration, t2))

    @pytest.mark.benchmark
    def test_select_strings(self, diverse_spool):
        """Time select non-dimensional selects."""
        spool = diverse_spool
        spool.select(tag="some_tag")
        spool.select(tag="wayout")

    @pytest.mark.benchmark
    def test_select_string_match(self, diverse_spool):
        """Time select non-dimensional selects with wildcards."""
        spool = diverse_spool
        spool.select(tag="some_*")
        spool.select(tag="wayou?")


class TestManyFileDirectoryBenchmarks:
    """Benchmarks for working with a directory of many files."""

    @pytest.fixture
    def indexed_spool(self, many_file_directory):
        """Get an indexed spool of the directory."""
        return dc.spool(many_file_directory).update(progress=None)

    @pytest.mark.benchmark
    def test_index_directory(self, many_file_directory):
        """Time indexing a directory of many files."""
        for index_path in many_file_directory.glob(".dascore*"):
            index_path.unlink()
        spool = dc.spool(many_file_directory).update(progress=None)
        assert len(spool)

    @pytest.mark.benchmark
    def test_merge_many_files(self, indexed_spool):
        """Time merging many files into a single patch."""
        merged = indexed_spool.chunk(time=None)
        patches = list(merged)
        assert len(patches) == 1

    @pytest.mark.benchmark
    def test_iterate_chunked(self, indexed_spool):
        """Time loading chunks which each span several files."""
        # each file spans 0.4s, so each chunk merges 4 files.
        chunked = indexed_spool.chunk(time=1.6)
        for patch in chunked:
            assert isinstance(patch, dc.Patch)


class TestHivePathAttrBenchmarks:
    """Benchmarks for hive-style path attribute extraction and loading."""

    @pytest.fixture(scope="class")
    def hive_directory(self, tmp_path_factory):
        """A hive-partitioned directory: 5 data sources x 5 files."""
        path = tmp_path_factory.mktemp("hive_benchmark")
        patches = _make_contiguous_patches(25)
        for i, patch in enumerate(patches):
            sub = path / f"acquisition_key=XX.R2D1..S{i % 5}" / "cable=A__tag=raw"
            sub.mkdir(parents=True, exist_ok=True)
            patch.io.write(sub / f"patch_{i:03d}.h5", "dasdae")
        return path

    @pytest.mark.benchmark
    def test_index_hive_directory(self, hive_directory):
        """Time indexing a hive tree (includes path-attr extraction)."""
        for index_path in hive_directory.glob(".dascore*"):
            index_path.unlink()
        spool = dc.spool(hive_directory).update(progress=None)
        assert len(spool.select(acquisition_key="XX.R2D1..S0")) == 5

    @pytest.mark.benchmark
    def test_load_patches_with_path_attrs(self, hive_directory):
        """Time loading patches that get path attrs stamped on."""
        spool = dc.spool(hive_directory).update(progress=None)
        for patch in spool:
            assert patch.attrs.acquisition_key.startswith("XX.")


def _make_gapped_patches(count, shape=(10, 20), time_step=0.01):
    """Make small patches separated by gaps so each is its own partition."""
    base = dc.get_example_patch(
        "random_das",
        time_min="2023-01-01",
        shape=shape,
        time_step=time_step,
        distance_step=1.0,
    ).update_attrs(history=[])
    duration = to_timedelta64(shape[1] * time_step)
    gap = to_timedelta64(5 * time_step)  # larger than the default tolerance
    start = np.datetime64("2023-01-01")
    stride = duration + gap
    return [
        base.update_coords(time_min=start + i * stride).update_attrs(history=[])
        for i in range(count)
    ]


class TestManyPartitionChunkBenchmarks:
    """
    Benchmarks for planning chunks across many gapped partitions.

    Every patch sits behind a gap larger than the merge tolerance, so
    each is its own partition; chunking is lazy, so these capture the
    planning cost, which scales with partition count rather than data
    volume (the regime a long gappy acquisition puts the planner in).
    """

    @pytest.fixture(scope="class")
    def many_partition_spool(self):
        """An in-memory spool where every patch is its own partition."""
        return dc.spool(_make_gapped_patches(100))

    @pytest.mark.benchmark
    def test_merge_many_partitions(self, many_partition_spool):
        """Time merge-mode planning over many partitions."""
        merged = many_partition_spool.chunk(time=None)
        assert len(merged) == 100

    @pytest.mark.benchmark
    def test_segment_many_partitions(self, many_partition_spool):
        """Time segment-mode planning over many partitions."""
        chunked = many_partition_spool.chunk(time=0.05)
        assert len(chunked) == 400


class TestMemorySpoolBenchmarks:
    """Benchmarks for in-memory spool creation and access."""

    @pytest.fixture(scope="class")
    def many_patches(self):
        """Get a list of many small contiguous patches."""
        return _make_contiguous_patches(100)

    @pytest.mark.benchmark
    def test_spool_from_patches_access(self, many_patches):
        """Time creating a spool from patches and accessing its contents."""
        spool = dc.spool(many_patches)
        assert len(spool) == len(many_patches)
        assert isinstance(spool[0], dc.Patch)
        assert isinstance(spool[-1], dc.Patch)
        for patch in spool:
            assert isinstance(patch, dc.Patch)

    @pytest.mark.benchmark
    def test_merge_many_patches(self, many_patches):
        """Time merging many in-memory patches into one."""
        merged = dc.spool(many_patches).chunk(time=None)
        assert len(merged) == 1
        assert isinstance(merged[0], dc.Patch)
