"""Benchmarks for spool operations using pytest-codspeed."""

from __future__ import annotations

import pytest

import dascore as dc


@pytest.fixture
def spool_no_gap():
    """Get test spool with no gaps."""
    return dc.get_example_spool("random_das", length=10)


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
        spool.select(station="wayout")

    @pytest.mark.benchmark
    def test_select_string_match(self, diverse_spool):
        """Time select non-dimensional selects with wildcards."""
        spool = diverse_spool
        spool.select(tag="some_*")
        spool.select(station="wayou?")
