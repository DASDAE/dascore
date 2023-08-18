"""Benchmark for generic memory spool operations."""
from __future__ import annotations

import dascore as dc


class ChunkSuite:
    """Benchmark for chunking patches inside spools."""

    def setup(self):
        """Get test spools."""
        self._spool_no_gap = dc.get_example_spool("random_das", length=10)
        self._spool_no_overlap = dc.get_example_spool(
            "random_das", length=10, time_gap=10
        )
        self._diverse_spool = dc.get_example_spool("diverse_das")

    def _chunk_n_check(self, spool, length: int | None = 1, time=None):
        """Helper function to merge and check the spool."""
        new = spool.chunk(time=time)
        if length is not None:
            assert len(new) == length
        # just in case we make spools more lazy, force iteration
        for patch in new:
            assert isinstance(patch, dc.Patch)

    def time_contiguous_merge(self):
        """Time merging contiguous patches from in-memory spool."""
        self._chunk_n_check(self._spool_no_gap)

    def time_no_overlap_merge(self):
        """Timing for trying to chunk patches that have no overlap."""
        self._chunk_n_check(self._spool_no_gap)

    def time_diverse_merge(self):
        """Time trying to merge the diverse spool."""
        self._chunk_n_check(self._diverse_spool, length=None)

    def time_1second_chunk(self):
        """Time chunking for one second along no gap spool."""
        self._chunk_n_check(self._spool_no_gap, time=1, length=None)

    def time_half_second_chunk(self):
        """Time chunking for 0.5 along no gap spool."""
        self._chunk_n_check(self._spool_no_gap, time=0.5, length=None)


class SelectSuite:
    """Suite of selection timing."""

    def setup(self):
        """Get test spools."""
        self._spool_no_gap = dc.get_example_spool("random_das", length=10)
        self._spool_no_gap_df = self._spool_no_gap.get_contents()
        self._diverse_spool = dc.get_example_spool("diverse_das")

    def time_select_full_range(self):
        """Timing selecting the full time range."""
        df, spool = self._spool_no_gap_df, self._spool_no_gap
        t1, t2 = df["time_min"].min(), df["time_max"].max()
        spool.select(time=(t1, t2))
        spool.select(time=(None, t2))
        spool.select(time=(t1, None))

    def time_select_half_range(self):
        """Time selecting and trimming."""
        df, spool = self._spool_no_gap_df, self._spool_no_gap
        t1, t2 = df["time_min"].min(), df["time_max"].max()
        duration = (t2 - t1) / 2
        spool.select(time=(t1, t2 - duration))
        spool.select(time=(t1 + duration, t2))

    def time_select_strings(self):
        """Time select non-dimensional selects."""
        spool = self._diverse_spool
        spool.select(tag="some_tag")
        spool.select(station="wayout")

    def time_select_string_match(self):
        """Time select non-dimensional selects."""
        spool = self._diverse_spool
        spool.select(tag="some_*")
        spool.select(station="wayou?")
