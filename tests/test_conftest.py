"""Tests for the shared test helpers defined in conftest."""

from __future__ import annotations

import threading

import pytest


class TestRunInThreads:
    """The concurrency helper must not hide what its workers do."""

    def test_returns_each_worker_result(self, run_in_threads):
        """Each worker's return value comes back in index order."""
        results = run_in_threads(lambda index: index * 2, count=3)
        assert results == [0, 2, 4]

    def test_worker_exception_propagates(self, run_in_threads):
        """A raising worker fails the test instead of leaving a None result."""

        def _raise_on_first(index):
            if index == 0:
                raise ValueError("worker failed")
            return index

        with pytest.raises(ValueError, match="worker failed"):
            run_in_threads(_raise_on_first, count=2)

    def test_several_worker_exceptions_surface_one(self, run_in_threads):
        """When several workers fail, one real failure reaches the test."""

        def _always_raise(index):
            raise ValueError(f"worker {index} failed")

        with pytest.raises(ValueError, match="worker . failed"):
            run_in_threads(_always_raise, count=3)

    def test_base_exception_from_worker_propagates(self, run_in_threads):
        """A BaseException-derived outcome, such as pytest.skip, is not swallowed.

        These cannot be members of an ExceptionGroup, so aggregating them
        would replace the outcome with a TypeError.
        """

        def _skip(index):
            pytest.fail("worker used a BaseException-derived outcome")

        with pytest.raises(BaseException, match="BaseException-derived"):
            run_in_threads(_skip, count=2)

    def test_workers_run_concurrently(self, run_in_threads):
        """The barrier releases every worker together.

        A worker waiting on a barrier the others cannot reach would time
        out, so passing it means they really did overlap.
        """
        barrier = threading.Barrier(3, timeout=30)
        results = run_in_threads(lambda index: barrier.wait() is not None, count=3)
        assert all(results)
