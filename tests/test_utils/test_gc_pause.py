"""Tests for the garbage-collection pause used by remote HDF5 reads."""

from __future__ import annotations

import gc
import io
import os
import threading
from contextlib import suppress
from types import SimpleNamespace

import pytest

import dascore.utils.remote_io as remote_io
from dascore.utils.hdf5 import (
    _is_loop_backed,
    _ManagedH5pyFile,
    _open_h5_fileobj,
)
from dascore.utils.remote_io import pause_gc, resume_gc


class _FakeFS:
    """Stand-in for an fsspec filesystem."""

    def __init__(self, async_impl):
        self.async_impl = async_impl


class _FakeRemoteFile(io.RawIOBase):
    """A file object whose filesystem serves reads on a loop thread."""

    def __init__(self):
        self.fs = _FakeFS(True)

    def readable(self):
        return True


def _raise_keyboard_interrupt(*args, **kwargs):
    """Stand in for an interrupt landing inside an uninterruptible step."""
    raise KeyboardInterrupt("interrupted")


def _open_paused(fileobj, constructor=lambda fileobj, **kwargs: object()):
    """Open a fake loop-backed file object the way the UPath branch does."""
    return _open_h5_fileobj(fileobj, constructor, "r", pause=True, close_on_error=True)


class TestDeadlockProperty:
    """The pause must actually prevent the h5py/loop-thread deadlock."""

    def test_pause_prevents_loop_thread_deadlock(self):
        """A collection on the loop thread cannot wedge a lock-holding reader.

        Models the real cycle: a reader holds h5py's global lock and waits on
        a loop thread, while that thread's collection finalizes an object
        needing the same lock. Without the pause this deadlocks.

        This calls pause_gc directly; that the HDF5 open paths reach it is
        covered by test_remote_h5_handle_pauses_gc and its neighbours.
        """
        phil = threading.RLock()
        request = threading.Semaphore(0)
        answer = threading.Semaphore(0)
        stop = threading.Event()

        class _NeedsPhil:
            """Its finalizer takes the lock, as a dead h5py object would."""

            def __init__(self):
                self.self_ref = self  # a cycle: only gc can free it

            def __del__(self):
                with phil:
                    pass

        def loop_thread():
            """Serve reads and make cyclic garbage while doing it."""
            while not stop.is_set():
                if not request.acquire(timeout=0.5):
                    continue
                for _ in range(200):
                    _NeedsPhil()
                answer.release()

        server = threading.Thread(target=loop_thread, daemon=True)
        server.start()
        pause_gc()
        try:
            for _ in range(20):
                with phil:  # h5py holds its lock across the fetch
                    request.release()
                    assert answer.acquire(timeout=20), "deadlocked"
        finally:
            stop.set()
            resume_gc()
            server.join(timeout=5)
        gc.collect()


class TestPauseAccounting:
    """The pause must never be left on, and never lifted early."""

    def test_nests(self):
        """Only the outermost resume re-enables collection."""
        pause_gc()
        pause_gc()
        try:
            assert not gc.isenabled()
            resume_gc()
            assert not gc.isenabled()
        finally:
            resume_gc()
        assert gc.isenabled()

    def test_unbalanced_resume_is_a_no_op(self):
        """A stray resume cannot enable collection during a live session."""
        resume_gc()
        pause_gc()
        try:
            assert not gc.isenabled()
        finally:
            resume_gc()

    def test_user_disabled_gc_is_not_re_enabled(self):
        """A caller who disabled collection keeps it disabled."""
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            pause_gc()
            resume_gc()
            assert not gc.isenabled()
        finally:
            if was_enabled:
                gc.enable()

    def test_leaked_handle_resumes(self):
        """Dropping a handle without closing it releases the pause."""
        _open_paused(_FakeRemoteFile())
        gc.collect()
        assert gc.isenabled()

    def test_stranded_cyclic_handle_is_recovered(self):
        """A handle leaked inside a cycle is healed by the next remote open."""
        holder = {}
        handle = _open_paused(_FakeRemoteFile())
        holder["handle"], holder["self"] = handle, holder  # unreachable cycle
        del handle, holder
        assert not gc.isenabled()
        remote_io._gc_collect_after = 0.0  # the valve is rate limited
        pause_gc()
        resume_gc()
        assert gc.isenabled()

    def test_interrupted_safety_collect_takes_no_pause(self, monkeypatch):
        """An interrupt during the safety collect cannot strand a pause."""
        fake_gc = SimpleNamespace(
            collect=_raise_keyboard_interrupt,
            isenabled=gc.isenabled,
            disable=gc.disable,
            enable=gc.enable,
        )
        monkeypatch.setattr(remote_io, "gc", fake_gc)
        remote_io._gc_collect_after = 0.0  # the valve is rate limited
        with pytest.raises(KeyboardInterrupt):
            pause_gc()
        assert gc.isenabled()

    def test_open_failure_releases_pause_and_fileobj(self):
        """A failed open leaves neither the pause nor the file object behind."""
        fileobj = _FakeRemoteFile()

        def _raise(*args, **kwargs):
            raise ValueError("no")

        with pytest.raises(ValueError):
            _open_paused(fileobj, _raise)
        assert gc.isenabled()
        assert fileobj.closed

    def test_teardown_error_still_resumes(self):
        """A BaseException from the owned file object cannot strand the pause."""

        class _BadClose:
            def close(self):
                raise KeyboardInterrupt("interrupted mid-close")

        class _Handle:
            def close(self):
                pass

        pause_gc()
        managed = _ManagedH5pyFile(_Handle(), _BadClose(), gc_paused=True)
        with suppress(KeyboardInterrupt):
            managed.close()
        assert gc.isenabled()

    def test_inherited_handle_does_not_resume_after_fork(self):
        """A handle carried through a fork cannot resume the child's session."""
        pause_gc()
        try:

            class _Handle:
                def close(self):
                    pass

            managed = _ManagedH5pyFile(_Handle(), None, gc_paused=True)
            managed._gc_paused_pid = os.getpid() + 1  # as if inherited
            managed.close()
            assert not gc.isenabled(), "an inherited close stole a live pause"
        finally:
            resume_gc()

    def test_fork_reset_clears_inherited_pause(self):
        """The fork hook drops a pause no child close can rebalance."""
        pause_gc()
        remote_io._reset_gc_pause_state()
        assert gc.isenabled()
        assert remote_io._gc_pause_depth == 0

    @pytest.mark.concurrency
    def test_concurrent_sessions_keep_the_count_exact(self, run_in_threads):
        """Overlapping pauses must not lose or double-count each other."""

        def _pause_and_resume(_index):
            for _ in range(50):
                pause_gc()
                assert not gc.isenabled()
                resume_gc()

        run_in_threads(_pause_and_resume)
        assert remote_io._gc_pause_depth == 0
        assert gc.isenabled()


class TestLoopBackedDetection:
    """Missing a loop-backed object leaves the deadlock window open."""

    def test_async_filesystem_detected(self):
        """A file object over an async filesystem is loop backed."""
        assert _is_loop_backed(_FakeRemoteFile())

    def test_wrapped_async_filesystem_detected(self):
        """Buffering wrappers hide the filesystem but not the loop thread."""
        wrapped = io.BufferedReader(_FakeRemoteFile())
        assert _is_loop_backed(wrapped)

    def test_plain_stream_is_not_loop_backed(self):
        """Local and in-memory streams must not pause collection."""
        assert not _is_loop_backed(io.BytesIO(b"abc"))

    def test_unavailable_backend_is_not_loop_backed(self):
        """A path whose backend is not installed raises on ``fs``, not here."""

        class _MissingBackend:
            @property
            def fs(self):
                raise ImportError("please install some-fs")

        assert not _is_loop_backed(_MissingBackend())

    def test_sync_filesystem_is_not_loop_backed(self):
        """A synchronous fsspec filesystem needs no pause."""

        class _SyncFile:
            fs = _FakeFS(False)

        assert not _is_loop_backed(_SyncFile())
