"""Tests for the garbage-collection pause used by remote HDF5 reads."""

from __future__ import annotations

import gc
import io
import os
import threading
import warnings
from contextlib import suppress
from types import SimpleNamespace

import pytest
from upath import UPath

import dascore as dc
import dascore.utils.hdf5 as hdf5_module
import dascore.utils.remote_io as remote_io
from dascore.config import config_context
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

    @pytest.mark.concurrency
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
        # A collection has to be due for the pause to be what prevents it.
        # Counting on the default gen-0 threshold means counting rounds
        # against a number CPython is free to change -- and 8 rounds of 200
        # under the current 2000 never reaches it, so the test would pass
        # whether or not gc was paused.
        threshold = gc.get_threshold()
        gc.set_threshold(100)
        pause_gc()
        try:
            for _ in range(8):
                with phil:  # h5py holds its lock across the fetch
                    request.release()
                    assert answer.acquire(timeout=20), "deadlocked"
        finally:
            stop.set()
            resume_gc()
            gc.set_threshold(*threshold)
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

    def test_stranded_cyclic_handle_is_recovered(self, monkeypatch):
        """A handle leaked inside a cycle is healed by the next remote open."""
        holder = {}
        handle = _open_paused(_FakeRemoteFile())
        holder["handle"], holder["self"] = handle, holder  # unreachable cycle
        del handle, holder
        assert not gc.isenabled()
        monkeypatch.setattr(remote_io, "_gc_collect_after", 0.0)  # rate limited
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
        monkeypatch.setattr(remote_io, "_gc_collect_after", 0.0)  # rate limited
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


class TestPauseWarning:
    """The pause is invisible otherwise, so it must announce itself once."""

    @pytest.fixture(autouse=True)
    def _unwarned(self, monkeypatch):
        """Start each test as though nothing had warned yet."""
        monkeypatch.setattr(remote_io, "_gc_pause_warned", False)

    def test_warns_once_per_process(self):
        """A spool over many remote files must not repeat the warning."""
        with pytest.warns(UserWarning, match="pauses Python's automatic garbage"):
            pause_gc()
        resume_gc()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a second warning would raise
            pause_gc()
            resume_gc()

    def test_can_be_silenced(self):
        """`warn_on_gc_pause=False` turns the warning off."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with config_context(warn_on_gc_pause=False):
                pause_gc()
                resume_gc()

    def test_raising_filter_cannot_steal_a_live_pause(self):
        """A warning turned into an error must not release someone else's pause.

        The caller resumes on any failure from pause_gc, so a warning which
        raises before the depth moves would release a pause this call never
        took, re-enabling collection under a handle still open.
        """
        with config_context(warn_on_gc_pause=False):
            pause_gc()  # stand in for a handle already open
        try:
            with pytest.raises(UserWarning), warnings.catch_warnings():
                warnings.simplefilter("error")
                pause_gc()
            resume_gc()  # what _open_h5_fileobj does on failure
            assert not gc.isenabled(), "the live pause was stolen"
            assert remote_io._gc_pause_depth == 1
        finally:
            resume_gc()

    @pytest.mark.concurrency
    @pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
    # Forking a multi-threaded process can wedge the child, and the repo sets
    # no global timeout, so bound it rather than hang the job for an hour.
    @pytest.mark.timeout(30)
    def test_fork_rearms_the_warning(self):
        """A pool worker which pauses collection should say so itself.

        Forks for real rather than calling the reset hook, so this also
        pins that the hook is registered with os.register_at_fork.
        """
        with pytest.warns(UserWarning, match="pauses Python's automatic garbage"):
            pause_gc()
        resume_gc()
        assert remote_io._gc_pause_warned
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:  # the child reports what it inherited, then leaves
            os.close(read_fd)
            os.write(write_fd, b"1" if remote_io._gc_pause_warned else b"0")
            os._exit(0)
        os.close(write_fd)
        inherited = os.read(read_fd, 1)
        os.close(read_fd)
        os.waitpid(pid, 0)
        assert inherited == b"0", "the child inherited the parent's warned flag"


class TestProbeSuppression:
    """Format detection must not be disturbed by the pause warning."""

    def test_probe_survives_warnings_as_errors(
        self, random_patch, tmp_path, monkeypatch
    ):
        """A warning raised while probing must not read as "wrong format".

        `_get_format` treats any exception from a probe as "not my format",
        so a filter turning the warning into an error would silently skip
        the reader which does match. `file_format` pins the probe to that
        reader; without it the once-per-process warning lands on an earlier
        miss and the failure hides.
        """
        path = tmp_path / "probe.h5"
        dc.write(random_patch, path, "dasdae")
        # Make a local file take the loop-backed branch, which pauses.
        monkeypatch.setattr(hdf5_module, "_is_loop_backed", lambda _resource: True)
        monkeypatch.setattr(remote_io, "_gc_pause_warned", False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = dc.get_format(UPath(path), file_format="DASDAE")
        assert out == ("DASDAE", "1")

    def test_probe_does_not_warn(self, random_patch, tmp_path, monkeypatch):
        """Probing claims no HDF5 read; the resource may not even be one."""
        path = tmp_path / "quiet.h5"
        dc.write(random_patch, path, "dasdae")
        monkeypatch.setattr(hdf5_module, "_is_loop_backed", lambda _resource: True)
        monkeypatch.setattr(remote_io, "_gc_pause_warned", False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dc.get_format(UPath(path), file_format="DASDAE")
        assert not [x for x in caught if "automatic garbage" in str(x.message)]


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

    def test_refused_attribute_is_not_loop_backed(self):
        """A wrapper may refuse an attribute with more than AttributeError."""

        class _Refusing:
            def __getattr__(self, name):
                raise io.UnsupportedOperation(name)

        assert not _is_loop_backed(_Refusing())

    def test_sync_filesystem_is_not_loop_backed(self):
        """A synchronous fsspec filesystem needs no pause."""

        class _SyncFile:
            fs = _FakeFS(False)

        assert not _is_loop_backed(_SyncFile())
