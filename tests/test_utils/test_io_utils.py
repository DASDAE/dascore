"""Tests for IO utilities."""

from __future__ import annotations

import gc
import threading
from contextlib import closing
from io import BufferedReader, BufferedWriter, BytesIO, StringIO, TextIOBase
from pathlib import Path

import h5py
import numpy as np
import pytest
from fsspec.asyn import AsyncFileSystem
from upath import UPath

import dascore as dc
import dascore.utils.hdf5 as hdf5_module
import dascore.utils.remote_io as remote_io
from dascore.config import config_context
from dascore.exceptions import PatchConversionError, RemoteCacheError
from dascore.utils.hdf5 import (
    H5Reader,
    H5Writer,
    LocalH5Reader,
    open_h5_resource,
)
from dascore.utils.io import (
    BinaryReader,
    BinaryWriter,
    IOResourceManager,
    LocalBinaryReader,
    LocalPath,
    TextReader,
    ensure_local_file,
    get_handle_from_resource,
    xarray_to_patch,
)
from dascore.utils.misc import suppress_warnings
from dascore.utils.remote_io import (
    _FallbackFileObj,
    _get_cached_local_file,
    _warn_remote_cache_download,
    clear_remote_file_cache,
    get_remote_cache_path,
    get_remote_cache_scope,
    is_no_range_http_error,
    remote_cache_scope,
)


class _DummyHandle:
    """A file-like stand-in that only records being closed."""

    closed = False

    def close(self):
        self.closed = True


class _FakeAsyncFS(AsyncFileSystem):
    """A stand-in async fsspec filesystem which needs no event loop."""

    def __init__(self):
        pass


class _FakeRemoteFile(BytesIO):
    """A file object whose fsspec filesystem serves reads on a loop thread."""

    fs = _FakeAsyncFS()


class _BadType:
    """A dummy type for testing."""


def _dummy_func(arg: Path, arg2: _BadType) -> int:
    """A dummy function."""


class _FailOnSeek(BytesIO):
    """A test handle which raises once on seek."""

    def __init__(self, data: bytes, exc: Exception):
        super().__init__(data)
        self._exc = exc
        self.triggered = False

    def seek(self, offset, whence=0):
        """Raise once, then defer to BytesIO."""
        if not self.triggered:
            self.triggered = True
            raise self._exc
        return super().seek(offset, whence)


class _NoTellHandle(BytesIO):
    """A test handle whose tell method fails."""

    def tell(self):
        """Raise to exercise fallback position tracking."""
        raise OSError("tell failed")


class _PlainHandle:
    """A simple handle without writable/flush helpers."""

    def __init__(self):
        self.closed = False
        self.extra_attr = "value"

    def read(self, _size=-1):
        return b""

    def seek(self, offset, _whence=0):
        return offset

    def tell(self):
        return 0

    def close(self):
        self.closed = True


class _WritableHandle(_PlainHandle):
    """A handle exposing a writable method."""

    def writable(self):
        return True


class TestGetHandleFromResource:
    """Tests for getting the file handle from specific resources."""

    def test_bad_type(self):
        """
        In order to not break anything, unsupported types should just
        return the original argument.
        """
        out = get_handle_from_resource("here", _BadType)
        assert out == "here"

    def test_path_to_buffered_reader(self, tmp_path):
        """Ensure we get a reader from tmp path reader."""
        path = tmp_path / "test_read_buffer.txt"
        path.touch()
        with closing(get_handle_from_resource(path, BinaryReader)) as handle:
            assert isinstance(handle, BufferedReader)

    def test_path_to_buffered_writer(self, tmp_path):
        """Ensure we get a reader from tmp path reader."""
        path = tmp_path / "test_buffered_writer.txt"
        with closing(get_handle_from_resource(path, BinaryWriter)) as handle:
            assert isinstance(handle, BufferedWriter)

    def test_path_to_text_reader(self, tmp_path):
        """Ensure text reader opens text streams."""
        path = tmp_path / "test_text_reader.txt"
        path.write_text("hello")
        with closing(get_handle_from_resource(path, TextReader)) as handle:
            assert isinstance(handle, TextIOBase)

    def test_stringio_to_text_reader(self):
        """Ensure StringIO is accepted by TextReader."""
        resource = StringIO("abc")
        out = get_handle_from_resource(resource, TextReader)
        assert out is resource

    def test_binary_stream_not_text_reader(self):
        """Ensure binary streams are rejected by TextReader."""
        with pytest.raises(NotImplementedError):
            get_handle_from_resource(BytesIO(b"abc"), TextReader)

    def test_path_to_hdf5_reader(self, generic_hdf5):
        """Ensure we get a reader from tmp path reader."""
        with closing(get_handle_from_resource(generic_hdf5, H5Reader)) as handle:
            assert "bob" in handle  # h5py-file-like

    def test_path_to_hdf5_writer(self, tmp_path):
        """Ensure we get a writer from tmp path."""
        path = tmp_path / "test_hdf_writer.h5"
        with closing(get_handle_from_resource(path, H5Writer)) as handle:
            handle.create_group("waveforms")
            assert "waveforms" in handle

    def test_get_path(self, tmp_path):
        """Ensure we can get a path."""
        path = get_handle_from_resource(tmp_path, Path)
        assert isinstance(path, Path)

    def test_get_str(self, tmp_path):
        """Unsupported string targets should keep richer path objects."""
        out = get_handle_from_resource(tmp_path, str)
        assert isinstance(out, Path)
        assert out == tmp_path

    def test_get_upath(self, tmp_path):
        """Ensure we can get a UPath."""
        path = get_handle_from_resource(tmp_path, UPath)
        assert isinstance(path, UPath)

    def test_already_file_handle(self, tmp_path):
        """Ensure an input that is already the requested type works."""
        path = tmp_path / "pass_back.txt"
        with open(path, "wb") as fi:
            out = get_handle_from_resource(fi, BinaryWriter)
            assert out is fi

    def test_binary_reader_from_upath(self, tmp_path):
        """Ensure binary readers can open UPath resources directly."""
        path = UPath(tmp_path / "upath.bin")
        path.write_bytes(b"abc")
        with closing(get_handle_from_resource(path, BinaryReader)) as handle:
            assert handle.read() == b"abc"

    def test_binary_reader_resets_buffered_stream(self):
        """Ensure BinaryReader resets offsets on binary streams."""
        resource = BytesIO(b"abc")
        _ = resource.read(1)
        out = BinaryReader.get_handle(resource)
        assert out is resource
        assert out.tell() == 0
        assert out.read(1) == b"a"

    def test_text_reader_from_upath(self, tmp_path):
        """Ensure text readers can open UPath resources directly."""
        path = UPath(tmp_path / "upath.txt")
        path.write_text("abc")
        with closing(get_handle_from_resource(path, TextReader)) as handle:
            assert handle.read() == "abc"

    def test_binary_writer_to_remote_upath(self):
        """Binary writers should create remote UPath files."""
        path = UPath("memory://dascore/upath-write.bin")
        with closing(get_handle_from_resource(path, BinaryWriter)) as handle:
            handle.write(b"abc")
        assert path.read_bytes() == b"abc"

    def test_local_binary_reader_passthrough_resets_offset(self):
        """Ensure LocalBinaryReader preserves passthrough stream behavior."""
        resource = BytesIO(b"abc")
        _ = resource.read(1)
        out = LocalBinaryReader.get_handle(resource)
        assert out is resource
        assert out.tell() == 0

    def test_h5_reader_from_open_file_handle(self, tmp_path):
        """Ensure h5py-backed readers support open file handles."""
        path = tmp_path / "handle.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("data", data=[1, 2, 3])
        with open(path, "rb") as raw:
            handle = H5Reader.get_handle(raw)
            try:
                assert type(handle).__name__ == "_ManagedH5pyFile"
                assert list(handle["data"][:]) == [1, 2, 3]
            finally:
                handle.close()

    def test_h5_reader_close_closes_owned_fileobj(self, tmp_path):
        """Closing the reader should close the file object passed to h5py."""
        path = tmp_path / "owned_handle.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("data", data=[1, 2, 3])
        raw = open(path, "rb")
        handle = H5Reader.get_handle(raw)
        assert not raw.closed
        handle.close()
        assert raw.closed

    def test_h5_reader_managed_handle_context_manager_and_closed(self, tmp_path):
        """Managed HDF5 handles should support context-manager helpers."""
        path = tmp_path / "managed_context.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("data", data=[1, 2, 3])
        raw = open(path, "rb")
        with H5Reader.get_handle(raw) as handle:
            assert "data" in handle
            assert list(iter(handle)) == ["data"]
            assert not handle.closed
        assert handle.closed
        assert raw.closed

    def test_h5_reader_prefers_existing_cached_local_file(self, monkeypatch, tmp_path):
        """Cached remote HDF5 resources should reopen locally, not remotely."""
        local_path = tmp_path / "cached.h5"
        with h5py.File(local_path, "w") as handle:
            handle.create_dataset("data", data=[1, 2, 3])

        path = UPath("http://example.com/cached.h5")
        monkeypatch.setattr(
            "dascore.utils.hdf5._get_cached_local_file", lambda _: local_path
        )
        monkeypatch.setattr(
            type(path),
            "open",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("remote open should not be used")
            ),
        )

        handle = H5Reader.get_handle(path)
        try:
            assert list(handle["data"][:]) == [1, 2, 3]
        finally:
            handle.close()

    def test_h5_reader_wraps_existing_h5py_handle(self, tmp_path):
        """Ensure h5py-backed readers wrap existing open handles consistently."""
        path = tmp_path / "passthrough.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("data", data=[1, 2, 3])
        with h5py.File(path, "r") as raw:
            handle = H5Reader.get_handle(raw)
            assert type(handle).__name__ == "_ManagedH5pyFile"
            assert list(handle["data"][:]) == [1, 2, 3]
            handle.close()
            assert raw.id.valid == 0

    def test_local_h5_reader_materializes_local_path(self, tmp_path):
        """Ensure LocalH5Reader can open a local path through its adapter."""
        path = tmp_path / "local_h5_reader.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("data", data=[1, 2, 3])
        handle = LocalH5Reader.get_handle(path)
        try:
            assert type(handle).__name__ == "_ManagedH5pyFile"
            assert list(handle["data"][:]) == [1, 2, 3]
        finally:
            handle.close()

    def test_h5_reader_wraps_local_path(self, tmp_path):
        """Local path opens should use the same managed HDF5 handle type."""
        path = tmp_path / "managed_local.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("data", data=[1, 2, 3])
        handle = H5Reader.get_handle(path)
        try:
            assert type(handle).__name__ == "_ManagedH5pyFile"
            assert list(handle["data"][:]) == [1, 2, 3]
        finally:
            handle.close()

    def test_open_h5_resource_passthrough_managed_handle(self, tmp_path):
        """The low-level helper should return managed handles unchanged."""
        path = tmp_path / "managed_passthrough.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("data", data=[1, 2, 3])
        handle = H5Reader.get_handle(path)
        try:
            out = open_h5_resource(
                handle,
                mode=H5Reader.mode,
                constructor=H5Reader.constructor,
                open_kwargs_getter=H5Reader._get_open_kwargs,
            )
            assert out is handle
        finally:
            handle.close()

    def test_h5_reader_passthrough_managed_handle(self, tmp_path):
        """Reader-level get_handle should return managed handles unchanged."""
        path = tmp_path / "managed_reader_passthrough.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("data", data=[1, 2, 3])
        handle = H5Reader.get_handle(path)
        try:
            assert H5Reader.get_handle(handle) is handle
        finally:
            handle.close()

    def test_open_h5_resource_raises_on_unsupported_resource(self):
        """Unsupported HDF5 resources should raise a clear error."""
        with pytest.raises(NotImplementedError, match="Couldn't get handle"):
            open_h5_resource(
                _BadType(),
                mode=H5Reader.mode,
                constructor=H5Reader.constructor,
                open_kwargs_getter=H5Reader._get_open_kwargs,
            )

    def test_h5_reader_closes_upath_handle_on_constructor_error(
        self, tmp_path, monkeypatch
    ):
        """Ensure constructor failures close UPath-opened file handles."""
        handle = _DummyHandle()
        path = UPath(tmp_path / "error.h5")
        path.write_bytes(b"not an hdf5")
        monkeypatch.setattr(type(path), "open", lambda self, *args, **kwargs: handle)
        monkeypatch.setattr(
            H5Reader,
            "constructor",
            staticmethod(
                lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
            ),
        )
        with pytest.raises(RuntimeError, match="boom"):
            H5Reader.get_handle(path)
        assert handle.closed

    @pytest.mark.parametrize(
        ("url", "options", "expected"),
        [
            (
                "s3://example-bucket/example.h5",
                {"anon": True},
                {"cache_type": "readahead"},
            ),
            (
                "http://example.com/example.h5",
                {},
                {"cache_type": "blockcache", "cache_options": {"maxblocks": 8}},
            ),
            (
                "https://example.com/example.h5",
                {},
                {"cache_type": "blockcache", "cache_options": {"maxblocks": 8}},
            ),
        ],
    )
    def test_remote_h5_open_kwargs_are_tuned(self, monkeypatch, url, options, expected):
        """Remote HDF5 opens must override backend defaults that overfetch.

        s3fs defaults to 50 MB readahead blocks. HTTP instead needs a block
        LRU: the h5py metadata probe alternates between the file header and
        footer, and a single-window cache refetches on every jump.
        """
        opened = {}
        path = UPath(url, **options)

        def _open(_self, _mode, **kwargs):
            opened.update(kwargs)
            return _DummyHandle()

        monkeypatch.setattr(type(path), "open", _open)
        monkeypatch.setattr(
            H5Reader,
            "constructor",
            staticmethod(lambda *args, **kwargs: _DummyHandle()),
        )
        with config_context(remote_hdf5_block_size=1234):
            H5Reader.get_handle(path).close()
        assert opened["block_size"] == 1234
        for key, value in expected.items():
            assert opened[key] == value

    def test_remote_h5_handle_pauses_gc(self, monkeypatch):
        """Automatic collection stays paused while a remote handle is open."""
        path = UPath("http://example.com/gc-pause.h5")
        # Stand in for the real filesystem, which needs aiohttp; some
        # platforms DASCore supports (wasm, free-threaded) do not have it.
        monkeypatch.setattr(type(path), "fs", property(lambda _self: _FakeAsyncFS()))
        monkeypatch.setattr(type(path), "open", lambda *a, **k: _DummyHandle())
        monkeypatch.setattr(
            H5Reader,
            "constructor",
            staticmethod(lambda *args, **kwargs: _DummyHandle()),
        )
        assert gc.isenabled()
        handle = H5Reader.get_handle(path)
        try:
            assert not gc.isenabled()
        finally:
            handle.close()
        assert gc.isenabled()
        # Closing twice must not unbalance the pause bookkeeping.
        handle.close()
        assert gc.isenabled()

    def test_dropped_wrapper_leaves_caller_handle_open(self, generic_hdf5):
        """Collecting a wrapper must not close the handle its caller owns."""
        file = h5py.File(generic_hdf5, "r")
        try:
            wrapper = H5Reader.get_handle(file)
            assert wrapper is not file
            del wrapper
            gc.collect()
            assert file  # h5py files are falsey once closed
        finally:
            file.close()

    def test_failed_open_leaves_caller_fileobj_usable(self):
        """A caller's file object must survive a failed HDF5 open.

        get_format hands the same object to every FiberIO in turn, so an
        HDF5 miss cannot close it out from under the next one.
        """
        buffer = BytesIO(b"not an hdf5 file")
        with pytest.raises(OSError):
            H5Reader.get_handle(buffer)
        assert not buffer.closed

    def test_local_upath_does_not_pause_gc(self, generic_hdf5):
        """A synchronous backend has no loop thread, so it must not pause gc."""
        handle = H5Reader.get_handle(UPath(generic_hdf5))
        try:
            assert gc.isenabled()
        finally:
            handle.close()

    def test_loop_backed_fileobj_pauses_gc(self, monkeypatch):
        """User-supplied fsspec async file objects need the GC pause too."""
        monkeypatch.setattr(
            H5Reader,
            "constructor",
            staticmethod(lambda *args, **kwargs: BytesIO()),
        )
        assert gc.isenabled()
        handle = H5Reader.get_handle(_FakeRemoteFile())
        try:
            assert not gc.isenabled()
        finally:
            handle.close()
        assert gc.isenabled()

    def test_loop_backed_fileobj_constructor_error_resumes_gc(self, monkeypatch):
        """A failed h5py construction must rebalance the GC pause."""

        def _explode(*args, **kwargs):
            raise ValueError("not an hdf5 fileobj")

        monkeypatch.setattr(H5Reader, "constructor", staticmethod(_explode))
        assert gc.isenabled()
        with pytest.raises(ValueError, match="not an hdf5 fileobj"):
            H5Reader.get_handle(_FakeRemoteFile())
        assert gc.isenabled()

    def test_h5_writer_to_remote_upath(self):
        """HDF5 writers should create remote UPath files via write-back."""
        path = UPath("memory://dascore/upath-write.h5")
        handle = H5Writer.get_handle(path)
        try:
            handle.create_dataset("data", data=[1, 2, 3])
        finally:
            handle.close()
        with path.open("rb") as raw:
            with h5py.File(raw, "r", driver="fileobj") as reopened:
                assert list(reopened["data"][:]) == [1, 2, 3]

    def test_h5_writer_to_remote_upath_aborts_on_context_error(self):
        """Remote HDF5 writers should not upload partial files on exceptions."""
        path = UPath("memory://dascore/upath-write-abort.h5")
        with pytest.raises(RuntimeError, match="boom"):
            with H5Writer.get_handle(path) as handle:
                handle.create_dataset("data", data=[1, 2, 3])
                raise RuntimeError("boom")
        assert not path.exists()

    def test_h5_writer_remote_context_commits(self):
        """Leaving the context without an error uploads the file."""
        path = UPath("memory://dascore/upath-write-commit.h5")
        with H5Writer.get_handle(path) as handle:
            handle.create_dataset("data", data=[1, 2, 3])
        assert path.exists()
        with path.open("rb") as raw, h5py.File(raw, "r", driver="fileobj") as reopened:
            assert list(reopened["data"][:]) == [1, 2, 3]

    def test_h5_writer_remote_append_keeps_existing(self):
        """Reopening an existing remote file downloads it before writing."""
        path = UPath("memory://dascore/upath-write-append.h5")
        with H5Writer.get_handle(path) as handle:
            handle.create_dataset("first", data=[1, 2, 3])
        with H5Writer.get_handle(path) as handle:
            handle.create_dataset("second", data=[4, 5, 6])
        with path.open("rb") as raw, h5py.File(raw, "r", driver="fileobj") as reopened:
            assert list(reopened["first"][:]) == [1, 2, 3]
            assert list(reopened["second"][:]) == [4, 5, 6]

    def test_h5_writer_remote_setitem_and_contains(self):
        """The remote writer proxies item assignment and membership."""
        path = UPath("memory://dascore/upath-write-setitem.h5")
        with H5Writer.get_handle(path) as handle:
            handle["data"] = [1, 2, 3]
            assert "data" in handle
            assert "missing" not in handle

    def test_h5_writer_remote_cleans_up_after_open_failure(self, monkeypatch):
        """A failed local open removes the temp file and raises."""
        created = []
        real_mkstemp = hdf5_module.tempfile.mkstemp

        def _tracking_mkstemp(*args, **kwargs):
            file_descriptor, name = real_mkstemp(*args, **kwargs)
            created.append(Path(name))
            return file_descriptor, name

        def _raise(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(hdf5_module.tempfile, "mkstemp", _tracking_mkstemp)
        monkeypatch.setattr(hdf5_module, "H5pyFile", _raise)
        path = UPath("memory://dascore/upath-write-open-failure.h5")
        with pytest.raises(RuntimeError, match="boom"):
            H5Writer.get_handle(path)
        assert created and not created[0].exists()

    def test_h5_writer_remote_abort_is_idempotent(self):
        """Remote writer aborts should be safe to call more than once."""
        path = UPath("memory://dascore/upath-write-abort-twice.h5")
        handle = H5Writer.get_handle(path)
        handle.create_dataset("data", data=[1, 2, 3])
        handle._abort()
        handle._abort()
        assert not path.exists()

    def test_not_implemented(self):
        """Tests for raising not implemented errors for types not supported."""
        bad_instance = _BadType()
        with pytest.raises(NotImplementedError):
            get_handle_from_resource(bad_instance, BinaryReader)
        with pytest.raises(NotImplementedError):
            get_handle_from_resource(bad_instance, BinaryWriter)
        with pytest.raises(NotImplementedError):
            get_handle_from_resource(bad_instance, H5Writer)
        with pytest.raises(NotImplementedError):
            get_handle_from_resource(bad_instance, H5Reader)


class TestIOResourceManager:
    """Tests for the IO resource manager."""

    @pytest.fixture(autouse=True)
    def clear_remote_cache(self):
        """Ensure remote cache state doesn't leak between tests."""
        with config_context(warn_on_remote_cache=False):
            clear_remote_file_cache()
            yield
            clear_remote_file_cache()

    def test_basic_context_manager(self, tmp_path):
        """Ensure it works as a context manager."""
        write_path = tmp_path / "io_writer"

        with IOResourceManager(write_path) as man:
            path_from_hint = man.get_resource(_dummy_func)
            assert isinstance(path_from_hint, Path)
            path = man.get_resource(Path)
            assert isinstance(path, Path)
            hf = man.get_resource(H5Writer)
            fi = man.get_resource(BinaryWriter)
            assert not hf.closed
            assert not fi.closed
        # after the context manager exits everything should be closed.
        assert hf.closed
        assert fi.closed

    def test_get_none_resource_returns_source(self):
        """Requesting no specific resource should return the original source."""
        source = object()
        with IOResourceManager(source) as man:
            assert man.get_resource(None) is source

    def test_error_in_context_aborts_handles(self):
        """An exception inside the context must abort, not commit, handles."""

        class _Recorder:
            aborted = False
            closed = False

            def abort(self):
                self.aborted = True

            def close(self):
                self.closed = True

        recorder = _Recorder()
        man = IOResourceManager("unused")
        man._cache["key"] = recorder
        with pytest.raises(ValueError, match="boom"):
            with man:
                raise ValueError("boom")
        assert recorder.aborted
        assert not recorder.closed
        # A clean exit closes normally.
        recorder2 = _Recorder()
        man2 = IOResourceManager("unused")
        man2._cache["key"] = recorder2
        with man2:
            pass
        assert recorder2.closed
        assert not recorder2.aborted

    def test_failed_abort_does_not_mask_original_error(self):
        """A cleanup failure must not replace the error which caused it."""

        class _BadAbort:
            def abort(self):
                raise OSError("abort failed")

        man = IOResourceManager("unused")
        man._cache["key"] = _BadAbort()
        with pytest.raises(ValueError, match="boom") as exc_info:
            with man:
                raise ValueError("boom")
        assert any("abort failed" in note for note in exc_info.value.__notes__)

    def test_close_all_survives_failing_handle(self):
        """One handle raising must not skip cleanup of the others."""

        class _Exploder:
            def close(self):
                raise OSError("close failed")

        class _Recorder:
            closed = False

            def close(self):
                self.closed = True

        recorder = _Recorder()
        man = IOResourceManager("unused")
        man._cache["bad"] = _Exploder()
        man._cache["good"] = recorder
        with pytest.raises(OSError, match="close failed"):
            man.close_all()
        assert recorder.closed

    def test_non_pathlike_resource_passthrough(self):
        """Non-pathlike resources should bypass path coercion entirely."""
        source = BytesIO(b"abc")
        with IOResourceManager(source) as man:
            out = man.get_resource(BinaryReader)
            assert out is source

    def test_nested_context(self, tmp_path):
        """Ensure nested context works as well."""
        write_path = tmp_path / "io_writer"
        with IOResourceManager(write_path) as man:
            fi1 = man.get_resource(BinaryWriter)
            with IOResourceManager(man):
                fi2 = man.get_resource(BinaryWriter)
                # nested IOManager should just return value from previous
                assert fi1 is fi2
            # on first exist the resource should remain open
            assert not fi2.closed
        # then closed.
        assert fi2.closed

    def test_closed_after_exception(self, tmp_path):
        """Ensure the file resources are closed after an exception."""
        path = tmp_path / "closed_resource_test.txt"
        path.touch()
        try:
            with IOResourceManager(path) as man:
                fi = man.get_resource(BinaryReader)
                raise ValueError("Waaagh!")
        except ValueError:
            assert fi.closed

    def test_remote_path_is_materialized_in_cache(self):
        """Remote resources that need local paths should be cache-backed."""
        path = UPath("memory://dascore/io_resource_test.txt")
        path.write_text("hello")
        with IOResourceManager(path) as man:
            local_path = man.get_resource(Path)
            assert isinstance(local_path, Path)
            assert local_path.exists()
            assert local_path.read_text() == "hello"
        assert local_path.exists()
        assert get_remote_cache_path() in local_path.parents

    def test_remote_cache_dir_comes_from_config(self, tmp_path):
        """Configured remote cache directories should be used for materialization."""
        path = UPath("memory://dascore/io_resource_test_custom_cache.txt")
        path.write_text("hello")
        cache_dir = tmp_path / "remote-cache"
        with config_context(remote_cache_dir=cache_dir):
            local_path = ensure_local_file(path)
        assert cache_dir in local_path.parents
        assert local_path.exists()

    def test_remote_path_can_return_upath(self):
        """Remote resources should be returned unchanged for UPath consumers."""
        path = UPath("memory://dascore/io_resource_test_upath.txt")
        path.write_text("hello")
        with IOResourceManager(path) as man:
            out = man.get_resource(UPath)
            assert isinstance(out, UPath)
            assert out == path

    def test_remote_path_as_string_preserves_remote_identity(self):
        """Remote string requests should preserve remote identity without caching."""
        path = UPath("memory://dascore/io_resource_test_str.txt")
        path.write_text("hello")
        with IOResourceManager(path) as man:
            out = man.get_resource(str)
            assert isinstance(out, UPath)
            assert out == path
        assert not list(get_remote_cache_path().rglob(path.name))

    def test_remote_path_to_binary_reader(self):
        """Binary readers should consume remote resources directly when possible."""
        path = UPath("memory://dascore/io_resource_test_binary.bin")
        path.write_bytes(b"abc")
        with IOResourceManager(path) as man:
            out = man.get_resource(BinaryReader)
            assert out.read() == b"abc"

    def test_remote_path_to_local_binary_reader(self):
        """Local binary readers should materialize remote resources once."""
        path = UPath("memory://dascore/io_resource_test_local_binary.bin")
        path.write_bytes(b"abc")
        with IOResourceManager(path) as man:
            out = man.get_resource(LocalBinaryReader)
            assert out.read() == b"abc"
        cached_files = list(
            get_remote_cache_path().rglob("io_resource_test_local_binary.bin")
        )
        assert len(cached_files) == 1

    def test_remote_path_to_local_path(self):
        """LocalPath should return a cache-backed local path."""
        path = UPath("memory://dascore/io_resource_test_local_path.bin")
        path.write_text("hello")
        with IOResourceManager(path) as man:
            out = man.get_resource(LocalPath)
            assert isinstance(out, Path)
            assert out.exists()
            assert out.read_text() == "hello"

    def test_remote_path_reuses_cached_local_file(self):
        """Repeated materialization of one remote file should reuse the cache entry."""
        path = UPath("memory://dascore/io_resource_test_reuse.txt")
        path.write_text("hello")
        with IOResourceManager(path) as man:
            first = man.get_resource(Path)
        with IOResourceManager(path) as man:
            second = man.get_resource(Path)
        assert first == second
        assert first.exists()

    def test_clear_remote_cache_removes_cached_files(self):
        """Clearing the remote cache should remove cached local artifacts."""
        path = UPath("memory://dascore/io_resource_test_clear.txt")
        path.write_text("hello")
        with IOResourceManager(path) as man:
            local_path = man.get_resource(Path)
            assert local_path.exists()
        clear_remote_file_cache()
        assert not local_path.exists()

    def test_ensure_local_file_reuses_cached_path(self):
        """Repeated ensure_local_file calls should return one stable local path."""
        path = UPath("memory://dascore/io_resource_test_ensure.txt")
        path.write_text("hello")
        first = ensure_local_file(path)
        second = ensure_local_file(path)
        assert first == second
        assert first.exists()
        assert first.read_text() == "hello"

    def test_get_cached_local_file_returns_existing_cached_path(self):
        """The cache helper should find already materialized remote resources."""
        path = UPath("memory://dascore/io_resource_test_cached_lookup.txt")
        path.write_text("hello")
        local_path = ensure_local_file(path)
        assert _get_cached_local_file(path) == local_path

    def test_ensure_local_file_respects_cache_dir_changes(self, tmp_path):
        """Changing the configured cache dir should change future materialization."""
        path = UPath("memory://dascore/io_resource_test_reconfigure.txt")
        path.write_text("hello")
        first_cache = tmp_path / "remote-cache-a"
        second_cache = tmp_path / "remote-cache-b"

        with config_context(remote_cache_dir=first_cache):
            first = ensure_local_file(path)
        with config_context(remote_cache_dir=second_cache):
            second = ensure_local_file(path)

        assert first_cache in first.parents
        assert second_cache in second.parents
        assert first != second
        assert first.exists()
        assert second.exists()
        assert first.read_text() == "hello"
        assert second.read_text() == "hello"

    def test_ensure_local_file_preserves_upath_storage_options(self, monkeypatch):
        """Remote cache materialization should keep UPath storage options intact."""
        path = UPath(
            "s3://gdr-data-lake/soda_lake/raw_seismic/2010/v1.0.0/F1000R1.SGY",
            anon=True,
        )
        seen = {}

        def _fake_download(resource, local_path):
            seen["storage_options"] = dict(resource.storage_options)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(b"abc")

        monkeypatch.setattr(remote_io, "_download_remote_file", _fake_download)
        local_path = ensure_local_file(path)
        assert local_path.exists()
        assert local_path.read_bytes() == b"abc"
        assert seen["storage_options"] == {"anon": True}

    def test_remote_download_uses_configured_block_size(self, monkeypatch):
        """Remote file materialization should honor configured read chunk size."""

        class _RemoteHandle:
            def __init__(self):
                self.read_sizes = []

            def read(self, size=-1):
                self.read_sizes.append(size)
                return b"a" if len(self.read_sizes) == 1 else b""

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        handle = _RemoteHandle()
        path = UPath("memory://dascore/io_resource_test_block_size.bin")
        monkeypatch.setattr(type(path), "open", lambda *_args, **_kwargs: handle)

        with config_context(remote_download_block_size=321):
            local_path = ensure_local_file(path)

        assert local_path.exists()
        assert local_path.read_bytes() == b"a"
        assert handle.read_sizes == [321, 321]

    def test_http_remote_download_uses_upath_open(self, monkeypatch, tmp_path):
        """HTTP cache downloads should preserve the fsspec transport."""

        class _HTTPResource:
            def __init__(self):
                self.protocol = "http"
                self.storage_options = {
                    "headers": {"User-Agent": "dascore-test"},
                    "client_kwargs": {"trust_env": True},
                }
                self.open_args = None

            def open(self, *args, **kwargs):
                self.open_args = (args, kwargs)
                return response

        class _HTTPResponse:
            def __init__(self):
                self._chunks = [b"ab", b"c", b""]
                self.read_sizes = []

            def read(self, size=-1):
                self.read_sizes.append(size)
                return self._chunks.pop(0)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        response = _HTTPResponse()
        resource = _HTTPResource()

        monkeypatch.setattr(remote_io, "coerce_to_upath", lambda resource: resource)
        with config_context(remote_download_block_size=2):
            local_path = tmp_path / "downloaded.bin"
            remote_io._download_remote_file(resource, local_path)

        assert local_path.read_bytes() == b"abc"
        assert resource.open_args == (("rb",), {"block_size": 0})
        assert response.read_sizes == [2, 2, 2]

    def test_ensure_local_file_can_unwrap_io_resource_manager(self):
        """ensure_local_file should accept IOResourceManager instances."""
        path = UPath("memory://dascore/io_resource_test_manager.txt")
        path.write_text("hello")
        with IOResourceManager(path) as man:
            local_path = ensure_local_file(man)
        assert local_path.exists()
        assert local_path.read_text() == "hello"

    def test_ensure_local_file_uses_local_named_resource(self, tmp_path):
        """Local named resources should resolve to their local file path."""
        path = tmp_path / "named_resource.txt"
        path.write_text("hello")
        with path.open() as handle:
            assert ensure_local_file(handle) == path

    def test_ensure_local_file_invalid_resource_raises(self):
        """Objects without local or remote path semantics should fail."""
        with pytest.raises(TypeError, match="Cannot ensure a local file"):
            ensure_local_file(object())

    def test_ensure_local_file_warns_on_first_remote_download(self):
        """First-time remote cache materialization should warn."""
        path = UPath("memory://dascore/io_resource_test_warn.txt")
        path.write_text("hello")
        with config_context(warn_on_remote_cache=True):
            with pytest.warns(
                UserWarning,
                match=r"Downloading remote file memory://\.\.\./io_resource_test_warn\.txt",
            ):
                local_path = ensure_local_file(path)
        assert local_path.exists()

    def test_ensure_local_file_reuse_is_silent_after_first_download(self):
        """Cache hits should not warn after a remote file is already cached."""
        path = UPath("memory://dascore/io_resource_test_warn_reuse.txt")
        path.write_text("hello")
        with config_context(warn_on_remote_cache=True):
            with pytest.warns(UserWarning, match="Downloading remote file"):
                first = ensure_local_file(path)
            with suppress_warnings(action="always", record=True) as record:
                second = ensure_local_file(path)
        assert not record
        assert first == second

    def test_ensure_local_file_warning_can_be_disabled(self):
        """Configured warning suppression should keep downloads silent."""
        path = UPath("memory://dascore/io_resource_test_warn_off.txt")
        path.write_text("hello")
        with config_context(warn_on_remote_cache=False):
            with suppress_warnings(action="always", record=True) as record:
                local_path = ensure_local_file(path)
        assert not record
        assert local_path.exists()

    def test_remote_warning_redacts_multi_protocol_names(self, monkeypatch):
        """Warning messages should redact tuple/list-style protocol values."""

        class _Resource:
            protocol = ("zip", "s3")
            name = "archive.h5"
            suffix = ".h5"

        monkeypatch.setattr(remote_io, "coerce_to_upath", lambda resource: resource)
        out = remote_io._redact_remote_resource(_Resource())
        assert out == "zip+s3://.../archive.h5"

    def test_ensure_local_file_raises_when_remote_cache_disabled(self):
        """Disabling remote caching should block local materialization."""
        path = UPath("memory://dascore/io_resource_test_disabled.txt")
        path.write_text("hello")
        with config_context(allow_remote_cache=False):
            with pytest.raises(RemoteCacheError, match="Remote caching is disabled"):
                ensure_local_file(path)
        assert not list(get_remote_cache_path().rglob(path.name))

    def test_metadata_scope_raises_when_metadata_cache_disabled(self):
        """Metadata scope should reject downloads unless explicitly enabled."""
        path = UPath("memory://dascore/io_resource_test_metadata_disabled.txt")
        path.write_text("hello")
        with remote_cache_scope("metadata"):
            with pytest.raises(
                RemoteCacheError, match="allow_remote_cache_for_metadata"
            ):
                ensure_local_file(path)
        assert not list(get_remote_cache_path().rglob(path.name))

    def test_metadata_scope_allows_download_when_enabled(self):
        """Metadata scope should permit downloads when opted in."""
        path = UPath("memory://dascore/io_resource_test_metadata_enabled.txt")
        path.write_text("hello")
        with config_context(
            allow_remote_cache_for_metadata=True, warn_on_remote_cache=False
        ):
            with remote_cache_scope("metadata"):
                local_path = ensure_local_file(path)
        assert local_path.exists()

    def test_read_scope_overrides_metadata_default(self):
        """Read scope should still allow downloads with default metadata policy."""
        path = UPath("memory://dascore/io_resource_test_read_scope.txt")
        path.write_text("hello")
        with remote_cache_scope("read"):
            assert get_remote_cache_scope() == "read"
            local_path = ensure_local_file(path)
        assert local_path.exists()

    def test_remote_cache_scope_restores_previous_value_after_exception(self):
        """Nested scope changes should unwind even when the body raises."""
        assert get_remote_cache_scope() == "default"
        with pytest.raises(RuntimeError, match="boom"):
            with remote_cache_scope("metadata"):
                assert get_remote_cache_scope() == "metadata"
                raise RuntimeError("boom")
        assert get_remote_cache_scope() == "default"

    def test_metadata_scope_download_warning_guidance(self, tmp_path):
        """Metadata scope yields metadata-specific download-warning guidance."""
        resource = UPath("memory://dascore/metadata_warning.txt")
        with remote_cache_scope("metadata"):
            with pytest.warns(UserWarning, match="allow_remote_cache_for_metadata"):
                _warn_remote_cache_download(resource, tmp_path / "metadata_warning.txt")


class TestRemoteIOFallback:
    """Tests for remote fallback helpers."""

    def test_no_range_error_predicate_matches_expected_message(self):
        """The helper should only match the known no-range HTTP failure."""
        exc = ValueError(
            "The HTTP server doesn't appear to support range requests. "
            "Only reading this file from the beginning is supported."
        )
        assert is_no_range_http_error(exc)
        assert not is_no_range_http_error(ValueError("different error"))
        assert not is_no_range_http_error(RuntimeError("range requests"))

    def test_no_range_error_predicate_matches_streaming_seek(self):
        """Seeking a streaming (size-less) HTTP file needs the same fallback."""
        exc = ValueError("Cannot seek streaming HTTP file")
        assert is_no_range_http_error(exc)
        assert not is_no_range_http_error(RuntimeError(str(exc)))


class TestFallbackFileObj:
    """Tests for switching failed remote handles to local cache files."""

    def test_fallback_file_obj_switches_once_and_preserves_position(self):
        """_FallbackFileObj should retry on the local file and preserve cursor."""
        remote = _FailOnSeek(
            b"abcdef",
            ValueError(
                "The HTTP server doesn't appear to support range requests. "
                "Only reading this file from the beginning is supported."
            ),
        )
        local_handles = []

        def _open_local():
            assert remote.closed
            handle = BytesIO(b"abcdef")
            local_handles.append(handle)
            return handle

        handle = _FallbackFileObj(
            remote_opener=lambda: remote,
            local_opener=_open_local,
            error_predicate=is_no_range_http_error,
        )
        try:
            assert handle.read(2) == b"ab"
            assert handle.seek(4) == 4
            assert handle.read(2) == b"ef"
            assert len(local_handles) == 1
        finally:
            handle.close()

    def test_fallback_file_obj_uses_fallback_position_when_tell_fails(self):
        """Fallback position should be used if the wrapped handle cannot tell."""
        handle = _FallbackFileObj(
            remote_opener=lambda: _NoTellHandle(b"abcdef"),
            local_opener=lambda: BytesIO(b"abcdef"),
            error_predicate=is_no_range_http_error,
        )
        handle._handle = _NoTellHandle(b"abcdef")
        handle._set_pos_from_handle(fallback=3)
        assert handle._pos == 3
        handle.close()

    def test_fallback_file_obj_switch_to_local_is_idempotent(self):
        """A second local switch should return immediately."""
        remote = _PlainHandle()
        local_handles = []

        def _open_local():
            handle = _PlainHandle()
            local_handles.append(handle)
            return handle

        handle = _FallbackFileObj(
            remote_opener=lambda: remote,
            local_opener=_open_local,
            error_predicate=is_no_range_http_error,
        )
        try:
            handle._switch_to_local()
            handle._switch_to_local()
            assert len(local_handles) == 1
        finally:
            handle.close()

    def test_fallback_file_obj_propagates_non_matching_errors(self):
        """_FallbackFileObj should not hide unrelated transport errors."""
        handle = _FallbackFileObj(
            remote_opener=lambda: _FailOnSeek(b"abcdef", RuntimeError("boom")),
            local_opener=lambda: BytesIO(b"abcdef"),
            error_predicate=is_no_range_http_error,
        )
        try:
            with pytest.raises(RuntimeError, match="boom"):
                handle.seek(1)
        finally:
            handle.close()

    def test_fallback_file_obj_exposes_basic_handle_state(self):
        """Basic helpers should proxy or report sensible state."""
        plain = _PlainHandle()
        handle = _FallbackFileObj(
            remote_opener=lambda: plain,
            local_opener=lambda: _PlainHandle(),
            error_predicate=is_no_range_http_error,
        )
        try:
            assert handle.seekable() is True
            assert handle.readable() is True
            assert handle.writable() is False
            assert handle.extra_attr == "value"
            assert handle.closed is False
            assert handle.flush() is None
            handle.close()
            assert handle.closed is True
            handle.close()
        finally:
            if not handle.closed:
                handle.close()

    def test_fallback_file_obj_uses_wrapped_writable_when_available(self):
        """The writable helper should defer to the wrapped handle when present."""
        handle = _FallbackFileObj(
            remote_opener=lambda: _WritableHandle(),
            local_opener=lambda: _WritableHandle(),
            error_predicate=is_no_range_http_error,
        )
        try:
            assert handle.writable() is True
        finally:
            handle.close()

    def test_h5_reader_warns_when_no_range_fallback_downloads(self, monkeypatch):
        """HDF5 remote fallback should warn when it materializes a local cache file."""
        clear_remote_file_cache()
        path = UPath("memory://dascore/io_resource_test_fallback_warn.h5")
        path.write_bytes(b"abcdef")
        monkeypatch.setattr(
            type(path),
            "open",
            lambda *_args, **_kwargs: _FailOnSeek(
                b"abcdef",
                ValueError(
                    "The HTTP server doesn't appear to support range requests. "
                    "Only reading this file from the beginning is supported."
                ),
            ),
        )
        monkeypatch.setattr(
            H5Reader,
            "constructor",
            staticmethod(lambda handle, **_kwargs: handle.seek(1) or object()),
        )

        with config_context(warn_on_remote_cache=True):
            with pytest.warns(UserWarning, match="Downloading remote file"):
                H5Reader.get_handle(path)

    def test_h5_reader_raises_when_no_range_fallback_cache_disabled(self, monkeypatch):
        """HDF5 remote fallback should fail fast when remote caching is disabled."""
        path = UPath("memory://dascore/io_resource_test_fallback_disabled.h5")
        path.write_bytes(b"abcdef")
        monkeypatch.setattr(
            type(path),
            "open",
            lambda *_args, **_kwargs: _FailOnSeek(
                b"abcdef",
                ValueError(
                    "The HTTP server doesn't appear to support range requests. "
                    "Only reading this file from the beginning is supported."
                ),
            ),
        )
        monkeypatch.setattr(
            H5Reader,
            "constructor",
            staticmethod(lambda handle, **_kwargs: handle.seek(1) or object()),
        )

        with config_context(allow_remote_cache=False):
            with pytest.raises(RemoteCacheError, match="Remote caching is disabled"):
                H5Reader.get_handle(path)


class TestTextReader:
    """Tests for TextReader behavior."""

    def test_get_handle_from_path_reads_text(self, tmp_path):
        """Ensure TextReader opens paths in text mode."""
        path = tmp_path / "text_reader_path.txt"
        path.write_text("line1\nline2\n")
        with closing(TextReader.get_handle(path)) as handle:
            assert isinstance(handle, TextIOBase)
            assert handle.readline() == "line1\n"

    def test_get_handle_stringio_resets_offset(self):
        """Ensure StringIO input has its offset reset."""
        resource = StringIO("abc")
        _ = resource.read(1)
        out = TextReader.get_handle(resource)
        assert out is resource
        assert out.tell() == 0
        assert out.read(1) == "a"

    def test_get_handle_text_file_resets_offset(self, tmp_path):
        """Ensure open text handles are accepted and reset."""
        path = tmp_path / "text_reader_reset.txt"
        path.write_text("abcdef")
        with open(path, encoding="utf-8") as fi:
            _ = fi.read(2)
            out = TextReader.get_handle(fi)
            assert out is fi
            assert out.tell() == 0
            assert out.read(1) == "a"


class TestXarray:
    """Tests for xarray conversions."""

    @pytest.fixture
    def data_array_from_patch(self, random_patch):
        """Get a data array from a patch."""
        pytest.importorskip("xarray")
        return random_patch.io.to_xarray()

    def test_convert_to_xarray(self, data_array_from_patch):
        """Tests for converting to xarray object."""
        import xarray as xr  # noqa: PLC0415

        assert isinstance(data_array_from_patch, xr.DataArray)

    def test_convert_from_xarray(self, data_array_from_patch):
        """Ensure xarray data arrays can be converted back."""
        out = xarray_to_patch(data_array_from_patch)
        assert isinstance(out, dc.Patch)

    def test_round_trip(self, random_patch, data_array_from_patch):
        """Converting to xarray should be lossless."""
        out = xarray_to_patch(data_array_from_patch)
        assert out == random_patch

    def test_convert_non_coord(self, random_patch):
        """Ensure a patch with non-coord can still be converted."""
        xr = pytest.importorskip("xarray")
        patch = random_patch.sum("time")
        dar = patch.io.to_xarray()
        assert isinstance(dar, xr.DataArray)
        # Ensure it round-trips
        patch2 = xarray_to_patch(dar)
        assert isinstance(patch2, dc.Patch)


class TestSpoolToXarray:
    """Tests for converting a spool to a dask-backed xarray DataTree."""

    @pytest.fixture(autouse=True)
    def _require_libs(self):
        """These tests need both optional libraries."""
        pytest.importorskip("xarray")
        pytest.importorskip("dask")

    @pytest.fixture
    def diverse_tree(self, diverse_spool):
        """Convert the diverse spool, skipping without xarray or dask."""
        return diverse_spool.io.to_xarray()

    def _leaves(self, tree):
        """Return the datasets holding a data variable."""
        return [node for node in tree.subtree if "data" in node.dataset]

    def test_tree_structure(self, diverse_tree):
        """Each leaf holds one lazy data variable with dim coordinates."""
        import dask.array as da  # noqa: PLC0415
        import xarray as xr  # noqa: PLC0415

        assert isinstance(diverse_tree, xr.DataTree)
        leaves = self._leaves(diverse_tree)
        assert leaves
        for leaf in leaves:
            data = leaf.dataset["data"]
            assert isinstance(data.data, da.Array)
            assert set(data.dims) <= set(data.coords)

    def test_matches_chunk(self, diverse_spool, diverse_tree):
        """Every leaf's values equal the equivalent chunk output patch."""
        expected = {}
        for patch in diverse_spool.chunk(time=None):
            coord = patch.get_coord("time")
            key = (
                patch.attrs.tag,
                patch.attrs.acquisition_key or "",
                np.datetime64(coord.min(), "ns"),
                patch.shape,
            )
            expected[key] = patch
        leaves = self._leaves(diverse_tree)
        assert len(leaves) == len(expected)
        for leaf in leaves:
            data = leaf.dataset["data"]
            key = (
                data.attrs["tag"],
                data.attrs.get("acquisition_key") or "",
                np.datetime64(data["time"].values.min(), "ns"),
                data.shape,
            )
            patch = expected.pop(key)
            np.testing.assert_array_equal(data.values, patch.data)
            for dim in patch.dims:
                np.testing.assert_array_equal(
                    data[dim].values, patch.get_coord(dim).values
                )
        assert not expected

    def test_builds_without_reading(self, diverse_spool_directory, monkeypatch):
        """Constructing the tree must not read any patch data."""
        from dascore.io.index.catalog import FileResolver, PatchCatalog  # noqa: PLC0415
        from dascore.io.index.planned import PlanResolver  # noqa: PLC0415

        spool = dc.spool(diverse_spool_directory).update()

        def _fail(*args, **kwargs):
            raise AssertionError("tree construction read patch data")

        monkeypatch.setattr(PatchCatalog, "resolve_row", _fail)
        monkeypatch.setattr(FileResolver, "resolve", _fail)
        monkeypatch.setattr(PlanResolver, "_load_member", _fail)
        monkeypatch.setattr(PlanResolver, "_load_member_array", _fail)
        tree = spool.io.to_xarray()
        assert len(self._leaves(tree))

    def test_compute_reads_only_needed_blocks(
        self, diverse_spool_directory, monkeypatch
    ):
        """A small selection loads only the member blocks it touches."""
        from dascore.io.index.planned import PlanResolver  # noqa: PLC0415

        spool = dc.spool(diverse_spool_directory).update()
        tree = spool.io.to_xarray()
        calls = []
        original = PlanResolver._load_member

        def _counting(self, kwargs):
            calls.append(1)
            return original(self, kwargs)

        monkeypatch.setattr(PlanResolver, "_load_member", _counting)
        # The DAS2.R2D1..RAW random segment merges three source patches;
        # slicing inside the first must load exactly one of the three,
        # and the loaded values must match the eagerly chunked patch.
        leaf = next(
            x
            for x in self._leaves(tree)
            if x.dataset["data"].attrs.get("acquisition_key") == "DAS2.R2D1..RAW"
        )
        data = leaf.dataset["data"]
        assert data.data.npartitions == 3
        small = data.isel(time=slice(0, 5)).compute()
        assert len(calls) == 1
        merged = spool.select(acquisition_key="DAS2.R2D1..RAW").chunk(time=None)[0]
        expected = merged.data[:, :5] if merged.dims[0] != "time" else merged.data[:5]
        np.testing.assert_array_equal(small.values, expected)

    def test_plan_backed_spool(self, random_spool):
        """A chunked spool converts and computes like its merged self."""
        tree = random_spool.chunk(time=2).io.to_xarray()
        leaves = self._leaves(tree)
        assert len(leaves) == 1
        merged = random_spool.chunk(time=None)[0]
        np.testing.assert_array_equal(leaves[0].dataset["data"].values, merged.data)

    @pytest.mark.parametrize("bad_dtype", [None, ""])
    def test_missing_dtype_raises(self, random_spool, monkeypatch, bad_dtype):
        """An index without a dtype cannot size the arrays; say so."""
        import dascore.utils.chunk_plan as chunk_plan_module  # noqa: PLC0415

        original = chunk_plan_module.build_chunk_plan

        def _null_dtype(*args, **kwargs):
            plan = original(*args, **kwargs)
            plan.outputs["_dtype"] = bad_dtype
            return plan

        monkeypatch.setattr(chunk_plan_module, "build_chunk_plan", _null_dtype)
        with pytest.raises(PatchConversionError, match="dtype"):
            random_spool.io.to_xarray()

    def test_tolerance_argument(self, diverse_spool):
        """A looser tolerance merges gaps the default keeps as segments."""
        sub = diverse_spool.select(tag="big_gaps")
        default = len(self._leaves(sub.io.to_xarray()))
        loose = len(self._leaves(sub.io.to_xarray(tolerance=10_000)))
        assert loose < default

    def test_stale_index_shape_raises(self, random_spool, monkeypatch):
        """A block whose loaded shape breaks its promise raises clearly."""
        from dascore.io.index.planned import PlanResolver  # noqa: PLC0415

        tree = random_spool.io.to_xarray()
        original = PlanResolver._load_member

        def _truncated(self, kwargs):
            patch = original(self, kwargs)
            return patch.select(time=(0, 5), samples=True)

        monkeypatch.setattr(PlanResolver, "_load_member", _truncated)
        leaf = self._leaves(tree)[0]
        with pytest.raises(PatchConversionError, match="promised"):
            leaf.dataset["data"].compute()

    def test_segment_names_follow_dim_order(self, diverse_spool):
        """segment_0..n are ordered along the merged dimension."""
        tree = diverse_spool.io.to_xarray()
        for node in tree.children.values():
            starts = [
                child.dataset["data"]["time"].values.min()
                for _, child in sorted(node.children.items())
            ]
            assert starts == sorted(starts)

    def test_sampling_jitter_steps(self, random_patch):
        """Members merged under sampling tolerance keep their own grids."""
        first = random_patch
        coord = first.get_coord("time")
        step = coord.step * 1.04  # within the 5% sampling tolerance
        second = first.update_coords(time_min=coord.max() + coord.step, time_step=step)
        spool = dc.spool([first, second])
        merged = spool.chunk(time=None)[0]
        leaf = self._leaves(spool.io.to_xarray())[0]
        data = leaf.dataset["data"]
        assert data.shape == merged.data.shape
        np.testing.assert_array_equal(data.values, merged.data)
        np.testing.assert_array_equal(
            data["time"].values, merged.get_coord("time").values
        )

    def test_off_grid_overlap(self, random_patch):
        """An overlap whose grids misalign still sizes blocks exactly."""
        coord = random_patch.get_coord("time")
        shifted = random_patch.update_coords(time_min=coord.max() - 9.3 * coord.step)
        spool = dc.spool([random_patch, shifted])
        merged = spool.chunk(time=None)[0]
        leaf = self._leaves(spool.io.to_xarray())[0]
        data = leaf.dataset["data"]
        assert data.shape == merged.data.shape
        np.testing.assert_array_equal(data.values, merged.data)
        np.testing.assert_array_equal(
            data["time"].values, merged.get_coord("time").values
        )

    def test_single_sample_non_dim(self, random_patch):
        """A one-sample non-merge dimension has no step yet converts."""
        thin = random_patch.select(distance=(0, 1), samples=True)
        thin = thin.update_coords(distance=np.array([5.0]))
        leaf = self._leaves(dc.spool([thin]).io.to_xarray())[0]
        assert leaf.dataset["data"].shape == thin.shape
        np.testing.assert_array_equal(leaf.dataset["data"].values, thin.data)
        np.testing.assert_array_equal(leaf.dataset["data"]["distance"].values, [5.0])

    def test_irregular_dim_raises(self, random_patch):
        """A multi-sample coordinate with no step cannot be sized."""
        time = random_patch.get_coord("time").values.copy()
        time[1] += np.timedelta64(1, "ms")
        wobbly = random_patch.update_coords(time=time)
        with pytest.raises(PatchConversionError, match="no sampling step"):
            dc.spool([wobbly]).io.to_xarray()

    def test_irregular_non_dim_raises(self, random_patch):
        """A stepless non-merge dimension cannot be sized either."""
        dist = random_patch.get_coord("distance").values.copy().astype(float)
        dist[1] += 0.5
        wobbly = random_patch.update_coords(distance=dist)
        with pytest.raises(PatchConversionError, match="no sampling step"):
            dc.spool([wobbly]).io.to_xarray()

    def test_transposed_member_load(self, random_spool, monkeypatch):
        """A member loading in another dim order is transposed to match."""
        from dascore.io.index.planned import PlanResolver  # noqa: PLC0415

        tree = random_spool.io.to_xarray()
        merged = random_spool.chunk(time=None)[0]
        original = PlanResolver._load_member

        def _transposed(self, kwargs):
            return original(self, kwargs).transpose()

        monkeypatch.setattr(PlanResolver, "_load_member", _transposed)
        leaf = self._leaves(tree)[0]
        np.testing.assert_array_equal(leaf.dataset["data"].values, merged.data)

    def test_no_group_attrs(self, random_spool):
        """With no grouping attributes the whole spool is one group."""
        with config_context(patch_kind_attrs=()):
            tree = random_spool.io.to_xarray()
        assert len(tree.children) == 1
        assert len(self._leaves(tree)) == 1

    def test_quantity_tolerance(self, diverse_spool):
        """A unit-bearing tolerance is handed to simplify as it stands."""
        sub = diverse_spool.select(tag="big_gaps")
        default = len(self._leaves(sub.io.to_xarray()))
        loose = len(self._leaves(sub.io.to_xarray(tolerance=dc.get_quantity("1 hour"))))
        assert loose < default

    def test_duplicate_node_names_raise(self, random_patch, monkeypatch):
        """Two groups resolving to one node name must not overwrite arrays."""
        import dascore.utils.display as display_module  # noqa: PLC0415

        patches = [
            random_patch.update_attrs(cable_id="a"),
            random_patch.update_attrs(cable_id="b"),
        ]
        monkeypatch.setattr(
            display_module, "group_names", lambda *args, **kwargs: ["same", "same"]
        )
        with pytest.raises(PatchConversionError, match="more than one group"):
            dc.spool(patches).io.to_xarray(group="cable_id")

    def test_assoc_coord_samples_select_refused(self, random_patch):
        """A samples selection on an associated coordinate cannot be sized."""
        n = len(random_patch.get_coord("distance"))
        patch = random_patch.update_coords(zone=("distance", np.arange(n)))
        sub = dc.spool([patch]).select(zone=(0, 5), samples=True)
        with pytest.raises(PatchConversionError, match="associated"):
            sub.io.to_xarray()

    def test_enriched_spool_refused(self, random_spool):
        """Pending inventory enrichment would be dropped; refuse instead."""
        sub = random_spool[0:2]
        sub._enrich_kwargs = {"coords": True}
        with pytest.raises(PatchConversionError, match="enrichment"):
            sub.io.to_xarray()

    def test_quantity_tolerance_with_units(self, random_patch):
        """A unit-bearing tolerance reads against the coordinate's units."""
        d = random_patch.get_coord("distance")
        gap = random_patch.update_coords(
            distance_min=d.max() + 5 * d.step  # a 5-step gap along distance
        )
        spool = dc.spool([random_patch, gap])
        tol = dc.get_quantity("10 m")
        merged = spool.chunk(distance=None, tolerance=tol)[0]
        tree = spool.io.to_xarray(dim="distance", tolerance=tol)
        leaves = self._leaves(tree)
        assert len(leaves) == 1
        data = leaves[0].dataset["data"]
        assert data.shape == merged.data.shape
        np.testing.assert_array_equal(data.values, merged.data)
        np.testing.assert_array_equal(
            data["distance"].values, merged.get_coord("distance").values
        )

    def test_value_select_refused(self, random_spool):
        """A pending value-range selection cannot be sized; it raises."""
        coord = random_spool[0].get_coord("time")
        sub = random_spool.select(time=(coord.min() + coord.step // 2, None))
        with pytest.raises(PatchConversionError, match="value selections"):
            sub.io.to_xarray()

    def test_samples_select_supported(self, random_spool):
        """A samples-based selection stays exact and converts."""
        sub = random_spool.select(time=(10, -10), samples=True)
        merged = sub.chunk(time=None)[0]
        leaf = self._leaves(sub.io.to_xarray())[0]
        data = leaf.dataset["data"]
        assert data.shape == merged.data.shape
        np.testing.assert_array_equal(data.values, merged.data)
        np.testing.assert_array_equal(
            data["time"].values, merged.get_coord("time").values
        )

    def test_descending_dim_raises(self, random_patch):
        """A descending merge dimension is refused with a clear message."""
        flipped = random_patch.update_coords(
            distance=random_patch.get_coord("distance").values[::-1]
        )
        with pytest.raises(PatchConversionError, match="descending"):
            dc.spool([flipped]).io.to_xarray(dim="distance")

    def test_descending_non_dim_coord(self, random_patch):
        """A descending non-merge coordinate keeps its order and values."""
        flipped = random_patch.update_coords(
            distance=random_patch.get_coord("distance").values[::-1]
        )
        leaf = self._leaves(dc.spool([flipped]).io.to_xarray())[0]
        data = leaf.dataset["data"]
        np.testing.assert_array_equal(
            data["distance"].values, flipped.get_coord("distance").values
        )
        np.testing.assert_array_equal(data.values, flipped.data)

    def test_mixed_dtype_upcasts(self, random_patch):
        """Blocks narrower than the combined dtype upcast at load."""
        coord = random_patch.get_coord("time")
        narrow = random_patch.new(data=random_patch.data.astype(np.float32))
        narrow = narrow.update_coords(time_min=coord.max() + coord.step)
        spool = dc.spool([random_patch, narrow])
        leaf = self._leaves(spool.io.to_xarray())[0]
        data = leaf.dataset["data"]
        assert data.dtype == np.float64
        # A slice touching only the narrow member must upcast in the
        # loader itself, not by concatenation with a wider block.
        assert data.isel(time=slice(-3, None)).compute().dtype == np.float64

    def test_single_group_spool(self, random_spool):
        """A homogeneous spool merges into one segment of one group."""
        tree = random_spool.io.to_xarray()
        leaves = self._leaves(tree)
        assert len(leaves) == 1
        merged = random_spool.chunk(time=None)[0]
        np.testing.assert_array_equal(leaves[0].dataset["data"].values, merged.data)

    def test_group_argument(self, diverse_spool):
        """An explicit group partitions the tree by that attribute."""
        tree = diverse_spool.io.to_xarray(group="tag", conflict="drop")
        tags = {x.dataset["data"].attrs.get("tag") for x in self._leaves(tree)}
        contents_tags = set(diverse_spool.get_contents()["tag"])
        assert tags == contents_tags
        # Grouping by tag alone merges kinds the default grouping keeps
        # apart (e.g. differing acquisition keys), so the node count must
        # equal the tag count, not the finer default partition.
        assert len(tree.children) == len(contents_tags)

    def test_bad_group_raises(self, diverse_spool):
        """A group attribute no patch has raises the standard query error."""
        from dascore.exceptions import InvalidSpoolQueryError  # noqa: PLC0415

        with pytest.raises(InvalidSpoolQueryError, match="do not exist"):
            diverse_spool.io.to_xarray(group="not_an_attr")

    def test_empty_spool(self):
        """An empty spool converts to an empty tree."""
        import xarray as xr  # noqa: PLC0415

        tree = dc.spool([]).io.to_xarray()
        assert isinstance(tree, xr.DataTree)
        assert not self._leaves(tree)

    def test_slash_in_group_name_raises(self, random_patch):
        """A group value which cannot name a tree node raises."""
        # Two values are needed: a lone group is named by its ordinal.
        patches = [
            random_patch.update_attrs(cable_id="a/b"),
            random_patch.update_attrs(cable_id="c/d"),
        ]
        with pytest.raises(PatchConversionError, match="cannot name"):
            dc.spool(patches).io.to_xarray(group="cable_id")


class TestObsPy:
    """Tests for converting patches to/from ObsPy streams."""

    @pytest.fixture
    def short_patch(self, random_patch):
        """Just shorten the patch distance dim to speed up these tests."""
        return random_patch.select(distance=(0, 10), samples=True)

    @pytest.fixture
    def stream_from_patch(self, short_patch):
        """Get a stream from a patch."""
        pytest.importorskip("obspy")
        st = short_patch.io.to_obspy()
        return st

    def test_convert_to_obspy(self, stream_from_patch):
        """Ensure a patch can be converted to a stream."""
        import obspy  # noqa: PLC0415

        assert isinstance(stream_from_patch, obspy.Stream)

    def test_obspy_to_patch(self, stream_from_patch):
        """Ensure we can convert back to patch from stream."""
        out = dc.io.obspy_to_patch(stream_from_patch)
        assert isinstance(out, dc.Patch)

    def test_patch_no_time_raises(self, random_patch):
        """Ensure a patch without time dimension raises."""
        pytest.importorskip("obspy")
        patch = random_patch.rename_coords(time="not_time")
        with pytest.raises(PatchConversionError):
            patch.io.to_obspy()

    def test_bad_stream_raises(self):
        """Ensure a stream without even length or require param raises."""
        obspy = pytest.importorskip("obspy")
        st = obspy.read()
        # since st doesn't have a value of "distance" in each of its traces
        # attrs dict this should raise.
        with pytest.raises(PatchConversionError):
            dc.io.obspy_to_patch(st)

    def test_empty_stream(self):
        """An empty Stream should return an empty Patch."""
        obspy = pytest.importorskip("obspy")
        st = obspy.Stream([])
        patch = dc.io.obspy_to_patch(st)
        assert not patch.dims

    def test_example_event(self, event_patch_2):
        """Ensure example event can be converted to stream."""
        obspy = pytest.importorskip("obspy")
        # make patch smaller to make test faster
        patch = event_patch_2.select(distance=(500, 550))
        st = patch.io.to_obspy()
        assert isinstance(st, obspy.Stream)
        assert len(st) == len(patch.get_coord("distance"))


class TestRemoteCacheConcurrency:
    """The remote cache serializes per resource, not globally."""

    @pytest.fixture(autouse=True)
    def isolated_cache(self, tmp_path, permanent_config):
        """
        Give each test its own cache directory.

        Uses the permanent config: worker threads start with a fresh
        context, so a scoped config_context override would not reach them.
        """
        with permanent_config(
            remote_cache_dir=tmp_path / "remote_cache", warn_on_remote_cache=False
        ):
            clear_remote_file_cache()
            yield
            clear_remote_file_cache()

    def _memory_file(self, name: str) -> UPath:
        """Write a small file into the in-memory filesystem."""
        path = UPath(f"memory://dascore/concurrent/{name}")
        with path.open("wb") as fi:
            fi.write(b"dascore" * 64)
        return path

    @pytest.mark.concurrency
    def test_racing_callers_download_once(self, monkeypatch, run_in_threads):
        """Callers wanting one resource agree on the path and download it once."""
        resource = self._memory_file("shared.bin")
        downloads = []
        original = remote_io._download_remote_file

        def _counted(path, local_path):
            downloads.append(local_path)
            return original(path, local_path)

        monkeypatch.setattr(remote_io, "_download_remote_file", _counted)
        results = run_in_threads(lambda _: ensure_local_file(resource))
        assert len({str(x) for x in results}) == 1
        assert len(downloads) == 1
        assert results[0].exists()

    @pytest.mark.concurrency
    def test_distinct_resources_are_not_serialized(self, monkeypatch, run_in_threads):
        """Unrelated downloads run at once; one global lock would time out here."""
        resources = [self._memory_file(f"file_{i}.bin") for i in range(4)]
        barrier = threading.Barrier(len(resources), timeout=30)
        original = remote_io._download_remote_file

        def _synchronized(path, local_path):
            # Every download has to be in flight together to get past this.
            barrier.wait()
            return original(path, local_path)

        monkeypatch.setattr(remote_io, "_download_remote_file", _synchronized)
        results = run_in_threads(lambda index: ensure_local_file(resources[index]))
        assert all(x is not None and x.exists() for x in results)

    def test_failed_download_is_retried(self, monkeypatch):
        """A failed download publishes nothing, so the next caller retries."""
        resource = self._memory_file("flaky.bin")
        calls = []
        original = remote_io._download_remote_file

        def _fail_once(path, local_path):
            calls.append(local_path)
            if len(calls) == 1:
                raise OSError("download failed")
            return original(path, local_path)

        monkeypatch.setattr(remote_io, "_download_remote_file", _fail_once)
        with pytest.raises(OSError, match="download failed"):
            ensure_local_file(resource)
        assert ensure_local_file(resource).exists()
        assert len(calls) == 2

    def test_unregistered_remote_id_is_coerced(self):
        """An id missing from the resource cache is rebuilt from the id itself."""
        resource = self._memory_file("unregistered.bin")
        remote_io._REMOTE_RESOURCE_CACHE.clear()
        cache_root = remote_io._normalize_cache_root(remote_io.get_remote_cache_path())
        local_path = remote_io._materialize_remote_file(str(resource), cache_root)
        assert local_path.exists()

    def test_reinit_drops_download_locks(self):
        """The fork hook drops locks a dead thread may have been holding."""
        resource = self._memory_file("forked.bin")
        ensure_local_file(resource)
        assert remote_io._REMOTE_KEY_LOCKS
        old_locks = remote_io._REMOTE_KEY_LOCKS
        remote_io._reinit_remote_cache_locks()
        assert not remote_io._REMOTE_KEY_LOCKS
        assert remote_io._REMOTE_KEY_LOCKS is not old_locks


class TestIOResourceManagerConcurrency:
    """One manager hands every caller the same handle per type."""

    @pytest.mark.concurrency
    def test_racing_callers_share_one_handle(self, tmp_path, run_in_threads):
        """get_resource opens each required type exactly once."""
        path = tmp_path / "concurrent_resource.bin"
        path.write_bytes(b"dascore")
        with IOResourceManager(path) as man:
            handles = run_in_threads(lambda _: man.get_resource(BinaryReader))
            assert len({id(x) for x in handles}) == 1
            assert not handles[0].closed
        assert handles[0].closed


class TestToXarrayReadArray:
    """Tests wiring the read_array fast path into to_xarray blocks."""

    @pytest.fixture(autouse=True)
    def _require_libs(self):
        """These tests need both optional libraries."""
        pytest.importorskip("xarray")
        pytest.importorskip("dask")

    @pytest.fixture(scope="class")
    def dasdae_directory(self, tmp_path_factory):
        """A directory of single-patch DASDAE files with distinct data.

        Distinct arrays per file, or a block reading the wrong member
        would still pass the parity assertions.
        """
        path = tmp_path_factory.mktemp("to_xarray_read_array")
        for num, patch in enumerate(dc.get_example_spool()):
            patch.new(data=patch.data + num).io.write(
                path / f"patch_{num}.h5", "dasdae"
            )
        return path

    @pytest.fixture
    def override_calls(self):
        """Give DASDAE a counting read_array override."""
        from dascore.io.core import FiberIO  # noqa: PLC0415
        from dascore.io.dasdae.core import DASDAEV1  # noqa: PLC0415

        calls = []

        def read_array(self, resource, windows, **kwargs):
            # a real override's caster wrapper consumes _pre_cast; this
            # raw function sees it and must not forward it to read
            kwargs.pop("_pre_cast", None)
            calls.append(windows)
            return FiberIO.read_array(self, resource, windows, **kwargs)

        # set and delete by hand: monkeypatch would restore the inherited
        # method as an own class attribute rather than remove it
        DASDAEV1.read_array = read_array
        yield calls
        del DASDAEV1.read_array

    def _leaf(self, tree):
        """The first dataset holding a data variable."""
        return next(node for node in tree.subtree if "data" in node.dataset)

    def test_fast_path_loads_blocks(
        self, dasdae_directory, override_calls, monkeypatch
    ):
        """With an override, computing never builds a member Patch."""
        from dascore.io.index.planned import PlanResolver  # noqa: PLC0415

        spool = dc.spool(dasdae_directory).update()
        eager = spool.chunk(time=None)[0].data
        tree = spool.io.to_xarray()

        def _fail(*args, **kwargs):
            raise AssertionError("fast path fell back to patch loading")

        monkeypatch.setattr(PlanResolver, "_load_member", _fail)
        out = self._leaf(tree)["data"].data.compute()
        assert np.array_equal(out, eager)
        assert len(override_calls) == len(spool)

    def test_residual_spool_falls_back(self, dasdae_directory, override_calls):
        """A samples-selected spool loads through the exact patch path."""
        spool = dc.spool(dasdae_directory).update()
        sub = spool.select(time=(2, 100), samples=True)
        out = self._leaf(sub.io.to_xarray())["data"].data.compute()
        assert np.array_equal(out, sub.chunk(time=None)[0].data)
        assert override_calls == []

    def test_chunked_spool_falls_back(self, dasdae_directory, override_calls):
        """A plan-backed spool's trimmed rows never take the fast path.

        Its collapsed member rows state trimmed envelopes, so a sample
        window computed against them is not a window on the file grid;
        the fast path must refuse or it reads the wrong samples.
        """
        spool = dc.spool(dasdae_directory).update().chunk(time=3)
        eager = spool.chunk(time=None)[0].data
        out = self._leaf(spool.io.to_xarray())["data"].data.compute()
        assert np.array_equal(out, eager)
        assert override_calls == []

    def test_interior_window_fast_path(self, tmp_path, override_calls):
        """An overlap-trimmed member reads an interior file window.

        Two half-overlapping files merge into one segment, so the second
        member's window starts mid-file — the case where a wrong window
        anchor would silently read the wrong samples.
        """
        first = dc.get_example_patch()
        time = first.get_coord("time")
        half = time.values[len(time) // 2]
        # distinct data so reading the wrong file cannot pass parity
        second = first.update_coords(time_min=half).new(data=first.data + 1)
        for num, patch in enumerate((first, second)):
            patch.update_attrs(history=[]).io.write(tmp_path / f"p{num}.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        eager = spool.chunk(time=None)[0].data
        out = self._leaf(spool.io.to_xarray())["data"].data.compute()
        assert np.array_equal(out, eager)
        # the trimmed member's window must not be anchored at the start
        starts = sorted(window["time"][0] for window in override_calls)
        assert len(override_calls) == 2
        assert starts[0] == 0 and starts[1] > 0

    def test_transposes_source_order(self):
        """A native-order array is transposed to the tree's dims.

        Three dimensions with a cyclic permutation, so the permutation
        differs from its inverse and a reversed mapping cannot pass.
        """
        from dascore.utils.io import _load_xarray_block  # noqa: PLC0415

        native = np.arange(24).reshape(2, 3, 4)

        class _Fake:
            def _load_member_array(self, row, windows):
                return native

        row = {"dims": "time,distance,depth", "source_path": "x"}
        out = _load_xarray_block(
            _Fake(),
            row,
            "time",
            (0, 1),
            ("distance", "depth", "time"),
            (3, 4, 2),
            native.dtype,
            (0, 2),
        )
        assert np.array_equal(out, native.transpose(1, 2, 0))

    def test_mismatched_dims_fall_back(self, random_patch):
        """A row stating different dims than the tree takes the patch path."""
        from dascore.utils.io import _load_xarray_block  # noqa: PLC0415

        patch = random_patch

        class _Fake:
            def _load_member_array(self, row, windows):
                raise AssertionError("fast path consulted with foreign dims")

            def _load_member(self, row):
                return patch

        coord = patch.get_coord("time")
        out = _load_xarray_block(
            _Fake(),
            {"dims": "depth,time", "source_path": "x"},
            "time",
            (coord.min(), coord.max()),
            patch.dims,
            patch.shape,
            patch.data.dtype,
            (0, len(coord)),
        )
        assert np.array_equal(out, patch.data)

    def test_stale_shape_raises(self):
        """An array which breaks the index's promise raises."""
        from dascore.exceptions import PatchConversionError  # noqa: PLC0415
        from dascore.utils.io import _load_xarray_block  # noqa: PLC0415

        class _Fake:
            def _load_member_array(self, row, windows):
                return np.zeros((2, 2))

        row = {"dims": "time,distance", "source_path": "x"}
        with pytest.raises(PatchConversionError, match="promised"):
            _load_xarray_block(
                _Fake(), row, "time", (0, 2), ("time", "distance"), (3, 4), "f8", (0, 3)
            )

    def test_row_without_dims_falls_back(self, random_patch):
        """A row which cannot state its dimension order takes the patch path."""
        from dascore.utils.io import _load_xarray_block  # noqa: PLC0415

        patch = random_patch

        class _Fake:
            def _load_member_array(self, row, windows):
                raise AssertionError("fast path consulted without dims")

            def _load_member(self, row):
                return patch

        coord = patch.get_coord("time")
        lims = (coord.min(), coord.max())
        out = _load_xarray_block(
            _Fake(),
            {"source_path": "x"},
            "time",
            lims,
            patch.dims,
            patch.shape,
            patch.data.dtype,
            (0, len(coord)),
        )
        assert np.array_equal(out, patch.data)
