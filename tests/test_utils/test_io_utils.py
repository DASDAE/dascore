"""Tests for IO utilities."""

from __future__ import annotations

import warnings
from contextlib import closing
from io import BufferedReader, BufferedWriter, BytesIO, StringIO, TextIOBase
from pathlib import Path

import h5py
import pytest
from tables import File
from upath import UPath

import dascore as dc
import dascore.utils.remote_io as remote_io
from dascore.config import set_config
from dascore.exceptions import PatchConversionError, RemoteCacheError
from dascore.utils.hdf5 import (
    H5Reader,
    H5Writer,
    HDF5Reader,
    HDF5Writer,
    LocalH5Reader,
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
)
from dascore.utils.remote_io import (
    FallbackFileObj,
    clear_remote_file_cache,
    get_remote_cache_path,
    get_remote_cache_scope,
    is_no_range_http_error,
    remote_cache_scope,
)


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
        with closing(get_handle_from_resource(generic_hdf5, HDF5Reader)) as handle:
            assert isinstance(handle, File)

    def test_path_to_hdf5_writer(self, tmp_path):
        """Ensure we get a reader from tmp path reader."""
        path = tmp_path / "test_hdf_writer.h5"
        with closing(get_handle_from_resource(path, HDF5Writer)) as handle:
            assert isinstance(handle, File)

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
                assert list(handle["data"][:]) == [1, 2, 3]
            finally:
                handle.close()

    def test_h5_reader_passthrough_h5py_handle(self, tmp_path):
        """Ensure h5py-backed readers return open handles unchanged."""
        path = tmp_path / "passthrough.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("data", data=[1, 2, 3])
        with h5py.File(path, "r") as raw:
            assert H5Reader.get_handle(raw) is raw

    def test_local_h5_reader_materializes_local_path(self, tmp_path):
        """Ensure LocalH5Reader can open a local path through its adapter."""
        path = tmp_path / "local_h5_reader.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("data", data=[1, 2, 3])
        handle = LocalH5Reader.get_handle(path)
        try:
            assert list(handle["data"][:]) == [1, 2, 3]
        finally:
            handle.close()

    def test_h5_reader_closes_upath_handle_on_constructor_error(
        self, tmp_path, monkeypatch
    ):
        """Ensure constructor failures close UPath-opened file handles."""

        class _DummyHandle:
            closed = False

            def close(self):
                self.closed = True

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

    def test_h5_reader_uses_small_blocks_for_s3_upath(self, monkeypatch):
        """S3-backed HDF5 readers should override s3fs's large default block."""

        class _DummyHandle:
            closed = False

            def close(self):
                self.closed = True

        opened = {}
        handle = _DummyHandle()
        path = UPath("s3://example-bucket/example.h5", anon=True)

        def _open(_self, _mode, **kwargs):
            opened.update(kwargs)
            return handle

        monkeypatch.setattr(type(path), "open", _open)
        monkeypatch.setattr(
            H5Reader,
            "constructor",
            staticmethod(lambda *args, **kwargs: object()),
        )
        with set_config(remote_hdf5_block_size=1234):
            H5Reader.get_handle(path)
        assert opened["block_size"] == 1234
        assert opened["cache_type"] == "readahead"

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
            get_handle_from_resource(bad_instance, HDF5Writer)
        with pytest.raises(NotImplementedError):
            get_handle_from_resource(bad_instance, HDF5Reader)


class TestIOResourceManager:
    """Tests for the IO resource manager."""

    @pytest.fixture(autouse=True)
    def clear_remote_cache(self):
        """Ensure remote cache state doesn't leak between tests."""
        with set_config(warn_on_remote_cache=False):
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
            hf = man.get_resource(HDF5Writer)
            fi = man.get_resource(BinaryWriter)
            # Why didn't pytables implement the stream like pythons?
            assert hf.isopen
            assert not fi.closed
        # after the context manager exists everything should be closed.
        assert not hf.isopen
        assert fi.closed

    def test_get_none_resource_returns_source(self):
        """Requesting no specific resource should return the original source."""
        source = object()
        with IOResourceManager(source) as man:
            assert man.get_resource(None) is source

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
        with set_config(remote_cache_dir=cache_dir):
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

    def test_ensure_local_file_respects_cache_dir_changes(self, tmp_path):
        """Changing the configured cache dir should change future materialization."""
        path = UPath("memory://dascore/io_resource_test_reconfigure.txt")
        path.write_text("hello")
        first_cache = tmp_path / "remote-cache-a"
        second_cache = tmp_path / "remote-cache-b"

        with set_config(remote_cache_dir=first_cache):
            first = ensure_local_file(path)
        with set_config(remote_cache_dir=second_cache):
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

        with set_config(remote_download_block_size=321):
            local_path = ensure_local_file(path)

        assert local_path.exists()
        assert local_path.read_bytes() == b"a"
        assert handle.read_sizes == [321, 321]

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
        with set_config(warn_on_remote_cache=True):
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
        with set_config(warn_on_remote_cache=True):
            with pytest.warns(UserWarning, match="Downloading remote file"):
                first = ensure_local_file(path)
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                second = ensure_local_file(path)
        assert not record
        assert first == second

    def test_ensure_local_file_warning_can_be_disabled(self):
        """Configured warning suppression should keep downloads silent."""
        path = UPath("memory://dascore/io_resource_test_warn_off.txt")
        path.write_text("hello")
        with set_config(warn_on_remote_cache=False):
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
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
        with set_config(allow_remote_cache=False):
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
        with set_config(
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

    def test_fallback_file_obj_switches_once_and_preserves_position(self):
        """FallbackFileObj should retry on the local file and preserve cursor."""
        remote = _FailOnSeek(
            b"abcdef",
            ValueError(
                "The HTTP server doesn't appear to support range requests. "
                "Only reading this file from the beginning is supported."
            ),
        )
        local_handles = []

        def _open_local():
            handle = BytesIO(b"abcdef")
            local_handles.append(handle)
            return handle

        handle = FallbackFileObj(
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
        handle = FallbackFileObj(
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

        handle = FallbackFileObj(
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
        """FallbackFileObj should not hide unrelated transport errors."""
        handle = FallbackFileObj(
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
        handle = FallbackFileObj(
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
        handle = FallbackFileObj(
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

        with set_config(warn_on_remote_cache=True):
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

        with set_config(allow_remote_cache=False):
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
        import xarray as xr

        assert isinstance(data_array_from_patch, xr.DataArray)

    def test_convert_from_xarray(self, data_array_from_patch):
        """Ensure xarray data arrays can be converted back."""
        out = dc.utils.io.xarray_to_patch(data_array_from_patch)
        assert isinstance(out, dc.Patch)

    def test_round_trip(self, random_patch, data_array_from_patch):
        """Converting to xarray should be lossless."""
        out = dc.utils.io.xarray_to_patch(data_array_from_patch)
        assert out == random_patch

    def test_convert_non_coord(self, random_patch):
        """Ensure a patch with non-coord can still be converted."""
        xr = pytest.importorskip("xarray")
        patch = random_patch.sum("time")
        dar = patch.io.to_xarray()
        assert isinstance(dar, xr.DataArray)
        # Ensure it round-trips
        patch2 = dc.utils.io.xarray_to_patch(dar)
        assert isinstance(patch2, dc.Patch)


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
        import obspy

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
