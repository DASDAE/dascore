"""Tests for indexing local file systems."""

from __future__ import annotations

import multiprocessing
import os
import platform
import shutil
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, suppress
from pathlib import Path

import pandas as pd
import pytest
from upath import UPath

from dascore.config import set_config
from dascore.exceptions import InvalidSpoolError
from dascore.io.index.indexer import DBDirectoryIndexer
from dascore.utils.misc import suppress_warnings
from dascore.utils.patch import get_patch_names


def _update_corrupt_index_map_process(cache_path, key, ready, start):
    """Update a corrupt index map in a child process."""
    from dascore.io.index.indexer import _update_index_map

    ready.set()
    if not start.wait(timeout=20):
        raise RuntimeError("timed out waiting to update index map")
    _update_index_map({key: f"index-{key}"}, cache_path=cache_path)


def _read_index_map_process(cache_path):
    """Read the index map in a child process."""
    from dascore.io.index.indexer import _get_index_map

    _get_index_map(cache_path)


@pytest.fixture(scope="class")
def basic_indexer(two_patch_directory):
    """Return an indexer on the basic spool directory."""
    indexer = DBDirectoryIndexer(two_patch_directory).update(progress=None)
    yield indexer
    indexer.close()


@pytest.fixture(scope="class")
def diverse_indexer(diverse_spool_directory):
    """Return an indexer on the diverse spool directory."""
    indexer = DBDirectoryIndexer(diverse_spool_directory).update(progress=None)
    yield indexer
    indexer.close()


@pytest.fixture(scope="class")
def diverse_df(diverse_indexer):
    """Return the contents of the diverse indexer."""
    return diverse_indexer()


@pytest.fixture()
def empty_index(tmp_path_factory):
    """Create an index around an empty directory."""
    path = tmp_path_factory.mktemp("index_created_test")
    indexer = DBDirectoryIndexer(path).update(progress=None)
    yield indexer
    indexer.close()


class TestFindIndex:
    """Tests for finding the index."""

    @pytest.fixture()
    def unwritable_directory(self, tmp_path_factory):
        """Return an un-writable directory."""
        if "windows" in platform.system().lower():
            pytest.skip("Cant run this test on windows")
        path = tmp_path_factory.mktemp("read_only_data_file")
        os.chmod(path, 0o444)
        yield path
        os.chmod(path, 0o755)

    @pytest.fixture()
    def directory_indexer_bad_cache(self, tmp_path_factory):
        """Create a bad index_map file."""
        path = tmp_path_factory.mktemp("corrupt_cache_test")
        cache_path = path / "corrupt_cache.sqlite3"
        cache_path.write_bytes(b"not a sqlite database")
        return cache_path

    def test_directory_cant_write(self, unwritable_directory):
        """Ensure correct path is found when a read-only directory is used."""
        dir_index = DBDirectoryIndexer(unwritable_directory)
        index_path = dir_index.index_path
        index_map_path = dir_index.index_map_path
        assert index_map_path.parent == index_path.parent

    def test_read_only_index_name_is_stable(self, unwritable_directory, tmp_path):
        """The read-only fallback index name must not depend on hash().

        hash() of a str/Path is randomized per process, so a hash-derived
        name would differ every session and orphan the prior index. The
        name is a stable digest of the directory path.
        """
        import hashlib

        map_path = tmp_path / "cache_paths.sqlite3"
        with set_config(directory_index_map_path=map_path):
            first = DBDirectoryIndexer(unwritable_directory).index_path
        digest = hashlib.sha256(str(unwritable_directory).encode()).hexdigest()[:16]
        assert first.name == f"_dascore_index_{digest}.sqlite3"

    def test_specify_index_path(self, tmp_path_factory):
        """Ensure specifying a Path works and is remembered."""
        data_path = tmp_path_factory.mktemp("data_dir")
        index_path = tmp_path_factory.mktemp("index_dir") / "index.sqlite"
        dir_index = DBDirectoryIndexer(data_path, index_path=index_path)
        assert dir_index.index_path == index_path
        # loading the same data dir should now remember where this is.
        dir_index2 = DBDirectoryIndexer(data_path)
        assert dir_index2.index_path == index_path

    def test_writeable_dir_index_not_there(self, tmp_path_factory):
        """Tests for when there is a writeable directory."""
        path = tmp_path_factory.mktemp("normal_indexer_test")
        dir_indexer = DBDirectoryIndexer(path)
        assert dir_indexer.index_path.parent == path

    def test_writable_dir_survives_unwritable_map(
        self, unwritable_directory, tmp_path_factory
    ):
        """Use a local index when the global map directory is read-only."""
        data_path = tmp_path_factory.mktemp("writable_data")
        map_path = unwritable_directory / "cache_paths.sqlite3"
        with set_config(directory_index_map_path=map_path):
            out = DBDirectoryIndexer(data_path)
        assert out.index_path.parent == data_path

    def test_writable_dir_survives_unavailable_sqlite_map(
        self, tmp_path_factory, monkeypatch
    ):
        """Use a local index when SQLite reports an unavailable map."""
        import dascore.io.index.indexer as indexer

        def unavailable_map(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(indexer, "_get_index_map", unavailable_map)
        data_path = tmp_path_factory.mktemp("writable_data")
        out = DBDirectoryIndexer(data_path)
        assert out.index_path.parent == data_path

    def test_writable_dir_index_exists(self, tmp_path_factory):
        """A test case where the index does exist."""
        path = tmp_path_factory.mktemp("normal_indexer_test")
        first = DBDirectoryIndexer(path)
        second = DBDirectoryIndexer(path)
        assert first.index_path == second.index_path
        assert first.index_path.exists()

    def test_corrupt_cache(self, directory_indexer_bad_cache, tmp_path_factory):
        """Ensure a corrupted cache doesn't crash indexing. See #508."""
        path = tmp_path_factory.mktemp("corrupt_cache_test")
        assert directory_indexer_bad_cache.exists()
        with set_config(directory_index_map_path=directory_indexer_bad_cache):
            DBDirectoryIndexer(path)
        assert directory_indexer_bad_cache.read_bytes().startswith(b"SQLite format 3")

    def test_remote_directory_not_supported(self):
        """Remote directory indexing should fail fast."""
        path = UPath("memory://dascore/indexer")
        (path / "file.txt").write_text("x")
        with pytest.raises(InvalidSpoolError, match="local filesystem"):
            DBDirectoryIndexer(path)

    def test_local_upath_normalized_to_path(self, tmp_path):
        """Local UPath inputs should normalize to pathlib.Path internally."""
        out = DBDirectoryIndexer(UPath(tmp_path))
        assert isinstance(out.path, Path)
        assert out.path == Path(tmp_path).absolute()

    def test_index_map_path_comes_from_config(self, tmp_path):
        """Index map paths should be sourced from runtime configuration."""
        index_map_path = tmp_path / "cache_paths.sqlite3"
        with set_config(directory_index_map_path=index_map_path):
            out = DBDirectoryIndexer(tmp_path)
            assert out.index_map_path == index_map_path


class TestIndexMap:
    """Tests for the index-location map helpers."""

    def test_default_map_path_uses_sqlite(self):
        """The default map path names the SQLite database directly."""
        from dascore.config import DascoreConfig

        assert DascoreConfig().directory_index_map_path.name == "cache_paths.sqlite3"

    def test_updates_use_sqlite_database(self, tmp_path):
        """Updates persist transactionally without temporary database files."""
        from dascore.io.index.indexer import _get_index_map, _update_index_map

        cache_path = tmp_path / "cache_paths.sqlite3"
        _update_index_map({"a": "1"}, cache_path=str(cache_path))
        _update_index_map({"b": "2"}, cache_path=str(cache_path))
        assert _get_index_map(str(cache_path)) == {"a": "1", "b": "2"}
        assert cache_path.read_bytes().startswith(b"SQLite format 3")
        assert {path.name for path in tmp_path.iterdir()} == {
            cache_path.name,
            f"{cache_path.name}.recovery.lock",
        }

    def test_corrupt_sqlite_is_rebuilt(self, tmp_path):
        """A corrupt disposable SQLite map is replaced transparently."""
        from dascore.io.index.indexer import _get_index_map, _update_index_map

        cache_path = tmp_path / "cache_paths.sqlite3"
        cache_path.write_bytes(b"not a sqlite database")
        assert _get_index_map(cache_path) == {}
        assert cache_path.read_bytes().startswith(b"SQLite format 3")
        _update_index_map({"a": "1"}, cache_path)
        assert _get_index_map(cache_path) == {"a": "1"}

    def test_corrupt_recovery_lock_does_not_block_rebuild(self, tmp_path):
        """Recovery locking must not depend on the lock file's contents."""
        from dascore.io.index.indexer import _get_index_map

        cache_path = tmp_path / "cache_paths.sqlite3"
        lock_path = tmp_path / "cache_paths.sqlite3.recovery.lock"
        cache_path.write_bytes(b"not a sqlite database")
        lock_path.write_bytes(b"arbitrary non-database contents")

        assert _get_index_map(cache_path) == {}
        assert cache_path.read_bytes().startswith(b"SQLite format 3")

    @pytest.mark.parametrize(
        "message",
        ["database disk image is malformed", "file is not a database"],
    )
    def test_corruption_detection_without_error_code(self, message):
        """Python 3.10 SQLite messages still identify disposable corruption."""
        from dascore.io.index.indexer import _is_corrupt_index_map_error

        error = sqlite3.DatabaseError(message)
        assert not hasattr(error, "sqlite_errorcode")
        assert _is_corrupt_index_map_error(error)

    def test_unrelated_database_error_propagates(self, tmp_path, monkeypatch):
        """Operational failures are not mistaken for disposable corruption."""
        import dascore.io.index.indexer as indexer

        error = sqlite3.OperationalError("database is locked")

        def fail_open(*args):
            raise error

        monkeypatch.setattr(indexer, "_open_index_map", fail_open)
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            indexer._get_index_map(tmp_path / "cache_paths.sqlite3")

    def test_corruption_during_select_is_rebuilt(self, tmp_path, monkeypatch):
        """Corruption raised by a query triggers a rebuild and retry."""
        import dascore.io.index.indexer as indexer

        cache_path = tmp_path / "cache_paths.sqlite3"
        indexer._update_index_map({"a": "1"}, cache_path)
        original_open = indexer._open_index_map
        open_calls = 0

        class CorruptOnSelect:
            """Make the first map connection fail when queried."""

            def __init__(self, connection):
                self.connection = connection

            def execute(self, statement, *args, **kwargs):
                if statement.lstrip().startswith("SELECT"):
                    self.connection.close()
                    cache_path.write_bytes(b"not a sqlite database")
                    error = sqlite3.DatabaseError("database disk image is malformed")
                    error.sqlite_errorcode = getattr(sqlite3, "SQLITE_CORRUPT", 11)
                    raise error
                return self.connection.execute(statement, *args, **kwargs)

            def close(self):
                self.connection.close()

        def open_map(*args):
            nonlocal open_calls
            open_calls += 1
            connection = original_open(*args)
            if open_calls == 1:
                return CorruptOnSelect(connection)
            return connection

        monkeypatch.setattr(indexer, "_open_index_map", open_map)
        assert indexer._get_index_map(cache_path) == {}
        assert cache_path.read_bytes().startswith(b"SQLite format 3")

    @pytest.mark.parametrize("_repeat", range(3))
    def test_concurrent_process_recovery_preserves_updates(self, tmp_path, _repeat):
        """Separate processes repeatedly coordinate corruption recovery."""
        from dascore.io.index.indexer import _get_index_map

        cache_path = tmp_path / "cache_paths.sqlite3"
        cache_path.write_bytes(b"not a sqlite database")
        context = multiprocessing.get_context("spawn")
        start = context.Event()
        ready = [context.Event(), context.Event()]
        keys = ("a", "b")
        processes = [
            context.Process(
                target=_update_corrupt_index_map_process,
                args=(str(cache_path), key, event, start),
            )
            for key, event in zip(keys, ready, strict=True)
        ]
        try:
            for process in processes:
                process.start()
            assert all(event.wait(timeout=20) for event in ready)
            start.set()
            for process in processes:
                process.join(timeout=30)
            assert [process.exitcode for process in processes] == [0, 0]
        finally:
            start.set()
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)

        assert _get_index_map(cache_path) == {"a": "index-a", "b": "index-b"}

    @pytest.mark.skipif(
        "fork" not in multiprocessing.get_all_start_methods(),
        reason="requires POSIX fork",
    )
    def test_recovery_guard_blocks_fork(self, tmp_path, monkeypatch):
        """Fork waits until no recovery file-lock descriptor can be inherited."""
        import dascore.io.index.indexer as indexer

        cache_path = tmp_path / "cache_paths.sqlite3"
        indexer._update_index_map({"a": "1"}, cache_path)

        class ObservedLock:
            """Expose when the at-fork callback waits on the guard lock."""

            def __init__(self):
                self._lock = threading.Lock()
                self._count_lock = threading.Lock()
                self.acquire_count = 0
                self.fork_waiting = threading.Event()

            def acquire(self):
                with self._count_lock:
                    self.acquire_count += 1
                    if self.acquire_count == 2:
                        self.fork_waiting.set()
                return self._lock.acquire()

            def release(self):
                self._lock.release()

            def __enter__(self):
                self.acquire()
                return self

            def __exit__(self, *_):
                self.release()

        observed_lock = ObservedLock()
        monkeypatch.setattr(indexer, "_INDEX_MAP_RECOVERY_LOCK", observed_lock)
        guard_held = threading.Event()
        release_guard = threading.Event()
        fork_errors = []

        def hold_guard():
            with indexer._index_map_recovery_guard(cache_path):
                guard_held.set()
                release_guard.wait(timeout=20)

        context = multiprocessing.get_context("fork")
        process = context.Process(
            target=_read_index_map_process,
            args=(str(cache_path),),
        )

        def start_child():
            try:
                with suppress_warnings(DeprecationWarning):
                    process.start()
                process.join(timeout=10)
            except BaseException as exc:
                fork_errors.append(exc)

        guard_thread = threading.Thread(target=hold_guard)
        fork_thread = threading.Thread(target=start_child)
        guard_thread.start()
        try:
            assert guard_held.wait(timeout=5)
            fork_thread.start()
            assert observed_lock.fork_waiting.wait(timeout=5)
            assert process.pid is None
            release_guard.set()
            fork_thread.join(timeout=15)
            assert not fork_thread.is_alive()
            assert not fork_errors
            assert process.exitcode == 0
        finally:
            release_guard.set()
            guard_thread.join(timeout=5)
            if fork_thread.is_alive():
                fork_thread.join(timeout=15)
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(timeout=5)

    def test_get_reads_fresh_each_call(self, tmp_path):
        """Reads are not cached, so out-of-band SQLite changes are visible."""
        from dascore.io.index.indexer import _get_index_map, _update_index_map

        cache_path = tmp_path / "cache_paths.sqlite3"
        _update_index_map({"a": "1"}, cache_path=str(cache_path))
        assert _get_index_map(str(cache_path)) == {"a": "1"}
        with closing(sqlite3.connect(cache_path)) as connection, connection:
            connection.execute(
                "INSERT INTO index_map (directory, index_path) VALUES (?, ?)",
                ("b", "2"),
            )
        assert _get_index_map(str(cache_path)) == {"a": "1", "b": "2"}

    def test_concurrent_updates_preserve_every_entry(self, tmp_path):
        """Concurrent callers preserve entries when they start together."""
        from dascore.io.index.indexer import _get_index_map, _update_index_map

        cache_path = tmp_path / "cache_paths.sqlite3"
        keys = [f"source-{number}" for number in range(8)]
        barrier = threading.Barrier(len(keys))

        def update(key):
            barrier.wait(timeout=5)
            _update_index_map({key: f"index-{key}"}, cache_path=cache_path)

        with ThreadPoolExecutor(max_workers=len(keys)) as pool:
            futures = [pool.submit(update, key) for key in keys]
            for future in futures:
                future.result(timeout=10)

        expected = {key: f"index-{key}" for key in keys}
        assert _get_index_map(cache_path) == expected

    def test_legacy_json_is_ignored(self, tmp_path):
        """A neighboring JSON map is neither read nor changed."""
        from dascore.io.index.indexer import _get_index_map

        legacy_path = tmp_path / "cache_paths.json"
        database_path = tmp_path / "cache_paths.sqlite3"
        legacy_contents = '{"a": "1", "b": "2"}'
        legacy_path.write_text(legacy_contents)

        assert _get_index_map(database_path) == {}
        assert database_path.read_bytes().startswith(b"SQLite format 3")
        assert legacy_path.read_text() == legacy_contents

    def test_configured_suffix_does_not_select_storage_format(self, tmp_path):
        """The configured map path is SQLite regardless of its suffix."""
        from dascore.io.index.indexer import _get_index_map, _update_index_map

        cache_path = tmp_path / "custom-index-map.json"
        _update_index_map({"external-data": "external-index.sqlite3"}, cache_path)

        assert cache_path.read_bytes().startswith(b"SQLite format 3")
        assert _get_index_map(cache_path) == {"external-data": "external-index.sqlite3"}


class TestBasics:
    """Basic tests for indexer."""

    def test_str_repr(self, basic_indexer):
        """Ensure a useful (not the default) str/repr is implemented."""
        out = str(basic_indexer)
        assert "object at" not in out

    def test_metadata(self, basic_indexer):
        """The index records its schema version and identity."""
        meta = basic_indexer._backend.get_metadata()
        assert meta["what_is_this"] == "dascore_spool_index"
        assert meta["index_version"] >= 1


class TestGetContents:
    """Test cases for getting contents of indexer as dataframes."""

    def test_get_contents(self, basic_indexer, two_patch_directory):
        """Ensure contents are returned."""
        out = basic_indexer()
        files = list(Path(two_patch_directory).rglob("*.hdf5"))
        assert isinstance(out, pd.DataFrame)
        assert len(out) == len(files)
        names_df = {x.split("/")[-1] for x in out["path"]}
        names_files = {x.name for x in files}
        assert names_df == names_files

    def test_filter_time_after(self, diverse_df, diverse_indexer):
        """Half-open time range keeps every file overlapping it."""
        max_starttime = diverse_df["time_min"].max()
        expected = diverse_df[diverse_df["time_max"] >= max_starttime]
        out = diverse_indexer(time=(max_starttime, None))
        assert len(out) == len(expected)

    def test_filter_time_before(self, diverse_df, diverse_indexer):
        """Half-open time range keeps every file overlapping it."""
        min_endtime = diverse_df["time_max"].min()
        expected = diverse_df[diverse_df["time_min"] <= min_endtime]
        out = diverse_indexer(time=(None, min_endtime))
        assert len(out) == len(expected)

    def test_filter_station_exact(self, diverse_df, diverse_indexer):
        """Ensure contents can be filtered on an attr."""
        exact_name = diverse_df["station"].unique()[0]
        new_df = diverse_indexer(station=exact_name)
        assert (new_df["station"] == exact_name).all()

    def test_filter_isin(self, diverse_df, diverse_indexer):
        """Ensure contents can be filtered with a collection."""
        # empty strings mean "attr missing" and are not queryable (spec).
        stations = [x for x in diverse_df["station"].unique() if x]
        new_df = diverse_indexer(station=stations[:2])
        assert set(new_df["station"]) <= set(stations[:2])
        assert len(new_df)

    def test_empty_index(self, empty_index):
        """An empty index should return an empty dataframe."""
        df = empty_index()
        assert df.empty


class TestUpdate:
    """Tests for updating the index."""

    @pytest.fixture(scope="class")
    def spool_directory_with_non_das_file(self, two_patch_directory, tmp_path_factory):
        """Create a directory with some das files and some non-das files."""
        new = tmp_path_factory.mktemp("unreadable_test") / "sub"
        shutil.copytree(two_patch_directory, new)
        with suppress(FileNotFoundError):
            for index in Path(new).glob(".dascore_index*"):
                index.unlink()
        with open(new / "not_das.open", "w") as fi:
            fi.write("cant be das, can it?")
        return new

    def test_add_one_patch(self, empty_index, random_patch):
        """Ensure a new patch added to the directory shows up."""
        path = empty_index.path / get_patch_names(random_patch).iloc[0]
        random_patch.io.write(path, file_format="dasdae")
        new_index = empty_index.update(progress=None)
        contents = new_index()
        assert len(contents) == 1

    def test_index_with_bad_file(self, spool_directory_with_non_das_file):
        """Ensure if one file is not readable index continues."""
        indexer = DBDirectoryIndexer(spool_directory_with_non_das_file)
        updated = indexer.update(progress=None)
        assert isinstance(updated, DBDirectoryIndexer)
        assert len(updated()) == 2

    def test_removed_file_dropped(self, two_patch_directory, tmp_path_factory):
        """A deleted file's rows disappear on the next update."""
        new = tmp_path_factory.mktemp("removed_file_test") / "sub"
        shutil.copytree(two_patch_directory, new)
        for index in Path(new).glob(".dascore_index*"):
            index.unlink()
        indexer = DBDirectoryIndexer(new).update(progress=None)
        assert len(indexer()) == 2
        next(iter(Path(new).glob("*.hdf5"))).unlink()
        assert len(indexer.update(progress=None)()) == 1

    def test_noop_update_rescans_nothing(self, basic_indexer):
        """Unchanged sources are not rescanned."""
        before = basic_indexer._backend.get_sources()["last_indexed_ns"].max()
        basic_indexer.update(progress=None)
        after = basic_indexer._backend.get_sources()["last_indexed_ns"].max()
        assert before == after

    def test_update_with_specific_paths(self, basic_indexer):
        """Updating with specific paths restricts the rescan."""
        files = sorted(basic_indexer.path.rglob("*.hdf5"))
        assert len(files) >= 2

        def _indexed_times():
            sources = basic_indexer._backend.get_sources().set_index("source_path")
            return sources["last_indexed_ns"].to_dict()

        before = _indexed_times()
        for path in files[:2]:
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

        first, second = (basic_indexer._rel(path) for path in files[:2])
        basic_indexer.update(paths=[files[0].name], progress=None)
        after_relative = _indexed_times()
        assert after_relative[first] > before[first]
        assert after_relative[second] == before[second]

        basic_indexer.update(paths=[str(files[1])], progress=None)
        after_absolute = _indexed_times()
        assert after_absolute[first] == after_relative[first]
        assert after_absolute[second] > after_relative[second]


class TestNameResolution:
    """Unknown names raise per the selector spec."""

    def test_unknown_name_raises(self, basic_indexer):
        """Names in neither namespace error clearly (#435)."""
        from dascore.io.index.query import InvalidSpoolQueryError

        with pytest.raises(InvalidSpoolQueryError, match="neither an attribute"):
            basic_indexer(bad_dimension=(1, 2))
