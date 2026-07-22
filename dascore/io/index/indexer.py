"""
A directory indexer backed by the generic spool index.

It walks a directory, detects new/changed/removed sources by per-source
(mtime, size) comparison, scans only what changed, and answers content
queries from the index backend. Also holds the machinery for tracking
index locations when the data directory itself is not writable (e.g.
read-only archives).
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from contextlib import contextmanager, suppress
from pathlib import Path
from threading import Lock

import pandas as pd
from typing_extensions import Self

import dascore as dc
from dascore.compat import UPath
from dascore.config import config_attr
from dascore.constants import PROGRESS_LEVELS
from dascore.exceptions import InvalidIndexVersionError
from dascore.io.index.backend import get_backend, resolve_query
from dascore.io.index.ingest import SourceRecord, summaries_to_records
from dascore.io.index.schema import SPOOL_HIDDEN_COLUMNS
from dascore.utils.misc import _iter_filesystem
from dascore.utils.paths import directory_writable, requires_local_directory

_INDEX_MAP_SCHEMA = """
CREATE TABLE IF NOT EXISTS index_map (
    directory TEXT PRIMARY KEY,
    index_path TEXT NOT NULL
)
"""
_INDEX_MAP_UPSERT = """
INSERT INTO index_map (directory, index_path)
VALUES (?, ?)
ON CONFLICT(directory) DO UPDATE SET index_path = excluded.index_path
"""
_INDEX_MAP_CORRUPTION_CODES = {
    getattr(sqlite3, "SQLITE_CORRUPT", 11),
    getattr(sqlite3, "SQLITE_NOTADB", 26),
}
_INDEX_MAP_CORRUPTION_MESSAGES = ("database disk image is malformed", "not a database")
_INDEX_MAP_RECOVERY_LOCK = Lock()


def _acquire_index_map_recovery_lock_before_fork():
    """Wait for the recovery guard before allowing a process fork."""
    _INDEX_MAP_RECOVERY_LOCK.acquire()


def _release_index_map_recovery_lock_after_fork():
    """Release the parent recovery lock acquired before a process fork."""
    _INDEX_MAP_RECOVERY_LOCK.release()


def _reset_index_map_recovery_lock():
    """Replace thread-lock state inherited by a forked child."""
    global _INDEX_MAP_RECOVERY_LOCK
    _INDEX_MAP_RECOVERY_LOCK = Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_acquire_index_map_recovery_lock_before_fork,
        after_in_parent=_release_index_map_recovery_lock_after_fork,
        after_in_child=_reset_index_map_recovery_lock,
    )


def _is_corrupt_index_map_error(exc: sqlite3.DatabaseError) -> bool:
    """Return whether an SQLite error means the disposable map is corrupt."""
    error_code = getattr(exc, "sqlite_errorcode", None)
    if error_code is not None:
        return error_code & 0xFF in _INDEX_MAP_CORRUPTION_CODES
    message = str(exc).lower()
    return any(part in message for part in _INDEX_MAP_CORRUPTION_MESSAGES)


def _open_index_map(database_path: Path) -> sqlite3.Connection:
    """Open and initialize one SQLite index-map connection."""
    connection = sqlite3.connect(database_path, timeout=30, isolation_level=None)
    try:
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute(_INDEX_MAP_SCHEMA)
        return connection
    except BaseException:
        connection.close()
        raise


def _acquire_recovery_file_lock(lock_file):
    """Acquire a content-independent process lock on an open file."""
    if os.name == "nt":  # pragma: no cover
        import errno
        import msvcrt
        import time

        lock_file.seek(0, os.SEEK_END)
        if not lock_file.tell():
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        transient_errors = {errno.EACCES, errno.EAGAIN, errno.EDEADLK}
        while True:
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                if exc.errno not in transient_errors:
                    raise
                time.sleep(0.05)
            else:
                break
    else:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)


def _release_recovery_file_lock(lock_file):
    """Release a process lock acquired by `_acquire_recovery_file_lock`."""
    if os.name == "nt":  # pragma: no cover
        import msvcrt

        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _index_map_recovery_guard(database_path: Path):
    """Keep map operations out of the way of destructive recovery."""
    lock_path = database_path.with_name(f"{database_path.name}.recovery.lock")
    with _INDEX_MAP_RECOVERY_LOCK, lock_path.open("a+b") as lock_file:
        _acquire_recovery_file_lock(lock_file)
        try:
            yield
        finally:
            _release_recovery_file_lock(lock_file)


def _index_map_is_healthy(database_path: Path) -> bool:
    """Validate the schema and all map leaf pages."""
    connection = None
    try:
        connection = _open_index_map(database_path)
        connection.execute("SELECT directory, index_path FROM index_map").fetchall()
    except sqlite3.DatabaseError as exc:
        if not _is_corrupt_index_map_error(exc):
            raise
        return False
    finally:
        if connection is not None:
            connection.close()
    return True


def _remove_index_map(database_path: Path) -> None:
    """Remove a corrupt map and SQLite sidecars before rebuilding."""
    for suffix in ("", "-journal", "-shm", "-wal"):
        Path(f"{database_path}{suffix}").unlink(missing_ok=True)


def _recover_index_map(database_path: Path) -> None:
    """Rebuild a corrupt map while the caller holds the recovery guard."""
    # A process that held the recovery lock before us may have fixed it.
    if _index_map_is_healthy(database_path):
        return
    _remove_index_map(database_path)
    _open_index_map(database_path).close()


def _run_index_map_operation(cache_path, operation):
    """Run one complete map operation, rebuilding and retrying on corruption."""
    database_path = Path(cache_path)
    database_path.parent.mkdir(exist_ok=True, parents=True)
    # Recovery removes and recreates the database, so participating access
    # is deliberately exclusive, including reads. This ensures no process retains
    # an open handle to the unlinked database or its name-based SQLite sidecars.
    # SQLite transactions also protect against connections outside this guard.
    with _index_map_recovery_guard(database_path):
        for attempt in range(2):
            connection = None
            try:
                connection = _open_index_map(database_path)
                return operation(connection)
            except sqlite3.DatabaseError as exc:
                if attempt or not _is_corrupt_index_map_error(exc):
                    raise
            finally:
                if connection is not None:
                    connection.close()
            _recover_index_map(database_path)
    raise AssertionError("unreachable")


def _get_index_map(cache_path) -> dict[str, str]:
    """
    Return a fresh dict of index locations read from the SQLite database.

    Read (not cached): another process may have updated the map.
    """

    def read(connection):
        rows = connection.execute(
            "SELECT directory, index_path FROM index_map"
        ).fetchall()
        return dict(rows)

    return _run_index_map_operation(cache_path, read)


def _update_index_map(updates, cache_path) -> dict[str, str]:
    """Transactionally upsert index locations without losing other writers."""
    rows = [(str(key), str(value)) for key, value in updates.items()]

    def update(connection):
        with connection:
            # Reserve the single writer slot before reading or updating rows.
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany(_INDEX_MAP_UPSERT, rows)
            data = connection.execute(
                "SELECT directory, index_path FROM index_map"
            ).fetchall()
        return dict(data)

    return _run_index_map_operation(cache_path, update)


class DBDirectoryIndexer:
    """
    Index a directory of fiber files with a database backend.

    Parameters
    ----------
    path
        The directory to index.
    index_path
        Where to keep the index; defaults to a hidden entry at the top of
        the data directory.
    """

    ext: str | None = None
    # user-level file tracking index locations for unwritable data dirs
    index_map_path: Path = config_attr("directory_index_map_path")

    def __init__(
        self,
        path: str | Path,
        index_path: str | Path | None = None,
    ):
        path = UPath(path).absolute() if isinstance(path, UPath) else Path(path)
        requires_local_directory(path, label="DBDirectoryIndexer")
        self.path = Path(path).absolute()
        self.index_path = Path(self._find_index_path(index_path))
        try:
            self._backend = get_backend(self.index_path)
        except InvalidIndexVersionError:
            # The index is a disposable cache and the file already
            # identified itself as a dascore spool index of another
            # schema version; rebuild it rather than asking the user to.
            self.index_path.unlink()
            self._backend = get_backend(self.index_path)
        # Schema creation alone is not a successful directory scan. Read the
        # transactional marker so a new process retries an interrupted first
        # update instead of trusting a merely nonempty SQLite file.
        metadata = self._backend.get_metadata()
        self._initial_update_done = bool(metadata["last_indexed_ns"])

    @property
    def _index_name(self) -> str:
        return ".dascore_index.sqlite3"

    @staticmethod
    def _is_legacy_or_foreign_index(path: Path) -> bool:
        """Return True if an existing file is not a SQLite database.

        Older DASCore versions recorded PyTables (.h5) index locations in
        the index map; passing those to sqlite3 fails with an opaque
        error instead of building the replacement index. Only the file
        header decides — users may legitimately choose any suffix for a
        custom index path.
        """
        if not path.exists():
            return False
        with suppress(OSError), open(path, "rb") as fh:
            header = fh.read(16)
            return len(header) >= 16 and not header.startswith(b"SQLite format 3")
        return False

    def _find_index_path(self, index_path=None) -> Path:
        """
        Find where the index lives (or should live).

        Mirrors the historic DirectoryIndexer behavior: in-directory by
        default; when the data directory is read-only the index lives in
        the dascore cache and its location is recorded in the index map.
        """
        map_key = str(self.path)
        if index_path:
            index_path = Path(index_path).absolute()
            update = {map_key: str(index_path)}
            _update_index_map(update, cache_path=str(self.index_map_path))
            return index_path
        expected = self.path / self._index_name
        with suppress(PermissionError):
            if expected.exists():
                return expected
        path_map = {}
        # A writable data directory can fall back to its local index when the
        # optional global map is unavailable, such as on a read-only cache.
        with suppress(OSError, sqlite3.OperationalError):
            path_map = _get_index_map(cache_path=str(self.index_map_path))
        if out := path_map.get(map_key):
            mapped = Path(out)
            # Index-map entries from older DASCore versions can point at
            # the retired PyTables (.h5) index; those are not usable and
            # a fresh SQLite index is built in their place.
            if not self._is_legacy_or_foreign_index(mapped):
                return mapped
        if not directory_writable(self.path):
            # A stable digest, not hash(): str/Path hashing is randomized
            # per process (PYTHONHASHSEED), so hash() would name a new
            # index file every session and orphan the previous one.
            digest = hashlib.sha256(str(self.path).encode()).hexdigest()[:16]
            name = f"_dascore_index_{digest}.sqlite3"
            index_path = self.index_map_path.parent / name
            _update_index_map(
                {map_key: str(index_path.absolute())},
                cache_path=str(self.index_map_path),
            )
            return index_path
        return expected

    def ensure_updated(self) -> bool:
        """Run the initial update if the index was never populated."""
        if self._initial_update_done:
            return False
        self.update(progress=None)
        return True

    def __str__(self) -> str:
        return f"{self.__class__.__name__} managing: {self.path}"

    __repr__ = __str__

    def __deepcopy__(self, memo) -> Self:
        """
        Derived spools share the indexer (and its live DB connection).

        Spool copies its state on select/chunk; the index
        connection is read-shared, matching the single-writer model.
        """
        return self

    def _rel(self, path: Path) -> str:
        """Relative posix path of a file under the spool root."""
        return Path(path).relative_to(self.path).as_posix()

    def _directory_format(self, path: Path) -> bool:
        """Return True when a directory is itself one FiberIO scan unit."""
        from dascore.io.core import is_directory_format

        return is_directory_format(path)

    @staticmethod
    def _directory_signature(path: Path) -> tuple[int, int]:
        """Return a stable 128-bit manifest signature as two SQLite ints."""
        members = sorted(
            (
                sub
                for sub in path.rglob("*")
                if sub.is_file() and not sub.name.startswith(".")
            ),
            key=lambda sub: sub.relative_to(path).as_posix(),
        )
        digest = hashlib.sha256()
        for member in members:
            stat = member.stat()
            relative = member.relative_to(path).as_posix().encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(stat.st_mtime_ns.to_bytes(8, "big", signed=True))
            digest.update(stat.st_size.to_bytes(8, "big"))
        fingerprint = digest.digest()
        return (
            int.from_bytes(fingerprint[:8], "big", signed=True),
            int.from_bytes(fingerprint[8:16], "big", signed=True),
        )

    def _walk(self) -> dict[str, tuple[int, int, Path]]:
        """
        Walk the spool directory, honoring directory-format scan units.

        Maps relative path -> (mtime_ns, size, abs path) for every scan
        unit. A directory-format unit (e.g. XMLBinary) appears as one
        entry keyed by the directory, with a 128-bit manifest fingerprint
        split across the two integer stat fields. The fingerprint covers
        every member's relative path, mtime, and size, so member changes
        cannot cancel each other out. Mirrors the skip protocol dc.scan
        uses so members are not offered individually.
        """
        files: dict[str, tuple[int, int, Path]] = {}
        gen = _iter_filesystem(self.path, ext=self.ext, include_directories=True)
        signal = None
        while True:
            try:
                # send(None) is equivalent to next() and also starts it
                candidate = gen.send(signal)
            except StopIteration:
                break
            signal = None
            if candidate is None:  # the reply to a "skip" send
                continue
            path = Path(candidate)
            if path.is_dir():
                if self._directory_format(path):
                    signal = "skip"
                    signature = self._directory_signature(path)
                    files[self._rel(path)] = (*signature, path)
                continue
            try:
                stat = path.stat()
            except OSError:
                # The file vanished between the walk yielding it and this
                # stat (a concurrent deletion); skip it rather than
                # crashing the whole index update.
                continue
            files[self._rel(path)] = (stat.st_mtime_ns, stat.st_size, path)
        return files

    def update(self, paths=None, progress: PROGRESS_LEVELS = "standard") -> Self:
        """
        Update the index: scan new/changed sources, drop removed ones.

        Change detection compares each source's stored (mtime_ns,
        size_bytes) against the filesystem — never a global watermark —
        and stale-source removal is folded in (the walk is the dominant
        cost; removal afterwards is nearly free). Directory-format scan
        units (e.g. XMLBinary) are rescanned whole when any member file
        changes.
        """
        files = self._walk()
        stored = {
            row.source_path: (
                None
                if pd.isnull(row.mtime_ns)
                else (int(row.mtime_ns), int(row.size_bytes))
            )
            for row in self._backend.source_stats().itertuples()
        }
        stale = [path for path in stored if path not in files]
        changed = [
            rel
            for rel, (mtime, size, _) in files.items()
            if stored.get(rel) != (mtime, size)
        ]
        if paths is not None:
            # restrict the rescan (not stale removal) to the given paths
            keep = set()
            for one in paths:
                one = Path(one)
                rel = (
                    one.relative_to(self.path).as_posix()
                    if one.is_absolute()
                    else one.as_posix()
                )
                keep.add(rel)
            changed = [rel for rel in changed if rel in keep]
        if stale:
            self._backend.delete_sources(stale)
        if changed:
            # Only changed paths are rescanned, so the stat maps handed to
            # summaries_to_records need only cover them — not the whole
            # archive (a large mostly-unchanged directory otherwise built
            # full-archive mtime/size maps for a tiny update).
            changed_stats = {rel: files[rel] for rel in changed}
            scan_paths = [stat[2] for stat in changed_stats.values()]
            summaries = dc.scan(scan_paths, progress=progress)
            # scan reports absolute source paths; stat maps use them too
            records = summaries_to_records(
                summaries,
                relative_to=str(self.path),
                mtimes_ns={str(p): m for (m, _, p) in changed_stats.values()},
                sizes_bytes={str(p): s for (_, s, p) in changed_stats.values()},
            )
            # Every visited path gets a sources row, even when scanning
            # produced no patches (e.g. a non-fiber file). Otherwise such
            # files look "new" on every update and force perpetual
            # rescans.
            recorded = {rec.source_path for rec in records}
            for rel in set(changed) - recorded:
                mtime, size, _ = files[rel]
                records.append(
                    SourceRecord(
                        source_path=rel,
                        source_format="",
                        format_version="",
                        mtime_ns=mtime,
                        size_bytes=size,
                    )
                )
            if records:
                self._backend.write_sources(records)
        if stale or changed or not self._initial_update_done:
            # Directory archives present in time order; ingest assigns
            # walk-order ordinals, so each sync renumbers to keep the
            # contract (iterate by ordinal) aligned with time. The
            # not-yet-marked-done case covers a process killed between
            # write_sources committing and this renumber: the retry sees
            # no stale/changed files but must still fix walk-order
            # ordinals before marking the initial update complete.
            self._backend.renumber_ordinals_by_time()
        if not self._initial_update_done:
            self._backend.mark_initial_update_done()
            self._initial_update_done = True
        return self

    def get_contents(self, _attrs=None, _coords=None, **kwargs) -> pd.DataFrame:
        """
        Query the index, returning the spool-facing flat relation.

        Bare kwargs resolve attrs-first then coords; `_attrs`/`_coords`
        disambiguate explicitly (see the selector semantics spec).
        """
        self.ensure_updated()
        query = resolve_query(self._backend, _attrs=_attrs, _coords=_coords, **kwargs)
        df = self._backend.query(query)
        df = df.drop(columns=list(SPOOL_HIDDEN_COLUMNS), errors="ignore")
        return df.rename(columns={"patch_id": "_patch_id"})

    __call__ = get_contents

    def close(self) -> None:
        """Close the backend."""
        self._backend.close()
