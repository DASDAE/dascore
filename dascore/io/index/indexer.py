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
import json
import math
import os
import stat
import tempfile
import warnings
from contextlib import suppress
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Self

import pandas as pd

import dascore as dc
from dascore.compat import UPath
from dascore.config import config_attr, config_context, get_config
from dascore.constants import INDEX_NAME, PROGRESS_LEVELS, ExecutorType
from dascore.exceptions import InvalidIndexVersionError
from dascore.io.core import _directory_members, is_directory_format
from dascore.io.index.backend import get_backend
from dascore.io.index.ingest import (
    SourceRecord,
    hive_path_attrs,
    summaries_to_records,
)
from dascore.utils.identity import _snap_names
from dascore.utils.misc import _iter_filesystem
from dascore.utils.paths import (
    coerce_to_local_path,
    directory_writable,
    requires_local_directory,
)
from dascore.utils.progress import track, validate_progress_level
from dascore.warnings import DASCoreWarning


def _directory_signature(path: Path) -> tuple[int, int]:
    """Return a stable 128-bit manifest signature as two SQLite ints."""
    digest = hashlib.sha256()
    for member in _directory_members(path):
        status = member.stat()
        relative = member.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(status.st_mtime_ns.to_bytes(8, "big", signed=True))
        digest.update(status.st_size.to_bytes(8, "big"))
    fingerprint = digest.digest()
    return (
        int.from_bytes(fingerprint[:8], "big", signed=True),
        int.from_bytes(fingerprint[8:16], "big", signed=True),
    )


def scan_unit_stats(path) -> tuple[int | None, int | None]:
    """
    Return the (mtime_ns, size_bytes) used to detect source changes.

    Index writers and readers share this check. Files return their stat values;
    directory formats return a manifest fingerprint because directory stats miss member
    content and size changes. Return (None, None) when filesystem access fails.
    """
    try:
        status = os.stat(path)
        if stat.S_ISDIR(status.st_mode):
            # A member removed between the listing and its stat is a change.
            return _directory_signature(Path(path))
    except OSError:
        return None, None
    return status.st_mtime_ns, status.st_size


def _scan_batch(paths, config, snap=None):
    """Scan a source batch using the caller's DASCore configuration."""
    with config_context(config):
        return dc.scan(paths, progress=None, snap=snap)


def snap_text(snap) -> str:
    """The JSON an index records a snap option as; null for each format's default."""
    return json.dumps(None if snap is None else _snap_names(snap))


def _scan_with_client(paths, client, progress, snap=None):
    """Collect ordered scan batches from a caller-owned executor."""
    # Standard-library pools expose capacity here. Map-only clients can omit
    # it; host CPU count still gives them several batches to schedule.
    workers = getattr(client, "_max_workers", None)
    if not isinstance(workers, int) or workers < 1:
        workers = os.cpu_count() or 1
    size = max(1, math.ceil(len(paths) / (4 * workers)))
    batches = [paths[start : start + size] for start in range(0, len(paths), size)]
    func = partial(_scan_batch, config=get_config(), snap=snap)
    results = client.map(func, batches)
    summaries = []
    try:
        for batch in track(
            results, "Scanning file batches", progress, length=len(batches)
        ):
            summaries.extend(batch)
    finally:
        # Retained exception tracebacks must not keep queued work alive.
        # Map-only clients are also allowed to return a plain iterable.
        if closer := getattr(results, "close", None):
            closer()
    return summaries


def _path_digest(path) -> str:
    """
    Return a stable SHA-256 path digest; Python's hash varies by process.

    Use the full digest to minimize collisions that would make directories share an
    index. ``os.fsencode`` preserves filesystem bytes, including non-UTF-8 names;
    unsupported inputs fall back to string encoding, allowing future remote paths.
    """
    try:
        encoded = os.fsencode(path)
    except (TypeError, ValueError, NotImplementedError):
        encoded = str(path).encode("utf-8", "surrogatepass")
    return hashlib.sha256(encoded).hexdigest()


def _map_entry_path(directory, map_dir) -> Path:
    """Return the entry file recording one data directory's index location."""
    return Path(map_dir) / f"{_path_digest(directory)}.json"


def _get_mapped_index_path(directory, map_dir) -> Path | None:
    """
    Return a directory's recorded index path, or None.

    Each mapping has its own JSON file. Treat corrupt or unreadable entries as misses;
    the next atomic write repairs them. Reads never delete entries, which could race a
    writer's repair. See #508.
    """
    entry = _map_entry_path(directory, map_dir)
    try:
        with entry.open("r") as fi:
            data = json.load(fi)
    except (OSError, ValueError):
        # Missing, unreadable, non-UTF-8, or non-JSON: treat as a miss.
        return None
    # Guard the (astronomically unlikely) digest collision and any
    # malformed payload; a mismatch or bad shape is treated as a miss.
    if not isinstance(data, dict) or data.get("directory") != str(directory):
        return None
    index_path = data.get("index_path")
    if not isinstance(index_path, str) or not index_path:
        return None
    return Path(index_path)


def _set_mapped_index_path(directory, index_path, map_dir) -> None:
    """
    Record a directory's external index location atomically.

    Replace the entry with a sibling temporary file so readers see complete files.
    Separate entries let writers update different directories independently.
    """
    entry = _map_entry_path(directory, map_dir)
    entry.parent.mkdir(exist_ok=True, parents=True)
    payload = {"directory": str(directory), "index_path": str(index_path)}
    fd, tmp = tempfile.mkstemp(dir=entry.parent, prefix=entry.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fi:
            json.dump(payload, fi)
        os.replace(tmp, entry)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


class DBDirectoryIndexer:
    """
    Index a directory of fiber files with a database backend.

    Parameters
    ----------
    path
        The directory to index.
    index_path
        Where to keep the index; defaults to a hidden entry at the top of
        the data directory, or the dascore cache when the directory is
        read-only or is itself one store (e.g. zarr).
    snap
        The dimensions each file's patches are read on a grid, as in
        [`read`](`dascore.read`). The index records it beside
        `dascore.config.max_shift`, and an index scanned under others is
        rescanned.
    """

    ext: str | None = None
    # cache dir holding per-data-directory index-location entries, used
    # when a data directory itself is not writable
    index_map_dir: Path = config_attr("directory_index_map_dir")

    def __init__(
        self,
        path: str | Path | UPath,
        index_path: str | Path | None = None,
        snap=None,
    ):
        path = UPath(path).absolute() if isinstance(path, UPath) else Path(path)
        requires_local_directory(path, label="DBDirectoryIndexer")
        self.path = Path(coerce_to_local_path(path)).absolute()
        self.index_path = Path(self._find_index_path(index_path))
        self.snap, self.max_shift = snap, get_config().max_shift
        try:
            self._backend = get_backend(self.index_path)
        except InvalidIndexVersionError:
            # The index is a disposable cache and the file already
            # identified itself as a dascore spool index of another
            # schema version; rebuild it rather than asking the user to.
            self.index_path.unlink()
            self._backend = get_backend(self.index_path)
            self._backend.clear()  # marks it rebuilt, not new
        # Schema creation alone is not a successful directory scan. Read the
        # transactional marker so a new process retries an interrupted first
        # update instead of trusting a merely nonempty SQLite file.
        self._check_decode()

    def _check_decode(self) -> None:
        """
        Start the index again where its files were scanned under another decode.

        Checked when opened and again before each update, since another
        spool of the directory may have changed it in between.
        """
        metadata = self._backend.get_metadata()
        # The marker is 0 for a new index, -1 for a cleared or rebuilt one,
        # and the time of the first complete update after that.
        marker = metadata["last_indexed_ns"]
        decode = (snap_text(self.snap), self.max_shift)
        stored = (metadata["snap"], metadata["max_shift"])
        if stored != decode and marker > 0:
            msg = (
                f"The index of {self.path} was scanned with snap={stored[0]} "
                f"and max_shift={stored[1]}; rescanning its files with "
                f"snap={decode[0]} and max_shift={decode[1]}."
            )
            warnings.warn(msg, UserWarning, stacklevel=5)
            self._backend.clear()
            marker = -1
        if stored != decode:
            self._backend.set_decode(*decode)
        self._initial_update_done = marker > 0 and stored == decode
        self._reset = marker < 0

    @property
    def _index_name(self) -> str:
        return INDEX_NAME

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

        In-directory by default; when the data directory is read-only or
        is itself one store, the index lives in the dascore cache. Only a
        custom location is recorded in the index map.
        """
        map_dir = self.index_map_dir
        if index_path:
            index_path = Path(index_path).absolute()
            # A custom location is the only case worth recording: it is not
            # otherwise rediscoverable. Read-only fallbacks are deterministic.
            _set_mapped_index_path(self.path, index_path, map_dir)
            return index_path
        expected = self.path / self._index_name
        # A directory one reader takes whole (e.g. a zarr store) is replaced
        # whole when rewritten, taking an index inside it along; its index
        # lives in the cache, as a read-only directory's does.
        unit = self._directory_format(self.path)
        with suppress(PermissionError):
            if not unit and expected.exists():
                return expected
        mapped = _get_mapped_index_path(self.path, map_dir)
        if mapped is not None and not self._is_legacy_or_foreign_index(mapped):
            # Index-map entries from older DASCore versions can point at
            # the retired PyTables (.h5) index; those are not usable and
            # a fresh SQLite index is built in their place.
            return mapped
        if unit or not directory_writable(self.path):
            # A stable digest, not hash(): str/Path hashing is randomized
            # per process (PYTHONHASHSEED), so hash() would name a new
            # index file every session and orphan the previous one. The
            # name is deterministic, so no map entry is needed to find it.
            map_dir.mkdir(parents=True, exist_ok=True)
            name = f"_dascore_index_{_path_digest(self.path)}.sqlite3"
            return map_dir / name
        return expected

    def ensure_updated(self) -> bool:
        """Run the initial update if the index was never populated."""
        if not self._initial_update_done:
            self._check_decode()  # another spool may have built it since
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
        return is_directory_format(path)

    def _walk(self, scopes=None) -> dict[str, tuple[int, int, Path]]:
        """
        Walk the spool directory (or only scopes), honoring scan units.

        Maps relative path -> (mtime_ns, size, abs path) for every scan
        unit. A directory-format unit (e.g. XMLBinary) appears as one
        entry keyed by the directory, with a 128-bit manifest fingerprint
        split across the two integer stat fields. The fingerprint covers
        the relative path, mtime, and size of each member
        `_directory_members` names, so member changes cannot cancel each
        other out. Mirrors the skip protocol dc.scan
        uses so members are not offered individually.
        """
        files: dict[str, tuple[int, int, Path]] = {}
        if scopes is None or not self.path.is_dir():
            # A vanished root raises here rather than reading as deletions.
            self._walk_root(self.path, files)
        for scope in scopes or ():
            # A scope that no longer exists contributes nothing, so its rows
            # are pruned as stale.
            if (root := self.path / scope).exists():
                self._walk_root(root, files)
        return files

    def _walk_root(self, root: Path, files: dict) -> None:
        """Add the scan units under one root to files."""
        gen = _iter_filesystem(root, ext=self.ext, include_directories=True)
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
            # Not Path(candidate): the generator's yield type includes UPath,
            # which is not os.PathLike unless it resolved to a local path.
            path = coerce_to_local_path(candidate)
            directory = path.is_dir()
            if directory and not self._directory_format(path):
                continue
            if directory:
                signal = "skip"
            mtime, size = scan_unit_stats(path)
            if mtime is None or size is None:
                # The source vanished between the walk yielding it and
                # this stat (a concurrent deletion); skip it rather than
                # crashing the whole index update.
                continue
            files[self._rel(path)] = (mtime, size, path)

    def _scopes(self, paths, stored) -> list[str]:
        """Relative posix scopes for update(paths=...), refusing bad ones."""
        if isinstance(paths, str | os.PathLike):
            paths = [paths]
        base = Path(os.path.abspath(self.path))
        scopes = []
        for one in paths:
            full = Path(os.path.abspath(base / one))
            if not full.is_relative_to(base):
                msg = f"{one} is not inside the spool directory {self.path}"
                raise ValueError(msg)
            rel = PurePosixPath(full.relative_to(base).as_posix())
            if any(x.startswith(".") for x in rel.parts):
                msg = f"{one} is hidden, and the index skips hidden paths"
                raise ValueError(msg)
            if source := next((x for x in rel.parents if str(x) in stored), None):
                msg = f"{one} is inside the indexed source {source}; name that"
                raise ValueError(msg)
            scopes.append(str(rel))
        return scopes

    def _detect_moves(
        self,
        stale: list[str],
        stored: dict[str, tuple[int, int] | None],
        files: dict[str, tuple[int, int, Path]],
    ) -> dict[str, str]:
        """
        Map old -> new relative path for unambiguous renames/moves.

        A moved source keeps its exact (mtime_ns, size_bytes) — for
        directory-format units, the rename-invariant manifest signature —
        so a stale path and a brand-new path with identical stats, each
        unique on its side, are the same bytes at a new location and the
        index can rewrite the path instead of re-reading contents. This
        makes attaching metadata by renaming hive-style directories cheap
        on large archives. Renames that *remove* a hive key are excluded:
        restoring the file's own value for the dropped attr requires a
        rescan.
        """
        old_by_stats: dict[tuple[int, int], list[str]] = {}
        for rel in stale:
            stats = stored.get(rel)
            if stats is None or rel == ".":
                continue
            old_by_stats.setdefault(stats, []).append(rel)
        new_by_stats: dict[tuple[int, int], list[str]] = {}
        for rel, (mtime, size, _) in files.items():
            if rel in stored:
                continue
            new_by_stats.setdefault((mtime, size), []).append(rel)
        moves: dict[str, str] = {}
        for stats, olds in old_by_stats.items():
            news = new_by_stats.get(stats, ())
            if len(olds) != 1 or len(news) != 1:
                continue  # ambiguous match: fall back to rescan
            old, new = olds[0], news[0]
            old_keys = set(hive_path_attrs(old, warn=False))
            if old_keys - set(hive_path_attrs(new, warn=False)):
                continue
            moves[old] = new
        return moves

    def _scoped_stale(self, scopes, stored, files) -> list[str]:
        """
        Stale sources for a scoped update.

        Stored sources inside the scopes and missing from the walk, plus
        sources outside them that moved in: gone from disk, with the stats
        of a new file in scope. Without the latter, a move into a scope
        would leave the old row behind until a full update.
        """
        inside = [
            path
            for path in stored
            if any(x == "." or path == x or path.startswith(f"{x}/") for x in scopes)
        ]
        stale = [path for path in inside if path not in files]
        new = {(m, s) for rel, (m, s, _) in files.items() if rel not in stored}
        if new:
            inside = set(inside)
            stale += [
                path
                for path, stat in stored.items()
                if stat in new
                and path not in inside
                and not (self.path / path).exists()
            ]
        return stale

    def update(
        self,
        paths=None,
        progress: PROGRESS_LEVELS = "standard",
        *,
        client: ExecutorType | None = None,
        retry_unreadable: bool = False,
    ) -> Self:
        """
        Update the index: scan new/changed sources, drop removed ones.

        Change detection compares each source's stored (mtime_ns,
        size_bytes) against the filesystem — never a global watermark —
        and stale-source removal is folded in (the walk is the dominant
        cost; removal afterwards is nearly free). Directory-format scan
        units (e.g. XMLBinary) are rescanned whole when any member file
        changes. Renamed/moved sources are detected by their unchanged
        stats and rewritten in place, never rescanned (see _detect_moves).
        An optional client scans batches of changed sources; all database
        changes stay in the calling thread. The caller owns the client.
        Paths (files or directories under the root) limit the walk, rescans
        and pruning to those scopes, except that a source moved into a scope
        drops its old row. None, or an index that was rebuilt or never
        finished its first update, walks everything.
        retry_unreadable also rescans walked sources that gave no patches.
        """
        progress = validate_progress_level(progress)
        self._check_decode()
        legacy = self.path / ".dascore_index.h5"
        if legacy.exists() and self._is_legacy_or_foreign_index(legacy):
            msg = (
                f"{self.path} holds a legacy .dascore_index.h5, which DASCore "
                f"no longer uses and can be deleted; the index is now "
                f"{self.index_path}, so delete that file to reset the index."
            )
            skip = (str(Path(dc.__file__).parent),)
            warnings.warn(msg, DASCoreWarning, skip_file_prefixes=skip)
        stats = self._backend.source_stats()
        stored = {
            path: (None if pd.isnull(mtime) else (int(mtime), int(size)))
            for path, mtime, size in zip(
                stats["source_path"],
                stats["mtime_ns"],
                stats["size_bytes"],
                strict=True,
            )
        }
        # Only a new, empty index may be built one scope at a time; one that
        # was rebuilt, cleared, or partly built must first be walked whole,
        # or it would be marked complete while holding only the scopes.
        unfinished = not self._initial_update_done and (stored or self._reset)
        if paths is not None and unfinished:
            msg = (
                f"The index of {self.path} was rebuilt or never finished its "
                f"first update; ignoring paths and walking the whole directory."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
        scoped = paths is not None and not unfinished
        scopes = self._scopes(paths, stored) if scoped else None
        files = self._walk(scopes)
        if scopes is None:
            stale = [path for path in stored if path not in files]
        else:
            stale = self._scoped_stale(scopes, stored, files)
        retry = self._backend.unread_sources() if retry_unreadable else set()
        changed = [
            rel
            for rel, (mtime, size, _) in files.items()
            if stored.get(rel) != (mtime, size) or rel in retry
        ]
        moves = self._detect_moves(stale, stored, files)
        # A retried source that moved drops its old row and rescans at its
        # new path, instead of carrying the empty row across.
        moves = {old: new for old, new in moves.items() if old not in retry}
        if moves:
            new_attrs = {old: hive_path_attrs(new) for old, new in moves.items()}
            self._backend.move_sources(moves, new_attrs)
            moved_targets = set(moves.values())
            stale = [rel for rel in stale if rel not in moves]
            changed = [rel for rel in changed if rel not in moved_targets]
        if stale:
            self._backend.delete_sources(stale)
        if changed:
            # Only changed paths are rescanned, so the stat maps handed to
            # summaries_to_records need only cover them — not the whole
            # archive (a large mostly-unchanged directory otherwise built
            # full-archive mtime/size maps for a tiny update).
            changed_stats = {rel: files[rel] for rel in changed}
            scan_paths = [stat[2] for stat in changed_stats.values()]
            # scanned under the decode the index records, as reads are
            with config_context(max_shift=self.max_shift):
                summaries = (
                    dc.scan(scan_paths, progress=progress, snap=self.snap)
                    if client is None
                    else _scan_with_client(scan_paths, client, progress, self.snap)
                )
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
                        path_attrs=hive_path_attrs(rel, warn=False) or None,
                    )
                )
            if records:
                self._backend.write_sources(records)
        if stale or changed or moves or not self._initial_update_done:
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

    def close(self) -> None:
        """Close the backend."""
        self._backend.close()
