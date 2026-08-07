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
import os
import tempfile
from contextlib import suppress
from pathlib import Path

import pandas as pd
from typing_extensions import Self

import dascore as dc
from dascore.compat import UPath
from dascore.config import config_attr
from dascore.constants import PROGRESS_LEVELS
from dascore.exceptions import InvalidIndexVersionError
from dascore.io.index.backend import get_backend, resolve_query
from dascore.io.index.ingest import (
    SourceRecord,
    hive_path_attrs,
    summaries_to_records,
)
from dascore.io.index.schema import SPOOL_HIDDEN_COLUMNS
from dascore.utils.misc import _iter_filesystem
from dascore.utils.paths import (
    coerce_to_local_path,
    directory_writable,
    requires_local_directory,
)


def _path_digest(path) -> str:
    """Stable per-path digest (hash() of a str/Path is per-process random).

    Uses the full sha256 so distinct paths never collide (a collision
    would make two read-only directories share one index file). Prefer
    os.fsencode so local filesystem paths (including non-UTF-8 filename
    bytes) digest exactly; fall back to a plain string encoding for inputs
    os.fsencode rejects, e.g. remote URL/UPath directories we may support
    later.
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
    Return the index path recorded for a data directory, or None.

    Each directory's mapping is its own small JSON file. A corrupt or
    unreadable entry simply reads as a miss; the next atomic write for
    that directory self-heals it. Reads never delete the entry, which
    would otherwise race a concurrent writer's repair. See #508.
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
    Record a data directory's external index location.

    The entry is written to a sibling temp file and swapped into place, so
    a concurrent reader never sees a half-written file. Different
    directories use different files, so concurrent writers to distinct
    directories cannot clobber one another.
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
        the data directory.
    """

    ext: str | None = None
    # cache dir holding per-data-directory index-location entries, used
    # when a data directory itself is not writable
    index_map_dir: Path = config_attr("directory_index_map_dir")

    def __init__(
        self,
        path: str | Path | UPath,
        index_path: str | Path | None = None,
    ):
        path = UPath(path).absolute() if isinstance(path, UPath) else Path(path)
        requires_local_directory(path, label="DBDirectoryIndexer")
        self.path = Path(coerce_to_local_path(path)).absolute()
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
        map_dir = self.index_map_dir
        if index_path:
            index_path = Path(index_path).absolute()
            # A custom location is the only case worth recording: it is not
            # otherwise rediscoverable. Read-only fallbacks are deterministic.
            _set_mapped_index_path(self.path, index_path, map_dir)
            return index_path
        expected = self.path / self._index_name
        with suppress(PermissionError):
            if expected.exists():
                return expected
        mapped = _get_mapped_index_path(self.path, map_dir)
        if mapped is not None and not self._is_legacy_or_foreign_index(mapped):
            # Index-map entries from older DASCore versions can point at
            # the retired PyTables (.h5) index; those are not usable and
            # a fresh SQLite index is built in their place.
            return mapped
        if not directory_writable(self.path):
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
            # Not Path(candidate): the generator's yield type includes UPath,
            # which is not os.PathLike unless it resolved to a local path.
            path = coerce_to_local_path(candidate)
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
        makes attaching metadata by renaming hive-style directories (or
        files) cheap on large archives. Renames that *remove* a hive key
        are excluded: restoring the file's own value for the dropped
        attr requires a rescan.
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

    def update(self, paths=None, progress: PROGRESS_LEVELS = "standard") -> Self:
        """
        Update the index: scan new/changed sources, drop removed ones.

        Change detection compares each source's stored (mtime_ns,
        size_bytes) against the filesystem — never a global watermark —
        and stale-source removal is folded in (the walk is the dominant
        cost; removal afterwards is nearly free). Directory-format scan
        units (e.g. XMLBinary) are rescanned whole when any member file
        changes. Renamed/moved sources are detected by their unchanged
        stats and rewritten in place, never rescanned (see _detect_moves).
        """
        files = self._walk()
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
        stale = [path for path in stored if path not in files]
        changed = [
            rel
            for rel, (mtime, size, _) in files.items()
            if stored.get(rel) != (mtime, size)
        ]
        # Moves apply to the unrestricted sets, like stale removal below:
        # the rows must track the filesystem even when a `paths` argument
        # narrows which sources get rescanned.
        moves = self._detect_moves(stale, stored, files)
        if moves:
            new_attrs = {old: hive_path_attrs(new) for old, new in moves.items()}
            self._backend.move_sources(moves, new_attrs)
            moved_targets = set(moves.values())
            stale = [rel for rel in stale if rel not in moves]
            changed = [rel for rel in changed if rel not in moved_targets]
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
