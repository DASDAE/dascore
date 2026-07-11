"""
Base indexer interface and index-location utilities.

The concrete directory indexer lives in `dascore.io.index.indexer`
(`DBDirectoryIndexer`); this module holds the abstract interface and the
machinery for tracking index locations when the data directory itself is
not writable (e.g. read-only archives).
"""

from __future__ import annotations

import abc
import json
import os
from contextlib import suppress
from functools import cache
from pathlib import Path

from typing_extensions import Self


@cache
def _get_index_map(cache_path) -> dict:
    """
    Get a dict of index locations.

    Note: this is purposefully mutable; handle with care.
    """
    path = Path(cache_path)
    out = {}
    successful_read = True
    if path.exists():
        try:
            with path.open("r") as fi:
                out = json.load(fi)
        # On rare occasions, the file can become corrupt. See #508.
        except (OSError, json.JSONDecodeError):
            successful_read = False
    if not isinstance(out, dict) or not successful_read:
        out = {}
        with suppress(FileNotFoundError, PermissionError):
            path.unlink(missing_ok=True)
    return out


def _update_index_map(updates, cache_path) -> dict:
    """Update index map to track new index."""
    data = _get_index_map(cache_path=cache_path)
    data.update(updates)
    Path(cache_path).parent.mkdir(exist_ok=True, parents=True)
    with open(cache_path, "w") as fi:
        json.dump(data, fi)
    return data


def _directory_writable(path):
    """Return True if the directory is writable else False."""
    name = "._dascore_write_test_delete_me"
    path = Path(path) / name
    path.parent.mkdir(exist_ok=True, parents=True)
    try:
        open(path, "w").close()
    except (PermissionError, IsADirectoryError):
        return False
    else:
        os.remove(path)
    return True


class AbstractIndexer:
    """
    A base class for indexers.

    This is primarily here for a place-holder.
    """

    path: Path

    @abc.abstractmethod
    def update(self) -> Self:
        """
        Updates the contents of the Indexer.

        Resets any previous selection.
        """

    def ensure_updated(self) -> bool:
        """
        Run the initial update if the index was never populated.

        Return True when an update actually ran. Indexers which track
        their initial-population state override this; by default nothing
        happens.
        """
        return False
