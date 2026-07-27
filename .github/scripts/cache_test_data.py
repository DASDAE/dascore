"""Populate the pooch cache with the default non-large DASCore test data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dascore.utils.downloader import (  # noqa: E402
    fetch,
    get_fetcher,
    get_registry_df,
)


def _prune_unregistered(registered: set[str]) -> None:
    """
    Delete cached files no longer in the registry.

    The cache is restored from an older key when the registry changes, so a
    file dropped from the registry would otherwise ride along into every
    later cache. Compares against the full registry, not the primed subset,
    so a large file that was legitimately fetched is kept.
    """
    root = Path(get_fetcher().path)
    if not root.is_dir():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.relative_to(root).as_posix() in registered:
            continue
        print(f"Pruning unregistered cache file: {path}")  # noqa
        path.unlink()


def main() -> None:
    """Fetch the default non-large registered test data into the local cache."""
    registry = get_registry_df(exclude_large=True)
    total = len(registry)
    print(f"Priming DASCore test-data cache with {total} files")  # noqa
    for index, name in enumerate(registry["name"], start=1):
        path = Path(fetch(name))
        print(f"[{index}/{total}] {name} -> {path}")  # noqa
    _prune_unregistered(set(get_registry_df()["name"]))
    print("Finished priming DASCore test-data cache")  # noqa


if __name__ == "__main__":
    main()
